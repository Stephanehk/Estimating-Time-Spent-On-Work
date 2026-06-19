import json
import sys
from pathlib import Path

import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from data_formatting.exposure_loading_utils import (
    SIMULATION_BASED_EXPOSURE_PATH,
    EXPOSURE_COL,
    MIN_OCCUPATION_EXPOSURE_COVERAGE,
    TASK_STATEMENTS_PATH,
    load_simulation_based_exposure_mapped_to_ratings,
)
from analysis.figure4_exposure_share_cdf import (
    RUBRIC_BASED_EXPOSURE_PATH,
    _task_id_to_str,
    load_rubric_based_exposure_excel,
)
from utils.constants import HOURS_PER_DAY

ONET_VERSION = "30.2"
EXTRA_DETAILS = f"gpt-5.2_ONET_{ONET_VERSION}_use_cps_constants_prob_thresh=0.7"

TASK_RATINGS_PATH = Path("data/onet_data/Task Ratings.xlsx")
OCCUPATION_RESULT_PATH_TEMPLATE = (
    "data/generated_data/occupation_time_per_task_{soc_code}"
    f"{EXTRA_DETAILS}_chosen_max_span_weigths.json"
)
# The main project output: per-(occupation, task) time-share estimates, written to the repo root.
OUTPUT_PATH = _REPO_ROOT / "task_time_share_estimates.xlsx"
SOC_DONES_PATH = _REPO_ROOT / "data" / "generated_data" / f"soc_code_dones_{EXTRA_DETAILS}.txt"

# Simulation-based continuous score and binary regimes (same scale as load_simulation_based_exposure_mapped_to_ratings).
SIMULATION_BASED_MOD_HIGH_THRESHOLD = 0.25
SIMULATION_BASED_HIGH_ONLY_THRESHOLD = 0.50


def _normalize_task_id(task_id):
    return int(float(task_id))


def _load_occupation_result(occupation_code):
    result_path = Path(OCCUPATION_RESULT_PATH_TEMPLATE.format(soc_code=occupation_code))
    if not result_path.exists():
        return None
    with result_path.open("r") as f:
        return json.load(f)


def _build_task_rows_from_task_ratings(ratings_path: Path) -> list[dict]:
    """Build one row per (occupation, task) from Task Ratings; Importance is NaN if missing."""
    df = pd.read_excel(ratings_path)
    assert "O*NET-SOC Code" in df.columns and "Task ID" in df.columns
    assert "Task" in df.columns and "Title" in df.columns and "Scale ID" in df.columns
    assert "Data Value" in df.columns

    rows = []
    for (soc_code, task_id), group in df.groupby(["O*NET-SOC Code", "Task ID"], dropna=False):
        if pd.isna(soc_code) or pd.isna(task_id):
            continue
        soc_code = str(soc_code)
        task_id_norm = _normalize_task_id(task_id)
        task_name = group["Task"].iloc[0]
        occupation_title = group["Title"].iloc[0]
        imp_rows = group[group["Scale ID"] == "IM"]
        if imp_rows.empty or imp_rows["Data Value"].isna().all():
            importance = pd.NA
        else:
            importance = float(imp_rows["Data Value"].dropna().iloc[0])
        rows.append({
            "Occupation ID": soc_code,
            "Occupation Name": occupation_title,
            "Task ID": task_id_norm,
            "Task Name": task_name,
            "Mean Importance": importance,
        })
    return rows


def _build_rubric_based_lookup(full_task_df: pd.DataFrame) -> dict[tuple[str, int], float]:
    """(ONETSOCCode, TaskID) -> auto_hl_bi. Only includes rows with non-null TaskID, auto_hl_bi, TaskType."""
    sub = full_task_df[["ONETSOCCode", "TaskID", "auto_hl_bi", "TaskType"]].dropna(
        subset=["TaskID", "auto_hl_bi", "TaskType"]
    )
    lookup = {}
    for _, row in sub.iterrows():
        key = (str(row["ONETSOCCode"]), _normalize_task_id(row["TaskID"]))
        lookup[key] = float(row["auto_hl_bi"])
    return lookup


def _build_rubric_based_task_type_lookup(full_task_df: pd.DataFrame) -> dict[tuple[str, int], str]:
    """(ONETSOCCode, TaskID) -> TaskType (Core/Supplemental)."""
    sub = full_task_df[["ONETSOCCode", "TaskID", "TaskType"]].dropna(
        subset=["TaskID", "TaskType"]
    )
    lookup = {}
    for _, row in sub.iterrows():
        key = (str(row["ONETSOCCode"]), _normalize_task_id(row["TaskID"]))
        lookup[key] = str(row["TaskType"])
    return lookup


def _build_hm_l_automation_lookups(rubric_based_path):
    """
    (ONETSOCCode, TaskID) -> 0.0/1.0 for Rubric-based task labels from automation_hl.

    Moderate + high: T2, T3, T4. High only: T3, T4.
    Assumes rubric_based_path has ONETSOCCode, TaskID, automation_hl.
    Deduplicates (ONETSOCCode, TaskID) like load_rubric_based_exposure_excel (first row wins).
    """
    df = pd.read_excel(rubric_based_path)
    assert {"ONETSOCCode", "TaskID", "automation_hl"}.issubset(df.columns), (
        f"Rubric-based file missing columns; got {list(df.columns)}"
    )
    sub = df[["ONETSOCCode", "TaskID", "automation_hl"]].dropna(subset=["TaskID"])
    sub = sub.drop_duplicates(subset=["ONETSOCCode", "TaskID"], keep="first")
    mod_high = {}
    high_only = {}
    for _, row in sub.iterrows():
        key = (str(row["ONETSOCCode"]).strip(), _normalize_task_id(row["TaskID"]))
        assert key not in mod_high
        ah = str(row["automation_hl"]).strip().upper()
        mod_high[key] = 1.0 if ah in {"T2", "T3", "T4"} else 0.0
        high_only[key] = 1.0 if ah in {"T3", "T4"} else 0.0
    return mod_high, high_only


def _build_simulation_based_regime_lookups(simulation_based_df, exposure_col):
    """
    From Simulation-based rows retained by load_simulation_based_exposure_mapped_to_ratings, build
    (soc, task_id) -> 0.0/1.0 for moderate+high (>= SIMULATION_BASED_MOD_HIGH_THRESHOLD) and
    high-only (>= SIMULATION_BASED_HIGH_ONLY_THRESHOLD) on the continuous score.
    """
    mod_high = {}
    high_only = {}
    for _, row in simulation_based_df.iterrows():
        key = (str(row["ONETSOCCode"]).strip(), float(row["TaskID"]))
        v = float(row[exposure_col])
        mod_high[key] = 1.0 if v >= SIMULATION_BASED_MOD_HIGH_THRESHOLD else 0.0
        high_only[key] = 1.0 if v >= SIMULATION_BASED_HIGH_ONLY_THRESHOLD else 0.0
    return mod_high, high_only


def _build_simulation_based_raw_median_lookup(simulation_based_df, exposure_col):
    """(ONETSOCCode, TaskID) -> raw_time_savings_median."""
    out = {}
    for _, row in simulation_based_df.iterrows():
        key = (str(row["ONETSOCCode"]).strip(), float(row["TaskID"]))
        out[key] = float(row[exposure_col])
    return out


def _build_expertise_lookup(rubric_based_path: Path) -> dict[tuple[str, int], int]:
    """
    (ONETSOCCode, TaskID) -> expertise_without (1–5) from rubric_based_exposure.xlsx.
    Assumes column expertise_without as in predict_exposure_and_expertise.py.
    """
    df = pd.read_excel(rubric_based_path)
    assert "ONETSOCCode" in df.columns and "TaskID" in df.columns
    assert "expertise_without" in df.columns
    sub = df[["ONETSOCCode", "TaskID", "expertise_without"]].dropna(
        subset=["TaskID", "expertise_without"]
    )
    lookup = {}
    for _, row in sub.iterrows():
        key = (str(row["ONETSOCCode"]), _normalize_task_id(row["TaskID"]))
        ev = int(round(float(row["expertise_without"])))
        assert 1 <= ev <= 5
        assert key not in lookup or lookup[key] == ev
        lookup[key] = ev
    return lookup


def _load_soc_codes_done():
    assert SOC_DONES_PATH.exists(), f"Missing SOC done list: {SOC_DONES_PATH}"
    with open(SOC_DONES_PATH, "r") as f:
        codes = [line.strip() for line in f.readlines() if line.strip()]
    assert len(codes) > 0, f"No SOC codes in {SOC_DONES_PATH}"
    return codes


def main():
    assert TASK_RATINGS_PATH.exists(), f"Task Ratings not found: {TASK_RATINGS_PATH}"
    assert RUBRIC_BASED_EXPOSURE_PATH.exists(), f"Rubric-based exposure data not found: {RUBRIC_BASED_EXPOSURE_PATH}"
    assert SIMULATION_BASED_EXPOSURE_PATH.exists(), f"Simulation-based exposure file not found: {SIMULATION_BASED_EXPOSURE_PATH}"
    task_rows = _build_task_rows_from_task_ratings(TASK_RATINGS_PATH)

    full_task_df = load_rubric_based_exposure_excel(
        RUBRIC_BASED_EXPOSURE_PATH, TASK_STATEMENTS_PATH
    )
    rubric_based_lookup = _build_rubric_based_lookup(full_task_df)
    rubric_based_task_type_lookup = _build_rubric_based_task_type_lookup(full_task_df)
    hm_l_mod_high_lookup, hm_l_high_only_lookup = _build_hm_l_automation_lookups(
        RUBRIC_BASED_EXPOSURE_PATH
    )
    expertise_lookup = _build_expertise_lookup(RUBRIC_BASED_EXPOSURE_PATH)

    simulation_based_df, _ = load_simulation_based_exposure_mapped_to_ratings(
        SIMULATION_BASED_EXPOSURE_PATH, TASK_RATINGS_PATH, TASK_STATEMENTS_PATH
    )
    simulation_based_raw_lookup = _build_simulation_based_raw_median_lookup(simulation_based_df, EXPOSURE_COL)
    simulation_based_lookup_mod_high, simulation_based_lookup_high_only = _build_simulation_based_regime_lookups(
        simulation_based_df, EXPOSURE_COL
    )
    simulation_based_task_type_lookup = _build_rubric_based_task_type_lookup(simulation_based_df)
    simulation_based_task_keys = set(simulation_based_raw_lookup.keys())
    assert len(simulation_based_task_keys) > 0, "No Simulation-based task keys remain after filtering."
    simulation_based_occupation_codes = {occ for occ, _ in simulation_based_task_keys}

    soc_codes_done = _load_soc_codes_done()
    soc_codes_set = set(soc_codes_done)
    retained_soc_codes = soc_codes_set & simulation_based_occupation_codes
    assert len(retained_soc_codes) > 0, (
        "No O*NET 30.2 SOC-done occupations remain after Simulation-based coverage filtering."
    )
    print(
        "Retained O*NET 30.2 occupations with at least "
        f"{MIN_OCCUPATION_EXPOSURE_COVERAGE:.0%} Simulation-based exposure-score coverage: "
        f"{len(retained_soc_codes)}"
    )


    output_rows = []
    last_occupation_code = None
    occupation_result = None
    warned_no_time_json_occ = set()

    for tr in task_rows:
        occupation_code = tr["Occupation ID"]
        task_id = tr["Task ID"]
        if occupation_code not in retained_soc_codes:
            continue
        if (occupation_code, task_id) not in simulation_based_task_keys:
            continue

        if occupation_code != last_occupation_code:
            last_occupation_code = occupation_code
            occupation_result = _load_occupation_result(occupation_code)
            if occupation_result is None and occupation_code not in warned_no_time_json_occ:
                warned_no_time_json_occ.add(occupation_code)
                print(
                    f"Warning: no time estimates JSON for occupation {occupation_code}; "
                    "per-task time spent left blank."
                )

        simulation_based_median = simulation_based_raw_lookup[(occupation_code, task_id)]
        simulation_based_class_mod_high = int(simulation_based_lookup_mod_high[(occupation_code, task_id)])
        simulation_based_class_high_only = int(simulation_based_lookup_high_only[(occupation_code, task_id)])
        hm_l_mod_high = hm_l_mod_high_lookup.get((occupation_code, task_id), pd.NA)
        hm_l_high_only = hm_l_high_only_lookup.get((occupation_code, task_id), pd.NA)
        expertise_class = expertise_lookup.get((occupation_code, task_id), pd.NA)

        time_spent_per_day = pd.NA
        if occupation_result is not None:
            time_per_task = occupation_result.get("Time per task", {})
            expected_freq_per_task = occupation_result.get("Expected freq per task", {})
            task_code_str = _task_id_to_str(task_id)
            if task_code_str in time_per_task and task_code_str in expected_freq_per_task:
                time_spent_per_day = float(time_per_task[task_code_str]) * float(
                    expected_freq_per_task[task_code_str]
                )

        output_rows.append({
            "Task ID": task_id,
            "Task Name": tr["Task Name"],
            "Occupation ID": occupation_code,
            "Occupation Name": tr["Occupation Name"],
            "Simulation-based exposure estimate": simulation_based_median,
            "Simulation-based - Moderate + High exposure classification": simulation_based_class_mod_high,
            "Simulation-based - High exposure classification only": simulation_based_class_high_only,
            "Rubric-based - Moderate + High exposure classification": (
                int(hm_l_mod_high) if not pd.isna(hm_l_mod_high) else pd.NA
            ),
            "Rubric-based - High exposure classification": (
                int(hm_l_high_only) if not pd.isna(hm_l_high_only) else pd.NA
            ),
            "Expertise classification (1-5)": expertise_class,
            "Time spent per day estimate": time_spent_per_day,
            "Mean Importance": tr["Mean Importance"],
        })

    assert len(output_rows) > 0, "No task rows remain after filtering to Simulation-based's task universe."
    output_df = pd.DataFrame(output_rows)
    imp_sum_per_occ = output_df.groupby("Occupation ID")["Mean Importance"].transform(
        lambda x: x.dropna().astype(float).sum()
    )
    has_imp = output_df["Mean Importance"].notna() & (imp_sum_per_occ > 0)
    output_df["Mean Importance Renormalized To Be Time Shares"] = pd.NA
    output_df.loc[has_imp, "Mean Importance Renormalized To Be Time Shares"] = (
        HOURS_PER_DAY
        * output_df.loc[has_imp, "Mean Importance"].astype(float)
        / imp_sum_per_occ.loc[has_imp]
    )
    for _, g in output_df.groupby("Occupation ID"):
        if g["Mean Importance"].dropna().empty:
            continue
        total = g["Mean Importance Renormalized To Be Time Shares"].dropna().astype(float).sum()
        assert abs(float(total) - HOURS_PER_DAY) < 1e-6, total
    output_df = output_df.sort_values(["Occupation ID", "Task ID"]).reset_index(drop=True)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_excel(OUTPUT_PATH, index=False)
    print(f"Saved {len(output_df)} rows to {OUTPUT_PATH}")
if __name__ == "__main__":
    main()
