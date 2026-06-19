import json
import sys
from pathlib import Path

import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from data_formatting.exposure_loading_utils import (
    ALEX_EXPOSURE_PATH,
    EXPOSURE_COL,
    MIN_OCCUPATION_EXPOSURE_COVERAGE,
    TASK_STATEMENTS_PATH,
    load_alex_exposure_mapped_to_ratings,
)
from analysis.figure4_exposure_share_cdf import (
    REPLICATED_EXPOSURE_PATH,
    _task_id_to_str,
    load_replicated_exposure_excel,
)
from utils.constants import HOURS_PER_DAY

ONET_VERSION = "30.2"
EXTRA_DETAILS = f"gpt-5.2_ONET_{ONET_VERSION}_use_cps_constants_prob_thresh=0.7"

TASK_RATINGS_PATH = Path("data/Task Ratings.xlsx")
OCCUPATION_RESULT_PATH_TEMPLATE = (
    "data/generated_data/occupation_time_per_task_{soc_code}"
    f"{EXTRA_DETAILS}_chosen_max_span_weigths.json"
)
OUTPUT_PATH = Path("data/analysis_results/current_per_task_estimates.xlsx")
OUTPUT_PATH_TASK_MISSING_DATA_KEPT = Path(
    "data/analysis_results/current_per_task_estimates_missing_data_kept.xlsx"
)
WAGE_DATA_PATH = Path("data/national_M2024_dl.xlsx")
OCCUPATION_OUTPUT_PATH = Path(
    "data/analysis_results/current_per_occupation_estimates.xlsx"
)
OCCUPATION_OUTPUT_PATH_MISSING_DATA_KEPT = Path(
    "data/analysis_results/current_per_occupation_estimates_missing_data_kept.xlsx"
)
SOC_DONES_PATH = _REPO_ROOT / "data" / f"soc_code_dones_{EXTRA_DETAILS}.txt"

# Alex continuous score and binary regimes (same scale as load_alex_exposure_mapped_to_ratings).
ALEX_MOD_HIGH_THRESHOLD = 0.25
ALEX_HIGH_ONLY_THRESHOLD = 0.50


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


def _build_hosseini_lookup(full_task_df: pd.DataFrame) -> dict[tuple[str, int], float]:
    """(ONETSOCCode, TaskID) -> auto_hl_bi. Only includes rows with non-null TaskID, auto_hl_bi, TaskType."""
    sub = full_task_df[["ONETSOCCode", "TaskID", "auto_hl_bi", "TaskType"]].dropna(
        subset=["TaskID", "auto_hl_bi", "TaskType"]
    )
    lookup = {}
    for _, row in sub.iterrows():
        key = (str(row["ONETSOCCode"]), _normalize_task_id(row["TaskID"]))
        lookup[key] = float(row["auto_hl_bi"])
    return lookup


def _build_hosseini_task_type_lookup(full_task_df: pd.DataFrame) -> dict[tuple[str, int], str]:
    """(ONETSOCCode, TaskID) -> TaskType (Core/Supplemental)."""
    sub = full_task_df[["ONETSOCCode", "TaskID", "TaskType"]].dropna(
        subset=["TaskID", "TaskType"]
    )
    lookup = {}
    for _, row in sub.iterrows():
        key = (str(row["ONETSOCCode"]), _normalize_task_id(row["TaskID"]))
        lookup[key] = str(row["TaskType"])
    return lookup


def _build_hm_l_automation_lookups(replicated_path):
    """
    (ONETSOCCode, TaskID) -> 0.0/1.0 for HM&L task labels from automation_hl.

    Moderate + high: T2, T3, T4. High only: T3, T4.
    Assumes replicated_path has ONETSOCCode, TaskID, automation_hl.
    Deduplicates (ONETSOCCode, TaskID) like load_replicated_exposure_excel (first row wins).
    """
    df = pd.read_excel(replicated_path)
    assert {"ONETSOCCode", "TaskID", "automation_hl"}.issubset(df.columns), (
        f"Replicated file missing columns; got {list(df.columns)}"
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


def _build_alex_regime_lookups(alex_df, exposure_col):
    """
    From Alex rows retained by load_alex_exposure_mapped_to_ratings, build
    (soc, task_id) -> 0.0/1.0 for moderate+high (>= ALEX_MOD_HIGH_THRESHOLD) and
    high-only (>= ALEX_HIGH_ONLY_THRESHOLD) on the continuous score.
    """
    mod_high = {}
    high_only = {}
    for _, row in alex_df.iterrows():
        key = (str(row["ONETSOCCode"]).strip(), float(row["TaskID"]))
        v = float(row[exposure_col])
        mod_high[key] = 1.0 if v >= ALEX_MOD_HIGH_THRESHOLD else 0.0
        high_only[key] = 1.0 if v >= ALEX_HIGH_ONLY_THRESHOLD else 0.0
    return mod_high, high_only


def _build_alex_raw_median_lookup(alex_df, exposure_col):
    """(ONETSOCCode, TaskID) -> raw_time_savings_median."""
    out = {}
    for _, row in alex_df.iterrows():
        key = (str(row["ONETSOCCode"]).strip(), float(row["TaskID"]))
        out[key] = float(row[exposure_col])
    return out


def _build_expertise_lookup(replicated_path: Path) -> dict[tuple[str, int], int]:
    """
    (ONETSOCCode, TaskID) -> expertise_without (1–5) from replicated_task_dataset.xlsx.
    Assumes column expertise_without as in predict_exposure_and_expertise.py.
    """
    df = pd.read_excel(replicated_path)
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


def _triple_exposure_shares(
    occupation_code,
    common_task_ids,
    task_id_to_importance,
    exposure_lookup,
    task_type_lookup,
    occupation_result_occ,
):
    """
    Task-type-weighted, time-weighted, and importance-weighted occupation exposure shares.
    exposure_lookup maps (soc, task_id) -> 0/1 (or float); task_type_lookup -> Core/Supplemental.

    Returns (share, time_weighted_share, importance_weighted_share, time_coverage_incomplete).
    time_coverage_incomplete is True iff occupation_result_occ is not None and at least one
    task in common_task_ids is missing from Time per task or Expected freq per task.

    JSON dict keys use the same normalization as run_weighted_exposure_analysis
    (figure4_exposure_share_cdf._task_id_to_str), not raw str(task_id).
    """
    weight_sum = 0.0
    weighted_exposed_sum = 0.0
    for task_id in common_task_ids:
        task_type = task_type_lookup.get((occupation_code, task_id))
        assert task_type in {"Core", "Supplemental"}, (
            f"Missing or invalid TaskType for ({occupation_code}, {task_id}): {task_type!r}"
        )
        assert (occupation_code, task_id) in exposure_lookup, (
            f"Missing exposure score for ({occupation_code}, {task_id})"
        )
        w = 1.0 if task_type == "Core" else 0.5
        exp = exposure_lookup[(occupation_code, task_id)]
        weight_sum += w
        weighted_exposed_sum += w * float(exp)
    share = (
        float(weighted_exposed_sum / weight_sum)
        if weight_sum > 0
        else pd.NA
    )

    time_coverage_incomplete = False
    if occupation_result_occ is None:
        time_weighted_share = pd.NA
    else:
        time_per_task = occupation_result_occ.get("Time per task", {})
        expected_freq_per_task = occupation_result_occ.get("Expected freq per task", {})
        exposed_sum = 0.0
        total_weight = 0.0
        for task_id in common_task_ids:
            task_code_str = _task_id_to_str(task_id)
            if task_code_str not in time_per_task or task_code_str not in expected_freq_per_task:
                time_coverage_incomplete = True
                continue
            time_weight = float(time_per_task[task_code_str]) * float(
                expected_freq_per_task[task_code_str]
            )
            assert (occupation_code, task_id) in exposure_lookup, (
                f"Missing exposure score for ({occupation_code}, {task_id})"
            )
            is_exposed = exposure_lookup[(occupation_code, task_id)]
            exposed_sum += time_weight * float(is_exposed)
            total_weight += time_weight
        time_weighted_share = (
            float(exposed_sum / total_weight)
            if total_weight > 0
            else pd.NA
        )

    importance_weighted_exposed = 0.0
    total_importance = 0.0
    for task_id in common_task_ids:
        imp = task_id_to_importance.get(task_id)
        if imp is None or pd.isna(imp):
            continue
        imp = float(imp)
        assert (occupation_code, task_id) in exposure_lookup, (
            f"Missing exposure score for ({occupation_code}, {task_id})"
        )
        exposure = exposure_lookup[(occupation_code, task_id)]
        importance_weighted_exposed += imp * float(exposure)
        total_importance += imp
    importance_weighted_share = (
        float(importance_weighted_exposed / total_importance)
        if total_importance > 0
        else pd.NA
    )

    return share, time_weighted_share, importance_weighted_share, time_coverage_incomplete


def _load_soc_codes_done():
    assert SOC_DONES_PATH.exists(), f"Missing SOC done list: {SOC_DONES_PATH}"
    with open(SOC_DONES_PATH, "r") as f:
        codes = [line.strip() for line in f.readlines() if line.strip()]
    assert len(codes) > 0, f"No SOC codes in {SOC_DONES_PATH}"
    return codes


def _collect_occupation_summary_rows(
    occ_to_task_rows,
    wage_df,
    alex_lookup_mod_high,
    alex_lookup_high_only,
    alex_task_type_lookup,
    hm_l_mod_high_lookup,
    hm_l_high_only_lookup,
    hosseini_lookup,
    hosseini_task_type_lookup,
    warned_no_time_json_occ,
    require_tr_task_ids_equal_alex,
):
    """
    Build one summary row per occupation in occ_to_task_rows.

    When require_tr_task_ids_equal_alex is True, Task Ratings rows must equal Alex's task set.
    When False, Task Ratings rows may be a strict superset (Alex tasks must still be a subset).
    """
    occupation_output_rows = []
    occ_incomplete_time = set()
    n_occ_hm_l_task_mismatch = 0
    n_occ_alex_task_mismatch = 0

    for occupation_code in sorted(occ_to_task_rows.keys()):
        rows_for_occ = occ_to_task_rows[occupation_code]
        occupation_name = rows_for_occ[0]["Occupation Name"]
        tr_task_ids = {r["Task ID"] for r in rows_for_occ}
        task_id_to_importance = {r["Task ID"]: r["Mean Importance"] for r in rows_for_occ}
        alex_task_ids = {tid for (occ, tid) in alex_lookup_mod_high if occ == occupation_code}
        if require_tr_task_ids_equal_alex:
            assert tr_task_ids == alex_task_ids, (
                f"Task rows for {occupation_code} must match Alex Wan's filtered task universe."
            )
        else:
            assert alex_task_ids.issubset(tr_task_ids), (
                f"Alex task IDs for {occupation_code} must be contained in Task Ratings tasks."
            )
        hosseini_task_ids = {
            tid for (occ, tid) in hosseini_lookup if occ == occupation_code
        }
        common_h = alex_task_ids
        n_missing_from_hosseini = len(alex_task_ids - hosseini_task_ids)

        common_a = alex_task_ids
        n_alex_missing_from_ours = len(alex_task_ids - tr_task_ids)

        can_hm_l = (
            bool(common_h)
            and n_missing_from_hosseini == 0
        )
        can_alex = (
            bool(common_a)
            and n_alex_missing_from_ours == 0
        )

        if not can_hm_l:
            n_occ_hm_l_task_mismatch += 1
        if not can_alex:
            n_occ_alex_task_mismatch += 1

        occ_json = _load_occupation_result(occupation_code)
        if occ_json is None and (can_hm_l or can_alex):
            if occupation_code not in warned_no_time_json_occ:
                warned_no_time_json_occ.add(occupation_code)
                print(
                    f"Warning: no time estimates JSON for occupation {occupation_code}; "
                    "time-weighted occupation exposure shares left blank."
                )

        if not can_hm_l:
            hm_l_share_mod = pd.NA
            hm_l_time_mod = pd.NA
            hm_l_imp_mod = pd.NA
            hm_l_share_high = pd.NA
            hm_l_time_high = pd.NA
            hm_l_imp_high = pd.NA
        else:
            hm_l_share_mod, hm_l_time_mod, hm_l_imp_mod, inc_h_mod = _triple_exposure_shares(
                occupation_code,
                common_h,
                task_id_to_importance,
                hm_l_mod_high_lookup,
                hosseini_task_type_lookup,
                occ_json,
            )
            hm_l_share_high, hm_l_time_high, hm_l_imp_high, inc_h_hi = _triple_exposure_shares(
                occupation_code,
                common_h,
                task_id_to_importance,
                hm_l_high_only_lookup,
                hosseini_task_type_lookup,
                occ_json,
            )
            assert inc_h_mod == inc_h_hi
            if inc_h_mod:
                occ_incomplete_time.add(occupation_code)

        if not can_alex:
            alex_share_mod = pd.NA
            alex_time_mod = pd.NA
            alex_imp_mod = pd.NA
            alex_share_high = pd.NA
            alex_time_high = pd.NA
            alex_imp_high = pd.NA
        else:
            alex_share_mod, alex_time_mod, alex_imp_mod, inc_a_mod = _triple_exposure_shares(
                occupation_code,
                common_a,
                task_id_to_importance,
                alex_lookup_mod_high,
                alex_task_type_lookup,
                occ_json,
            )
            alex_share_high, alex_time_high, alex_imp_high, inc_a_hi = _triple_exposure_shares(
                occupation_code,
                common_a,
                task_id_to_importance,
                alex_lookup_high_only,
                alex_task_type_lookup,
                occ_json,
            )
            assert inc_a_mod == inc_a_hi
            if inc_a_mod:
                occ_incomplete_time.add(occupation_code)

        wage_row = wage_df[wage_df["OCC_CODE"] == occupation_code.split(".")[0]]
        if wage_row.empty:
            wage = pd.NA
            employment = pd.NA
        else:
            wage = wage_row["A_MEAN"].values[0]
            employment = wage_row["TOT_EMP"].values[0]

        occupation_output_rows.append({
            "Occupation ID": occupation_code,
            "Occupation Name": occupation_name,
            "Wage": wage,
            "Employment": employment,
            "Alex Wan, Occupation Exposure Share (Moderate + High exposure)": alex_share_mod,
            "Alex Wan, Occupation Exposure Share (High exposure)": alex_share_high,
            "HM&L, Occupation Exposure Share (Moderate + High exposure)": hm_l_share_mod,
            "HM&L, Occupation Exposure Share (High exposure)": hm_l_share_high,
            "Alex Wan, Occupation Exposure Time Share (Moderate + High exposure)": alex_time_mod,
            "Alex Wan, Occupation Exposure Time Share (High exposure)": alex_time_high,
            "HM&L, Occupation Exposure Time Share (Moderate + High exposure)": hm_l_time_mod,
            "HM&L, Occupation Exposure Time Share (High exposure)": hm_l_time_high,
            "Alex Wan, Occupation Exposure Importance Weighted (Moderate + High exposure)": (
                alex_imp_mod
            ),
            "Alex Wan, Occupation Exposure Importance Weighted (High exposure)": alex_imp_high,
            "HM&L, Occupation Exposure Importance Weighted (Moderate + High exposure)": (
                hm_l_imp_mod
            ),
            "HM&L, Occupation Exposure Importance Weighted (High exposure)": hm_l_imp_high,
        })

    return (
        occupation_output_rows,
        occ_incomplete_time,
        n_occ_hm_l_task_mismatch,
        n_occ_alex_task_mismatch,
    )


def main():
    assert TASK_RATINGS_PATH.exists(), f"Task Ratings not found: {TASK_RATINGS_PATH}"
    assert REPLICATED_EXPOSURE_PATH.exists(), f"Replicated HM&L data not found: {REPLICATED_EXPOSURE_PATH}"
    assert ALEX_EXPOSURE_PATH.exists(), f"Alex Wan exposure file not found: {ALEX_EXPOSURE_PATH}"
    task_rows = _build_task_rows_from_task_ratings(TASK_RATINGS_PATH)

    full_task_df = load_replicated_exposure_excel(
        REPLICATED_EXPOSURE_PATH, TASK_STATEMENTS_PATH
    )
    hosseini_lookup = _build_hosseini_lookup(full_task_df)
    hosseini_task_type_lookup = _build_hosseini_task_type_lookup(full_task_df)
    hm_l_mod_high_lookup, hm_l_high_only_lookup = _build_hm_l_automation_lookups(
        REPLICATED_EXPOSURE_PATH
    )
    expertise_lookup = _build_expertise_lookup(REPLICATED_EXPOSURE_PATH)

    alex_df, _ = load_alex_exposure_mapped_to_ratings(
        ALEX_EXPOSURE_PATH, TASK_RATINGS_PATH, TASK_STATEMENTS_PATH
    )
    alex_raw_lookup = _build_alex_raw_median_lookup(alex_df, EXPOSURE_COL)
    alex_lookup_mod_high, alex_lookup_high_only = _build_alex_regime_lookups(
        alex_df, EXPOSURE_COL
    )
    alex_task_type_lookup = _build_hosseini_task_type_lookup(alex_df)
    alex_task_keys = set(alex_raw_lookup.keys())
    assert len(alex_task_keys) > 0, "No Alex Wan task keys remain after filtering."
    alex_occupation_codes = {occ for occ, _ in alex_task_keys}

    soc_codes_done = _load_soc_codes_done()
    soc_codes_set = set(soc_codes_done)
    retained_soc_codes = soc_codes_set & alex_occupation_codes
    assert len(retained_soc_codes) > 0, (
        "No O*NET 30.2 SOC-done occupations remain after Alex coverage filtering."
    )
    print(
        "Retained O*NET 30.2 occupations with at least "
        f"{MIN_OCCUPATION_EXPOSURE_COVERAGE:.0%} Alex exposure-score coverage: "
        f"{len(retained_soc_codes)}"
    )

    wage_df = pd.read_excel(WAGE_DATA_PATH)
    assert "OCC_CODE" in wage_df.columns
    assert "A_MEAN" in wage_df.columns
    assert "TOT_EMP" in wage_df.columns

    output_rows = []
    last_occupation_code = None
    occupation_result = None
    warned_no_time_json_occ = set()

    for tr in task_rows:
        occupation_code = tr["Occupation ID"]
        task_id = tr["Task ID"]
        if occupation_code not in retained_soc_codes:
            continue
        if (occupation_code, task_id) not in alex_task_keys:
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

        alex_median = alex_raw_lookup[(occupation_code, task_id)]
        alex_class_mod_high = int(alex_lookup_mod_high[(occupation_code, task_id)])
        alex_class_high_only = int(alex_lookup_high_only[(occupation_code, task_id)])
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
            "Alex Wan's exposure estimate": alex_median,
            "Alex Wan - Moderate + High exposure classification": alex_class_mod_high,
            "Alex Wan - High exposure classification only": alex_class_high_only,
            "HM&L - Moderate + High exposure classification": (
                int(hm_l_mod_high) if not pd.isna(hm_l_mod_high) else pd.NA
            ),
            "HM&L - High exposure classification": (
                int(hm_l_high_only) if not pd.isna(hm_l_high_only) else pd.NA
            ),
            "Expertise classification (1-5)": expertise_class,
            "Time spent per day estimate": time_spent_per_day,
            "Mean Importance": tr["Mean Importance"],
        })

    assert len(output_rows) > 0, "No task rows remain after filtering to Alex Wan's task universe."
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

    # Task Ratings × SOC-dones: include tasks without Alex scores (NaN) for DEL_viz / Connacher alignment.
    output_rows_mk = []
    last_occupation_code_mk = None
    occupation_result_mk = None
    for tr in task_rows:
        occupation_code = tr["Occupation ID"]
        task_id = tr["Task ID"]
        if occupation_code not in retained_soc_codes:
            continue

        if occupation_code != last_occupation_code_mk:
            last_occupation_code_mk = occupation_code
            occupation_result_mk = _load_occupation_result(occupation_code)
            if occupation_result_mk is None and occupation_code not in warned_no_time_json_occ:
                warned_no_time_json_occ.add(occupation_code)
                print(
                    f"Warning: no time estimates JSON for occupation {occupation_code}; "
                    "per-task time spent left blank."
                )

        if (occupation_code, task_id) in alex_task_keys:
            alex_median = alex_raw_lookup[(occupation_code, task_id)]
            alex_class_mod_high = int(alex_lookup_mod_high[(occupation_code, task_id)])
            alex_class_high_only = int(alex_lookup_high_only[(occupation_code, task_id)])
        else:
            alex_median = pd.NA
            alex_class_mod_high = pd.NA
            alex_class_high_only = pd.NA
        hm_l_mod_high = hm_l_mod_high_lookup.get((occupation_code, task_id), pd.NA)
        hm_l_high_only = hm_l_high_only_lookup.get((occupation_code, task_id), pd.NA)
        expertise_class = expertise_lookup.get((occupation_code, task_id), pd.NA)

        time_spent_per_day = pd.NA
        if occupation_result_mk is not None:
            time_per_task = occupation_result_mk.get("Time per task", {})
            expected_freq_per_task = occupation_result_mk.get("Expected freq per task", {})
            task_code_str = _task_id_to_str(task_id)
            if task_code_str in time_per_task and task_code_str in expected_freq_per_task:
                time_spent_per_day = float(time_per_task[task_code_str]) * float(
                    expected_freq_per_task[task_code_str]
                )

        output_rows_mk.append({
            "Task ID": task_id,
            "Task Name": tr["Task Name"],
            "Occupation ID": occupation_code,
            "Occupation Name": tr["Occupation Name"],
            "Alex Wan's exposure estimate": alex_median,
            "Alex Wan - Moderate + High exposure classification": alex_class_mod_high,
            "Alex Wan - High exposure classification only": alex_class_high_only,
            "HM&L - Moderate + High exposure classification": (
                int(hm_l_mod_high) if not pd.isna(hm_l_mod_high) else pd.NA
            ),
            "HM&L - High exposure classification": (
                int(hm_l_high_only) if not pd.isna(hm_l_high_only) else pd.NA
            ),
            "Expertise classification (1-5)": expertise_class,
            "Time spent per day estimate": time_spent_per_day,
            "Mean Importance": tr["Mean Importance"],
        })

    assert len(output_rows_mk) > 0, "No task rows for SOC-dones missing-data-kept export."
    output_df_mk = pd.DataFrame(output_rows_mk)
    imp_sum_per_occ_mk = output_df_mk.groupby("Occupation ID")["Mean Importance"].transform(
        lambda x: x.dropna().astype(float).sum()
    )
    has_imp_mk = output_df_mk["Mean Importance"].notna() & (imp_sum_per_occ_mk > 0)
    output_df_mk["Mean Importance Renormalized To Be Time Shares"] = pd.NA
    output_df_mk.loc[has_imp_mk, "Mean Importance Renormalized To Be Time Shares"] = (
        HOURS_PER_DAY
        * output_df_mk.loc[has_imp_mk, "Mean Importance"].astype(float)
        / imp_sum_per_occ_mk.loc[has_imp_mk]
    )
    for _, g in output_df_mk.groupby("Occupation ID"):
        if g["Mean Importance"].dropna().empty:
            continue
        total = g["Mean Importance Renormalized To Be Time Shares"].dropna().astype(float).sum()
        assert abs(float(total) - HOURS_PER_DAY) < 1e-6, total
    output_df_mk = output_df_mk.sort_values(["Occupation ID", "Task ID"]).reset_index(drop=True)
    OUTPUT_PATH_TASK_MISSING_DATA_KEPT.parent.mkdir(parents=True, exist_ok=True)
    output_df_mk.to_excel(OUTPUT_PATH_TASK_MISSING_DATA_KEPT, index=False)
    print(f"Saved {len(output_df_mk)} rows to {OUTPUT_PATH_TASK_MISSING_DATA_KEPT}")

    # Occupation-level: one row per occupation retained by Alex Wan's filtered exposure universe.
    occ_to_task_rows = {}
    for tr in task_rows:
        occ = tr["Occupation ID"]
        if occ not in retained_soc_codes:
            continue
        if (occ, tr["Task ID"]) not in alex_task_keys:
            continue
        if occ not in occ_to_task_rows:
            occ_to_task_rows[occ] = []
        occ_to_task_rows[occ].append(tr)

    (
        occupation_output_rows,
        occ_incomplete_time,
        n_occ_hm_l_task_mismatch,
        n_occ_alex_task_mismatch,
    ) = _collect_occupation_summary_rows(
        occ_to_task_rows,
        wage_df,
        alex_lookup_mod_high,
        alex_lookup_high_only,
        alex_task_type_lookup,
        hm_l_mod_high_lookup,
        hm_l_high_only_lookup,
        hosseini_lookup,
        hosseini_task_type_lookup,
        warned_no_time_json_occ,
        True,
    )

    assert len(occupation_output_rows) > 0, (
        "No occupation rows remain after filtering to Alex Wan's occupation universe."
    )
    occupation_output_df = pd.DataFrame(occupation_output_rows)
    occupation_output_df = occupation_output_df.sort_values(["Occupation ID"]).reset_index(drop=True)
    occupation_output_df.to_excel(OCCUPATION_OUTPUT_PATH, index=False)
    print(f"Saved {len(occupation_output_df)} rows to {OCCUPATION_OUTPUT_PATH}")

    occ_to_task_rows_mk = {}
    for tr in task_rows:
        occ = tr["Occupation ID"]
        if occ not in retained_soc_codes:
            continue
        if occ not in occ_to_task_rows_mk:
            occ_to_task_rows_mk[occ] = []
        occ_to_task_rows_mk[occ].append(tr)

    (
        occupation_output_rows_mk,
        _occ_inc_mk,
        n_occ_hm_l_task_mismatch_mk,
        n_occ_alex_task_mismatch_mk,
    ) = _collect_occupation_summary_rows(
        occ_to_task_rows_mk,
        wage_df,
        alex_lookup_mod_high,
        alex_lookup_high_only,
        alex_task_type_lookup,
        hm_l_mod_high_lookup,
        hm_l_high_only_lookup,
        hosseini_lookup,
        hosseini_task_type_lookup,
        warned_no_time_json_occ,
        False,
    )
    assert len(occupation_output_rows_mk) > 0, (
        "No occupation rows for missing-data-kept workbook."
    )
    occupation_output_df_mk = pd.DataFrame(occupation_output_rows_mk)
    occupation_output_df_mk = occupation_output_df_mk.sort_values(["Occupation ID"]).reset_index(drop=True)
    OCCUPATION_OUTPUT_PATH_MISSING_DATA_KEPT.parent.mkdir(parents=True, exist_ok=True)
    occupation_output_df_mk.to_excel(OCCUPATION_OUTPUT_PATH_MISSING_DATA_KEPT, index=False)
    print(f"Saved {len(occupation_output_df_mk)} rows to {OCCUPATION_OUTPUT_PATH_MISSING_DATA_KEPT}")
    print(
        "[missing_data_kept occupations] HM&L task mismatches (Alex-retained tasks missing HM&L): "
        f"{n_occ_hm_l_task_mismatch_mk}; Alex universe mismatches: {n_occ_alex_task_mismatch_mk}"
    )

    print(
        f"Occupations with no time-estimate JSON (warned): {len(warned_no_time_json_occ)}"
    )
    print(
        "Occupations without full task coverage in time JSON "
        f"(at least one common task missing from Time per task or Expected freq per task): "
        f"{len(occ_incomplete_time)}"
    )
    print(
        "Alex Wan–retained occupations missing at least one HM&L label for an Alex Wan–retained task: "
        f"{n_occ_hm_l_task_mismatch}"
    )
    print(
        "Occupations with Alex Wan exposure task codes outside Task Ratings task set: "
        f"{n_occ_alex_task_mismatch}"
    )


if __name__ == "__main__":
    main()
