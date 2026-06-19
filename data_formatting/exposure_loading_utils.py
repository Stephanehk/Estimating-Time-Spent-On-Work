"""
Shared loading utilities for Simulation-based's task-level exposure data.

These helpers are the single place where the simulation-based exposure spreadsheet is mapped
onto O*NET task ratings. They assume:
- the Simulation-based file has one row per occupation/task text pair;
- exposure scores live in raw_time_savings_median;
- a score is usable only when synthetic_generation_success and
  exposure_prediction_success are TRUE;
- occupations with usable scores for fewer than 80% of their mapped tasks are
  excluded entirely.
"""

from pathlib import Path

import numpy as np
import pandas as pd


SIMULATION_BASED_EXPOSURE_PATH = Path("data/exposure_scores/simulation_based_exposure.xlsx")
TASK_RATINGS_PATH = Path("data/onet_data/Task Ratings.xlsx")
TASK_STATEMENTS_PATH = Path("data/onet_data/Task Statements.xlsx")

EXPOSURE_COL = "raw_time_savings_median"
SYNTHETIC_SUCCESS_COL = "synthetic_generation_success"
PREDICTION_SUCCESS_COL = "exposure_prediction_success"
TASK_DESC_COL = "Task"
OCC_DESC_COL = "Title"
MIN_OCCUPATION_EXPOSURE_COVERAGE = 0.8


def _build_ratings_lookup(ratings_path):
    """
    Build (occupation_description, task_description) -> (ONETSOCCode, TaskID).

    Assumes each stripped (Title, Task) pair maps to exactly one O*NET task.
    """
    ratings_path = Path(ratings_path)
    assert ratings_path.exists(), f"Task Ratings not found: {ratings_path}"
    df = pd.read_excel(ratings_path)
    assert "O*NET-SOC Code" in df.columns and "Task ID" in df.columns
    assert "Task" in df.columns and "Title" in df.columns
    grouped = df.groupby(["O*NET-SOC Code", "Task ID"], dropna=False).first().reset_index()
    grouped = grouped[["O*NET-SOC Code", "Task ID", "Title", "Task"]].dropna(subset=["Title", "Task"])
    lookup = {}
    for _, row in grouped.iterrows():
        key = (str(row["Title"]).strip(), str(row["Task"]).strip())
        value = (str(row["O*NET-SOC Code"]), float(row["Task ID"]))
        assert key not in lookup or lookup[key] == value, (
            f"(Title, Task) maps to more than one (SOC, TaskID): {key} -> {lookup[key]} and {value}"
        )
        lookup[key] = value
    return lookup


def _build_ratings_task_counts(ratings_path):
    """
    Build ONETSOCCode -> number of distinct tasks in Task Ratings.

    Assumes Task Ratings is the denominator for occupation-level Simulation-based coverage.
    """
    ratings_path = Path(ratings_path)
    assert ratings_path.exists(), f"Task Ratings not found: {ratings_path}"
    df = pd.read_excel(ratings_path)
    assert "O*NET-SOC Code" in df.columns and "Task ID" in df.columns
    grouped = df[["O*NET-SOC Code", "Task ID"]].dropna().drop_duplicates()
    counts = grouped.groupby("O*NET-SOC Code")["Task ID"].nunique()
    out = {str(k): int(v) for k, v in counts.items()}
    assert len(out) > 0, "No occupation task counts loaded from Task Ratings."
    return out


def _load_task_type_lookup(statements_path):
    """
    Build (ONETSOCCode, TaskID) -> TaskType.

    Assumes Task Type values are exactly Core or Supplemental.
    """
    statements_path = Path(statements_path)
    assert statements_path.exists(), f"Task Statements not found: {statements_path}"
    df = pd.read_excel(statements_path)
    assert "O*NET-SOC Code" in df.columns and "Task ID" in df.columns and "Task Type" in df.columns
    df = df[["O*NET-SOC Code", "Task ID", "Task Type"]].dropna(subset=["Task Type"])
    core_vals = {"Core", "Supplemental"}
    assert set(df["Task Type"].unique()).issubset(core_vals), (
        f"Unexpected Task Type: {df['Task Type'].unique()}"
    )
    out = {}
    for _, row in df.iterrows():
        key = (str(row["O*NET-SOC Code"]), float(row["Task ID"]))
        out[key] = str(row["Task Type"])
    return out


def _normalize_success_flag(value):
    """
    Normalize an Excel success flag to bool.

    Assumes flags are stored as booleans, TRUE/FALSE strings, or 1/0 values.
    """
    assert not pd.isna(value), "Success flag is missing."
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, str):
        cleaned = value.strip().upper()
        assert cleaned in {"TRUE", "FALSE"}, f"Unexpected success flag string: {value!r}"
        return cleaned == "TRUE"
    if isinstance(value, (int, np.integer)):
        assert int(value) in {0, 1}, f"Unexpected success flag integer: {value!r}"
        return int(value) == 1
    if isinstance(value, (float, np.floating)):
        assert float(value) in {0.0, 1.0}, f"Unexpected success flag float: {value!r}"
        return float(value) == 1.0
    raise TypeError(f"Unexpected success flag type: {type(value)}")


def _is_usable_exposure_row(row):
    """
    Return whether an Simulation-based row has a usable exposure score.

    Assumes score usability requires both success flags to be TRUE and a non-null
    raw_time_savings_median value.
    """
    if pd.isna(row[EXPOSURE_COL]):
        return False
    return (
        _normalize_success_flag(row[SYNTHETIC_SUCCESS_COL])
        and _normalize_success_flag(row[PREDICTION_SUCCESS_COL])
    )


def load_simulation_based_exposure_mapped_to_ratings(
    simulation_based_path,
    ratings_path,
    statements_path,
    exposure_threshold=0.5,
    min_occupation_coverage=MIN_OCCUPATION_EXPOSURE_COVERAGE,
):
    """
    Load Simulation-based exposure scores and map them to O*NET task ratings.

    Assumptions:
    - simulation_based_path has Title, Task, raw_time_savings_median,
      synthetic_generation_success, and exposure_prediction_success.
    - raw_time_savings_median is the exposure score to threshold.
    - rows with either success flag not TRUE have no exposure score.
    - occupations with usable scores for fewer than min_occupation_coverage of
      their mapped tasks are excluded entirely.

    Returns (DataFrame, n_occupations_not_all_tasks_in_ratings), where DataFrame
    has ONETSOCCode, TaskID, auto_hl_bi, TaskType, and raw_time_savings_median.
    """
    simulation_based_path = Path(simulation_based_path)
    assert 0.0 <= exposure_threshold <= 1.0, (
        f"exposure_threshold must be in [0, 1], got {exposure_threshold}"
    )
    assert 0.0 < min_occupation_coverage <= 1.0, (
        f"min_occupation_coverage must be in (0, 1], got {min_occupation_coverage}"
    )
    assert simulation_based_path.exists(), f"Simulation-based exposure file not found: {simulation_based_path}"
    df = pd.read_excel(simulation_based_path)
    required_cols = {
        TASK_DESC_COL,
        OCC_DESC_COL,
        EXPOSURE_COL,
        SYNTHETIC_SUCCESS_COL,
        PREDICTION_SUCCESS_COL,
    }
    assert required_cols.issubset(df.columns), (
        f"Simulation-based exposure file missing columns: {required_cols - set(df.columns)}"
    )

    ratings_lookup = _build_ratings_lookup(ratings_path)
    ratings_task_counts = _build_ratings_task_counts(ratings_path)
    task_type_lookup = _load_task_type_lookup(statements_path)

    occ_to_simulation_based_tasks = {}
    mapped_rows = []
    for _, row in df.iterrows():
        occ_desc = row[OCC_DESC_COL]
        task_desc = row[TASK_DESC_COL]
        if pd.isna(occ_desc) or pd.isna(task_desc):
            continue
        occ_stripped = str(occ_desc).strip()
        task_stripped = str(task_desc).strip()
        if occ_stripped not in occ_to_simulation_based_tasks:
            occ_to_simulation_based_tasks[occ_stripped] = set()
        occ_to_simulation_based_tasks[occ_stripped].add(task_stripped)
        key = (occ_stripped, task_stripped)
        if key not in ratings_lookup:
            continue
        onet_soc_code, task_id = ratings_lookup[key]
        task_type = task_type_lookup.get((onet_soc_code, task_id))
        assert task_type is not None, f"Task type not found for {key}"
        has_usable_score = _is_usable_exposure_row(row)
        score = pd.NA
        if has_usable_score:
            score = float(row[EXPOSURE_COL])
            assert np.isfinite(score), f"Exposure value must be finite, got {row[EXPOSURE_COL]!r}"
            assert 0.0 <= score <= 1.0, f"Expected Simulation-based exposure score in [0, 1], got {score}"
        mapped_rows.append({
            "SimulationBasedOccupationTitle": occ_stripped,
            "ONETSOCCode": onet_soc_code,
            "TaskID": task_id,
            "auto_hl_bi": pd.NA if pd.isna(score) else int(score >= exposure_threshold),
            EXPOSURE_COL: score,
            "TaskType": task_type,
        })

    n_occupations_not_all_tasks_in_ratings = 0
    for occ_stripped, task_set in occ_to_simulation_based_tasks.items():
        if any((occ_stripped, task) not in ratings_lookup for task in task_set):
            n_occupations_not_all_tasks_in_ratings += 1
    print(
        f"[Simulation-based exposure] Occupations with at least one Simulation-based task not in Task Ratings "
        f"(not all tasks covered): {n_occupations_not_all_tasks_in_ratings}"
    )

    mapped = pd.DataFrame(mapped_rows)
    assert len(mapped) > 0, "No rows mapped from Simulation-based file to Task Ratings + Task Statements."

    task_coverage = mapped.groupby(["ONETSOCCode", "TaskID"], as_index=False).agg(
        has_filled_exposure_score=(EXPOSURE_COL, lambda s: bool(s.notna().any())),
    )
    coverage = task_coverage.groupby("ONETSOCCode", as_index=False).agg(
        n_filled_exposure_scores=("has_filled_exposure_score", "sum"),
    )
    coverage["n_rating_tasks"] = coverage["ONETSOCCode"].map(ratings_task_counts)
    assert coverage["n_rating_tasks"].notna().all(), "Coverage contains SOC codes absent from Task Ratings."
    coverage["filled_share"] = coverage["n_filled_exposure_scores"] / coverage["n_rating_tasks"]
    excluded_occupations = set(
        coverage.loc[
            coverage["filled_share"] < float(min_occupation_coverage),
            "ONETSOCCode",
        ].astype(str)
    )
    print(
        "[Simulation-based exposure] Occupations excluded for < "
        f"{min_occupation_coverage:.0%} task exposure-score coverage: {len(excluded_occupations)}"
    )

    data = mapped[
        (~mapped["ONETSOCCode"].astype(str).isin(excluded_occupations))
        & mapped[EXPOSURE_COL].notna()
    ].copy()
    assert len(data) > 0, "No Simulation-based exposure rows remain after success and occupation coverage filters."
    data = data.drop_duplicates(subset=["ONETSOCCode", "TaskID"], keep="first")
    return (
        data[["ONETSOCCode", "TaskID", "auto_hl_bi", "TaskType", EXPOSURE_COL]],
        n_occupations_not_all_tasks_in_ratings,
    )


def load_simulation_based_labels(simulation_based_exposure_threshold=0.5):
    """
    Load binary Simulation-based task-level labels mapped to O*NET IDs.

    Assumes the shared Simulation-based loader's filtering rules are the correct source of
    truth for valid exposure scores.
    """
    simulation_based_df, _ = load_simulation_based_exposure_mapped_to_ratings(
        SIMULATION_BASED_EXPOSURE_PATH,
        TASK_RATINGS_PATH,
        TASK_STATEMENTS_PATH,
        exposure_threshold=simulation_based_exposure_threshold,
    )
    simulation_based_df = simulation_based_df[["ONETSOCCode", "TaskID", "auto_hl_bi", EXPOSURE_COL]].dropna(
        subset=["ONETSOCCode", "TaskID", "auto_hl_bi"]
    ).copy()
    simulation_based_df["ONETSOCCode"] = simulation_based_df["ONETSOCCode"].astype(str)
    simulation_based_df["TaskID"] = simulation_based_df["TaskID"].astype(float)
    simulation_based_df["auto_hl_bi"] = simulation_based_df["auto_hl_bi"].astype(int)
    unique_labels = set(simulation_based_df["auto_hl_bi"].unique().tolist())
    assert unique_labels.issubset({0, 1}), f"Expected binary auto_hl_bi for Simulation-based, got {unique_labels}"
    return simulation_based_df
