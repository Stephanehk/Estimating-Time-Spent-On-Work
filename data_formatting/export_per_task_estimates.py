"""Export the headline output of this project: per-task time-share estimates.

Writes `task_time_share_estimates.xlsx` to the repository root with one row per
(occupation, task) and exactly these columns:

    Task ID, Task Name, Occupation ID, Occupation Name,
    Time spent per day estimate (w(t,o) · E[f(t,o)]),
    Time per single instance (w(t,o)),
    Expected daily frequency (E[f(t,o)]).

It reads the per-occupation LP solution files written by find_time_per_task.py
(which already store the single-instance time w(t,o) and the expected daily
frequency E[f(t,o)]) and the O*NET Task Statements for task names.

Run from the repository root:
  python3 -m data_formatting.export_per_task_estimates
"""

import json
from pathlib import Path

import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parent.parent

EXTRA_DETAILS = "gpt-5.2_ONET_30.2_use_cps_constants_prob_thresh=0.7"
WEIGHTS_PATH_TEMPLATE = (
    "data/generated_data/occupation_time_per_task_{soc_code}"
    f"{EXTRA_DETAILS}_chosen_max_span_weigths.json"
)
TASK_STATEMENTS_PATH = _REPO_ROOT / "data" / "onet_data" / "Task Statements.xlsx"
SOC_DONES_PATH = _REPO_ROOT / "data" / "generated_data" / f"soc_code_dones_{EXTRA_DETAILS}.txt"
OUTPUT_PATH = _REPO_ROOT / "task_time_share_estimates.xlsx"

COL_TIME_PER_DAY = "Time spent per day estimate (w(t,o) · E[f(t,o)])"
COL_SINGLE_INSTANCE = "Time per single instance (w(t,o))"
COL_EXPECTED_FREQ = "Expected daily frequency (E[f(t,o)])"


def _task_name_lookup():
    """(O*NET-SOC Code, Task ID) -> task statement text, from O*NET Task Statements."""
    df = pd.read_excel(TASK_STATEMENTS_PATH)
    assert {"O*NET-SOC Code", "Task ID", "Task"}.issubset(df.columns)
    lookup = {}
    for _, row in df.iterrows():
        lookup[(str(row["O*NET-SOC Code"]), int(row["Task ID"]))] = str(row["Task"])
    return lookup


def _load_soc_codes_done():
    assert SOC_DONES_PATH.exists(), f"Missing SOC done list: {SOC_DONES_PATH}"
    with open(SOC_DONES_PATH) as f:
        codes = [line.strip() for line in f if line.strip()]
    assert len(codes) > 0, f"No SOC codes in {SOC_DONES_PATH}"
    return codes


def main():
    task_names = _task_name_lookup()

    rows = []
    for soc_code in _load_soc_codes_done():
        weights_path = _REPO_ROOT / WEIGHTS_PATH_TEMPLATE.format(soc_code=soc_code)
        if not weights_path.exists():
            print(f"Warning: no time estimates JSON for occupation {soc_code}; skipping.")
            continue
        with open(weights_path) as f:
            occupation_result = json.load(f)

        occupation_name = occupation_result["Title"]
        time_per_task = occupation_result["Time per task"]
        expected_freq_per_task = occupation_result["Expected freq per task"]

        for task_code in time_per_task:
            task_id = int(task_code)
            single_instance_time = float(time_per_task[task_code])
            expected_frequency = float(expected_freq_per_task[task_code])
            rows.append({
                "Task ID": task_id,
                "Task Name": task_names.get((soc_code, task_id), pd.NA),
                "Occupation ID": soc_code,
                "Occupation Name": occupation_name,
                COL_TIME_PER_DAY: single_instance_time * expected_frequency,
                COL_SINGLE_INSTANCE: single_instance_time,
                COL_EXPECTED_FREQ: expected_frequency,
            })

    assert len(rows) > 0, "No (occupation, task) rows produced."
    output_df = pd.DataFrame(rows, columns=[
        "Task ID",
        "Task Name",
        "Occupation ID",
        "Occupation Name",
        COL_TIME_PER_DAY,
        COL_SINGLE_INSTANCE,
        COL_EXPECTED_FREQ,
    ])
    output_df = output_df.sort_values(["Occupation ID", "Task ID"]).reset_index(drop=True)
    output_df.to_excel(OUTPUT_PATH, index=False)
    print(f"Saved {len(output_df)} rows for {output_df['Occupation ID'].nunique()} occupations to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
