"""Reproduce the numbers behind Figure 1 (Massage Therapists, O*NET 31-9011.00).

Figure 1 contrasts two exposure verdicts for the same task list under the rubric-based
(rubric-based) exposure measure at the moderate+high tier (automation_hl in
{T2, T3, T4}):
  - core/supplemental task-type weighting  -> 44% of tasks exposed
  - our time-share weighting s(t, o)        -> 16% of working time exposed

It also prints each task's share of working time (the percentages shown in the figure's
table) and marks the exposed tasks.

Run from the repository root:
  python3 -m analysis.figure1_massage_therapist_example
"""

import json
import sys
from pathlib import Path

import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from data_formatting.exposure_loading_utils import TASK_STATEMENTS_PATH
from analysis.figure4_exposure_share_cdf import (
    RUBRIC_BASED_EXPOSURE_PATH,
    _task_id_to_str,
    load_rubric_based_exposure_excel,
    run_weighted_exposure_analysis,
)

MASSAGE_SOC = "31-9011.00"
EXTRA_DETAILS = "gpt-5.2_ONET_30.2_use_cps_constants_prob_thresh=0.7"
# Figure 1 uses the rubric-based measure at the moderate+high tier (T2 is the first tier
# whose entire 50-80% range lies above the 25% "moderately exposed" cutoff).
EXPOSED_TIERS = {"T2", "T3", "T4"}


def main():
    exposure_df = load_rubric_based_exposure_excel(
        RUBRIC_BASED_EXPOSURE_PATH, TASK_STATEMENTS_PATH, exposed_tiers=EXPOSED_TIERS
    )

    result = run_weighted_exposure_analysis(
        exposure_df,
        extra_details=EXTRA_DETAILS,
        source_label="rubric_based_rubric_based_moderate_plus_high",
        update_website_json=False,
        quiet=True,
        use_max_span_time_weights=True,
    )

    task_weighted = result["onet2fraction_exposed"][MASSAGE_SOC]
    time_weighted = result["onet2fraction_exposed_time_weighted"][MASSAGE_SOC]

    # Per-task share of working time s(t, o) = time_per_task * expected_freq (normalized).
    occ_path = (
        _REPO_ROOT / "data" / "generated_data"
        / f"occupation_time_per_task_{MASSAGE_SOC}{EXTRA_DETAILS}_chosen_max_span_weigths.json"
    )
    with open(occ_path, "r") as f:
        occ = json.load(f)

    statements = pd.read_excel(TASK_STATEMENTS_PATH)
    statements = statements[statements["O*NET-SOC Code"].astype(str) == MASSAGE_SOC]
    task_code2name = {
        _task_id_to_str(row["Task ID"]): str(row["Task"])
        for _, row in statements.iterrows()
    }

    exposed_codes = {
        _task_id_to_str(r["TaskID"])
        for _, r in exposure_df[exposure_df["ONETSOCCode"] == MASSAGE_SOC].iterrows()
        if int(r["auto_hl_bi"]) == 1
    }

    time_per_task = occ["Time per task"]
    expected_freq = occ["Expected freq per task"]
    daily = {tc: time_per_task[tc] * expected_freq[tc] for tc in time_per_task}
    total = sum(daily.values())
    shares = {tc: v / total for tc, v in daily.items()}

    print(f"Figure 1 — Massage Therapists (O*NET {MASSAGE_SOC}), rubric-based moderate+high\n")
    print(f"{'share':>7}  {'exposed':>7}  task")
    for tc in sorted(shares, key=lambda x: -shares[x]):
        mark = "EXP" if tc in exposed_codes else ""
        name = task_code2name.get(tc, f"(task {tc})")
        print(f"{shares[tc]*100:6.1f}%  {mark:>7}  {name}")

    print()
    print(f"Tasks exposed (core/supplemental weighting): {task_weighted*100:.0f}%   (paper: 44%)")
    print(f"Working time exposed (time-share weighting): {time_weighted*100:.0f}%   (paper: 16%)")


if __name__ == "__main__":
    main()
