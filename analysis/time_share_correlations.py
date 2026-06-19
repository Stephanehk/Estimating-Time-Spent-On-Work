"""Correlate our time-share estimates with O*NET descriptors and external time data.

Reproduces the in-text correlation numbers from the Analysis section of the paper:

  1. Per-occupation Kendall's tau between our time-share s(t,o) and O*NET mean importance.
  2. Same, restricted to core tasks, against importance x expected-frequency.
  3. Per-occupation Kendall's tau between O*NET frequency and our single-instance w(t,o).
  4. Spearman correlation of our w(t,o) against the per-task human completion times of
     Tamkin & McCrory [2025] (the Anthropic Economic Index), over the tasks common to both.

For (1)-(3) we report, across occupations, how many have a significant correlation
(p < 0.05) and the mean tau among those.

Run from the repository root:
  python3 -m analysis.time_share_correlations --use_cps_constants \
      --extra_details gpt-5.2_ONET_30.2_use_cps_constants_prob_thresh=0.7 \
      --use_max_span_time_weights
"""

import argparse
import json

import numpy as np
from scipy.stats import kendalltau, spearmanr

from utils.constants import FREQ_TO_TIME_PER_DAY
from utils.compute_onet_hours import compute_soc_constants
from find_time_per_task import get_onet_task_data
from analysis.anthropic_index_task_times import get_anthropic_task_time_mapping


def _expected_frequency(tasks, freq_to_time_per_day):
    """Expected daily task frequency E[f(t,o)] for each task (O*NET freq bins x duration)."""
    freqs = []
    for task in tasks:
        freqs.append(
            sum(
                freq_to_time_per_day[freq["Category"]] * (freq["Data Value"] / 100)
                for freq in task["Ratings"]["Frequency"]
            )
        )
    return freqs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--extra_details", type=str, default="")
    parser.add_argument("--use_cps_constants", action="store_true", default=False)
    parser.add_argument("--use_max_span_time_weights", action="store_true", default=True)
    args = parser.parse_args()

    extra_details = args.extra_details
    max_span_suffix = "_max_span_weigths" if args.use_max_span_time_weights else ""
    if args.use_cps_constants:
        assert "_use_cps_constants" in extra_details, "extra_details must contain _use_cps_constants"

    with open(f"data/generated_data/soc_code_dones_{extra_details}.txt") as f:
        processed_codes = [line.strip() for line in f]

    anthropic_task_time_mapping = get_anthropic_task_time_mapping()

    importance_taus = []      # Kendall tau (s(t,o) vs importance) for every processed occupation
    importance_sig_pos = []   # ... the significant positive subset
    core_sig_pos = []         # core tasks: s(t,o) vs importance x freq, significant positive
    frequency_sig = []        # frequency vs w(t,o), significant subset (any sign)

    # Per-task pairing of our w(t,o) with Tamkin & McCrory completion times (for the Spearman).
    task2predicted_time = {}
    task2anthropic_time = {}

    for soc in processed_codes:
        weights_path = (
            f"data/generated_data/occupation_time_per_task_{soc}{extra_details}_chosen{max_span_suffix}.json"
        )
        try:
            with open(weights_path) as f:
                occupation_result = json.load(f)
        except FileNotFoundError:
            print(f"*** No time-per-task data for O*NET-SOC Code {soc}; skipping ***")
            continue

        task_data, task_statements = get_onet_task_data(soc, return_statements=True)
        freq_to_time_per_day = FREQ_TO_TIME_PER_DAY
        if args.use_cps_constants:
            freq_to_time_per_day = compute_soc_constants(soc)["FREQ_TO_TIME_PER_DAY"]

        tasks = task_data["Tasks"]
        task_codes = [str(task["Task ID"]) for task in tasks]
        importance = [task["Ratings"]["Importance"]["Data Value"] for task in tasks]
        expected_frequency = _expected_frequency(tasks, freq_to_time_per_day)

        single_instance_time = []        # w(t,o)
        time_share = []                  # s(t,o) = w(t,o) * E[f(t,o)]
        core_time_share = []
        core_importance_times_freq = []

        for i, task_code in enumerate(task_codes):
            w = occupation_result["Time per task"][task_code]
            freq = occupation_result["Expected freq per task"][task_code]
            single_instance_time.append(w)
            time_share.append(w * freq)

            statement_row = task_statements[task_statements["Task ID"] == int(task_code)]
            assert len(statement_row) == 1
            if statement_row["Task Type"].values[0] == "Core":
                core_time_share.append(w * freq)
                core_importance_times_freq.append(importance[i] * freq)

            anthropic_rows = anthropic_task_time_mapping[
                (anthropic_task_time_mapping["O*NET-SOC Code"] == soc)
                & (anthropic_task_time_mapping["Task ID"] == int(task_code))
            ]
            if len(anthropic_rows) == 1:
                # Task IDs are not globally unique, but a paired (SOC, task) appears once.
                assert task_code not in task2predicted_time
                task2predicted_time[task_code] = w
                task2anthropic_time[task_code] = anthropic_rows["human_only_time_mean"].values[0]

        tau_importance, p_importance = kendalltau(time_share, importance)
        tau_core, p_core = kendalltau(core_time_share, core_importance_times_freq)
        tau_frequency, p_frequency = kendalltau(expected_frequency, single_instance_time)

        importance_taus.append(tau_importance)
        if tau_importance > 0 and p_importance < 0.05 and not np.isnan(tau_importance):
            importance_sig_pos.append(tau_importance)
        if tau_core > 0 and p_core < 0.05:
            core_sig_pos.append(tau_core)
        if p_frequency < 0.05:
            frequency_sig.append(tau_frequency)

    n_occupations = len(importance_taus)
    print(f"\nOccupations with time-per-task data: {n_occupations}\n")
    print(
        f"Importance vs s(t,o):              "
        f"{len(importance_sig_pos)}/{n_occupations} significant, "
        f"mean tau = {np.mean(importance_sig_pos):.3f}"
    )
    print(
        f"Importance x freq (core) vs s(t,o): "
        f"{len(core_sig_pos)}/{n_occupations} significant, "
        f"mean tau = {np.mean(core_sig_pos):.3f}"
    )
    print(
        f"Frequency vs w(t,o):               "
        f"{len(frequency_sig)}/{n_occupations} significant, "
        f"mean tau = {np.mean(frequency_sig):.3f}"
    )

    common_codes = sorted(task2predicted_time)
    rho = spearmanr(
        [task2predicted_time[c] for c in common_codes],
        [task2anthropic_time[c] for c in common_codes],
    )
    print(
        f"\nSpearman w(t,o) vs Tamkin & McCrory completion times: "
        f"rho = {rho.correlation:.3f} (p = {rho.pvalue:.2e}, N = {len(common_codes)} tasks)"
    )


if __name__ == "__main__":
    main()
