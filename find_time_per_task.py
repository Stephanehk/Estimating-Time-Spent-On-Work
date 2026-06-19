"""Estimate the share of time spent per task for each O*NET occupation (the LP method).

This is the core entrypoint for the paper "Estimating Time Spent on Work Tasks".

For every occupation we:
  1. Read O*NET task frequencies and convert them to an expected number of task
     instances per day (`get_onet_task_data` + `impute_never_frequency`), optionally
     using per-SOC CPS normalization constants (`--use_cps_constants`).
  2. Elicit pairwise "which single instance takes longer" judgements from an LM for
     every task pair (cached on disk; see `time_classification.py`), aggregate them into
     a ranking with the Copeland method, and turn that ranking into ordinal constraints.
  3. Solve a linear program (scipy `linprog`, HiGHS) for the per-instance time weights
     `w(t, o)` subject to: the daily-time budget (sum of w*freq == 7h), the ranking
     constraints, and non-negativity. Two objectives are supported:
       - default: maximize the minimum daily time-on-task (most uniform allocation), and
       - `--find_max_span_weights` (the paper default): maximize the spread between the
         highest- and lowest-ranked task.
     If the full constraint set is infeasible, ranking constraints are relaxed starting
     from the lowest-ranked task (`try_fixing_lp_constraints`).

Outputs per occupation, written to `data/generated_data/`:
  - occupation_time_per_task_<SOC><extra_details>_chosen[_max_span_weigths].json  (weights)
  - occupation_time_per_task_<SOC><extra_details>_chosen_LP[_max_span_weigths].json (LP matrices)

The pairwise LM judgements are cached, so re-running with the shipped cache present
performs no LM/API calls — the LP solve is fully deterministic and reproduces the
released weight files exactly.

Example (paper configuration):
  python3 -m find_time_per_task --model_name gpt-5.2 --n_trials 1 --prob_threshold 0.7 \
      --use_cps_constants --find_max_span_weights --start_index 0 --end_index 879
"""

import pandas as pd
import json
import os
import copy
import traceback
import argparse
from functools import lru_cache
import numpy as np
import itertools
from scipy.optimize import linprog, OptimizeResult
from time_classification import get_time_preference_with_probability_for_pairs


@lru_cache(maxsize=1)
def _load_task_ratings():
    """Cached read of the (static) O*NET Task Ratings workbook.

    get_onet_task_data is called once per occupation; caching the 10MB read makes
    full-occupation runs fast. The returned DataFrame is never mutated by callers
    (they only filter it), so sharing one cached copy is safe and output-identical.
    """
    return pd.read_excel(os.path.join('data', 'Task Ratings.xlsx'))


@lru_cache(maxsize=1)
def _load_task_statements():
    """Cached read of the (static) O*NET Task Statements workbook."""
    return pd.read_excel(os.path.join('data', 'Task Statements.xlsx'))
from utils.constants import (
    FREQ_TO_TIME_PER_DAY,
    HOURS_PER_DAY,
    IMPORTANCE_TOLERANCE,
    MIN_TIME_PER_TASK,
)
from utils.compute_onet_hours import compute_soc_constants
from utils.llm_utils import DEFAULT_MODEL


def get_onet_task_data(onet_soc_code, return_statements=False):
    """
    Extract task frequency, importance, and relevance ratings for an O*NET-SOC code
    from `data/Task Ratings.xlsx` (O*NET version 30.2).

    Returns a dict {"O*NET-SOC Code", "Title", "Tasks": [...]} where each task carries
    its raw frequency distribution (Scale ID FT), importance (IM), and relevance (RT).
    When return_statements is True, also returns the Task Statements rows for this
    occupation (used by analysis/time_share_correlations.py).
    """

    # Assumes the script is run from the repo root where data/ lives.
    df_ratings = _load_task_ratings()
    if return_statements:
        df_statements = _load_task_statements()
        df_statements = df_statements[df_statements['O*NET-SOC Code'].astype(str) == str(onet_soc_code)]
    # Filter ratings for this occupation
    occupation_ratings = df_ratings[df_ratings['O*NET-SOC Code'].astype(str) == str(onet_soc_code)]
    if occupation_ratings.empty:
        print(f"in get_onet_task_data(), No data found for O*NET-SOC Code: {onet_soc_code}")
        return json.dumps({"error": f"No data found for O*NET-SOC Code: {onet_soc_code}"})

    # Get Occupation Title (from the first row)
    occupation_title = occupation_ratings.iloc[0]['Title']

    tasks_list = []

    # Iterate through tasks found in ratings
    task_groups = occupation_ratings.groupby(['Task ID', 'Task'], dropna=False)
    for (task_id, task_text), task_ratings_subset in task_groups:
        task_type = None

        ratings_data = {
            "Frequency": [],
            "Importance": None,
            "Relevance": None
        }

        # Process Frequency (Scale ID = FT)
        # Categories 1-7 representing frequency distribution
        freq_rows = task_ratings_subset[task_ratings_subset['Scale ID'] == 'FT'].sort_values('Category')

        for _, freq_row in freq_rows.iterrows():
            # we won't save distributional data for frequency because we rescale the frequency distribution to account for the "Never" category, which invalidates this distributional data
            entry = {
                "Category": int(freq_row['Category']) if not pd.isna(freq_row['Category']) else None,
                "Data Value": float(freq_row['Data Value']) if not pd.isna(freq_row['Data Value']) else None,
            }
            ratings_data["Frequency"].append(entry)

        # Process Importance (Scale ID = IM)
        imp_rows = task_ratings_subset[task_ratings_subset['Scale ID'] == 'IM']

        if not imp_rows.empty:
            # Typically one row for Importance per task
            imp_row = imp_rows.iloc[0]
            ratings_data["Importance"] = {
                "Data Value": float(imp_row['Data Value']) if not pd.isna(imp_row['Data Value']) else None,
                "N": int(imp_row['N']) if not pd.isna(imp_row['N']) else None,
                "Standard Error": float(imp_row['Standard Error']) if not pd.isna(imp_row['Standard Error']) else None,
                "Lower CI Bound": float(imp_row['Lower CI Bound']) if not pd.isna(imp_row['Lower CI Bound']) else None,
                "Upper CI Bound": float(imp_row['Upper CI Bound']) if not pd.isna(imp_row['Upper CI Bound']) else None
            }

        # Process Relevance (Scale ID = RT)
        rel_rows = task_ratings_subset[task_ratings_subset['Scale ID'] == 'RT']

        if not rel_rows.empty:
            # Typically one row for Relevance per task
            rel_row = rel_rows.iloc[0]
            ratings_data["Relevance"] = {
                "Data Value": float(rel_row['Data Value']) if not pd.isna(rel_row['Data Value']) else None,
                "N": int(rel_row['N']) if not pd.isna(rel_row['N']) else None,
                "Standard Error": float(rel_row['Standard Error']) if not pd.isna(rel_row['Standard Error']) else None,
                "Lower CI Bound": float(rel_row['Lower CI Bound']) if not pd.isna(rel_row['Lower CI Bound']) else None,
                "Upper CI Bound": float(rel_row['Upper CI Bound']) if not pd.isna(rel_row['Upper CI Bound']) else None
            }
        else:
            # if there is no relevance data, we cannot impute the "Never" category
            raise ValueError(f"No relevance data found for task {task_id} {task_text}")

        tasks_list.append({
            "Task ID": int(task_id),
            "Task": task_text,
            "Task Type": task_type,
            "Ratings": ratings_data
        })

    result = {
        "O*NET-SOC Code": onet_soc_code,
        "Title": occupation_title,
        "Tasks": tasks_list
    }

    if return_statements:
        return result, df_statements
    return result


def impute_never_frequency(task_data):
    """
    Imputes the frequency for the "Never" category for a given task.

    From: https://www.ons.gov.uk/economy/environmentalaccounts/articles/developingamethodformeasuringtimespentongreentasks/march2022
    The distribution of responses to the frequency question came only from respondents who said the task was relevant.
    As such, there was implicitly an eighth frequency category, which was "Never"
    This relates to those respondents who said the task was not relevant. As such, we imputed this eighth frequency category
    as 100 minus the relevance score, and rescaled the original frequency distribution accordingly, such that the adjusted
    frequency distribution again summed to 100. An example is given in Table 1.
    """
    for task in task_data["Tasks"]:
        task_relevance = task["Ratings"]["Relevance"]["Data Value"]
        never_frequency = 100 - task_relevance
        task["Ratings"]["Frequency"].append({
            "Category": 0,
            "Data Value": never_frequency,
        })

        # renormalise the frequency distribution
        total_frequency = sum(frequency["Data Value"] for frequency in task["Ratings"]["Frequency"])
        for frequency in task["Ratings"]["Frequency"]:
            frequency["Data Value"] = (frequency["Data Value"] / total_frequency) * 100
    return task_data


def try_fixing_lp_constraints(task_i2copeland_score, A_ub_pre_ordering, b_ub_pre_ordering, A_ranking, b_ranking, A_eq, b_eq, bounds, n, c):
    """
    Repair infeasible LPs by dropping ranking rows for an increasing prefix of tasks
    ordered by Copeland score (lowest first, tie-break by task index).
    """
    assert n == len(task_i2copeland_score)
    max_k = n - 1
    task_pairs = list(itertools.combinations(range(n), 2))
    assert len(task_pairs) == A_ranking.shape[0]
    pair_i = np.fromiter((p[0] for p in task_pairs), dtype=np.intp)
    pair_j = np.fromiter((p[1] for p in task_pairs), dtype=np.intp)

    n_pre_rows = A_ub_pre_ordering.shape[0]
    n_rank_rows = A_ranking.shape[0]
    n_cols = A_ranking.shape[1]
    assert A_ub_pre_ordering.shape[1] == n_cols

    def solve_for_removal(removed_tasks, keep_pair):
        """
        Zero out all ranking constraints involving any task in removed_tasks, except
        the single boundary constraint between the last removed task and the next task
        in the Copeland ordering (keep_pair).

        Assumptions:
        - removed_tasks is a tuple of task indices corresponding to order[:k].
        - keep_pair is (order[k-1], order[k]), the boundary pair to preserve.
        - keep_pair always corresponds to exactly one row in pair_i/pair_j.
        """
        removed_arr = np.asarray(removed_tasks, dtype=np.intp)
        A_ranking_cpy = A_ranking.copy()
        b_ranking_cpy = b_ranking.copy()
        row_mask = np.isin(pair_i, removed_arr) | np.isin(pair_j, removed_arr)

        ka, kb = min(keep_pair), max(keep_pair)
        keep_row_mask = (pair_i == ka) & (pair_j == kb)
        assert keep_row_mask.sum() == 1, f"Keep pair {keep_pair} not found in task_pairs"
        row_mask &= ~keep_row_mask

        A_ranking_cpy[row_mask] = 0.0
        b_ranking_cpy[row_mask] = 0.0

        A_ub_retry = np.empty((n_pre_rows + n_rank_rows, n_cols), dtype=A_ub_pre_ordering.dtype)
        np.copyto(A_ub_retry[:n_pre_rows], A_ub_pre_ordering)
        np.copyto(A_ub_retry[n_pre_rows:], A_ranking_cpy)

        b_ub_retry = np.empty(n_pre_rows + n_rank_rows, dtype=b_ub_pre_ordering.dtype)
        np.copyto(b_ub_retry[:n_pre_rows], b_ub_pre_ordering)
        np.copyto(b_ub_retry[n_pre_rows:], b_ranking_cpy)

        res = linprog(
            c,
            A_eq=A_eq,
            b_eq=b_eq,
            A_ub=A_ub_retry,
            b_ub=b_ub_retry,
            bounds=bounds,
            method="highs",
        )
        return res, A_ub_retry, b_ub_retry

    order = sorted(range(n), key=lambda i: (task_i2copeland_score[i], i))
    assert len(order) == n
    for k in range(1, n):
        removed = tuple(order[:k])
        keep_pair = (order[k - 1], order[k])
        res_fb, A_ub_fb, b_ub_fb = solve_for_removal(removed, keep_pair)
        if res_fb.success:
            print(
                "Fallback: feasible LP by removing constraints for "
                f"k={k} task(s) ordered by lowest Copeland score; "
                f"indices {removed}"
            )
            return res_fb, A_ub_fb, b_ub_fb

    print("No feasible solution found after trying cumulative removals up to k =", max_k)
    return (
        OptimizeResult(
            x=None,
            fun=None,
            success=False,
            status=2,
            message=(
                "No feasible solution after lowest-Copeland-score cumulative ranking-row removals."
            ),
            nit=0,
        ),
        None,
        None,
    )


def find_satisfying_weights(
    task_data,
    extra_details="",
    model_name=DEFAULT_MODEL,
    n_trials=1,
    beam_size=1000,
    prob_threshold=0.7,
    freq_to_time_per_day=None,
    cache_details_suffix="",
    find_max_span_weights=False,
):
    """
    Fits time-per-task weights with LLM pairwise ordering (logprob classifier + prob_threshold).

    Assumptions:
    - n_trials must be 1 (one logprob classification per pair).
    - The pairwise judgements are read from / written to the on-disk cache, so with the
      shipped cache present no LM calls are made and the LP solve is deterministic.

    Writes the chosen weights and LP matrices to data/generated_data/ and returns the
    cumulative LLM query cost (0.0 when fully cached).
    """
    assert n_trials == 1, "Only n_trials=1 is supported (logprob classifier per pair)."
    if freq_to_time_per_day is None:
        freq_to_time_per_day = FREQ_TO_TIME_PER_DAY

    extra_details += f"_prob_thresh={prob_threshold:g}"

    time_in_cat = HOURS_PER_DAY

    task_expected_freqs = []
    task_descriptions = []
    task_importances = []
    for task in task_data["Tasks"]:
        task_descriptions.append(task["Task"])

        task_importance = task["Ratings"]["Importance"]
        task_frequency = task["Ratings"]["Frequency"]
        # compute the expected time spent on the task per day as the sum of the percentage of respondants who said the task we perfomed at frequency f (freq["Data Value"]) X the mapping of f to time per day (via FREQ_TO_TIME_PER_DAY)
        expected_freq = sum(freq_to_time_per_day[freq["Category"]] * (freq["Data Value"] / 100) for freq in task_frequency)
        task_expected_freqs.append(expected_freq)
        task_importances.append(task_importance)
    # Default objective: maximize the minimum daily time-on-task (w_i * freq_i),
    # giving a uniform time-per-day allocation across tasks.
    # Alternate objective (find_max_span_weights): maximize the spread between a highest-ranked and lowest-ranked task.
    task_expected_freqs = np.array(task_expected_freqs)
    n = len(task_expected_freqs)
    c = np.zeros(n + 1)
    A_eq = np.zeros((1, n + 1))
    A_eq[0, :n] = task_expected_freqs
    b_eq = [time_in_cat]
    A_ub = np.zeros((n, n + 1))
    A_ub[:, :n] = -np.diag(task_expected_freqs)   # -w_i * freq_i + t <= 0  (i.e. t <= w_i * freq_i)
    A_ub[:, -1] = 1

    b_ub = np.zeros(n)

    total_cost = 0

    n_ordering_constraints = 0
    n_pairs = n * (n - 1) // 2
    A_ranking = np.zeros((n_pairs, n + 1))
    b_ranking = np.zeros(n_pairs)

    task_pairs = list(itertools.combinations(range(n), 2))
    task_i2copeland_score = {i: 0 for i in range(n)}
    undetermined_pairs = []
    pair2judgement = {}

    pair_inputs = [
        (
            task_data["Tasks"][i]["Task ID"],
            task_data["Tasks"][j]["Task ID"],
            task_data["Tasks"][i]["Task"],
            task_data["Tasks"][j]["Task"],
        )
        for i, j in task_pairs
    ]
    preference_entries, cost = get_time_preference_with_probability_for_pairs(
        pair_inputs=pair_inputs,
        occupation_title=task_data["Title"],
        onet_soc_code=task_data["O*NET-SOC Code"],
        per_day=False,
        use_cache=True,
        model_name=model_name,
        n_trials=n_trials,
        beam_size=beam_size,
        prob_threshold=prob_threshold,
        recompute_agreement_less_conservative=False,
        cache_details_suffix=cache_details_suffix,
    )
    total_cost += cost

    for constraint_index, task_pair in enumerate(task_pairs):
        i, j = task_pair
        entry = preference_entries[constraint_index]
        time_judgement = entry["label"]
        pair2judgement[(i, j)] = time_judgement
        if time_judgement in {"EQUAL_TIME", "CANNOT_DECIDE"}:
            undetermined_pairs.append(task_pair)
        elif time_judgement == "TASK_1":
            task_i2copeland_score[i] += 1
        elif time_judgement == "TASK_2":
            task_i2copeland_score[j] += 1
    queried_pair_set = set(pair2judgement.keys())

    print(f"Total cost of LLM queries for {len(queried_pair_set)} queried pairs: ", total_cost)

    undetermined_pair_set = set(undetermined_pairs)
    for constraint_index, task_pair in enumerate(task_pairs):
        i, j = task_pair

        '''
        We have to skip adding constraints for pairs where we cannot establish a preference via the LLM; otherwise, Copeland scores will imply the wrong ordering with the incomplete data.
        Example:
        True (unobservable) ranking: A>B>C>D>E
        LLM preferences: A>C>D>E, B>D
        Copeland scores: A = 3, C= 2, D = 1, B = 1, E = 0
        Implied ranking by copeland scores: A>C>D=B>E
            - this ranking is incorrect, as a resulting constraint would be C>B
        If we instead ignore the undetermined pairs, we get the following constraints:
        A>C>D>E, B > E
        '''
        if task_pair not in queried_pair_set or task_pair in undetermined_pair_set:
            continue
        coeff_i = 1
        coeff_j = 1

        if task_i2copeland_score[i] > task_i2copeland_score[j]:
            n_ordering_constraints += 1
            b_ranking[constraint_index] = -IMPORTANCE_TOLERANCE
            A_ranking[constraint_index, i] = -coeff_i
            A_ranking[constraint_index, j] = coeff_j
        elif task_i2copeland_score[j] > task_i2copeland_score[i]:
            n_ordering_constraints += 1
            b_ranking[constraint_index] = -IMPORTANCE_TOLERANCE
            A_ranking[constraint_index, i] = coeff_i
            A_ranking[constraint_index, j] = -coeff_j

    if find_max_span_weights:
        assert n >= 2, "find_max_span_weights requires at least two tasks."
        highest_ranked_task_i = max(
            range(n), key=lambda task_i: (task_i2copeland_score[task_i], -task_i)
        )
        lowest_ranked_task_i = min(
            range(n), key=lambda task_i: (task_i2copeland_score[task_i], task_i)
        )
        assert highest_ranked_task_i != lowest_ranked_task_i
        # Maximize w_high - w_low by minimizing -(w_high - w_low)
        c[highest_ranked_task_i] = -1
        c[lowest_ranked_task_i] = 1
        print(
            "Using max-span objective with highest-ranked task index "
            f"{highest_ranked_task_i} (score={task_i2copeland_score[highest_ranked_task_i]}) "
            "and lowest-ranked task index "
            f"{lowest_ranked_task_i} (score={task_i2copeland_score[lowest_ranked_task_i]})."
        )
    else:
        c[-1] = -1  # maximize t -> minimize -t

    print(task_i2copeland_score)
    print("n_ordering_constraints: ", n_ordering_constraints)
    A_ub_pre_ordering = copy.deepcopy(A_ub)
    b_ub_pre_ordering = copy.deepcopy(b_ub)
    A_ub = np.vstack([A_ub, A_ranking])
    b_ub = np.concatenate([b_ub, b_ranking])
    print("***Added number of ordering constraints: ", n_ordering_constraints)

    bounds = [(MIN_TIME_PER_TASK, HOURS_PER_DAY)] * n + [(0, None)]
    # takes the form of min c^T x s.t. A_eq x = b_eq, A_ub x <= b_ub, bounds
    res = linprog(
        c,
        A_eq=A_eq, b_eq=b_eq,
        A_ub=A_ub, b_ub=b_ub,
        bounds=bounds,
        method="highs"
    )
    if not res.success:
        print("LP failed to solve, trying to fix constraints")
        res, A_ub, b_ub = try_fixing_lp_constraints(task_i2copeland_score, A_ub_pre_ordering, b_ub_pre_ordering, A_ranking, b_ranking, A_eq, b_eq, bounds, n, c)
    if not res.success:
        print(f"Too many tasks to fix (n = {n}) or no solution found, returning None")
        return

    occupation_result = {
        "O*NET-SOC Code": task_data["O*NET-SOC Code"],
        "Title": task_data["Title"],
    }

    task_code2expected_freq = {}
    for task_i, task_description in enumerate(task_descriptions):
        task_code2expected_freq[task_data["Tasks"][task_i]["Task ID"]] = task_expected_freqs[task_i]
    occupation_result["Expected freq per task"] = task_code2expected_freq

    w = res.x[:n]  # the units are hours per day
    t = res.x[-1]

    assert np.isclose(np.dot(w, task_expected_freqs), time_in_cat), f"Expected {time_in_cat} hours per day, got {np.dot(w, task_expected_freqs)} hours per day"

    task_code2time = {}
    for task_i, task_description in enumerate(task_descriptions):
        task_code2time[task_data["Tasks"][task_i]["Task ID"]] = w[task_i]
    occupation_result["Time per task"] = task_code2time

    LP_json = {
        "A_eq": A_eq.tolist(),
        "b_eq": b_eq,
        "A_ub": A_ub.tolist(),
        "b_ub": b_ub.tolist(),
        "bounds": bounds,
        "c": c.tolist(),
    }

    output_suffix = "_max_span_weigths" if find_max_span_weights else ""
    with open(
        f"data/generated_data/occupation_time_per_task_{task_data['O*NET-SOC Code']}{extra_details}_chosen{output_suffix}.json",
        "w",
    ) as f:
        json.dump(occupation_result, f)

    with open(
        f"data/generated_data/occupation_time_per_task_{task_data['O*NET-SOC Code']}{extra_details}_chosen_LP{output_suffix}.json",
        "w",
    ) as f:
        json.dump(LP_json, f)

    return total_cost


if __name__ == "__main__":

    # get soc code indices as command line arguments via argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--end_index", type=int, default=879)
    parser.add_argument("--extra_details", type=str, default="")
    parser.add_argument("--model_name", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--n_trials", type=int, default=1)
    parser.add_argument("--max_cost", type=float, default=20)
    parser.add_argument("--beam_size", type=int, default=1000, help="Beam size segment in prob-threshold cache filename.")
    parser.add_argument("--prob_threshold", type=float, default=0.7)
    parser.add_argument("--use_cps_constants", action="store_true", default=False)
    parser.add_argument("--find_max_span_weights", action="store_true", default=False)
    args = parser.parse_args()

    start_index = args.start_index
    end_index = args.end_index
    extra_details = args.extra_details
    extra_details += args.model_name
    extra_details += "_ONET_30.2"
    if args.use_cps_constants:
        extra_details += "_use_cps_constants"
    model_name = args.model_name
    n_trials = args.n_trials
    max_cost = args.max_cost

    df = _load_task_ratings()
    soc_codes = df['O*NET-SOC Code'].unique().tolist()

    cost_so_far = 0
    for code in soc_codes[start_index:end_index]:
        try:
            print(f"Testing with O*NET-SOC Code: {code}")

            task_data = get_onet_task_data(code)
            task_data = impute_never_frequency(task_data)
            print("Occupation title: ", task_data["Title"])
            freq_to_time_per_day = FREQ_TO_TIME_PER_DAY
            cache_details_suffix = "_ONET_30.2"
            if args.use_cps_constants:
                cps_constants = compute_soc_constants(code)
                freq_to_time_per_day = cps_constants["FREQ_TO_TIME_PER_DAY"]
                print(
                    f"Using CPS constants for {code}: "
                    f"DAYS_PER_YEAR={cps_constants['DAYS_PER_YEAR']:.4f}, "
                    f"HOURS_WORKED_PER_YEAR={cps_constants['HOURS_WORKED_PER_YEAR']:.4f}"
                )

            cost = find_satisfying_weights(
                task_data,
                extra_details=extra_details,
                model_name=model_name,
                n_trials=n_trials,
                beam_size=args.beam_size,
                prob_threshold=args.prob_threshold,
                freq_to_time_per_day=freq_to_time_per_day,
                cache_details_suffix=cache_details_suffix,
                find_max_span_weights=args.find_max_span_weights,
            )
            if cost is None:
                print(
                    f"Skipping {code}: find_satisfying_weights returned no result "
                    "(e.g. infeasible LP or missing data path)."
                )
                continue
            cost_so_far += cost
            print("cost_so_far: ", cost_so_far)
            if cost_so_far > max_cost:
                raise ValueError(f"Total cost of LLM queries exceeded max cost of {max_cost}")
        except ValueError as exc:
            if "Total cost of LLM queries exceeded max cost" in str(exc):
                raise
            print(f"ValueError for O*NET-SOC Code {code}: {exc!r}")
            traceback.print_exc()
        except Exception as exc:
            print(f"Error for O*NET-SOC Code {code}: {exc!r}")
            traceback.print_exc()
