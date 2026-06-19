"""
Analyze user-study CSV exports: comprehension filters (twin-pair consistency,
attention check, practice accuracy) and mean pairwise Kendall tau-b; lists pairwise p-values
and counts how many are below 0.05.
Also reports attention-only: attentionCheck.selectedAnswer == FIRST with no other filters.
Mean session wall time uses CSV columns started_at and submitted_at (not payload timestamps).
Each detailed filter block prints payload userOccupation as a JSON array in CSV session order.

For each filter: (1) mean pairwise Kendall tau-b across participants' ranking rows;
(2) Kendall tau-b between Copeland-aggregated human ranks and Copeland scores from LLM
pairwise labels in the beam cache; (3) Kendall tau-b between Copeland-aggregated human
ranks and the LP 'Time per task' ranking from
data/generated_data/occupation_time_per_task_<SOC><EXTRA_DETAILS>_chosen.json;
(4) per-participant pairwise accuracy vs the LLM on
pairs where exp(label_logprob) > LLM_PROB_THRESHOLD (reference = effective label after the
same threshold rule as time_classification).

Beam JSON matches get_time_preference_with_probability_for_pairs /
find_time_per_task.find_satisfying_weights (gpt-5.2, n_trials=1, beam 1000, prob 0.7,
per_day=False, cache_details_suffix _ONET_30.2). Task ids come from user_study/data/onet2task.json.

Twin pairs: mainStepId ending in _reversed paired with the id without that suffix.
Consistency: preference_verdict(taskIdsPresented, selectedAnswer) matches on both
(SAME matches SAME; CANT_TELL matches CANT_TELL; FIRST/SECOND imply the same preferred task id).

Practice: PRACTICE_GRADED indices match user_study/app.js createBaseSteps order;
indices 3 and 6 are judgement-only and excluded.

After the detailed (n_twin, z) report blocks, prints a compact table for z in {5,6,7,8}
and n_twin in {2,3,4} when --show_all_filter_results is passed.

By default, only two filter sections print: attention check passed, and twin pairs>=3 with
practice>=7. Use --show_all_filter_results for every (n_twin, z) block and the compact grid.

CLI: --soc_code sets the O*NET-SOC code for onet2task.json, beam cache, and LP chosen JSON.
"""

import argparse
import csv
import itertools
import json
import math
import re
from datetime import datetime
from pathlib import Path

# started_at / submitted_at: space date-time, optional fractional seconds (any length), Z or ±offset.
_CSV_TS = re.compile(
    r"^(\d{4}-\d{2}-\d{2})[ T](\d{2}):(\d{2}):(\d{2})(?:\.(\d+))?(Z|[+-]\d{2}(?::\d{2})?)$"
)

from scipy.stats import kendalltau

# Release layout: this file lives at validation/user_study_analysis/, so the repo root is three levels up.
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
CSV_PATH = Path(__file__).resolve().parent / "user_responses" / "user_study_responses_rows.csv"
# onet2task.json ships alongside this script in the release.
ONET2TASK_PATH = Path(__file__).resolve().parent / "onet2task.json"

EXTRA_DETAILS = "gpt-5.2_ONET_30.2_use_cps_constants_prob_thresh=0.7"
DEFAULT_USER_STUDY_ONET_SOC_CODE = "43-6014.00"
LOOP_ALL_SOC_CODES = [
    "23-1011.00",
    "13-1111.00",
    "43-6014.00",
    "43-4051.00",
    "15-1252.00",
    "11-3121.00"
]
_ONET_SOC_CODE_RE = re.compile(r"^\d{2}-\d{4}\.\d{2}$")
LLM_MODEL_NAME = "gpt-5.2"
LLM_N_TRIALS = 1
LLM_BEAM_SIZE = 1000
LLM_PROB_THRESHOLD = 0.7
LLM_CACHE_DETAILS_SUFFIX = "_ONET_30.2"

# Indices into practiceQuestions (0-based) with objective correctAnswer in app.js; skips barber_q2, nurse_q2.
PRACTICE_GRADED = [
    (0, "FIRST"),
    (1, "SECOND"),
    (2, "FIRST"),
    (4, "SECOND"),
    (5, "FIRST"),
    (7, "SECOND"),
    (8, "FIRST"),
    (9, "SECOND"),
]

REV_SUFFIX = "_reversed"
EXPECTED_TWIN_PAIRS = 5

# Default report: twin consistency >= 3 and practice correct >= 7 (plus attention-only block).
DEFAULT_FILTER_CONFIGS = [(3, 7)]

FULL_FILTER_CONFIGS = [
    (4, 8),
    (4, 7),
    (4, 5),
    (4, 4),
    (4, 3),
    (3, 8),
    (3, 7),
    (3, 6),
    (3, 5),
    (3, 4),
    (3, 3),
    (2, 8),
    (2, 7),
    (2, 6),
    (2, 5),
    (2, 4),
    (2, 3),
]


def preference_verdict(task_ids, selected_answer):
    """Map a comparison answer to a verdict: preferred task id, SAME, or CANT_TELL."""
    a, b = task_ids[0], task_ids[1]
    assert len(task_ids) == 2
    if selected_answer == "FIRST":
        return a
    if selected_answer == "SECOND":
        return b
    if selected_answer == "SAME":
        return "SAME"
    if selected_answer == "CANT_TELL":
        return "CANT_TELL"
    assert False, f"unexpected selectedAnswer: {selected_answer!r}"


def twin_consistent_count(main_questions):
    """Count how many of the five *_reversed / base pairs have matching verdicts. Returns 0 if structure is wrong."""
    by_id = {q["mainStepId"]: q for q in main_questions}
    reversed_ids = [k for k in by_id if k.endswith(REV_SUFFIX)]
    if len(reversed_ids) != EXPECTED_TWIN_PAIRS:
        return 0
    n_ok = 0
    for rid in reversed_ids:
        base = rid[: -len(REV_SUFFIX)]
        if base not in by_id:
            return 0
        q1 = by_id[base]
        q2 = by_id[rid]
        v1 = preference_verdict(q1["taskIdsPresented"], q1["selectedAnswer"])
        v2 = preference_verdict(q2["taskIdsPresented"], q2["selectedAnswer"])
        if v1 == v2:
            n_ok += 1
    return n_ok


def practice_correct_count(practice_questions):
    """Count correct answers among PRACTICE_GRADED slots only (8 questions)."""
    
    n = 0
    for idx, exp in PRACTICE_GRADED:
        got = practice_questions[idx]["selectedAnswer"]
        if got == exp:
            n += 1
    return n


def ranking_row_ranks_or_none(ranking_rows_task_ids):
    """Map each task id to its row index rank, or None if not exactly 10 unique ids in 10 rows."""
    
    flat = []
    for row in ranking_rows_task_ids:
        if not isinstance(row, list):
            return None
        for tid in row:
            flat.append(tid)
    if len(flat) != 10 or len(set(flat)) != 10:
        return None
    rank_by_task = {}
    for r, row in enumerate(ranking_rows_task_ids):
        for tid in row:
            rank_by_task[tid] = r
    return rank_by_task


def mean_pairwise_kendall(rank_maps):
    """Pairwise Kendall tau-b (two-sided p from scipy); returns (mean_tau, p_values)."""
    n = len(rank_maps)
    if n < 2:
        return float("nan"), []
    task_ids = sorted(rank_maps[0].keys())
    for m in rank_maps:
        assert sorted(m.keys()) == task_ids
    vecs = [[m[t] for t in task_ids] for m in rank_maps]
    taus = []
    ps = []
    for va, vb in itertools.combinations(vecs, 2):
        tau, p = kendalltau(va, vb, variant="b")
        taus.append(tau)
        ps.append(p)
    return sum(taus) / len(taus), ps


def mean_pairwise_jaccard_top_bottom(rank_maps, k=3):
    n = len(rank_maps)
    if n < 2:
        return float("nan"), float("nan")
    ordered_task_lists = []
    for rank_by_task in rank_maps:
        assert len(rank_by_task) >= k
        ordered = sorted(rank_by_task.keys(), key=lambda task_id: (rank_by_task[task_id], task_id))
        ordered_task_lists.append(ordered)
    top_jaccards = []
    bottom_jaccards = []
    for list_a, list_b in itertools.combinations(ordered_task_lists, 2):
        top_a = set(list_a[:k])
        top_b = set(list_b[:k])
        bottom_a = set(list_a[-k:])
        bottom_b = set(list_b[-k:])
        top_jaccards.append(len(top_a & top_b) / len(top_a | top_b))
        bottom_jaccards.append(len(bottom_a & bottom_b) / len(bottom_a | bottom_b))
    assert top_jaccards and bottom_jaccards
    return sum(top_jaccards) / len(top_jaccards), sum(bottom_jaccards) / len(bottom_jaccards)


def copeland_scores_from_rank_map(rank_by_task):
    """rank_by_task: lower row index = more time per instance. Pairwise win = higher time."""
    tasks = sorted(rank_by_task.keys())
    scores = {t: 0.0 for t in tasks}
    for i in range(len(tasks)):
        for j in range(i + 1, len(tasks)):
            a, b = tasks[i], tasks[j]
            ra, rb = rank_by_task[a], rank_by_task[b]
            if ra < rb:
                scores[a] += 1.0
            elif rb < ra:
                scores[b] += 1.0
            else:
                scores[a] += 0.5
                scores[b] += 0.5
    return scores


def aggregate_scores_to_ranks(agg_scores):
    """Higher aggregate Copeland score -> rank 0 (most time)."""
    items = sorted(agg_scores.items(), key=lambda x: (-x[1], x[0]))
    ranks = {}
    row = 0
    i = 0
    while i < len(items):
        sc = items[i][1]
        group = []
        while i < len(items) and items[i][1] == sc:
            group.append(items[i][0])
            i += 1
        for t in group:
            ranks[t] = row
        row += 1
    return ranks


def aggregate_copeland_rank_maps(rank_maps):
    """Sum Copeland scores across participants; return dict task_id -> rank row (0 = top)."""
    task_ids = sorted(rank_maps[0].keys())
    for m in rank_maps:
        assert sorted(m.keys()) == task_ids
    total = {t: 0.0 for t in task_ids}
    for rm in rank_maps:
        cs = copeland_scores_from_rank_map(rm)
        for t in task_ids:
            total[t] += cs[t]
    return aggregate_scores_to_ranks(total)


_REVERSE_LABEL_MAPPING = {
    "TASK_1": "TASK_2",
    "TASK_2": "TASK_1",
    "EQUAL_TIME": "EQUAL_TIME",
    "CANNOT_DECIDE": "CANNOT_DECIDE",
}


def _llm_label_from_cache_entry(entry):
    llm = entry.get("llm_label", entry.get("label"))
    assert llm is not None, "Cache entry missing llm_label and label."
    return llm


def _cache_entry_usable_for_prob_cache(cached):
    if "label_logprob" not in cached or "p_task_1" not in cached or "uncertainty" not in cached:
        return False
    return "llm_label" in cached or "label" in cached


def _effective_label_after_prob_threshold(llm_label, label_logprob, prob_threshold):
    if label_logprob is None:
        return llm_label
    lp = float(label_logprob)
    if math.exp(lp) >= float(prob_threshold):
        return llm_label
    return "CANNOT_DECIDE"


def beam_prob_threshold_cache_path(soc_code):
    cache_details = f"_{LLM_MODEL_NAME}_n_trials={LLM_N_TRIALS}{LLM_CACHE_DETAILS_SUFFIX}"
    suffix = "task2time_preference_prob_threshold"
    return (
        _REPO_ROOT
        / "data"
        / "generated_data"
        / f"{soc_code}_{cache_details}_beam_size={LLM_BEAM_SIZE}_{suffix}.json"
    )


def load_onet_study_task_lookup(soc_code):
    dataset = json.loads(ONET2TASK_PATH.read_text(encoding="utf-8"))
    occ = dataset[soc_code]
    codes = occ["taskCodes"]
    names = occ["taskNames"]
    assert len(codes) == len(names)
    id_to_text = {str(codes[i]): names[i] for i in range(len(codes))}
    return occ["occupationName"], id_to_text


def build_pair_inputs(id_to_text, task_ids_sorted):
    pair_inputs = []
    # print ("id_to_text: ", id_to_text)
    # print ("task_ids_sorted: ", task_ids_sorted)
    for a, b in itertools.combinations(task_ids_sorted, 2):
        assert a in id_to_text and b in id_to_text
        pair_inputs.append((a, b, id_to_text[a], id_to_text[b]))
    return pair_inputs


def raw_llm_fields_for_pair(pair, occupation_title, beam_cache):
    """Return (llm_label, label_logprob) in the orientation of pair (task_id_1, task_id_2, ...)."""
    valid_labels = {"TASK_1", "TASK_2", "EQUAL_TIME", "CANNOT_DECIDE"}
    task_id_1, task_id_2, _, _ = pair
    if task_id_1 == task_id_2:
        return "EQUAL_TIME", None
    key = f"{occupation_title}_{task_id_1}_{task_id_2}"
    swapped_key = f"{occupation_title}_{task_id_2}_{task_id_1}"
    if key in beam_cache:
        cached = beam_cache[key]
        if _cache_entry_usable_for_prob_cache(cached):
            llm_label = _llm_label_from_cache_entry(cached)
            assert llm_label in valid_labels
            return llm_label, cached["label_logprob"]
    if swapped_key in beam_cache:
        cached = beam_cache[swapped_key]
        if _cache_entry_usable_for_prob_cache(cached):
            llm_stored = _llm_label_from_cache_entry(cached)
            assert llm_stored in valid_labels
            llm_label = _REVERSE_LABEL_MAPPING[llm_stored]
            return llm_label, cached["label_logprob"]
    raise ValueError(
        f"Missing usable beam cache for pair keys {occupation_title}_{task_id_1}_{task_id_2}"
    )


def preference_entry_for_pair(pair, occupation_title, beam_cache, prob_threshold):
    """Match time_classification.get_time_preference_with_probability_for_pairs cache branch."""
    llm_raw, lp = raw_llm_fields_for_pair(pair, occupation_title, beam_cache)
    eff = _effective_label_after_prob_threshold(llm_raw, lp, prob_threshold)
    return {"label": eff}


def load_llm_pairwise_preferences(id_to_text, occupation_title, task_ids_sorted, soc_code):
    pair_inputs = build_pair_inputs(id_to_text, task_ids_sorted)
    cache_path = beam_prob_threshold_cache_path(soc_code)
    beam_cache = json.loads(cache_path.read_text(encoding="utf-8"))
    preference_entries = [
        preference_entry_for_pair(p, occupation_title, beam_cache, LLM_PROB_THRESHOLD)
        for p in pair_inputs
    ]
    return pair_inputs, preference_entries, beam_cache


def build_filtered_high_confidence_pair_specs(pair_inputs, occupation_title, beam_cache):
    """
    Pairs where exp(label_logprob) > LLM_PROB_THRESHOLD (strict). Each item is
    (pair_index, effective_label_after_threshold).
    """
    out = []
    for i, pair in enumerate(pair_inputs):
        llm_raw, lp = raw_llm_fields_for_pair(pair, occupation_title, beam_cache)
        if lp is None:
            continue
        if math.exp(float(lp)) <= LLM_PROB_THRESHOLD:
            continue
        eff = _effective_label_after_prob_threshold(llm_raw, lp, LLM_PROB_THRESHOLD)
        out.append((i, eff))
    return out


def human_pairwise_label_from_rank(rank_by_task, task_id_1, task_id_2):
    """Same semantics as LLM: TASK_1 = first task id in pair takes longer per instance."""
    ra = rank_by_task[task_id_1]
    rb = rank_by_task[task_id_2]
    if ra < rb:
        return "TASK_1"
    if rb < ra:
        return "TASK_2"
    return "EQUAL_TIME"


def copeland_ranks_from_main_questions(main_questions, task_ids):
    """
    Build Copeland rank rows from a participant's mainQuestions pairwise answers.
    TASK winner gets +1; SAME gives +0.5 to both; CANT_TELL gives +0 to both.

    Assumptions:
    - task_ids are exactly the 10 ranked study task ids for this participant.
    - every main question compares exactly two task ids from task_ids.
    """
    tid_set = set(task_ids)
    scores = {t: 0.0 for t in task_ids}
    for q in main_questions:
        pair = q["taskIdsPresented"]
        assert len(pair) == 2
        a, b = str(pair[0]), str(pair[1])
        assert a in tid_set and b in tid_set
        verdict = preference_verdict(pair, q["selectedAnswer"])
        if verdict == a:
            scores[a] += 1.0
        elif verdict == b:
            scores[b] += 1.0
        elif verdict == "SAME":
            scores[a] += 0.5
            scores[b] += 0.5
        else:
            assert verdict == "CANT_TELL"
    return aggregate_scores_to_ranks(scores)


def per_participant_mainq_copeland_kendall(passing_rows):
    """
    Returns list of (participant_id, tau, p_value).
    tau/p_value are nan when participant has no valid rankingRowsTaskIds.
    """
    rows_out = []
    for row in passing_rows:
        p = parse_payload(row)
        pid = participant_id_from_row(row)
        rm = ranking_row_ranks_or_none(p.get("rankingRowsTaskIds"))
        if rm is None:
            rows_out.append((pid, float("nan"), float("nan")))
            continue
        task_ids = sorted(rm.keys())
        main_q = p["mainQuestions"]
        mainq_copeland_ranks = copeland_ranks_from_main_questions(main_q, task_ids)
        assert sorted(mainq_copeland_ranks.keys()) == task_ids
        v_rank = [rm[t] for t in task_ids]
        v_mainq = [mainq_copeland_ranks[t] for t in task_ids]
        tau, p_value = kendalltau(v_rank, v_mainq, variant="b")
        rows_out.append((pid, tau, p_value))
    return rows_out


def participant_id_from_row(row):
    pid = (row.get("prolific_id") or "").strip()
    if pid:
        return pid
    return row.get("session_id", "?")


def user_occupations_in_csv_order(passing_rows):
    """
    Values of payload['userOccupation'] in CSV row order (one entry per passing session).
    Null in JSON when the field is absent or empty (app stores null when unset).
    """
    out = []
    for row in passing_rows:
        p = parse_payload(row)
        uo = p.get("userOccupation")
        assert uo is None or isinstance(uo, str)
        out.append(uo if uo else None)
    return out


def rank_rows_from_rank_map(rank_by_task):
    """
    Convert task->row map to sorted row groups for readable printing.
    Returns list of (row_index, [task_ids_sorted_within_row]).
    """
    by_row = {}
    for task_id, row in rank_by_task.items():
        if row not in by_row:
            by_row[row] = []
        by_row[row].append(str(task_id))
    out = []
    for row in sorted(by_row.keys()):
        out.append((row, sorted(by_row[row])))
    return out


def format_rank_rows(rank_rows):
    """
    Render rank rows as: 0:[a,b] | 1:[c] | ...
    """
    parts = []
    for row_idx, tasks in rank_rows:
        parts.append(f"{row_idx}:[{', '.join(tasks)}]")
    return " | ".join(parts)


def per_participant_pairwise_accuracy(passing_rows, pair_inputs, filtered_specs):
    """
    Returns list of (participant_id, n_correct, n_total, accuracy).
    n_total is len(filtered_specs). Participants without a valid ranking get accuracy nan.
    """
    if pair_inputs is None or not filtered_specs:
        return []
    n_cmp = len(filtered_specs)
    rows_out = []
    for row in passing_rows:
        p = parse_payload(row)
        rm = ranking_row_ranks_or_none(p.get("rankingRowsTaskIds"))
        pid = participant_id_from_row(row)
        if rm is None:
            rows_out.append((pid, 0, n_cmp, float("nan")))
            continue
        ok = 0
        for pair_idx, eff in filtered_specs:
            pair = pair_inputs[pair_idx]
            a, b = str(pair[0]), str(pair[1])
            hum = human_pairwise_label_from_rank(rm, a, b)
            if hum == eff:
                ok += 1
        rows_out.append((pid, ok, n_cmp, ok / n_cmp))
    return rows_out


def filter_rows_for_occupation(csv_rows, expected_occupation_title):
    """
    Keep only rows whose payload occupation exactly matches expected_occupation_title.
    """
    kept = []
    for row in csv_rows:
        p = parse_payload(row)
        occupation = p.get("occupation")
        assert isinstance(occupation, str), "payload.occupation must be a string"
        if occupation == expected_occupation_title:
            kept.append(row)
    return kept


def first_valid_task_ids_from_rows(csv_rows, expected_task_ids):
    """
    Return sorted task ids from the first valid ranking row that exactly matches
    expected_task_ids. If no such row exists, return None.
    """
    expected = set(expected_task_ids)
    for row in csv_rows:
        p = parse_payload(row)
        rm = ranking_row_ranks_or_none(p.get("rankingRowsTaskIds"))
        if rm is None:
            continue
        found_soc_task_ids = set(rm.keys())
        assert len(found_soc_task_ids) == 10
        for task_id in found_soc_task_ids:
            assert task_id in expected
        return sorted(rm.keys())
    return None


def llm_copeland_ranks_from_pairwise(task_ids, pair_inputs, preference_entries):
    """Copeland scores from LLM pairwise labels (TASK_1 / TASK_2 / ties); then rank rows."""
    tid_set = set(task_ids)
    scores = {t: 0.0 for t in task_ids}
    for entry, pair in zip(preference_entries, pair_inputs):
        a, b = str(pair[0]), str(pair[1])
        assert a in tid_set and b in tid_set
        lab = entry["label"]
        if lab == "TASK_1":
            scores[a] += 1.0
        elif lab == "TASK_2":
            scores[b] += 1.0
        elif lab == "EQUAL_TIME":
            scores[a] += 0.5
            scores[b] += 0.5
        else:
            assert lab == "CANNOT_DECIDE"
    return aggregate_scores_to_ranks(scores)


def kendall_aggregate_vs_llm_pairwise(rank_maps, pair_inputs, preference_entries):
    """Kendall tau-b: aggregated user Copeland vs Copeland from LLM pairwise prefs."""
    # print ("rank_maps: ", rank_maps)
    # print ("pair_inputs: ", pair_inputs)
    # print ("preference_entries: ", preference_entries)
    if not rank_maps or pair_inputs is None:
        return float("nan"), float("nan")
    task_ids = sorted(rank_maps[0].keys())
    for m in rank_maps:
        assert sorted(m.keys()) == task_ids
    ids_from_pairs = set()
    for p in pair_inputs:
        ids_from_pairs.add(str(p[0]))
        ids_from_pairs.add(str(p[1]))
    assert ids_from_pairs == set(task_ids)
    user_ranks = aggregate_copeland_rank_maps(rank_maps)
    model_ranks = llm_copeland_ranks_from_pairwise(task_ids, pair_inputs, preference_entries)
    # print ("user_ranks: ", user_ranks)
    # print ("model_ranks: ", model_ranks)
    u = [user_ranks[t] for t in task_ids]
    v = [model_ranks[t] for t in task_ids]
    return kendalltau(u, v, variant="b")


def rank_map_higher_time_better(time_by_task, task_ids):
    """
    Build rank rows aligned with participant rankings: higher predicted time per instance
    gets a lower rank index (0 = most time). Ties share the same rank row; ordering matches
    aggregate_scores_to_ranks tie-breaking (by task id).

    Inputs: time_by_task maps task id string -> float hours (or compatible); task_ids lists
    the study tasks (same keys as rank maps). Assumes every task id is present in time_by_task.
    """
    tasks = sorted(task_ids)
    for t in tasks:
        assert str(t) in time_by_task
    items = sorted(tasks, key=lambda tid: (-float(time_by_task[str(tid)]), str(tid)))
    ranks = {}
    row = 0
    i = 0
    while i < len(items):
        v = float(time_by_task[str(items[i])])
        group = []
        while i < len(items) and float(time_by_task[str(items[i])]) == v:
            group.append(items[i])
            i += 1
        for t in group:
            ranks[t] = row
        row += 1
    return ranks


def kendall_copeland_human_vs_lp(rank_maps, occupation_result):
    """
    Kendall tau-b between Copeland-aggregated human rank rows and the LP solution ranking
    induced by the occupation_result Time per task map (higher time -> rank 0).

    Assumes occupation_result has a Time per task entry for each task in rank_maps.
    """
    if not rank_maps:
        return float("nan"), float("nan")
    task_ids = sorted(rank_maps[0].keys())
    for m in rank_maps:
        assert sorted(m.keys()) == task_ids
    time_per_task = occupation_result["Time per task"]
    user_ranks = aggregate_copeland_rank_maps(rank_maps)
    lp_ranks = rank_map_higher_time_better(time_per_task, task_ids)
    u = [user_ranks[t] for t in task_ids]
    v = [lp_ranks[t] for t in task_ids]
    return kendalltau(u, v, variant="b")


def copeland_aggregate_vs_individual_significance_counts(rank_maps):
    """
    Compare Copeland-aggregated human ranks to each individual participant ranking.
    Returns (n_p_lt_005, n_tested, n_total_valid_rankings, mean_tau_over_significant).
    n_tested excludes nan p-values from scipy.
    """
    if not rank_maps:
        return 0, 0, 0, float("nan")
    task_ids = sorted(rank_maps[0].keys())
    for m in rank_maps:
        assert sorted(m.keys()) == task_ids
    agg = aggregate_copeland_rank_maps(rank_maps)
    v_agg = [agg[t] for t in task_ids]
    n_sig = 0
    n_tested = 0
    sig_taus = []
    for rm in rank_maps:
        v_ind = [rm[t] for t in task_ids]
        tau, p = kendalltau(v_agg, v_ind, variant="b")
        if p == p:
            n_tested += 1
            if p < 0.05:
                n_sig += 1
                if tau == tau:
                    sig_taus.append(tau)
    mean_tau_sig = (
        sum(sig_taus) / len(sig_taus) if sig_taus else float("nan")
    )
    return n_sig, n_tested, len(rank_maps), mean_tau_sig


def occupation_lp_chosen_json_path(soc_code):
    """Chosen LP weights under the max-span objective (the default; --find_max_span_weights)."""
    return (
        _REPO_ROOT
        / "data"
        / "generated_data"
        / f"occupation_time_per_task_{soc_code}{EXTRA_DETAILS}_chosen_max_span_weigths.json"
    )


def load_rows():
    rows = []
    with CSV_PATH.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def parse_payload(row):
    return json.loads(row["payload"])


def parse_csv_timestamp(raw):
    """
    Parse CSV wall-clock timestamps. Pads or truncates fractional seconds to 6 digits so
    datetime.fromisoformat succeeds (e.g. Python 3.10 rejects '.44+00:00' without padding).
    """
    s = raw.strip()
    if s.endswith("+00") and not s.endswith("+00:00"):
        s = s[:-3] + "+00:00"
    m = _CSV_TS.match(s)
    assert m is not None, f"unparseable timestamp: {raw!r}"
    ymd, hh, mm, ss, frac_digits, tz = m.groups()
    if tz == "Z":
        tz = "+00:00"
    if frac_digits is None:
        iso = f"{ymd}T{hh}:{mm}:{ss}{tz}"
    else:
        assert frac_digits.isdigit()
        if len(frac_digits) < 6:
            fd = frac_digits + "0" * (6 - len(frac_digits))
        else:
            fd = frac_digits[:6]
        iso = f"{ymd}T{hh}:{mm}:{ss}.{fd}{tz}"
    return datetime.fromisoformat(iso)


def duration_seconds_from_row(row):
    start = parse_csv_timestamp(row["started_at"])
    end = parse_csv_timestamp(row["submitted_at"])
    return (end - start).total_seconds()


def mean_duration_minutes(csv_rows):
    if not csv_rows:
        return float("nan")
    secs = [duration_seconds_from_row(r) for r in csv_rows]
    return sum(secs) / len(secs) / 60.0


def passes_attention_only(payload):
    ac = payload.get("attentionCheck")
    return ac is not None and ac.get("selectedAnswer") == "FIRST"


def passes_filters(payload, n_twin_need, z_practice_need):
    m = payload["mainQuestions"]
    if twin_consistent_count(m) < n_twin_need:
        return False
    ac = payload.get("attentionCheck")
    if ac is None or ac.get("selectedAnswer") != "FIRST":
        return False
    if practice_correct_count(payload["practiceQuestions"]) < z_practice_need:
        return False
    return True


def report_block(
    label,
    passing_rows,
    pair_inputs,
    preference_entries,
    filtered_high_conf_specs,
    occupation_result,
):
    k = len(passing_rows)
    mean_min = mean_duration_minutes(passing_rows)
    rank_maps = []
    for row in passing_rows:
        p = parse_payload(row)
        rm = ranking_row_ranks_or_none(p.get("rankingRowsTaskIds"))
        if rm is not None:
            rank_maps.append(rm)
    mean_tau, p_values = mean_pairwise_kendall(rank_maps)
    mean_top3_jaccard, mean_bottom3_jaccard = mean_pairwise_jaccard_top_bottom(rank_maps, k=3)
    n_valid_rank = len(rank_maps)
    n_p_lt_005 = sum(1 for p in p_values if p < 0.05)
    tau_llm, p_llm = kendall_aggregate_vs_llm_pairwise(rank_maps, pair_inputs, preference_entries)
    tau_lp, p_lp = kendall_copeland_human_vs_lp(rank_maps, occupation_result)
    n_sig_agg_vs_ind, n_tested_agg_vs_ind, n_total_agg_vs_ind, mean_tau_sig_agg_vs_ind = (
        copeland_aggregate_vs_individual_significance_counts(rank_maps)
    )
    self_tau_rows = per_participant_mainq_copeland_kendall(passing_rows)
    self_tau_finite = [tau for _, tau, _ in self_tau_rows if tau == tau]
    mean_self_tau = (
        sum(self_tau_finite) / len(self_tau_finite) if self_tau_finite else float("nan")
    )
    acc_rows = per_participant_pairwise_accuracy(
        passing_rows, pair_inputs, filtered_high_conf_specs
    )
    print(f"--- {label} ---")
    print(f"  workers (rows passing): {k}")
    uo_list = user_occupations_in_csv_order(passing_rows)
    print(f"  userOccupation (JSON array, session order): {json.dumps(uo_list, ensure_ascii=False)}")
    print("  participant rankings (from rankingRowsTaskIds):")
    for row in passing_rows:
        p = parse_payload(row)
        pid = participant_id_from_row(row)
        rm = ranking_row_ranks_or_none(p.get("rankingRowsTaskIds"))
        if rm is None:
            print(f"    {pid}: n/a (invalid rankingRowsTaskIds)")
            continue
        print(f"    {pid}: {format_rank_rows(rank_rows_from_rank_map(rm))}")
    if k > 0:
        print(f"  mean session duration (submitted_at - started_at): {mean_min:.2f} min")
    else:
        print("  mean session duration (submitted_at - started_at): n/a")
    # print(f"  valid rankings for KT: {n_valid_rank} (excluded if not 10 unique task ids)")
    print(f"  mean pairwise KT between participant rankings: {mean_tau}")
    print(f"  mean pairwise Jaccard index for top-3 ranked tasks: {mean_top3_jaccard}")
    print(f"  mean pairwise Jaccard index for bottom-3 ranked tasks: {mean_bottom3_jaccard}")
    if p_values:
        print(f"  pairwise KT p-values between participant rankings < 0.05: {n_p_lt_005} / {len(p_values)}")
        # print("  pairwise KT p-values:")
        # for i, p in enumerate(p_values, start=1):
        #     print(f"    {i}: {p}")
    else:
        print("  pairwise p-values < 0.05: n/a (fewer than 2 valid rankings)")
    print(f"  KT correlation between Copeland-aggregated human rankings and LLM-produced ranking: {tau_llm}")
    print(f"  p-value: {p_llm}")
    print(
        f"  KT correlation between Copeland-aggregated human rankings and LP \"Time per task\" ranking: {tau_lp}"
    )
    print(f"  p-value (LP): {p_lp}")
    print(
        "  mean per-participant KT between rankingRowsTaskIds and "
        f"Copeland(mainQuestions): {mean_self_tau}"
    )
    if n_total_agg_vs_ind == 0:
        print(
            "  participants with p<0.05 for KT(Copeland-aggregated human ranking vs individual human ranking): "
            "n/a (no valid rankings)"
        )
    else:
        mean_tau_sig_s = (
            f"{mean_tau_sig_agg_vs_ind:.4f}"
            if mean_tau_sig_agg_vs_ind == mean_tau_sig_agg_vs_ind
            else "n/a"
        )
        print(
            "  participants with p<0.05 for KT(Copeland-aggregated human ranking vs individual human ranking): "
            f"{n_sig_agg_vs_ind} / {n_tested_agg_vs_ind} "
            f"(tested; {n_total_agg_vs_ind} valid rankings total); "
            f"mean KT among p<0.05: {mean_tau_sig_s}"
        )
    if filtered_high_conf_specs and acc_rows:
        # print(
        #     f"  pairwise accuracy vs LLM (exp(label_logprob) > {LLM_PROB_THRESHOLD} only; "
        #     f"ref = effective label after threshold rule): "
        #     f"{len(filtered_high_conf_specs)} comparisons per respondent with valid ranking"
        # )
        print("    Pairwise accuracy between participant preferences and LLM pairwise prefs:")
        for pid, n_ok, n_tot, acc in acc_rows:
            if acc == acc:
                print(f"    {pid}: {n_ok}/{n_tot} = {acc:.4f}")
            else:
                print(f"    {pid}: n/a (no valid ranking)")
        finite = [acc for _, _, _, acc in acc_rows if acc == acc]
        if finite:
            print(f"  mean of per-participant accuracies: {sum(finite) / len(finite):.4f}")
    else:
        print("  pairwise accuracy vs LLM (high-confidence pairs): n/a")
    print()


def print_high_practice_threshold_grid(
    all_rows, pair_inputs, preference_entries, filtered_high_conf_specs, occupation_result
):
    """
    Compact table: twin thresholds 2, 3, 4 and practice thresholds 5..8 (same rules as
    passes_filters). Columns match the one-off summary: workers, duration, pairwise KT,
    Copeland vs LLM tau, mean high-confidence pairwise accuracy.
    When pair_inputs is None (no valid ranking in CSV), LLM-dependent columns are n/a.
    """
    twin_vals = [2, 3, 4]
    practice_vals = [5, 6, 7, 8]
    print("=== Practice thresholds z >= 5, 6, 7, 8 (twin >= 2, 3, 4) — compact table ===")
    print()
    header = (
        f"{'n_twin>=':>8}  {'z>=':>3}  {'workers':>7}  {'mean_min':>8}  "
        f"{'n_valid_rank':>12}  {'mean_pairwise_tau':>17}  {'p_lt_0.05':>11}  "
        f"{'tau_Copeland_vs_LLM':>20}  {'p_llm':>8}  "
        f"{'tau_Copeland_vs_LP':>20}  {'p_lp':>8}  "
        f"{'mean_self_mainq_tau':>20}  {'mean_pairwise_acc':>18}"
    )
    print(header)
    print("-" * len(header))
    for n_twin in twin_vals:
        for z in practice_vals:
            passing_rows = []
            for row in all_rows:
                p = parse_payload(row)
                if passes_filters(p, n_twin, z):
                    passing_rows.append(row)
            k = len(passing_rows)
            if k == 0:
                print(
                    f"{n_twin:>8}  {z:>3}  {k:>7}  {'n/a':>8}  {'n/a':>12}  "
                    f"{'n/a':>17}  {'n/a':>11}  {'n/a':>20}  {'n/a':>8}  "
                    f"{'n/a':>20}  {'n/a':>8}  {'n/a':>20}  {'n/a':>18}"
                )
                continue
            mean_min = mean_duration_minutes(passing_rows)
            rank_maps = []
            for row in passing_rows:
                p = parse_payload(row)
                rm = ranking_row_ranks_or_none(p.get("rankingRowsTaskIds"))
                if rm is not None:
                    rank_maps.append(rm)
            mean_tau, p_values = mean_pairwise_kendall(rank_maps)
            n_p_lt = sum(1 for p in p_values if p < 0.05)
            n_p_tot = len(p_values)
            p_lt_s = f"{n_p_lt}/{n_p_tot}" if n_p_tot else "n/a"
            tau_s = f"{mean_tau:.4f}" if mean_tau == mean_tau else "n/a"
            if rank_maps:
                tau_lp, p_lp = kendall_copeland_human_vs_lp(rank_maps, occupation_result)
                tau_lp_s = f"{tau_lp:.4f}" if tau_lp == tau_lp else "n/a"
                p_lp_s = f"{p_lp:.4g}" if p_lp == p_lp else "n/a"
            else:
                tau_lp_s = "n/a"
                p_lp_s = "n/a"
            self_tau_rows = per_participant_mainq_copeland_kendall(passing_rows)
            self_tau_finite = [tau for _, tau, _ in self_tau_rows if tau == tau]
            mean_self_tau = (
                sum(self_tau_finite) / len(self_tau_finite)
                if self_tau_finite
                else float("nan")
            )
            mean_self_tau_s = f"{mean_self_tau:.4f}" if mean_self_tau == mean_self_tau else "n/a"
            if pair_inputs is not None and preference_entries is not None:
                tau_llm, p_llm = kendall_aggregate_vs_llm_pairwise(
                    rank_maps, pair_inputs, preference_entries
                )
                acc_rows = per_participant_pairwise_accuracy(
                    passing_rows, pair_inputs, filtered_high_conf_specs
                )
                finite = [acc for _, _, _, acc in acc_rows if acc == acc]
                mean_acc = (
                    sum(finite) / len(finite) if finite else float("nan")
                )
                tau_llm_s = f"{tau_llm:.4f}" if tau_llm == tau_llm else "n/a"
                p_llm_s = f"{p_llm:.4g}" if p_llm == p_llm else "n/a"
                acc_s = f"{mean_acc:.4f}" if mean_acc == mean_acc else "n/a"
            else:
                tau_llm_s = "n/a"
                p_llm_s = "n/a"
                acc_s = "n/a"
            print(
                f"{n_twin:>8}  {z:>3}  {k:>7}  {mean_min:>8.2f}  {len(rank_maps):>12}  "
                f"{tau_s:>17}  {p_lt_s:>11}  {tau_llm_s:>20}  {p_llm_s:>8}  "
                f"{tau_lp_s:>20}  {p_lp_s:>8}  {mean_self_tau_s:>20}  {acc_s:>18}"
            )
        print()


def parse_args():
    parser = argparse.ArgumentParser(
        description="User-study CSV analysis: comprehension filters, rankings, LLM/LP agreement."
    )
    parser.add_argument(
        "--soc_code",
        default=DEFAULT_USER_STUDY_ONET_SOC_CODE,
        metavar="SOC",
        help=(
            "O*NET-SOC code for the study occupation (key in user_study/data/onet2task.json; "
            "also selects beam cache and occupation_time_per_task_<SOC>..._chosen.json). "
            f"Default: {DEFAULT_USER_STUDY_ONET_SOC_CODE}."
        ),
    )
    parser.add_argument(
        "--show_all_filter_results",
        action="store_true",
        help=(
            "Print every twin/practice threshold report block and the compact z×n_twin grid. "
            "Default: only (2) attention check passed and twin>=3 with practice>=7."
        ),
    )
    parser.add_argument(
        "--loop_all_socs",
        action="store_true",
        help=(
            "Run the full analysis once per SOC code in "
            f"{LOOP_ALL_SOC_CODES}. When set, --soc_code is ignored."
        ),
    )
    return parser.parse_args()


def run_analysis_for_soc(args, all_rows, soc_code):
    assert _ONET_SOC_CODE_RE.match(soc_code), (
        f"invalid O*NET SOC code {soc_code!r}; expected form like 43-6014.00"
    )
    lp_path = occupation_lp_chosen_json_path(soc_code)
    assert lp_path.is_file(), f"missing {lp_path}"
    occupation_result = json.loads(lp_path.read_text(encoding="utf-8"))
    occupation_title, id_to_text = load_onet_study_task_lookup(soc_code)
    rows_for_soc = filter_rows_for_occupation(all_rows, occupation_title)
    expected_task_ids = sorted(id_to_text.keys())
    # print (rows_for_soc)
    # print (expected_task_ids)
    # assert False
    ref_task_ids = first_valid_task_ids_from_rows(rows_for_soc, expected_task_ids)
    # print ("ref_task_ids: ", ref_task_ids)
    # assert False
    if ref_task_ids is not None:
        pair_inputs, preference_entries, beam_cache = load_llm_pairwise_preferences(
            id_to_text, occupation_title, ref_task_ids, soc_code
        )
        filtered_high_conf_specs = build_filtered_high_confidence_pair_specs(
            pair_inputs, occupation_title, beam_cache
        )
    else:
        pair_inputs, preference_entries, filtered_high_conf_specs = None, None, None
    if args.show_all_filter_results:
        filter_configs = FULL_FILTER_CONFIGS
    else:
        filter_configs = DEFAULT_FILTER_CONFIGS
    print(f"CSV: {CSV_PATH}")
    print(f"total rows (sessions): {len(all_rows)}")
    print(f"rows for selected occupation ({occupation_title}): {len(rows_for_soc)}")
    print(f"O*NET SOC (study occupation): {soc_code}")
    print(
        f"LLM pairwise cache: model={LLM_MODEL_NAME}, n_trials={LLM_N_TRIALS}, "
        f"beam_size={LLM_BEAM_SIZE}, prob_threshold={LLM_PROB_THRESHOLD}, "
        f"cache_details_suffix={LLM_CACHE_DETAILS_SUFFIX!r}, per_day=False"
    )
    print(f"EXTRA_DETAILS (occupation LP label): {EXTRA_DETAILS!r}")
    print(f"LP chosen JSON: {lp_path}")
    if args.show_all_filter_results:
        print("Filter output: all threshold blocks + compact grid (--show_all_filter_results).\n")
    else:
        print(
            "Filter output: attention-only and (twin>=3, practice>=7) only. "
            "Pass --show_all_filter_results for every threshold and the grid.\n"
        )
    attention_only = []
    for row in rows_for_soc:
        p = parse_payload(row)
        if passes_attention_only(p):
            attention_only.append(row)
    report_block(
        "(2) attention check passed",
        attention_only,
        pair_inputs,
        preference_entries,
        filtered_high_conf_specs,
        occupation_result,
    )
    for n_twin, z_practice in filter_configs:
        passing_rows = []
        for row in rows_for_soc:
            p = parse_payload(row)
            if passes_filters(p, n_twin, z_practice):
                passing_rows.append(row)
        label = f"(1) # of consistent twin pairs>={n_twin}, (2) attention check passed, (3) # of correct practice answers>={z_practice}"
        report_block(
            label,
            passing_rows,
            pair_inputs,
            preference_entries,
            filtered_high_conf_specs,
            occupation_result,
        )
    if args.show_all_filter_results:
        print_high_practice_threshold_grid(
            rows_for_soc,
            pair_inputs,
            preference_entries,
            filtered_high_conf_specs,
            occupation_result,
        )


def main():
    args = parse_args()
    assert CSV_PATH.is_file(), f"missing {CSV_PATH}"
    all_rows = load_rows()
    if args.loop_all_socs:
        soc_codes = LOOP_ALL_SOC_CODES
    else:
        soc_codes = [args.soc_code.strip()]
    for idx, soc_code in enumerate(soc_codes):
        if idx > 0:
            print("\n" + "=" * 100 + "\n")
        run_analysis_for_soc(args, all_rows, soc_code)


if __name__ == "__main__":
    main()
