"""LLM pairwise single-instance time judgements (the constraints feeding the LP).

For each pair of tasks within an occupation we ask an LM which task's single instance
typically takes longer (Figure 6 in the paper). We use one logprob-classifier call per
pair (`send_prompt_for_label_with_logprob`) and accept the label only if the chosen
token's probability is at least `prob_threshold` (default 0.7); otherwise the pair is
treated as CANNOT_DECIDE.

Every judgement is cached on disk at
  data/generated_data/<SOC>_<model>_n_trials=1<suffix>_beam_size=<beam>_task2time_preference_prob_threshold.json
The cache is shared across `prob_threshold` values (the raw label + logprob are stored,
thresholding happens at read time). With the shipped cache present, no API calls are made.
"""

import json
import os
import math
from utils.llm_utils import (
    DEFAULT_MODEL,
    send_prompt_for_label_with_logprob,
)


_REVERSE_LABEL_MAPPING = {
    "TASK_1": "TASK_2",
    "TASK_2": "TASK_1",
    "EQUAL_TIME": "EQUAL_TIME",
    "CANNOT_DECIDE": "CANNOT_DECIDE",
}


def _load_json_dict(path):
    if not os.path.exists(path):
        return {}
    with open(path, "r") as f:
        return json.load(f)


def _atomic_write_json(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w") as f:
        json.dump(data, f)
    os.replace(tmp_path, path)


def _build_time_preference_prompt_label_only(task_description_1, task_description_2, occupation_title, onet_soc_code):
    return (
        "You are a careful annotation model. You are given two O*NET tasks and you need to classify which *single-instance duration* of the tasks is typically more time-consuming for a typical worker in this occupation, under normal working conditions.\n"
        "\n"
        "You will be given:\n"
        "(1) an O*NET occupation title\n"
        "(2) two O*NET task statements.\n"
        "\n"
        "Your job: Classify which *single-instance duration* of the tasks is typically more time-consuming for a typical worker in this occupation, under normal working conditions."
        "\n"
        "IMPORTANT RULES:\n"
        "1) Disregard frequency entirely. Do NOT use how often the task is performed (daily/weekly/rarely) to infer duration.\n"
        "2) Classify only the duration of one completion of the task as written (one run-through).\n"
        "     Example: If the task is cleaning home, a single instance of this task is cleaning a home once.\n"
        "     Example: If the task is something that takes multiple sessions of work to complete, such as writing a book, then the time spent for a single instance of a task refers to the amount of time a typical worker would spend to make meaningful progress on the task in a day.\n"
        "     Example: If the task requires short and frequent instances of work, such as monitoring employees for safety violations or answering phone calls, a single instance means one run-through of monitoring employees or answering one phone call.\n"
        "3) Exclude waiting/idle time unless the person must actively monitor or be continuously engaged.\n"
        "4) Assume the typical worker performing this task does not have access to Generative AI assistance.\n"
        "\n\n"
        "OUTPUT:\n"
        "Output only a single label and nothing else: Set label to 0 if a single instance of task 1 is typically more time-consuming, 1 if a single instance of task 2 is typically more time-consuming, 2 if the tasks are typically roughly equally time-consuming, and 3 if you cannot decide.\n"
        "Also select 3 if you would need more context to decide or the label is context-dependent."

        "\n\nEXAMPLES:\n"
        "Occupation: Baber\n"
        "Task 1: Cutting hair\n"
        "Task 2: Answering phone calls\n"
        "Label: 0\n"
        "Explanation: A single instance of cutting hair (e.g., cutting one head of hair) typically takes longer than a single instance of answering phone calls (e.g., answering one phone call).\n\n"

        "Occupation: Barber\n"
        "Task 1: Managing payment from customers\n"
        "Task 2: Researching new hairstyles\n"
        "Label: 1\n"
        "Explanation: A single instance of managing payment from customers (e.g., taking one payment) typically takes less time than a single instance of researching new hairstyles (e.g., one research session).\n\n"

        "Occupation: Barber\n"
        "Task 1: Managing payment from customers\n"
        "Task 2: Answering booking phone-calls\n"
        "Label: 3\n"
        "Explanation: It is not clear which single instance of a task is more time-consuming, so you cannot decide.\n\n"

        "INPUT:\n"
        f"Occupation: {occupation_title} (O*NET-SOC: {onet_soc_code})\n"
        f"Task 1: {task_description_1}\n"
        f"Task 2: {task_description_2}\n"
        "\n"

        "Output exactly one label and nothing else:\n"
        "0 if a single instance of task 1 is typically more time-consuming, 1 if a single instance of task 2 is typically more time-consuming, 2 if the tasks are typically roughly equally time-consuming, and 3 if you cannot decide.\n"
    )


def _cache_path_beam(onet_soc_code, cache_details, per_day, beam_size, cache_mode):
    suffix = (
        f"task2time_per_day_preference_{cache_mode}"
        if per_day
        else f"task2time_preference_{cache_mode}"
    )
    return f"data/generated_data/{onet_soc_code}_{cache_details}_beam_size={beam_size}_{suffix}.json"


def _entropy_binary(p):
    assert 0.0 <= p <= 1.0, f"Invalid probability: {p}"
    if p == 0.0 or p == 1.0:
        return 0.0
    return -p * math.log(p) - (1 - p) * math.log(1 - p)


def _prob_from_token_logprob(label_logprob):
    """
    Map API token logprob to a probability in [0, 1].

    Assumes label_logprob is log P(chosen token). exp(label_logprob) should lie in [0, 1];
    allow tiny numerical slop, then clamp.
    """
    p_raw = math.exp(label_logprob)
    return min(1.0, max(0.0, p_raw))


def _label_and_logprob_to_p_task_1(label, label_logprob):
    if label in {"EQUAL_TIME", "CANNOT_DECIDE"}:
        return 0.5
    if label == "TASK_1":
        return _prob_from_token_logprob(label_logprob)
    assert label == "TASK_2", f"Unexpected label: {label}"
    p_token = _prob_from_token_logprob(label_logprob)
    return 1.0 - p_token


def _llm_label_from_cache_entry(entry):
    """Raw LLM label; supports legacy cache keys named 'label'."""
    llm = entry.get("llm_label", entry.get("label"))
    assert llm is not None, "Cache entry missing llm_label and label."
    return llm


def _effective_label_after_prob_threshold(llm_label, label_logprob, prob_threshold):
    """
    Return CANNOT_DECIDE when exp(label_logprob) < prob_threshold; else llm_label.
    """
    if label_logprob is None:
        return llm_label
    lp = float(label_logprob)
    if math.exp(lp) >= float(prob_threshold):
        return llm_label
    return "CANNOT_DECIDE"


def _p_unc_after_threshold(effective_label, llm_label, p_task_1, uncertainty):
    """When threshold forces CANNOT_DECIDE from a decisive LLM label, use neutral p_task_1."""
    if effective_label == "CANNOT_DECIDE" and llm_label != "CANNOT_DECIDE":
        return 0.5, _entropy_binary(0.5)
    return p_task_1, uncertainty


def _result_dict_for_pair(llm_label, label_logprob, p_task_1, uncertainty, prob_threshold):
    eff = _effective_label_after_prob_threshold(llm_label, label_logprob, prob_threshold)
    p_out, u_out = _p_unc_after_threshold(eff, llm_label, p_task_1, uncertainty)
    return {
        "label": eff,
        "llm_label": llm_label,
        "label_logprob": label_logprob,
        "p_task_1": p_out,
        "uncertainty": u_out,
    }


def _cache_entry_usable_for_prob_cache(cached):
    if "label_logprob" not in cached or "p_task_1" not in cached or "uncertainty" not in cached:
        return False
    return "llm_label" in cached or "label" in cached


def get_time_preference_with_probability_for_pairs(
    pair_inputs,
    occupation_title,
    onet_soc_code,
    per_day,
    use_cache=True,
    n_trials=1,
    model_name=DEFAULT_MODEL,
    beam_size=1000,
    only_use_cache=False,
    recompute_agreement_less_conservative=False,
    cache_details_suffix="",
    prob_threshold=0.7,
):
    """
    One logprob-classifier call per pair; cache is shared across prob_threshold values.
    prob_threshold: accept llm_label only if exp(label_logprob) >= prob_threshold, else CANNOT_DECIDE.

    Assumptions: n_trials == 1, per_day is False (single-instance comparison), and
    recompute_agreement_less_conservative is False. With the shipped cache present every
    pair is served from cache and no API calls are made (total_cost == 0.0).
    """
    assert n_trials == 1, "Prob-threshold mode requires exactly one trial per queried pair."
    assert not per_day, "per_day preference mode is not used in this release."
    assert not recompute_agreement_less_conservative, (
        "recompute_agreement_less_conservative is unsupported for logprob preference pairs."
    )
    assert prob_threshold is not None, "prob_threshold is required."

    cache_details = f"_{model_name}_n_trials={n_trials}{cache_details_suffix}"
    cache_path = _cache_path_beam(
        onet_soc_code=onet_soc_code,
        cache_details=cache_details,
        per_day=per_day,
        beam_size=beam_size,
        cache_mode="prob_threshold",
    )
    beam_cache = _load_json_dict(cache_path)

    results = [None] * len(pair_inputs)
    total_cost = 0.0
    valid_labels = {"TASK_1", "TASK_2", "EQUAL_TIME", "CANNOT_DECIDE"}

    for pair_index, (task_id_1, task_id_2, task_description_1, task_description_2) in enumerate(pair_inputs):
        if task_id_1 == task_id_2:
            results[pair_index] = _result_dict_for_pair(
                "EQUAL_TIME", None, 0.5, _entropy_binary(0.5), prob_threshold
            )
            continue

        key = f"{occupation_title}_{task_id_1}_{task_id_2}"
        swapped_key = f"{occupation_title}_{task_id_2}_{task_id_1}"
        if key in beam_cache and use_cache:
            cached = beam_cache[key]
            if _cache_entry_usable_for_prob_cache(cached):
                llm_label = _llm_label_from_cache_entry(cached)
                assert llm_label in valid_labels, f"Invalid cached llm_label for key {key}: {llm_label}"
                lp = cached["label_logprob"]
                p1 = float(cached["p_task_1"])
                unc = float(cached["uncertainty"])
                results[pair_index] = _result_dict_for_pair(llm_label, lp, p1, unc, prob_threshold)
                continue
        if swapped_key in beam_cache and use_cache:
            cached = beam_cache[swapped_key]
            if _cache_entry_usable_for_prob_cache(cached):
                llm_stored = _llm_label_from_cache_entry(cached)
                assert llm_stored in valid_labels, (
                    f"Invalid cached llm_label for key {swapped_key}: {llm_stored}"
                )
                llm_label = _REVERSE_LABEL_MAPPING[llm_stored]
                lp = cached["label_logprob"]
                p1 = 1.0 - float(cached["p_task_1"])
                unc = _entropy_binary(p1)
                results[pair_index] = _result_dict_for_pair(llm_label, lp, p1, unc, prob_threshold)
                continue

        if only_use_cache:
            raise ValueError(
                f"Key {key} not found in cache; only_use_cache=True requires all queried pairs to exist."
            )

        prompt = _build_time_preference_prompt_label_only(
            task_description_1=task_description_1,
            task_description_2=task_description_2,
            occupation_title=occupation_title,
            onet_soc_code=onet_soc_code,
        )
        label, label_logprob, cost = send_prompt_for_label_with_logprob(
            prompt=prompt,
            model_name=model_name,
        )
        assert label in valid_labels, f"Invalid label returned by model: {label}"
        p_task_1 = _label_and_logprob_to_p_task_1(label, label_logprob)
        uncertainty = _entropy_binary(p_task_1)
        entry_cache = {
            "llm_label": label,
            "label_logprob": float(label_logprob),
            "p_task_1": p_task_1,
            "uncertainty": uncertainty,
        }
        results[pair_index] = _result_dict_for_pair(
            label, float(label_logprob), p_task_1, uncertainty, prob_threshold
        )
        total_cost += cost
        if use_cache:
            beam_cache[key] = entry_cache

    if use_cache:
        _atomic_write_json(cache_path, beam_cache)

    if any(result is None for result in results):
        raise RuntimeError("Internal error: missing logprob preference result for one or more pairs.")
    return results, total_cost
