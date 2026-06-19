"""
Concise summary: how many occupations enter/exit the top-N (and bottom-N) when
switching from task-category weighting to max-span time weighting, across the
four exposure spec combinations (Alex vs Hosseini, high vs moderate+high).

Reports only set-level changes (ignoring within-top order), for N in {25, 50}.
"""
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from data_formatting.exposure_loading_utils import (
    ALEX_EXPOSURE_PATH,
    TASK_RATINGS_PATH,
    TASK_STATEMENTS_PATH,
    load_alex_exposure_mapped_to_ratings,
)
from analysis.figure4_exposure_share_cdf import (
    REPLICATED_EXPOSURE_PATH,
    load_replicated_exposure_excel,
    run_weighted_exposure_analysis,
)
from analysis.exposure_helpers import (
    _alex_task_keys,
    _restrict_hosseini_to_alex_task_universe,
    load_hosseini_exposure_t2_t3_t4,
)


EXTRA_DETAILS = "gpt-5.2_ONET_30.2_use_cps_constants_prob_thresh=0.7"
ALEX_THRESHOLD_HIGH = 0.5
ALEX_THRESHOLD_MODERATE_PLUS_HIGH = 0.25
GENERATED_DATA_DIR = _REPO_ROOT / "data" / "generated_data"
N_VALUES = (25, 50)


def _ranked(d, descending):
    items = [(str(k), float(v)) for k, v in d.items()]
    items.sort(key=lambda kv: (-kv[1] if descending else kv[1], kv[0]))
    return items


def _set_diff(d_task, d_time, n, descending):
    a = {soc for soc, _ in _ranked(d_task, descending=descending)[:n]}
    b = {soc for soc, _ in _ranked(d_time, descending=descending)[:n]}
    return {
        "n_common": len(d_task.keys() & d_time.keys()),
        "n": n,
        "in_task_only": len(a - b),
        "in_time_only": len(b - a),
        "overlap": len(a & b),
    }


def _summary(label, d_task, d_time):
    rows = []
    for n in N_VALUES:
        top = _set_diff(d_task, d_time, n, descending=True)
        bot = _set_diff(d_task, d_time, n, descending=False)
        rows.append((n, top, bot))
    print(f"\n=== {label} ===")
    print(f"{'N':<5} {'top: only-task':<16} {'top: only-time':<16} {'top: overlap':<14}  | "
          f"{'bot: only-task':<16} {'bot: only-time':<16} {'bot: overlap':<14}")
    for n, top, bot in rows:
        print(
            f"{n:<5} "
            f"{top['in_task_only']:<16} {top['in_time_only']:<16} {top['overlap']:<14}  | "
            f"{bot['in_task_only']:<16} {bot['in_time_only']:<16} {bot['overlap']:<14}"
        )


def main():
    exposure_alex_high, _ = load_alex_exposure_mapped_to_ratings(
        ALEX_EXPOSURE_PATH, TASK_RATINGS_PATH, TASK_STATEMENTS_PATH,
        exposure_threshold=ALEX_THRESHOLD_HIGH,
    )
    alex_task_universe = _alex_task_keys(exposure_alex_high)
    exposure_alex_mod, _ = load_alex_exposure_mapped_to_ratings(
        ALEX_EXPOSURE_PATH, TASK_RATINGS_PATH, TASK_STATEMENTS_PATH,
        exposure_threshold=ALEX_THRESHOLD_MODERATE_PLUS_HIGH,
    )
    assert _alex_task_keys(exposure_alex_mod) == alex_task_universe

    exposure_h_high = load_replicated_exposure_excel(REPLICATED_EXPOSURE_PATH, TASK_STATEMENTS_PATH)
    exposure_h_high = _restrict_hosseini_to_alex_task_universe(exposure_h_high, alex_task_universe)
    exposure_h_mod = load_hosseini_exposure_t2_t3_t4(REPLICATED_EXPOSURE_PATH, TASK_STATEMENTS_PATH)
    exposure_h_mod = _restrict_hosseini_to_alex_task_universe(exposure_h_mod, alex_task_universe)

    res = {}
    for key, exp_df, src in [
        ("alex_high", exposure_alex_high, "alex_high"),
        ("alex_mod", exposure_alex_mod, "alex_mod"),
        ("h_high", exposure_h_high, "h_high"),
        ("h_mod", exposure_h_mod, "h_mod"),
    ]:
        res[key] = run_weighted_exposure_analysis(
            exp_df, EXTRA_DETAILS, source_label=src,
            update_website_json=False, generated_data_dir=GENERATED_DATA_DIR, quiet=True,
        )

    _summary(
        "Alex (simulation), high only (threshold 0.5)",
        res["alex_high"]["onet2fraction_exposed"],
        res["alex_high"]["onet2fraction_exposed_time_weighted"],
    )
    _summary(
        "Alex (simulation), moderate+high (threshold 0.25)",
        res["alex_mod"]["onet2fraction_exposed"],
        res["alex_mod"]["onet2fraction_exposed_time_weighted"],
    )
    _summary(
        "Hosseini (rubric), high only (T3+T4)",
        res["h_high"]["onet2fraction_exposed"],
        res["h_high"]["onet2fraction_exposed_time_weighted"],
    )
    _summary(
        "Hosseini (rubric), moderate+high (T2+T3+T4)",
        res["h_mod"]["onet2fraction_exposed"],
        res["h_mod"]["onet2fraction_exposed_time_weighted"],
    )


if __name__ == "__main__":
    main()
