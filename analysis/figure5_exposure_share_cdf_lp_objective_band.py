"""
Same as analysis/figure4_exposure_share_cdf.py, except that for each
exposure source (Simulation-based simulation and rubric-based rubric) we load
occupation-level results under BOTH use_max_span_time_weights=True and =False, and
shade the area between the two time-weighted curves on the histogram and CCDF plots.

The task-weighted curves are identical between the two max-span variants (the task
weights do not depend on LP time-per-task), so they are drawn only once per source.

Website JSON is not updated by this script; this is a plot-only comparison.
"""
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from data_formatting.exposure_loading_utils import (
    SIMULATION_BASED_EXPOSURE_PATH,
    TASK_RATINGS_PATH,
    TASK_STATEMENTS_PATH,
    load_simulation_based_exposure_mapped_to_ratings,
)
from analysis.figure4_exposure_share_cdf import (
    RUBRIC_BASED_EXPOSURE_PATH,
    _ccdf_xy,
    _values_from_fraction_dict,
    load_rubric_based_exposure_excel,
    run_weighted_exposure_analysis,
)


def _step_survival_eval(sorted_x, survival, x_grid):
    """
    Evaluate a right-continuous survival step function (from `_ccdf_xy`) on `x_grid`.

    Assumptions:
    - sorted_x is sorted ascending and strictly aligned with survival values returned by
      `_ccdf_xy` (survival[k] = fraction of values >= sorted_x[k]).
    - For x < sorted_x[0] the survival is 1.0 (all values are >= x).
    - For x > sorted_x[-1] the survival is 0.0.
    """
    assert len(sorted_x) == len(survival)
    assert np.all(np.diff(sorted_x) >= 0)
    idx = np.searchsorted(sorted_x, x_grid, side="left")
    out = np.empty_like(x_grid, dtype=float)
    below = idx >= len(sorted_x)
    out[below] = 0.0
    in_range = ~below
    out[in_range] = survival[idx[in_range]]
    return out


def _save_max_span_band_plots(
    simulation_based_result_true,
    simulation_based_result_false,
    rubric_based_result_true,
    rubric_based_result_false,
    x_axis_label,
    filename_suffix="",
):
    """
    Two figures (histogram and CCDF) overlaying Simulation-based vs Rubric-based
    exposure distributions. Task-weighted curves drawn once per source. Time-weighted
    curves drawn for use_max_span_time_weights=True (solid) and =False (dashed) with
    the band between them shaded.
    """
    import matplotlib.pyplot as plt

    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Times New Roman", "Nimbus Roman", "Liberation Serif"]
    plt.rcParams["mathtext.fontset"] = "stix"
    plt.rcParams["font.size"] = 16
    plt.rcParams["axes.labelsize"] = 20
    plt.rcParams["xtick.labelsize"] = 16
    plt.rcParams["ytick.labelsize"] = 16
    plt.rcParams["legend.fontsize"] = 12

    Path("plots").mkdir(parents=True, exist_ok=True)

    a_task = _values_from_fraction_dict(simulation_based_result_true["onet2fraction_exposed"])
    h_task = _values_from_fraction_dict(rubric_based_result_true["onet2fraction_exposed"])
    a_time_t = _values_from_fraction_dict(simulation_based_result_true["onet2fraction_exposed_time_weighted"])
    a_time_f = _values_from_fraction_dict(simulation_based_result_false["onet2fraction_exposed_time_weighted"])
    h_time_t = _values_from_fraction_dict(rubric_based_result_true["onet2fraction_exposed_time_weighted"])
    h_time_f = _values_from_fraction_dict(rubric_based_result_false["onet2fraction_exposed_time_weighted"])

    assert len(a_task) > 0 and len(h_task) > 0, "Task-weighted exposure: no values to plot."
    assert len(a_time_t) > 0 and len(a_time_f) > 0, "Simulation time-weighted: no values to plot."
    assert len(h_time_t) > 0 and len(h_time_f) > 0, "Rubric time-weighted: no values to plot."

    sim_dark = "#08519c"
    sim_mid = "#4292c6"
    rub_dark = "#a50f15"
    rub_mid = "#ef3b2c"

    sim_task_label = "Simulation-based (task category weights)"
    sim_time_t_label = "Simulation-based (time share, max-span)"
    sim_time_f_label = "Simulation-based (time share, uniform-min)"
    rub_task_label = "Rubric-based (task category weights)"
    rub_time_t_label = "Rubric-based (time share, max-span)"
    rub_time_f_label = "Rubric-based (time share, uniform-min)"

    combined = np.concatenate([a_task, a_time_t, a_time_f, h_task, h_time_t, h_time_f])
    lo = float(np.min(combined))
    hi = float(np.max(combined))
    assert hi >= lo
    bins = np.linspace(lo, hi, 101)
    bin_left = bins[:-1]

    plt.figure(figsize=(12, 7), dpi=300)
    plt.hist(a_task, bins=bins, histtype="step", color=sim_dark, linestyle="-", linewidth=3.0, label=sim_task_label)
    plt.hist(h_task, bins=bins, histtype="step", color=rub_dark, linestyle="-", linewidth=3.0, label=rub_task_label)

    a_time_t_counts, _ = np.histogram(a_time_t, bins=bins)
    a_time_f_counts, _ = np.histogram(a_time_f, bins=bins)
    h_time_t_counts, _ = np.histogram(h_time_t, bins=bins)
    h_time_f_counts, _ = np.histogram(h_time_f, bins=bins)

    plt.fill_between(
        bin_left,
        np.minimum(a_time_t_counts, a_time_f_counts),
        np.maximum(a_time_t_counts, a_time_f_counts),
        step="post",
        color=sim_mid,
        alpha=0.3,
    )
    plt.step(bin_left, a_time_t_counts, where="post", color=sim_mid, linestyle="-", linewidth=2.5, label=sim_time_t_label)
    plt.step(bin_left, a_time_f_counts, where="post", color=sim_mid, linestyle="--", linewidth=2.5, label=sim_time_f_label)

    plt.fill_between(
        bin_left,
        np.minimum(h_time_t_counts, h_time_f_counts),
        np.maximum(h_time_t_counts, h_time_f_counts),
        step="post",
        color=rub_mid,
        alpha=0.3,
    )
    plt.step(bin_left, h_time_t_counts, where="post", color=rub_mid, linestyle="-", linewidth=2.5, label=rub_time_t_label)
    plt.step(bin_left, h_time_f_counts, where="post", color=rub_mid, linestyle="--", linewidth=2.5, label=rub_time_f_label)

    plt.xlabel(x_axis_label)
    plt.ylabel("Occupation count")
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(
        f"plots/histogram_fraction_tasks_exposed_simulation_based_vs_rubric_based_max_span_band{filename_suffix}.png",
        dpi=300,
    )
    plt.close()

    plt.figure(figsize=(12, 7), dpi=300)
    sa, fa = _ccdf_xy(a_task)
    sh, fh = _ccdf_xy(h_task)
    sta_t, fta_t = _ccdf_xy(a_time_t)
    sta_f, fta_f = _ccdf_xy(a_time_f)
    sth_t, fth_t = _ccdf_xy(h_time_t)
    sth_f, fth_f = _ccdf_xy(h_time_f)

    plt.step(sa, fa, where="post", color=sim_dark, linestyle="-", linewidth=3.0, label=sim_task_label)
    plt.step(sh, fh, where="post", color=rub_dark, linestyle="-", linewidth=3.0, label=rub_task_label)

    common_x_simulation_based = np.linspace(
        float(min(sta_t[0], sta_f[0])), float(max(sta_t[-1], sta_f[-1])), 1000
    )
    common_x_hos = np.linspace(
        float(min(sth_t[0], sth_f[0])), float(max(sth_t[-1], sth_f[-1])), 1000
    )
    simulation_based_t_eval = _step_survival_eval(sta_t, fta_t, common_x_simulation_based)
    simulation_based_f_eval = _step_survival_eval(sta_f, fta_f, common_x_simulation_based)
    hos_t_eval = _step_survival_eval(sth_t, fth_t, common_x_hos)
    hos_f_eval = _step_survival_eval(sth_f, fth_f, common_x_hos)

    plt.fill_between(
        common_x_simulation_based,
        np.minimum(simulation_based_t_eval, simulation_based_f_eval),
        np.maximum(simulation_based_t_eval, simulation_based_f_eval),
        color=sim_mid,
        alpha=0.3,
    )
    plt.step(sta_t, fta_t, where="post", color=sim_mid, linestyle="-", linewidth=2.5, label=sim_time_t_label)
    plt.step(sta_f, fta_f, where="post", color=sim_mid, linestyle="--", linewidth=2.5, label=sim_time_f_label)

    plt.fill_between(
        common_x_hos,
        np.minimum(hos_t_eval, hos_f_eval),
        np.maximum(hos_t_eval, hos_f_eval),
        color=rub_mid,
        alpha=0.3,
    )
    plt.step(sth_t, fth_t, where="post", color=rub_mid, linestyle="-", linewidth=2.5, label=rub_time_t_label)
    plt.step(sth_f, fth_f, where="post", color=rub_mid, linestyle="--", linewidth=2.5, label=rub_time_f_label)

    plt.xlabel(x_axis_label)
    plt.ylabel("Fraction of occupations at or above")
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(
        f"plots/fraction_occupations_at_or_above_fraction_tasks_exposed_simulation_based_vs_rubric_based_max_span_band{filename_suffix}.png",
        dpi=300,
    )
    plt.close()


def main():
    ONET_VERSION = "30.2"
    extra_details = f"gpt-5.2_ONET_{ONET_VERSION}_use_cps_constants_prob_thresh=0.7"

    from analysis.exposure_helpers import (
        _simulation_based_task_keys,
        _restrict_rubric_based_to_simulation_based_task_universe,
    )

    exposure_simulation_based, _ = load_simulation_based_exposure_mapped_to_ratings(
        SIMULATION_BASED_EXPOSURE_PATH, TASK_RATINGS_PATH, TASK_STATEMENTS_PATH
    )
    simulation_based_task_universe = _simulation_based_task_keys(exposure_simulation_based)

    result_simulation_based_true = run_weighted_exposure_analysis(
        exposure_simulation_based,
        extra_details,
        source_label="simulation_based_max_span_true",
        update_website_json=False,
        use_max_span_time_weights=True,
    )
    result_simulation_based_false = run_weighted_exposure_analysis(
        exposure_simulation_based,
        extra_details,
        source_label="simulation_based_max_span_false",
        update_website_json=False,
        use_max_span_time_weights=False,
    )

    exposure_rubric_based = load_rubric_based_exposure_excel(
        RUBRIC_BASED_EXPOSURE_PATH, TASK_STATEMENTS_PATH
    )
    exposure_rubric_based = _restrict_rubric_based_to_simulation_based_task_universe(
        exposure_rubric_based, simulation_based_task_universe
    )
    result_rubric_based_true = run_weighted_exposure_analysis(
        exposure_rubric_based,
        extra_details,
        source_label="rubric_based_max_span_true",
        update_website_json=False,
        use_max_span_time_weights=True,
    )
    result_rubric_based_false = run_weighted_exposure_analysis(
        exposure_rubric_based,
        extra_details,
        source_label="rubric_based_max_span_false",
        update_website_json=False,
        use_max_span_time_weights=False,
    )

    _save_max_span_band_plots(
        result_simulation_based_true,
        result_simulation_based_false,
        result_rubric_based_true,
        result_rubric_based_false,
        x_axis_label="Share of highly exposed tasks",
    )
    print(
        "Saved max-span band plots: "
        "plots/histogram_fraction_tasks_exposed_simulation_based_vs_rubric_based_max_span_band.png"
    )
    print(
        "Saved max-span band plots: "
        "plots/fraction_occupations_at_or_above_fraction_tasks_exposed_simulation_based_vs_rubric_based_max_span_band.png"
    )

    exposure_simulation_based_mod, _ = load_simulation_based_exposure_mapped_to_ratings(
        SIMULATION_BASED_EXPOSURE_PATH, TASK_RATINGS_PATH, TASK_STATEMENTS_PATH, exposure_threshold=0.25
    )
    assert _simulation_based_task_keys(exposure_simulation_based_mod) == simulation_based_task_universe, (
        "Simulation-based retained task universe changed when threshold dropped to 0.25"
    )
    result_simulation_based_mod_true = run_weighted_exposure_analysis(
        exposure_simulation_based_mod,
        extra_details,
        source_label="simulation_based_moderate_max_span_true",
        update_website_json=False,
        quiet=True,
        use_max_span_time_weights=True,
    )
    result_simulation_based_mod_false = run_weighted_exposure_analysis(
        exposure_simulation_based_mod,
        extra_details,
        source_label="simulation_based_moderate_max_span_false",
        update_website_json=False,
        quiet=True,
        use_max_span_time_weights=False,
    )

    exposure_rubric_based_mod = load_rubric_based_exposure_excel(
        RUBRIC_BASED_EXPOSURE_PATH, TASK_STATEMENTS_PATH, exposed_tiers={"T2", "T3", "T4"}
    )
    exposure_rubric_based_mod = _restrict_rubric_based_to_simulation_based_task_universe(
        exposure_rubric_based_mod, simulation_based_task_universe
    )
    result_rubric_based_mod_true = run_weighted_exposure_analysis(
        exposure_rubric_based_mod,
        extra_details,
        source_label="rubric_based_moderate_max_span_true",
        update_website_json=False,
        quiet=True,
        use_max_span_time_weights=True,
    )
    result_rubric_based_mod_false = run_weighted_exposure_analysis(
        exposure_rubric_based_mod,
        extra_details,
        source_label="rubric_based_moderate_max_span_false",
        update_website_json=False,
        quiet=True,
        use_max_span_time_weights=False,
    )

    _save_max_span_band_plots(
        result_simulation_based_mod_true,
        result_simulation_based_mod_false,
        result_rubric_based_mod_true,
        result_rubric_based_mod_false,
        x_axis_label="Share of moderately exposed tasks",
        filename_suffix="_moderate_exposure",
    )
    print(
        "Saved moderate-exposure max-span band plots: "
        "plots/histogram_fraction_tasks_exposed_simulation_based_vs_rubric_based_max_span_band_moderate_exposure.png"
    )
    print(
        "Saved moderate-exposure max-span band plots: "
        "plots/fraction_occupations_at_or_above_fraction_tasks_exposed_simulation_based_vs_rubric_based_max_span_band_moderate_exposure.png"
    )


if __name__ == "__main__":
    main()
