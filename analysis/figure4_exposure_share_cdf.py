"""
Load exposure scores from either:
  - data/exposure_scores/simulation_based_exposure.xlsx, loaded through
    data_formatting.exposure_loading_utils and mapped to O*NET-SOC code
    and task ID via Task Ratings; TaskType from Task Statements.
  - data/exposure_scores/rubric_based_exposure.xlsx (ONETSOCCode, TaskID, auto_hl_bi;
    TaskType from file or Task Statements).

For each occupation with generated time-per-task JSON, require the exposure table and
occupation_result["Time per task"] to have the exact same set of task codes (no intersection
fallback). Compute task-weighted and time-weighted exposure shares.

Website JSON (occupation_data2saveforwebsite) is updated only when using the rubric-based exposure file.
"""
import json
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from data_formatting.exposure_loading_utils import (
    SIMULATION_BASED_EXPOSURE_PATH,
    EXPOSURE_COL,
    TASK_RATINGS_PATH,
    TASK_STATEMENTS_PATH,
    _load_task_type_lookup,
    load_simulation_based_exposure_mapped_to_ratings,
)

RUBRIC_BASED_EXPOSURE_PATH = Path("data/exposure_scores/rubric_based_exposure.xlsx")


def _task_id_to_str(t):
    """Normalize task ID to string for matching (e.g. 1.0 -> '1', 1.5 -> '1.5')."""
    if isinstance(t, (int, np.integer)):
        return str(t)
    if isinstance(t, (float, np.floating)):
        if np.isnan(t):
            raise ValueError("TaskID is NaN")
        return str(int(t)) if t == int(t) else str(t)
    return str(t)


def _normalize_auto_hl_bi(val):
    """Map exposure value to 0/1 int."""
    if pd.isna(val):
        raise ValueError("auto_hl_bi / exposure value is NaN")
    return 1 if int(round(float(val))) == 1 else 0


def load_rubric_based_exposure_excel(rubric_based_path, statements_path, exposed_tiers=None):
    """
    Load data/exposure_scores/rubric_based_exposure.xlsx.
    Requires ONETSOCCode, TaskID, auto_hl_bi, automation_hl. Uses TaskType column if present;
    otherwise Task Statements.

    exposed_tiers: set of automation_hl tier labels (subset of {"T0","T1","T2","T3","T4"}) to
    treat as exposed. When None, uses the file's existing auto_hl_bi (which is T3+T4, i.e. >=80%).
    Set to {"T2","T3","T4"} for "moderately exposed (>25%)" since T2 is the first tier whose
    entire range (50-80%) lies above 25%.

    Returns DataFrame with ONETSOCCode, TaskID, auto_hl_bi (0/1), TaskType. One row per (SOC, TaskID).
    """
    assert rubric_based_path.exists(), f"Rubric-based exposure file not found: {rubric_based_path}"
    df = pd.read_excel(rubric_based_path)
    assert "ONETSOCCode" in df.columns and "TaskID" in df.columns and "auto_hl_bi" in df.columns

    valid_tiers = {"T0", "T1", "T2", "T3", "T4"}
    if exposed_tiers is not None:
        assert "automation_hl" in df.columns, "automation_hl column required when exposed_tiers is set"
        assert set(exposed_tiers).issubset(valid_tiers), (
            f"exposed_tiers must be a subset of {valid_tiers}, got {exposed_tiers}"
        )

    task_type_lookup = _load_task_type_lookup(statements_path)
    rows = []
    for _, row in df.iterrows():
        onet_soc_code = str(row["ONETSOCCode"]).strip()
        task_id = row["TaskID"]
        if pd.isna(task_id):
            raise ValueError(f"TaskID is NaN for ONETSOCCode {onet_soc_code!r}")
        task_id_f = float(task_id)
        if "TaskType" in df.columns and pd.notna(row["TaskType"]):
            task_type = str(row["TaskType"]).strip()
            assert task_type in {"Core", "Supplemental"}, f"Unexpected TaskType: {task_type!r}"
        else:
            task_type = task_type_lookup.get((onet_soc_code, task_id_f))
            assert task_type is not None, f"Task type not found for ({onet_soc_code}, {task_id_f})"
        if exposed_tiers is None:
            bi = _normalize_auto_hl_bi(row["auto_hl_bi"])
        else:
            tier = str(row["automation_hl"]).strip()
            assert tier in valid_tiers, f"Unexpected automation_hl tier {tier!r}"
            bi = 1 if tier in exposed_tiers else 0
        rows.append({
            "ONETSOCCode": onet_soc_code,
            "TaskID": task_id_f,
            "auto_hl_bi": bi,
            "TaskType": task_type,
        })
    out = pd.DataFrame(rows)
    assert len(out) > 0, "No rows loaded from rubric_based_exposure.xlsx"
    out = out.drop_duplicates(subset=["ONETSOCCode", "TaskID"], keep="first")
    return out


def run_weighted_exposure_analysis(
    exposure_df,
    extra_details,
    source_label,
    update_website_json,
    website_json_in_path=None,
    website_json_out_path=None,
    website_mirror_path=None,
    generated_data_dir=None,
    quiet=False,
    use_max_span_time_weights=True,
):
    """
    For each occupation in exposure_df that has occupation_time_per_task JSON, require
    occupation_result['Time per task'] keys to match exposure task IDs exactly (as strings).
    Computes task-type-weighted and time-weighted exposure shares.

    If update_website_json is True, website_json_in_path, website_json_out_path, and
    website_mirror_path must be set; loads the input JSON, merges exposure fields for
    matching SOCs, writes both output paths.

    generated_data_dir: directory containing occupation_time_per_task_*_chosen*.json files.
    If None, uses data/generated_data relative to the process working directory.

    quiet: if True, skip per-occupation and summary prints (for batch callers such as
    frank_paper_reproduction/run_regression_test.py).

    Returns dict with onet2fraction_exposed, onet2fraction_exposed_time_weighted,
    number_of_occupations_processed, data2save_for_website (if updated, else None).
    """
    assert source_label, "source_label must be non-empty"
    assert isinstance(exposure_df, pd.DataFrame)
    assert {"ONETSOCCode", "TaskID", "auto_hl_bi", "TaskType"}.issubset(exposure_df.columns)
    if update_website_json:
        assert website_json_in_path is not None
        assert website_json_out_path is not None
        assert website_mirror_path is not None
    assert isinstance(use_max_span_time_weights, bool)

    onet2fraction_exposed = {}
    onet2fraction_exposed_time_weighted = {}
    number_of_occupations_processed = 0
    data2save_for_website = None

    if update_website_json:
        assert website_json_in_path.exists(), f"Website JSON not found: {website_json_in_path}"
        with open(website_json_in_path, "r") as f:
            data2save_for_website = json.load(f)

    gen_dir = Path("data/generated_data") if generated_data_dir is None else Path(generated_data_dir)

    max_span_suffix = "_max_span_weigths" if use_max_span_time_weights else ""

    for onet_soc_code in exposure_df["ONETSOCCode"].unique():
        if not quiet:
            print("ONET-SOC Code: ", onet_soc_code)
        occ_json_path = (
            gen_dir
            / f"occupation_time_per_task_{onet_soc_code}{extra_details}_chosen{max_span_suffix}.json"
        )
        if not occ_json_path.exists():
            if not quiet:
                print(f"***No data found for O*NET-SOC Code: {onet_soc_code}***\n")
            continue
        with open(occ_json_path, "r") as f:
            occupation_result = json.load(f)

        number_of_occupations_processed += 1
        data_for_occupation = exposure_df[exposure_df["ONETSOCCode"] == onet_soc_code]

        time_task_codes = set(occupation_result["Time per task"].keys())
        exposure_task_ids = data_for_occupation["TaskID"].dropna().unique()
        exposure_task_codes = {_task_id_to_str(t) for t in exposure_task_ids}

        allow_partial_exposure_scores = EXPOSURE_COL in data_for_occupation.columns
        if allow_partial_exposure_scores:
            assert exposure_task_codes.issubset(time_task_codes), (
                f"Exposure has task codes absent from time-per-task for {onet_soc_code}: "
                f"{sorted(exposure_task_codes - time_task_codes)}"
            )
        elif time_task_codes != exposure_task_codes:
            raise ValueError(
                f"Task set mismatch for {onet_soc_code}: "
                f"time_per_task has {len(time_task_codes)} tasks, exposure has {len(exposure_task_codes)}. "
                f"Only in time: {sorted(time_task_codes - exposure_task_codes)}. "
                f"Only in exposure: {sorted(exposure_task_codes - time_task_codes)}."
            )

        task_id_str = data_for_occupation["TaskID"].apply(
            lambda t: _task_id_to_str(t) if pd.notna(t) else None
        )
        data_common = data_for_occupation[
            task_id_str.notna() & task_id_str.isin(time_task_codes)
        ].dropna(subset=["auto_hl_bi", "TaskType"])
        expected_rows = len(exposure_task_codes) if allow_partial_exposure_scores else len(time_task_codes)
        assert len(data_common) == expected_rows, (
            f"Expected one exposure row per scored task for {onet_soc_code}, got {len(data_common)} vs {expected_rows}"
        )

        task_weights_common = data_common["TaskType"].map({"Core": 1.0, "Supplemental": 0.5})
        weighted_exposed_common = (data_common["auto_hl_bi"] == 1).astype(float) * task_weights_common
        share = weighted_exposed_common.sum() / task_weights_common.sum()
        onet2fraction_exposed[onet_soc_code] = share

        exposed_sum = 0.0
        total_weight = 0.0
        task_code2is_exposed = {}
        for task_code in sorted(exposure_task_codes, key=lambda x: (len(x), x)):
            predicted_time = occupation_result["Time per task"][task_code]
            expected_freq = occupation_result["Expected freq per task"][task_code]
            time_weight = predicted_time * expected_freq

            mask = data_for_occupation["TaskID"].apply(
                lambda tid: _task_id_to_str(tid) if pd.notna(tid) else ""
            ) == task_code
            task_row = data_for_occupation[mask]
            assert len(task_row) == 1, (
                f"Expected exactly one exposure row for task {task_code} in {onet_soc_code}, got {len(task_row)}"
            )
            is_exposed = int(task_row["auto_hl_bi"].values[0])

            exposed_sum += time_weight * is_exposed
            total_weight += time_weight
            task_code2is_exposed[task_code] = is_exposed

        assert total_weight > 0, f"No positive time weight for {onet_soc_code}"
        time_weighted_share = exposed_sum / total_weight
        onet2fraction_exposed_time_weighted[onet_soc_code] = time_weighted_share

        if update_website_json and onet_soc_code in data2save_for_website:
            data2save_for_website[onet_soc_code]["time_task_codes==rubric_based_task_codes"] = True
            data2save_for_website[onet_soc_code]["Time weighted exposure share"] = time_weighted_share
            data2save_for_website[onet_soc_code]["Unweighted exposure share"] = share
            data2save_for_website[onet_soc_code]["task_code2is_exposed"] = task_code2is_exposed

    if update_website_json:
        assert data2save_for_website is not None
        if not quiet:
            print("# of occupations in data2save_for_website: ", len(data2save_for_website.keys()))
        website_json_out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(website_json_out_path, "w") as f:
            json.dump(data2save_for_website, f, indent=4)
        website_mirror_path.parent.mkdir(parents=True, exist_ok=True)
        with open(website_mirror_path, "w") as f:
            json.dump(data2save_for_website, f, indent=4)

    if not quiet:
        print(f"[{source_label}] Number of occupations processed: {number_of_occupations_processed}")
        print(f"[{source_label}] Mean fraction of tasks exposed (task-weighted): ", np.mean(list(onet2fraction_exposed.values())))
        print(f"[{source_label}] Median fraction of tasks exposed (task-weighted): ", np.median(list(onet2fraction_exposed.values())))
        print("------------------------------")
        print(f"[{source_label}] Mean fraction of hours exposed: ", np.mean(list(onet2fraction_exposed_time_weighted.values())))
        print(f"[{source_label}] Median fraction of hours exposed: ", np.median(list(onet2fraction_exposed_time_weighted.values())))

    return {
        "onet2fraction_exposed": onet2fraction_exposed,
        "onet2fraction_exposed_time_weighted": onet2fraction_exposed_time_weighted,
        "number_of_occupations_processed": number_of_occupations_processed,
        "data2save_for_website": data2save_for_website,
    }


def _values_from_fraction_dict(d):
    v = np.array(list(d.values()), dtype=float)
    return v[~np.isnan(v)]


def _ccdf_xy(values):
    """Sorted x and survival fraction y for a step-CDF plot of `values`."""
    assert len(values) > 0
    s = np.sort(values)
    f = np.arange(len(s), 0, -1) / len(s)
    return s, f


def _save_combined_exposure_plots(
    simulation_based_result,
    rubric_based_result,
    x_axis_label,
    filename_suffix="",
):
    """
    Overlay Simulation-based vs Rubric-based exposure distributions on two figures.
    Methods are distinguished by color (Simulation = blue, Rubric = red); weighting
    variant by shade (task category weights = dark, time share weights = mid). All
    lines are solid.

    Time-share weights are the max-span LP weights (i.e., the new default).

    x_axis_label: label used on the x-axis of both figures (e.g. "Share of highly
    exposed tasks").
    filename_suffix: appended before the .png extension (e.g. "_moderate_exposure").
    """
    import matplotlib.pyplot as plt

    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Times New Roman", "Nimbus Roman", "Liberation Serif"]
    plt.rcParams["mathtext.fontset"] = "stix"
    plt.rcParams["font.size"] = 16
    plt.rcParams["axes.labelsize"] = 20
    plt.rcParams["xtick.labelsize"] = 16
    plt.rcParams["ytick.labelsize"] = 16
    plt.rcParams["legend.fontsize"] = 14

    Path("plots").mkdir(parents=True, exist_ok=True)

    a_task = _values_from_fraction_dict(simulation_based_result["onet2fraction_exposed"])
    a_time = _values_from_fraction_dict(simulation_based_result["onet2fraction_exposed_time_weighted"])
    h_task = _values_from_fraction_dict(rubric_based_result["onet2fraction_exposed"])
    h_time = _values_from_fraction_dict(rubric_based_result["onet2fraction_exposed_time_weighted"])

    assert len(a_task) > 0 and len(a_time) > 0, "Simulation-based exposure: no values to plot."
    assert len(h_task) > 0 and len(h_time) > 0, "Rubric-based exposure: no values to plot."

    sim_dark = "#08519c"
    sim_mid = "#4292c6"
    rub_dark = "#a50f15"
    rub_mid = "#ef3b2c"

    sim_task_label = "Simulation-based exposure (task category task weights)"
    sim_time_label = "Simulation-based exposure (time share weights)"
    rub_task_label = "Rubric-based exposure (task category task weights)"
    rub_time_label = "Rubric-based exposure (time share weights)"

    combined = np.concatenate([a_task, a_time, h_task, h_time])
    lo = float(np.min(combined))
    hi = float(np.max(combined))
    assert hi >= lo
    bins = np.linspace(lo, hi, 101)

    plt.figure(figsize=(12, 7), dpi=300)
    plt.hist(a_task, bins=bins, histtype="step", color=sim_dark, linestyle="-", linewidth=3.0, label=sim_task_label)
    plt.hist(a_time, bins=bins, histtype="step", color=sim_mid, linestyle="-", linewidth=3.0, label=sim_time_label)
    plt.hist(h_task, bins=bins, histtype="step", color=rub_dark, linestyle="-", linewidth=3.0, label=rub_task_label)
    plt.hist(h_time, bins=bins, histtype="step", color=rub_mid, linestyle="-", linewidth=3.0, label=rub_time_label)
    plt.xlabel(x_axis_label)
    plt.ylabel("Occupation count")
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(f"plots/histogram_fraction_tasks_exposed_simulation_based_vs_rubric_based{filename_suffix}.png", dpi=300)
    plt.close()

    plt.figure(figsize=(12, 7), dpi=300)
    sa, fa = _ccdf_xy(a_task)
    sta, fta = _ccdf_xy(a_time)
    sh, fh = _ccdf_xy(h_task)
    sth, fth = _ccdf_xy(h_time)

    plt.step(sa, fa, where="post", color=sim_dark, linestyle="-", linewidth=3.0, label=sim_task_label)
    plt.step(sta, fta, where="post", color=sim_mid, linestyle="-", linewidth=3.0, label=sim_time_label)
    plt.step(sh, fh, where="post", color=rub_dark, linestyle="-", linewidth=3.0, label=rub_task_label)
    plt.step(sth, fth, where="post", color=rub_mid, linestyle="-", linewidth=3.0, label=rub_time_label)
    plt.xlabel(x_axis_label)
    plt.ylabel("Fraction of occupations at or above")
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(f"plots/fraction_occupations_at_or_above_fraction_tasks_exposed_simulation_based_vs_rubric_based{filename_suffix}.png", dpi=300)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_max_span_time_weights", action="store_true", default=True)
    args = parser.parse_args()
    assert args.use_max_span_time_weights, "Only max-span time weights are supported."

    ONET_VERSION = "30.2"
    extra_details = f"gpt-5.2_ONET_{ONET_VERSION}_use_cps_constants_prob_thresh=0.7"

    # Imported inside main to avoid the module-level import cycle:
    # exposure_helpers imports from this module.
    from analysis.exposure_helpers import (
        _simulation_based_task_keys,
        _restrict_rubric_based_to_simulation_based_task_universe,
    )

    exposure_simulation_based, _ = load_simulation_based_exposure_mapped_to_ratings(
        SIMULATION_BASED_EXPOSURE_PATH, TASK_RATINGS_PATH, TASK_STATEMENTS_PATH
    )
    simulation_based_task_universe = _simulation_based_task_keys(exposure_simulation_based)
    result_simulation_based = run_weighted_exposure_analysis(
        exposure_simulation_based,
        extra_details,
        source_label="simulation_based",
        update_website_json=False,
    )

    exposure_rubric_based = load_rubric_based_exposure_excel(
        RUBRIC_BASED_EXPOSURE_PATH, TASK_STATEMENTS_PATH
    )
    exposure_rubric_based = _restrict_rubric_based_to_simulation_based_task_universe(
        exposure_rubric_based, simulation_based_task_universe
    )
    result_rubric_based = run_weighted_exposure_analysis(
        exposure_rubric_based,
        extra_details,
        source_label="rubric_based",
        update_website_json=False,
    )

    _save_combined_exposure_plots(
        result_simulation_based,
        result_rubric_based,
        x_axis_label="Share of highly exposed tasks",
    )
    print("Saved combined plots: plots/histogram_fraction_tasks_exposed_simulation_based_vs_rubric_based.png")
    print("Saved combined plots: plots/fraction_occupations_at_or_above_fraction_tasks_exposed_simulation_based_vs_rubric_based.png")

    # Moderate-exposure variant: a task counts as exposed if its exposure > 25%.
    # Simulation-based uses raw_time_savings_median >= 0.25. Rubric-based uses tier >= T2 (T2 is the
    # first tier whose entire range, 50-80%, lies above 25%).
    exposure_simulation_based_mod, _ = load_simulation_based_exposure_mapped_to_ratings(
        SIMULATION_BASED_EXPOSURE_PATH, TASK_RATINGS_PATH, TASK_STATEMENTS_PATH, exposure_threshold=0.25
    )
    assert _simulation_based_task_keys(exposure_simulation_based_mod) == simulation_based_task_universe, (
        "Simulation-based retained task universe changed when threshold dropped to 0.25"
    )
    result_simulation_based_mod = run_weighted_exposure_analysis(
        exposure_simulation_based_mod,
        extra_details,
        source_label="simulation_based_moderate",
        update_website_json=False,
        quiet=True,
    )

    exposure_rubric_based_mod = load_rubric_based_exposure_excel(
        RUBRIC_BASED_EXPOSURE_PATH, TASK_STATEMENTS_PATH, exposed_tiers={"T2", "T3", "T4"}
    )
    exposure_rubric_based_mod = _restrict_rubric_based_to_simulation_based_task_universe(
        exposure_rubric_based_mod, simulation_based_task_universe
    )
    result_rubric_based_mod = run_weighted_exposure_analysis(
        exposure_rubric_based_mod,
        extra_details,
        source_label="rubric_based_moderate",
        update_website_json=False,
        quiet=True,
    )

    _save_combined_exposure_plots(
        result_simulation_based_mod,
        result_rubric_based_mod,
        x_axis_label="Share of moderately exposed tasks",
        filename_suffix="_moderate_exposure",
    )
    print("Saved moderate-exposure plots: plots/histogram_fraction_tasks_exposed_simulation_based_vs_rubric_based_moderate_exposure.png")
    print("Saved moderate-exposure plots: plots/fraction_occupations_at_or_above_fraction_tasks_exposed_simulation_based_vs_rubric_based_moderate_exposure.png")


if __name__ == "__main__":
    main()
