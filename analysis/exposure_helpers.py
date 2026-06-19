"""
CDF of task-type-weighted exposure shares (Core / Supplemental weights): Alex vs Hosseini.

  - Alex: continuous raw_time_savings_median in [0, 1]; two curves at thresholds 0.25 and 0.50
    (same construction as data_formatting.exposure_loading_utils.load_alex_exposure_mapped_to_ratings).

  - Hosseini: two curves:
      * default binary measure — exposed iff T3 or T4
      * moderate-plus measure — exposed iff T2, T3, or T4

Time weights are not used: only onet2fraction_exposed from run_weighted_exposure_analysis.

Run from repository root, e.g.:
  python3 -m analysis.figure4_exposure_share_cdf
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_FINAL = Path(__file__).resolve().parent
_REPO_ROOT = _FINAL.parent
if str(_FINAL) not in sys.path:
    sys.path.insert(0, str(_FINAL))
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import matplotlib.pyplot as plt

from data_formatting.exposure_loading_utils import (
    ALEX_EXPOSURE_PATH,
    EXPOSURE_COL,
    TASK_RATINGS_PATH,
    TASK_STATEMENTS_PATH,
    load_alex_exposure_mapped_to_ratings,
)
from figure4_exposure_share_cdf import (
    REPLICATED_EXPOSURE_PATH,
    _values_from_fraction_dict,
    load_replicated_exposure_excel,
    run_weighted_exposure_analysis,
)

EXTRA_DETAILS = "gpt-5.2_ONET_30.2_use_cps_constants_prob_thresh=0.7"
ALEX_THRESHOLDS = (0.25, 0.50)
OURS_COLORS = {
    0.25: "#8FD694",
    0.50: "#167A3B",
}
HML_MODERATE_HIGH_COLOR = "#B89AE8"
HML_HIGH_ONLY_COLOR = "#5B2A86"


def load_hosseini_exposure_t2_t3_t4(replicated_path, statements_path):
    """
    Load the replicated Hosseini file and mark tasks exposed when automation_hl is T2/T3/T4.

    Assumes replicated_path contains one automation_hl label per (ONETSOCCode, TaskID).
    Reuses load_replicated_exposure_excel for TaskType normalization and task-key handling.
    Returns a DataFrame with ONETSOCCode, TaskID, auto_hl_bi, and TaskType.
    """
    base = load_replicated_exposure_excel(replicated_path, statements_path)

    raw = pd.read_excel(replicated_path)
    assert {"ONETSOCCode", "TaskID", "automation_hl"}.issubset(raw.columns)
    automation = raw[["ONETSOCCode", "TaskID", "automation_hl"]].copy()
    automation["ONETSOCCode"] = automation["ONETSOCCode"].astype(str).str.strip()
    automation["TaskID"] = automation["TaskID"].astype(float)
    automation["automation_hl"] = automation["automation_hl"].astype(str).str.strip().str.upper()
    valid_levels = {"T0", "T1", "T2", "T3", "T4", "TF"}
    assert set(automation["automation_hl"].unique()).issubset(valid_levels), (
        f"Unexpected automation_hl values: {sorted(automation['automation_hl'].unique())}"
    )
    automation = automation.drop_duplicates(subset=["ONETSOCCode", "TaskID"], keep="first")

    out = base.merge(automation, on=["ONETSOCCode", "TaskID"], how="inner", validate="one_to_one")
    assert len(out) == len(base), "Missing automation_hl rows after merging Hosseini task labels."
    out["auto_hl_bi"] = out["automation_hl"].isin({"T2", "T3", "T4"}).astype(int)
    return out[["ONETSOCCode", "TaskID", "auto_hl_bi", "TaskType"]]


def _alex_task_keys(alex_df):
    """
    Return the set of (ONETSOCCode, TaskID) keys retained by Alex's exposure loader.

    Assumes alex_df has already applied Alex's success and occupation-coverage filters.
    """
    required = {"ONETSOCCode", "TaskID"}
    assert required.issubset(alex_df.columns), (
        f"Alex dataframe missing columns: {required - set(alex_df.columns)}"
    )
    keys = set(
        zip(
            alex_df["ONETSOCCode"].astype(str),
            alex_df["TaskID"].astype(float),
        )
    )
    assert len(keys) > 0, "No Alex task keys available for Hosseini filtering."
    return keys


def _restrict_hosseini_to_alex_task_universe(hosseini_df, alex_task_keys):
    """
    Keep exactly the Hosseini rows whose (ONETSOCCode, TaskID) keys are retained by Alex.

    Assumes every Alex-retained key has a Hosseini label. The marker column lets
    run_weighted_exposure_analysis know this is an intentionally reduced task set.
    """
    required = {"ONETSOCCode", "TaskID", "auto_hl_bi", "TaskType"}
    assert required.issubset(hosseini_df.columns), (
        f"Hosseini dataframe missing columns: {required - set(hosseini_df.columns)}"
    )
    normalized = hosseini_df.copy()
    normalized["ONETSOCCode"] = normalized["ONETSOCCode"].astype(str)
    normalized["TaskID"] = normalized["TaskID"].astype(float)
    duplicate_keys = normalized.duplicated(subset=["ONETSOCCode", "TaskID"], keep=False)
    assert not duplicate_keys.any(), (
        "Hosseini dataframe has duplicate task keys: "
        f"{normalized.loc[duplicate_keys, ['ONETSOCCode', 'TaskID']].to_dict('records')[:10]}"
    )
    normalized_keys = set(zip(normalized["ONETSOCCode"], normalized["TaskID"]))
    missing_keys = alex_task_keys - normalized_keys
    assert len(missing_keys) == 0, (
        f"Hosseini labels missing for Alex-retained task keys: {sorted(missing_keys)[:10]}"
    )
    filtered = normalized[
        normalized.apply(lambda row: (row["ONETSOCCode"], row["TaskID"]) in alex_task_keys, axis=1)
    ].copy()
    assert len(filtered) == len(alex_task_keys), (
        f"Expected one Hosseini row per Alex-retained key, got {len(filtered)} vs {len(alex_task_keys)}"
    )
    filtered[EXPOSURE_COL] = 1.0
    return filtered


def _plot_cdf(alex_by_thresh, hosseini_default_result, hosseini_t2_t3_t4_result, out_path, alex_thresholds):
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update({
        "font.family": "Times New Roman",
        "axes.titlesize": 14,
        "axes.labelsize": 13,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 10,
    })

    plt.figure(figsize=(7.2, 4.8))
    all_vals = []

    for i, thresh in enumerate(alex_thresholds):
        assert thresh in OURS_COLORS, f"Missing color for Alex threshold {thresh}"
        label = (
            "Ours (moderate + high exposure tasks)"
            if thresh == 0.25
            else "Ours (high exposure tasks only)"
        )
        fr = _values_from_fraction_dict(alex_by_thresh[thresh]["onet2fraction_exposed"])
        assert len(fr) > 0, f"No Alex occupation shares at threshold {thresh}"
        all_vals.append(fr)
        s = np.sort(fr)
        f = np.arange(len(s), 0, -1) / len(s)
        plt.step(
            s,
            f,
            where="post",
            color=OURS_COLORS[thresh],
            linestyle="-",
            linewidth=2.4,
            label=label,
        )

    fr_h = _values_from_fraction_dict(hosseini_default_result["onet2fraction_exposed"])
    assert len(fr_h) > 0, "No Hosseini occupation shares"
    all_vals.append(fr_h)

    fr_h_moderate = _values_from_fraction_dict(hosseini_t2_t3_t4_result["onet2fraction_exposed"])
    assert len(fr_h_moderate) > 0, "No Hosseini T2/T3/T4 occupation shares"
    all_vals.append(fr_h_moderate)
    sh_moderate = np.sort(fr_h_moderate)
    fh_moderate = np.arange(len(sh_moderate), 0, -1) / len(sh_moderate)
    plt.step(
        sh_moderate,
        fh_moderate,
        where="post",
        color=HML_MODERATE_HIGH_COLOR,
        linestyle="-",
        linewidth=2.4,
        label="HM&L (moderate + high exposure tasks)",
    )

    sh = np.sort(fr_h)
    fh = np.arange(len(sh), 0, -1) / len(sh)
    plt.step(
        sh,
        fh,
        where="post",
        color=HML_HIGH_ONLY_COLOR,
        linestyle="-",
        linewidth=2.4,
        label="HM&L (high exposure tasks only)",
    )

    stacked = np.concatenate(all_vals)
    lo = float(np.min(stacked))
    hi = float(np.max(stacked))
    assert hi >= lo
    margin = (hi - lo) * 0.02 if hi > lo else 0.02
    plt.xlim(lo - margin, hi + margin)

    plt.grid(axis="y", color="#D8D8D8", linewidth=0.7, alpha=0.7)
    plt.xlabel("Share of tasks exposed to AI")
    plt.ylabel("Fraction of occupations at or above")
    plt.title("Task exposure intensity across the 762 occupations")
    plt.legend(loc="upper right", frameon=False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def main():
    alex_by_thresh = {}
    alex_task_universe = None
    for thresh in ALEX_THRESHOLDS:
        alex_df, _ = load_alex_exposure_mapped_to_ratings(
            ALEX_EXPOSURE_PATH,
            TASK_RATINGS_PATH,
            TASK_STATEMENTS_PATH,
            exposure_threshold=thresh,
        )
        keys_for_threshold = _alex_task_keys(alex_df)
        if alex_task_universe is None:
            alex_task_universe = keys_for_threshold
        else:
            assert alex_task_universe == keys_for_threshold, (
                f"Alex retained task universe changed at threshold {thresh}."
            )
        alex_by_thresh[thresh] = run_weighted_exposure_analysis(
            alex_df,
            EXTRA_DETAILS,
            source_label=f"alex_thresh_{thresh}",
            update_website_json=False,
            quiet=True,
        )
    assert alex_task_universe is not None

    hosseini_df = load_replicated_exposure_excel(REPLICATED_EXPOSURE_PATH, TASK_STATEMENTS_PATH)
    hosseini_df = _restrict_hosseini_to_alex_task_universe(hosseini_df, alex_task_universe)
    
    hosseini_result = run_weighted_exposure_analysis(
        hosseini_df,
        EXTRA_DETAILS,
        source_label="hosseini_T3_T4",
        update_website_json=False,
        quiet=True,
    )

    hosseini_t2_t3_t4_df = load_hosseini_exposure_t2_t3_t4(
        REPLICATED_EXPOSURE_PATH,
        TASK_STATEMENTS_PATH,
    )
    hosseini_t2_t3_t4_df = _restrict_hosseini_to_alex_task_universe(
        hosseini_t2_t3_t4_df,
        alex_task_universe,
    )
    hosseini_t2_t3_t4_result = run_weighted_exposure_analysis(
        hosseini_t2_t3_t4_df,
        EXTRA_DETAILS,
        source_label="hosseini_T2_T3_T4",
        update_website_json=False,
        quiet=True,
    )

    out_png = _REPO_ROOT / "plots" / "cdf_exposure_shares_alex_thresh_25_50_hosseini_T2T3T4.png"
    _plot_cdf(alex_by_thresh, hosseini_result, hosseini_t2_t3_t4_result, out_png, ALEX_THRESHOLDS)
    print(f"Wrote {out_png}")


if __name__ == "__main__":
    main()
