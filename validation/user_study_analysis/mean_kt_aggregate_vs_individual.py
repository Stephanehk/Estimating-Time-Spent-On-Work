"""
Ad-hoc: mean Kendall tau-b between the Copeland-aggregated human ranking and each
individual human's ranking, per occupation, under the paper's (twin>=3, practice>=7)
filter. Reuses table2_human_ranking_agreement so the worker set / aggregation
match the human-validation table exactly.
"""

import table2_human_ranking_agreement as A
from scipy.stats import kendalltau


def mean_kt_agg_vs_individual(rank_maps):
    """Per-worker tau-b and p-value of each individual ranking vs the Copeland aggregate."""
    task_ids = sorted(rank_maps[0].keys())
    for m in rank_maps:
        assert sorted(m.keys()) == task_ids
    agg = A.aggregate_copeland_rank_maps(rank_maps)
    v_agg = [agg[t] for t in task_ids]
    taus = []
    pvals = []
    for rm in rank_maps:
        v_ind = [rm[t] for t in task_ids]
        tau, p = kendalltau(v_agg, v_ind, variant="b")
        assert tau == tau, "nan tau (degenerate individual ranking)"
        assert p == p, "nan p-value"
        taus.append(tau)
        pvals.append(p)
    return sum(taus) / len(taus), taus, pvals


def main():
    all_rows = A.load_rows()
    n_twin, z_practice = 3, 7
    print(f"Filter: twin>={n_twin}, attention passed, practice>={z_practice}\n")
    hdr = (
        f"{'Occupation':<48} {'N':>3} {'mean_KT_indiv':>13} {'n_sig(p<.05)':>12} "
        f"{'frac_sig':>9} {'tau_LP':>8} {'p_LP':>9}"
    )
    print(hdr)
    print("-" * len(hdr))
    import json
    for soc_code in A.LOOP_ALL_SOC_CODES:
        lp_path = A.occupation_lp_chosen_json_path(soc_code)
        occupation_result = json.loads(lp_path.read_text(encoding="utf-8"))
        occupation_title, _ = A.load_onet_study_task_lookup(soc_code)
        rows_for_soc = A.filter_rows_for_occupation(all_rows, occupation_title)
        rank_maps = []
        for row in rows_for_soc:
            p = A.parse_payload(row)
            if not A.passes_filters(p, n_twin, z_practice):
                continue
            rm = A.ranking_row_ranks_or_none(p.get("rankingRowsTaskIds"))
            if rm is not None:
                rank_maps.append(rm)
        n = len(rank_maps)
        mean_kt, taus, pvals = mean_kt_agg_vs_individual(rank_maps)
        n_sig = sum(1 for p in pvals if p < 0.05)
        frac_sig = n_sig / n
        tau_lp, p_lp = A.kendall_copeland_human_vs_lp(rank_maps, occupation_result)
        title = (occupation_title[:45] + "...") if len(occupation_title) > 48 else occupation_title
        print(
            f"{title:<48} {n:>3} {mean_kt:>13.4f} {n_sig:>5}/{n:<6} "
            f"{frac_sig:>9.2f} {tau_lp:>8.4f} {p_lp:>9.4g}"
        )
        per_worker = ", ".join(f"{t:.2f}(p={p:.3f})" for t, p in zip(taus, pvals))
        print(f"    per-worker tau(p): {per_worker}")


if __name__ == "__main__":
    main()
