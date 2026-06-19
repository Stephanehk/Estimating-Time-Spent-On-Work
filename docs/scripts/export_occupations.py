"""
Export per-occupation task time-share + exposure data for the paper website.

Joins two real data sources (no fabrication, no interpolation):
  (1) website/data/analysis_results/occupation_data2saveforwebsite.json
      -> per-occupation O*NET task titles, task codes, and our estimated
         predicted-time-per-day per task (which gives the share of working time
         when divided by the 7-hour daily budget).
  (2) data/onet_features_with_exposure.xlsx
      -> per (onet_soc_code, task_id) exposure flags:
           hl_is_exposed  = rubric-based exposure (Eloundou/HM&L recomputed)
           alex_is_exposed = simulation-based exposure (Wan et al.)

Output: paper_website/data/occupations.json
  {
    "default": "31-9011.00",
    "index": [ {"code","title"}, ... ],           # sorted, for the selector
    "occupations": {
       code: {
         "code","title",
         "rubric": {"share_tasks","share_time"},
         "sim":    {"share_tasks","share_time"},
         "tasks": [ {"label","share","rubric","sim"}, ... ]   # share in [0,1] of working time
       }, ...
    }
  }

Assumptions (validated below):
  - Each occupation's predicted times per day sum to ~7 (the daily budget).
  - Task codes in (1) match task_id in (2) for the same SOC code.
  - Exposure flags are 0/1 or missing (null) -- missing flags are excluded from
    the exposed numerator and denominator-agnostic share counts.
"""
import json
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
WEBSITE_JSON = REPO / "website/data/analysis_results/occupation_data2saveforwebsite.json"
EXPOSURE_XLSX = REPO / "data/onet_features_with_exposure.xlsx"
OUT = REPO / "paper_website/data/occupations.json"

DEFAULT_CODE = "31-9011.00"  # Massage Therapists (paper Figure 1)

# Short, human-readable labels for the Figure 1 default occupation, matching the
# phrasing used in the paper's Figure 1. Keyed by O*NET task code.
MASSAGE_SHORT_LABELS = {
    "7878": "Confer with clients on medical history",
    "7879": "Apply finger & hand pressure",
    "7880": "Massage & knead muscles / soft tissue",
    "7881": "Maintain treatment records",
    "7882": "Provide postural / stretching guidance",
    "7883": "Assess soft-tissue condition & range of motion",
    "7884": "Develop client treatment plans",
    "7885": "Refer clients to other therapists",
    "7886": "Use complementary aids (lamps, compresses)",
    "7887": "Treat clients on-site / travel to them",
    "7888": "Consult other health professionals",
    "7889": "Prepare & blend oils, apply to skin",
    "20757": "Perform other adjunctive therapies",
    "23928": "Maintain massage areas (restock / sanitize)",
}


def flag_value(v):
    """Return 0/1 int for a flag, or None if missing."""
    if v is None or pd.isna(v):
        return None
    return int(round(float(v)))


def main():
    web = json.loads(WEBSITE_JSON.read_text())

    df = pd.read_excel(EXPOSURE_XLSX)
    df["onet_soc_code"] = df["onet_soc_code"].astype(str).str.strip()
    df["task_id"] = df["task_id"].astype(str).str.strip()
    # (soc, task_id) -> (hl, alex)
    rubric_lookup = {}
    sim_lookup = {}
    for soc, tid, hl, al in zip(
        df["onet_soc_code"], df["task_id"], df["hl_is_exposed"], df["alex_is_exposed"]
    ):
        rubric_lookup[(soc, tid)] = flag_value(hl)
        sim_lookup[(soc, tid)] = flag_value(al)

    occupations = {}
    index = []
    for code, rec in web.items():
        title = rec["Title"]
        codes = [str(c) for c in rec["Task codes"]]
        titles = rec["Task Titles"]
        tpd = rec["Predicted times per day per task"]
        total = sum(tpd)
        assert total > 0, f"{code} has non-positive total time"
        assert abs(total - 7.0) < 0.5, f"{code} total time {total} not ~7h"
        assert len(codes) == len(titles) == len(tpd), f"{code} length mismatch"

        is_massage = code == DEFAULT_CODE
        tasks = []
        # exposed-time/task accumulators per measure, counted only over tasks
        # that have a non-null flag for that measure
        acc = {
            "rubric": {"t_exp": 0.0, "t_all": 0.0, "n_exp": 0, "n_all": 0},
            "sim": {"t_exp": 0.0, "t_all": 0.0, "n_exp": 0, "n_all": 0},
        }
        for c, ttl, t in zip(codes, titles, tpd):
            share = t / total
            rub = rubric_lookup.get((code, c))
            sim = sim_lookup.get((code, c))
            label = MASSAGE_SHORT_LABELS.get(c, ttl) if is_massage else ttl
            tasks.append(
                {
                    "label": label,
                    "share": round(share, 5),
                    "rubric": rub,
                    "sim": sim,
                }
            )
            for key, fl in (("rubric", rub), ("sim", sim)):
                if fl is None:
                    continue
                acc[key]["t_all"] += t
                acc[key]["n_all"] += 1
                if fl == 1:
                    acc[key]["t_exp"] += t
                    acc[key]["n_exp"] += 1

        def shares(a):
            st = a["t_exp"] / a["t_all"] if a["t_all"] > 0 else None
            sk = a["n_exp"] / a["n_all"] if a["n_all"] > 0 else None
            return {
                "share_time": round(st, 5) if st is not None else None,
                "share_tasks": round(sk, 5) if sk is not None else None,
            }

        # sort tasks by descending time share for display
        tasks.sort(key=lambda x: x["share"], reverse=True)
        occupations[code] = {
            "code": code,
            "title": title,
            "rubric": shares(acc["rubric"]),
            "sim": shares(acc["sim"]),
            "tasks": tasks,
        }
        index.append({"code": code, "title": title})

    index.sort(key=lambda x: x["title"])
    out = {"default": DEFAULT_CODE, "index": index, "occupations": occupations}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, separators=(",", ":")))

    # ---- validation / report ----
    m = occupations[DEFAULT_CODE]
    print("occupations exported:", len(occupations))
    print("default:", m["title"])
    print("  rubric:", m["rubric"])
    print("  sim:   ", m["sim"])
    print("  top task:", m["tasks"][0]["label"], round(m["tasks"][0]["share"] * 100, 1), "%")
    print("  out bytes:", OUT.stat().st_size)


if __name__ == "__main__":
    main()
