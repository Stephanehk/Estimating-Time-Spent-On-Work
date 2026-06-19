#!/usr/bin/env python3
"""Compute per-SOC CPS hours constants used to convert O*NET frequency bins to task
instances per day (the --use_cps_constants path).

Maps CPS ASEC workers to O*NET occupations via:
    CPS OCC (Census 4-digit codes) -> 2018 SOC -> O*NET-SOC 2019

The Census crosswalk contains ~100 broad SOC codes (ending in 0, e.g. "25-2020") and
~38 wildcard codes (with X placeholders, e.g. "29-12XX") that don't match any O*NET code
exactly. We expand broad/wildcard codes into their matching detailed SOC codes via prefix
matching, and apply residual-aware filtering so that "Other X" categories don't contaminate
estimates for occupations that have their own dedicated Census OCC code with CPS workers.

The single public entrypoint is `compute_soc_constants(soc_code)`, which returns a dict with
a per-SOC FREQ_TO_TIME_PER_DAY mapping (and DAYS_PER_YEAR / HOURS_WORKED_PER_YEAR), falling
back to the global defaults when CPS data is unavailable for the occupation. Reads:
    data/cps_data/cps_asec.csv, data/crosswalks/2018-occupation-code-list-and-crosswalk.xlsx, data/crosswalks/2019_to_SOC_Crosswalk.xlsx
"""
from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path
import numpy as np

import pandas as pd

from utils.constants import (
    DAYS_PER_YEAR,
    FREQ_TO_TIME_PER_DAY,
    HOURS_PER_DAY,
    HOURS_WORKED_PER_YEAR,
    TOTAL_HOURS_PER_DAY,
)

SOC_BASE_RE = re.compile(r"^\d{2}-\d{4}(?:\.\d{2})?$")
ONET_RE = re.compile(r"^\d{2}-\d{4}\.\d{2}$")
OCC_RE = re.compile(r"^\d{4}$")


def _clean_text(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def normalize_soc(value: object) -> str:
    text = _clean_text(value)
    if not text:
        return ""
    text = text.replace(" ", "")
    if text.endswith(".00"):
        text = text[:-3]
    return text


def normalize_onet(value: object) -> str:
    text = _clean_text(value)
    if not text:
        return ""
    text = text.replace(" ", "")
    if re.fullmatch(r"\d{2}-\d{4}", text):
        text = f"{text}.00"
    if re.fullmatch(r"\d{2}-\d{4}\.\d$", text):
        text = f"{text}0"
    return text


def normalize_occ(value: object) -> str:
    text = _clean_text(value)
    if not text:
        return ""
    if re.fullmatch(r"\d+\.0+", text):
        text = text.split(".", 1)[0]
    if re.fullmatch(r"\d+", text):
        text = text.zfill(4)
    return text


def find_sheet_and_header_row(excel_path: Path, workbook_kind: str) -> tuple[str, int]:
    xls = pd.ExcelFile(excel_path)
    found: list[tuple[str, int]] = []
    for sheet in xls.sheet_names:
        preview = pd.read_excel(excel_path, sheet_name=sheet, header=None, nrows=120, dtype=str)
        for row_idx in range(len(preview)):
            row_values = [_clean_text(v).lower() for v in preview.iloc[row_idx].tolist()]
            row_text = " | ".join([v for v in row_values if v])
            if not row_text:
                continue
            if workbook_kind == "census":
                if ("census" in row_text) and ("soc" in row_text) and ("code" in row_text):
                    found.append((sheet, row_idx))
            elif workbook_kind == "onet":
                if ("o*net" in row_text or "onet" in row_text) and ("soc" in row_text) and ("code" in row_text):
                    found.append((sheet, row_idx))
            else:
                raise ValueError(f"Unknown workbook kind: {workbook_kind}")
    if not found:
        raise ValueError(f"Could not infer sheet/header row for {workbook_kind} workbook: {excel_path}")
    # Prefer earliest header row in the sheet where a signal appears.
    found.sort(key=lambda x: x[1])
    return found[0]


def infer_columns(df: pd.DataFrame, workbook_kind: str) -> tuple[str, str]:
    cleaned_cols = {col: _clean_text(col).lower() for col in df.columns}
    if workbook_kind == "census":
        occ_candidates = [
            c for c, lc in cleaned_cols.items()
            if ("census" in lc and "code" in lc and "soc" not in lc)
        ]
        soc_candidates = [c for c, lc in cleaned_cols.items() if ("soc" in lc and "code" in lc)]
        if not occ_candidates:
            occ_candidates = [c for c, lc in cleaned_cols.items() if ("code" in lc and "soc" not in lc)]
        if not soc_candidates:
            soc_candidates = [c for c, lc in cleaned_cols.items() if "soc" in lc]
        if not occ_candidates or not soc_candidates:
            raise ValueError("Could not infer Census OCC/SOC columns.")
        return occ_candidates[0], soc_candidates[0]

    if workbook_kind == "onet":
        onet_candidates = [
            c for c, lc in cleaned_cols.items()
            if ("o*net" in lc and "code" in lc)
        ]
        soc_candidates = [
            c for c, lc in cleaned_cols.items()
            if ("soc" in lc and "code" in lc and "o*net" not in lc)
        ]
        if not onet_candidates:
            onet_candidates = [c for c, lc in cleaned_cols.items() if ("o*net" in lc and "soc" in lc)]
        if not soc_candidates:
            soc_candidates = [c for c, lc in cleaned_cols.items() if ("soc" in lc and "o*net" not in lc)]
        if not onet_candidates or not soc_candidates:
            raise ValueError("Could not infer O*NET/SOC columns.")
        return onet_candidates[0], soc_candidates[0]

    raise ValueError(f"Unknown workbook kind: {workbook_kind}")


def _expand_soc_to_detailed(soc_raw: str, valid_soc_codes: set[str]) -> list[str]:
    """Expand a Census SOC code (which may be broad or contain X wildcards)
    into all matching detailed SOC codes from the O*NET crosswalk.

    Examples:
        "25-2020" (broad)  -> ["25-2021", "25-2022", "25-2023", ...]
        "13-20XX" (wildcard) -> ["13-2011", "13-2022", ...]
        "11-1011" (detailed) -> ["11-1011"]  (if it exists in valid_soc_codes)
    """
    cleaned = normalize_soc(soc_raw)
    if not cleaned:
        return []

    # Case 1: exact match — already a detailed SOC code in the O*NET crosswalk
    if cleaned in valid_soc_codes:
        return [cleaned]

    # Case 2: contains X wildcards (e.g., "13-20XX", "37-201X", "51-4XXX")
    # Strip Xs and the dash to form a prefix, then match.
    if "X" in cleaned.upper() or "x" in cleaned:
        prefix = cleaned.upper().split("X")[0]  # e.g., "13-20XX" -> "13-20"
        prefix = prefix.rstrip("-")
        matches = sorted(s for s in valid_soc_codes if s.startswith(prefix))
        return matches

    # Case 3: broad SOC code ending in 0 (e.g., "25-2020", "53-3030")
    # Try progressively shorter prefixes until we find matches.
    if SOC_BASE_RE.match(cleaned):
        # Try 6-char prefix first (e.g., "25-202" from "25-2020")
        for prefix_len in (6, 5, 4):
            prefix = cleaned[:prefix_len]
            matches = sorted(s for s in valid_soc_codes if s.startswith(prefix))
            if matches:
                return matches

    return []


def load_census_occ_to_soc(
    census_path: Path,
    valid_soc_codes: set[str] | None = None,
    cps_occ_codes: set[str] | None = None,
) -> pd.DataFrame:
    """Load Census 2018 OCC -> SOC crosswalk, expanding broad and wildcard
    SOC codes into detailed codes when *valid_soc_codes* is provided.

    When *cps_occ_codes* is also provided, applies **residual-aware** expansion:
    wildcard/broad Census OCC codes only expand to detailed SOC codes that are
    NOT already "actively claimed" by a specific Census OCC code that has CPS
    workers.  This prevents "Other X" categories from contaminating estimates
    for occupations that have their own dedicated Census OCC code with real
    survey respondents.  SOC codes whose specific OCC has zero CPS workers are
    kept in the wildcard expansion (they would otherwise be lost entirely).

    If *valid_soc_codes* is None, falls back to the original strict-match
    behaviour (only keeps SOC codes that already match SOC_BASE_RE exactly).
    """
    sheet, header_row = find_sheet_and_header_row(census_path, "census")
    df = pd.read_excel(census_path, sheet_name=sheet, header=header_row, dtype=str)
    occ_col, soc_col = infer_columns(df, "census")

    raw = df[[occ_col, soc_col]].copy()
    raw.columns = ["occ_code_raw", "soc_code_raw"]
    raw["occ_code"] = raw["occ_code_raw"].map(normalize_occ)

    # Keep only rows with valid 4-digit Census OCC codes.
    raw = raw[raw["occ_code"].str.match(OCC_RE, na=False)]

    if valid_soc_codes is not None:
        # --- Pass 1: classify each Census row as specific vs broad/wildcard
        #     and collect the SOC codes "actively claimed" by specific OCCs
        #     that actually appear in CPS.
        actively_claimed: set[str] = set()
        row_classifications: list[tuple[str, str, str, list[str]]] = []  # (occ, soc_raw, type, expanded)

        for _, r in raw.iterrows():
            occ = r["occ_code"]
            soc_raw = r["soc_code_raw"]
            cleaned = normalize_soc(soc_raw)
            expanded = _expand_soc_to_detailed(soc_raw, valid_soc_codes)

            # Classify: "specific" if the cleaned SOC is already a valid detailed code
            if cleaned in valid_soc_codes:
                row_classifications.append((occ, soc_raw, "specific", expanded))
                if cps_occ_codes is not None and occ in cps_occ_codes:
                    actively_claimed.update(expanded)
            else:
                row_classifications.append((occ, soc_raw, "nonspecific", expanded))

        # --- Pass 2: build output, filtering overlaps for non-specific rows
        rows: list[dict] = []
        n_expanded = 0
        n_residual_filtered = 0
        for occ, soc_raw, cls, expanded in row_classifications:
            if not expanded:
                continue

            if cls == "specific":
                for soc in expanded:
                    rows.append({"occ_code": occ, "soc_code": soc})
            else:
                # Non-specific: apply residual filtering if we have CPS info
                if cps_occ_codes is not None:
                    residual = [s for s in expanded if s not in actively_claimed]
                    if not residual:
                        # All expanded SOCs are actively claimed — keep full
                        # expansion as fallback (rare edge case).
                        residual = expanded
                    filtered_count = len(expanded) - len(residual)
                    if filtered_count > 0:
                        n_residual_filtered += 1
                    for soc in residual:
                        rows.append({"occ_code": occ, "soc_code": soc})
                else:
                    for soc in expanded:
                        rows.append({"occ_code": occ, "soc_code": soc})
                n_expanded += 1

        out = pd.DataFrame(rows).drop_duplicates()
        if n_expanded:
            print(f"Expanded {n_expanded} broad/wildcard Census SOC codes into detailed O*NET SOC codes.")
        if n_residual_filtered:
            print(f"Applied residual filtering to {n_residual_filtered} wildcard/broad codes (removed SOCs already claimed by specific OCCs with CPS workers).")
    else:
        # Original strict-match behaviour.
        raw["soc_code"] = raw["soc_code_raw"].map(normalize_soc)
        out = raw[raw["soc_code"].str.match(SOC_BASE_RE, na=False)][["occ_code", "soc_code"]].drop_duplicates()

    if out.empty:
        raise ValueError("Census OCC->SOC mapping is empty after cleaning.")
    return out


def load_soc_to_onet(onet_path: Path) -> pd.DataFrame:
    sheet, header_row = find_sheet_and_header_row(onet_path, "onet")
    df = pd.read_excel(onet_path, sheet_name=sheet, header=header_row, dtype=str)
    onet_col, soc_col = infer_columns(df, "onet")

    out = df[[onet_col, soc_col]].copy()
    out.columns = ["onet_code_raw", "soc_code_raw"]
    out["onet_code"] = out["onet_code_raw"].map(normalize_onet)
    out["soc_code"] = out["soc_code_raw"].map(normalize_soc)

    out = out[out["onet_code"].str.match(ONET_RE, na=False)]
    out = out[out["soc_code"].str.match(SOC_BASE_RE, na=False)]
    out = out[["soc_code", "onet_code"]].drop_duplicates()
    if out.empty:
        raise ValueError("SOC->O*NET mapping is empty after cleaning.")
    return out


def load_cps(cps_path: Path) -> pd.DataFrame:
    cps = pd.read_csv(cps_path)
    required = ["OCC", "UHRSWORK1", "WKSWORK1", "ASECWT"]
    missing = [c for c in required if c not in cps.columns]
    if missing:
        raise ValueError(f"CPS file missing required columns: {missing}")

    cps = cps.reset_index(drop=False).rename(columns={"index": "cps_row_id"})

    # CPS UHRSWORK1 contains special values like 997/999 (not valid numeric hours worked).
    # Keep only plausible positive values for hours and weeks.
    before = len(cps)
    cps = cps[cps["ASECWT"].notna()]
    cps = cps[cps["UHRSWORK1"].notna() & (cps["UHRSWORK1"] > 0) & (cps["UHRSWORK1"] < 97)]
    cps = cps[cps["WKSWORK1"].notna() & (cps["WKSWORK1"] > 0) & (cps["WKSWORK1"] <= 52)]
    invalid_removed = before - len(cps)
    print(
        "Removed invalid CPS rows "
        "(ASECWT missing, UHRSWORK1 <= 0 or >= 97, WKSWORK1 <= 0 or > 52): "
        f"{invalid_removed}"
    )

    cps["occ_code"] = cps["OCC"].map(normalize_occ)
    occ_before = len(cps)
    cps = cps[cps["occ_code"].str.match(OCC_RE, na=False)]
    occ_removed = occ_before - len(cps)
    print(f"Removed rows with non-parseable OCC codes: {occ_removed}")

    if cps.empty:
        raise ValueError("No CPS rows remain after filtering.")
    return cps


def join_cps_occ_to_onet_long(
    cps: pd.DataFrame,
    occ_to_soc: pd.DataFrame,
    soc_to_onet: pd.DataFrame,
    weight_col: str,
) -> pd.DataFrame:
    """
    Map CPS rows to O*NET-SOC using the same OCC→2018 SOC→O*NET pipeline as compute_estimates.

    *cps* must include ``cps_row_id``, ``occ_code`` (from ``OCC`` via ``normalize_occ``), and
    *weight_col* (e.g. ``ASECWT`` for ASEC, ``WTFINL`` for monthly person weight).

    *occ_to_soc* comes from ``load_census_occ_to_soc``; *soc_to_onet* from ``load_soc_to_onet``.
    Returns one row per (CPS record, O*NET-SOC) with ``weight_frac`` = weight / ``n_onet``,
    where ``n_onet`` is the count of distinct O*NET codes for that record after deduplication.
    ``onet_code`` is passed through ``normalize_onet`` for consistent strings.
    """
    if weight_col not in cps.columns:
        raise ValueError(f"CPS DataFrame missing weight column {weight_col!r}")

    joined = cps.merge(occ_to_soc, how="inner", on="occ_code")
    joined = joined.merge(soc_to_onet, how="inner", on="soc_code")
    if joined.empty:
        raise ValueError("No rows after OCC->SOC->O*NET joins.")

    # If multiple paths map one CPS record to the same O*NET code, count it once.
    joined = joined.drop_duplicates(subset=["cps_row_id", "onet_code"])

    onet_count = joined.groupby("cps_row_id", as_index=False)["onet_code"].nunique()
    onet_count = onet_count.rename(columns={"onet_code": "n_onet"})
    if (onet_count["n_onet"] <= 0).any():
        raise ValueError("Found CPS rows with zero O*NET mappings after join.")

    joined = joined.merge(onet_count, on="cps_row_id", how="left")
    joined["weight_frac"] = joined[weight_col] / joined["n_onet"]
    joined["onet_code"] = joined["onet_code"].map(normalize_onet)
    return joined


def compute_estimates(
    cps: pd.DataFrame,
    occ_to_soc: pd.DataFrame,
    soc_to_onet: pd.DataFrame,
    requested_codes: list[str],
) -> tuple[pd.DataFrame, list[str], pd.DataFrame]:
    joined = join_cps_occ_to_onet_long(cps, occ_to_soc, soc_to_onet, "ASECWT")
    joined["annual_hours"] = joined["UHRSWORK1"] * joined["WKSWORK1"]

    joined["w_uhrs"] = joined["weight_frac"] * joined["UHRSWORK1"]
    joined["w_wks"] = joined["weight_frac"] * joined["WKSWORK1"]
    joined["w_ann"] = joined["weight_frac"] * joined["annual_hours"]

    grouped = joined.groupby("onet_code", as_index=False).agg(
        pop_weight_sum=("weight_frac", "sum"),
        sample_rows=("cps_row_id", "size"),
        sample_unique_cps_records=("cps_row_id", "nunique"),
        sum_w_uhrs=("w_uhrs", "sum"),
        sum_w_wks=("w_wks", "sum"),
        sum_w_ann=("w_ann", "sum"),
    )

    grouped["mean_uhrswork1"] = grouped["sum_w_uhrs"] / grouped["pop_weight_sum"]
    grouped["mean_wkswork1"] = grouped["sum_w_wks"] / grouped["pop_weight_sum"]
    grouped["mean_annual_hours"] = grouped["sum_w_ann"] / grouped["pop_weight_sum"]

    output = grouped[[
        "onet_code",
        "mean_uhrswork1",
        "mean_wkswork1",
        "mean_annual_hours",
        "pop_weight_sum",
        "sample_rows",
        "sample_unique_cps_records",
    ]].copy()
    output = output.rename(columns={"onet_code": "ONET_CODE"})

    missing_requested: list[str] = []
    if requested_codes:
        available = set(output["ONET_CODE"].tolist())
        missing_requested = [c for c in requested_codes if c not in available]
        output = output[output["ONET_CODE"].isin(requested_codes)].copy()
        order = {code: i for i, code in enumerate(requested_codes)}
        output["__order"] = output["ONET_CODE"].map(order)
        output = output.sort_values(["__order", "ONET_CODE"]).drop(columns=["__order"])
    else:
        output = output.sort_values("ONET_CODE")

    return output, missing_requested, joined


@lru_cache(maxsize=1)
def _all_soc_estimates() -> pd.DataFrame:
    cps_path = Path("data/cps_data/cps_asec.csv")
    census_path = Path("data/crosswalks/2018-occupation-code-list-and-crosswalk.xlsx")
    onet_path = Path("data/crosswalks/2019_to_SOC_Crosswalk.xlsx")

    cps = load_cps(cps_path)
    soc_to_onet = load_soc_to_onet(onet_path)
    valid_soc_codes = set(soc_to_onet["soc_code"].unique())
    cps_occ_codes = set(cps["occ_code"].unique())
    occ_to_soc = load_census_occ_to_soc(census_path, valid_soc_codes=valid_soc_codes, cps_occ_codes=cps_occ_codes)

    result, _, _ = compute_estimates(
        cps=cps,
        occ_to_soc=occ_to_soc,
        soc_to_onet=soc_to_onet,
        requested_codes=[],
    )
    if result.empty:
        raise ValueError("No CPS-based estimates found in compute_estimates.")
    return result.set_index("ONET_CODE")


def _hours_to_freq_constants(mean_hours_per_week: float, mean_weeks_per_year: float) -> dict:
    """Derive FREQ_TO_TIME_PER_DAY and related constants from CPS hours stats."""
    days_per_week = np.round(mean_hours_per_week / TOTAL_HOURS_PER_DAY, 0)
    days_per_year = np.round(mean_weeks_per_year * days_per_week, 0)
    days_per_month = np.round(days_per_year / 12, 0)
    hours_worked_per_year = np.round(mean_hours_per_week * mean_weeks_per_year, 0)

    freq_to_time_per_day = {
        0: 0,
        1: 1/(days_per_year*2), # yearly or less
        2: 1/days_per_year, # more than yearly (this is expected number of days worked as assumed by ATUS)
        3: 1/days_per_month, # more than monthly
        4: 1/days_per_week, # more than weekly
        5: 1, # daily
        6: 2, # several times daily (computed as twice daily)
        7: HOURS_PER_DAY# hourly or more (computed as 7 times daily); average workday is 8 hours, but here we subtract an hour because this is spread out over O*NET tasks and doesn't account for non-work related acitivities during work (i.e., eating lunch, taking breaks, etc.)
    }
    return {
        "FREQ_TO_TIME_PER_DAY": freq_to_time_per_day,
        "DAYS_PER_YEAR": days_per_year,
        "DAYS_PER_WEEK": days_per_week,
        "HOURS_WORKED_PER_YEAR": hours_worked_per_year,
        "mean_uhrswork1": mean_hours_per_week,
        "mean_wkswork1": mean_weeks_per_year,
    }


def compute_soc_constants(soc_code: str) -> dict:
    normalized_code = normalize_onet(soc_code)
    all_estimates = _all_soc_estimates()

    if normalized_code not in all_estimates.index:
        print(
            "Warning: No CPS-based estimates found for SOC/O*NET code "
            f"{soc_code}. Falling back to original FREQ_TO_TIME_PER_DAY constants."
        )
        return {
            "FREQ_TO_TIME_PER_DAY": dict(FREQ_TO_TIME_PER_DAY),
            "DAYS_PER_YEAR": DAYS_PER_YEAR,
            # Standard 5-day work week, consistent with the default DAYS_PER_YEAR=260.
            "DAYS_PER_WEEK": 5.0,
            "HOURS_WORKED_PER_YEAR": HOURS_WORKED_PER_YEAR,
            "mean_uhrswork1": float("nan"),
            "mean_wkswork1": float("nan"),
        }

    row = all_estimates.loc[normalized_code]
    return _hours_to_freq_constants(
        float(row["mean_uhrswork1"]),
        float(row["mean_wkswork1"]),
    )
