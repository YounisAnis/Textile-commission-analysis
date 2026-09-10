"""
Data cleaning for the Softwood commission dataset.

The raw export is a finance spreadsheet, so it carries the usual spreadsheet
problems: numbers stored as text, dash placeholders, entity names typed
inconsistently, and a few rows that break basic business rules.

Everything this module does is driven by something we actually verified in the
raw file. Nothing is guessed. Rows are flagged rather than dropped wherever the
row is still usable, and every drop is counted in the audit report.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Column naming
# ---------------------------------------------------------------------------

# The raw header has trailing spaces on some names and mixes conventions.
# We map to snake_case once, here, so nothing downstream has to guess.
COLUMN_MAP = {
    "Team_Head": "team_head",
    "Supplier": "supplier",
    "Customer": "customer",
    "Contract_Date": "contract_date",
    "Dispatch_Date": "dispatch_date",
    "Payment_Days": "payment_days",
    "Expected_Payment": "expected_payment_date",
    "Style_Name": "style_name",
    "Rate": "unit_rate_pkr",
    "Dollar_Exchange_Rate": "pkr_to_usd",
    "Dollar_Rate": "unit_rate_usd",
    "Contract_#": "contract_no",
    "Contract_Qty.": "contract_qty",
    "Shipped_Quantity": "shipped_qty",
    "Bal": "balance_qty",
    "Sales_Volume": "deal_value_usd",
    "Exchange": "usd_to_pkr",
    "Commission_Dollar_Value": "commission_usd",
    "Commission_Percentage": "commission_pct",
    "Commission_PKR_Value": "commission_pkr",
    "Commission_SW_Percentage": "sw_pct",
    "SW_Commission_PKR": "sw_commission_pkr",
    "Tax_%": "tax_pct",
    "SW_After_Tax_Value": "sw_after_tax_pkr",
    "Received": "commission_after_tax_pkr",
    "SW_Rcvd_after_Tax": "sw_received_after_tax_pkr",
}

NUMERIC_COLS = [
    "payment_days", "unit_rate_pkr", "pkr_to_usd", "unit_rate_usd",
    "contract_qty", "shipped_qty", "balance_qty", "deal_value_usd",
    "usd_to_pkr", "commission_usd", "commission_pct", "commission_pkr",
    "sw_pct", "sw_commission_pkr", "tax_pct", "sw_after_tax_pkr",
    "commission_after_tax_pkr", "sw_received_after_tax_pkr",
]

DATE_COLS = ["contract_date", "dispatch_date", "expected_payment_date"]

# ---------------------------------------------------------------------------
# Entity name fixes
# ---------------------------------------------------------------------------

# Only merges where the two spellings are the same string apart from case, or a
# single obvious transposition. Counts are from the raw file at the time of
# writing and are re-checked by the audit.
NAME_FIXES = {
    "supplier": {
        "indigo": "Indigo",              # 'indigo' (2) vs 'Indigo' (84)
        "safa tex": "Safa Tex",          # 5 vs 6
        "shorts quality": "Shorts Quality",  # 2 vs 30
    },
    "customer": {
        "tiffosi fab": "Tiffosi Fab",    # 'Tiffosi fab' (24) vs 'Tiffosi Fab' (203)
        "tifossi fab": "Tiffosi Fab",    # single-row transposition (1)
        "tiffosi knit": "Tiffosi Knit",  # 3 vs 10
        "get over": "Get Over",          # 3 vs 26
        "bizbee": "Bizzbee",             # 1 vs 45
        "cdrl fab": "CRDL Fab",          # 1 vs 2
    },
}

# Suffixes on the Customer name that mark which product line the deal sits in.
# This is not cosmetic: 'Fab' deals sit at a fixed 2% commission while 'Knit'
# deals are negotiated around 5%. See notebook 01 for the evidence.
DIVISION_RE = re.compile(
    r"[\s\-]+(fab|fabric|knit|knits|denim|woven|garment|garments|apparel)s?$",
    re.IGNORECASE,
)

# team_head holds two different kinds of value: first names of Softwood staff,
# and company names. We cannot tell what the company names mean, so we label
# them and let the analysis decide, rather than deleting or guessing.
ORG_TEAM_HEADS = {
    "SWU", "Interloop", "Eastern", "Rajby", "Lucky", "Gull Ahmed", "Azgard9",
    "Artistic", "Shafi", "Cotton Web", "TFA", "Three Star", "Active Apparel",
    "CW", "B1", "vista",
}

# ---------------------------------------------------------------------------
# Business rules used to decide what is valid
# ---------------------------------------------------------------------------

MIN_EXCHANGE, MAX_EXCHANGE = 100.0, 350.0   # PKR per USD over 2019-2025
MAX_LEAD_DAYS = 730                          # contract -> dispatch
HIGH_RATE_FLAG = 0.20                        # commission % worth a second look


@dataclass
class AuditEntry:
    step: str
    rule: str
    rows: int
    action: str


@dataclass
class CleaningResult:
    data: pd.DataFrame
    audit: list[AuditEntry] = field(default_factory=list)

    @property
    def audit_frame(self) -> pd.DataFrame:
        return pd.DataFrame([vars(a) for a in self.audit])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def to_number(series: pd.Series) -> pd.Series:
    """Turn a spreadsheet text column into numbers.

    Handles thousands separators, currency symbols, and the '-' / '$ -'
    placeholders Excel writes for an empty accounting cell.
    """
    cleaned = (
        series.astype(str)
        .str.replace(r"[,$\s]", "", regex=True)
        .replace({"-": np.nan, "": np.nan, "nan": np.nan, "None": np.nan})
    )
    return pd.to_numeric(cleaned, errors="coerce")


def split_customer(name: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Split 'Takko Fab' into brand 'Takko' and division 'fab'."""
    division = name.str.extract(DIVISION_RE, expand=False).str.lower()
    brand = name.str.replace(DIVISION_RE, "", regex=True).str.strip()
    return brand, division.fillna("garment")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def clean(raw: pd.DataFrame) -> CleaningResult:
    df = raw.copy()
    df.columns = [c.strip() for c in df.columns]
    df = df.rename(columns=COLUMN_MAP)
    audit: list[AuditEntry] = []
    start_rows = len(df)
    audit.append(AuditEntry("load", "rows in raw export", start_rows, "kept"))

    # --- types -------------------------------------------------------------
    for col in NUMERIC_COLS:
        if col in df.columns:
            before = df[col].isna().sum()
            df[col] = to_number(df[col])
            gained = int(df[col].isna().sum() - before)
            if gained:
                audit.append(AuditEntry(
                    "types", f"{col}: text placeholders could not be parsed",
                    gained, "set to missing"))

    for col in DATE_COLS:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    # --- entity names ------------------------------------------------------
    for col, fixes in NAME_FIXES.items():
        original = df[col].astype(str).str.strip()
        key = original.str.lower()
        df[col] = np.where(key.isin(fixes), key.map(fixes), original)
        changed = int((df[col] != original).sum())   # only rows whose spelling actually moved
        if changed:
            audit.append(AuditEntry(
                "names", f"{col}: case / spelling variants merged",
                changed, "renamed to majority spelling"))

    df["style_name"] = df["style_name"].astype(str).str.strip()

    # --- derived structure -------------------------------------------------
    df["customer_brand"], df["product_division"] = split_customer(df["customer"])
    df["team_head_type"] = np.where(
        df["team_head"].isin(ORG_TEAM_HEADS), "organization", "person")
    audit.append(AuditEntry(
        "structure", "team_head values that are company names, not staff names",
        int((df.team_head_type == "organization").sum()), "labelled, kept"))

    df["lead_days"] = (df.dispatch_date - df.contract_date).dt.days
    df["credit_days"] = (df.expected_payment_date - df.dispatch_date).dt.days
    df["contract_year"] = df.contract_date.dt.year
    df["contract_month"] = df.contract_date.dt.month
    df["contract_quarter"] = df.contract_date.dt.quarter

    # --- flags (row stays, we just mark it) --------------------------------
    df["flag_overshipped"] = df.shipped_qty > df.contract_qty
    df["flag_dispatch_before_contract"] = df.lead_days < 0
    df["flag_long_lead"] = df.lead_days > MAX_LEAD_DAYS
    df["flag_high_rate"] = df.commission_pct > HIGH_RATE_FLAG

    for col, label in [
        ("flag_overshipped", "shipped quantity exceeds contract quantity"),
        ("flag_dispatch_before_contract", "dispatch date earlier than contract date"),
        ("flag_long_lead", f"lead time over {MAX_LEAD_DAYS} days"),
        ("flag_high_rate", f"commission rate above {HIGH_RATE_FLAG:.0%}"),
    ]:
        audit.append(AuditEntry("flags", label, int(df[col].sum()), "flagged, kept"))

    # A negative lead time is a date entry error, so the duration itself is not
    # usable even though the rest of the row is.
    df.loc[df.flag_dispatch_before_contract, "lead_days"] = np.nan

    # --- drops (row cannot be used for commission analysis) ----------------
    drop_rules = {
        "commission rate is zero or missing":
            df.commission_pct.isna() | (df.commission_pct <= 0),
        "commission rate is 100% or more (impossible)":
            df.commission_pct >= 1,
        "contract quantity is zero or negative":
            df.contract_qty.isna() | (df.contract_qty <= 0),
        f"USD/PKR rate outside {MIN_EXCHANGE:.0f}-{MAX_EXCHANGE:.0f}":
            df.usd_to_pkr.isna() | ~df.usd_to_pkr.between(MIN_EXCHANGE, MAX_EXCHANGE),
        "contract date missing":
            df.contract_date.isna(),
    }
    to_drop = pd.Series(False, index=df.index)
    for rule, mask in drop_rules.items():
        newly = mask & ~to_drop           # count each row against its first rule
        audit.append(AuditEntry("drops", rule, int(newly.sum()), "dropped"))
        to_drop |= mask
    df = df[~to_drop].copy()

    # --- exact duplicates --------------------------------------------------
    dupes = df.duplicated().sum()
    if dupes:
        df = df.drop_duplicates()
    audit.append(AuditEntry("drops", "exact duplicate rows", int(dupes), "dropped"))

    # --- consistency of the accounting columns -----------------------------
    # deal value and commission are supposed to follow fixed formulas. We do not
    # correct them; we only measure how often the file disagrees with itself.
    expected_value = df.contract_qty * df.unit_rate_usd
    expected_comm = df.deal_value_usd * df.commission_pct
    for label, calculated, stated in [
        ("deal value != qty x unit rate", expected_value, df.deal_value_usd),
        ("commission != deal value x rate", expected_comm, df.commission_usd),
    ]:
        usable = stated.notna() & calculated.notna() & (stated.abs() > 1e-9)
        off = usable & ((calculated - stated).abs() / stated.abs() > 0.01)
        audit.append(AuditEntry(
            "consistency", f"{label} (>1% apart)", int(off.sum()), "measured only"))

    audit.append(AuditEntry("result", "rows in clean dataset", len(df), "kept"))
    df = df.sort_values("contract_date").reset_index(drop=True)
    return CleaningResult(df, audit)


def run(raw_path: str, out_path: str, audit_path: str) -> CleaningResult:
    result = clean(pd.read_csv(raw_path))
    result.data.to_csv(out_path, index=False)
    result.audit_frame.to_csv(audit_path, index=False)
    return result


if __name__ == "__main__":
    res = run(
        "data/raw/textile_sales.csv",
        "data/processed/deals_clean.csv",
        "reports/data_quality_audit.csv",
    )
    print(res.audit_frame.to_string(index=False))
