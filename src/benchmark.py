"""
Commission rate benchmark for Softwood.

The question this answers is narrow on purpose: given a deal that is about to be
signed, what commission rate have comparable past deals actually carried, and is
the rate on the table below that?

Two design choices are worth stating up front, because both were decided by
testing rather than by preference.

1. **Only the negotiated product lines are covered.** Fabric deals sit at a
   fixed contractual 2% (98.9% of them exactly, inter-quartile range zero), so
   there is nothing to benchmark. Denim is excluded for a different reason: with
   58 deals it is too thin to give a supplier a trustworthy band. Both are
   deliberate scope decisions, written as a constant rather than inferred at
   runtime, so the scope cannot drift silently when the data changes.

2. **The peer group is the supplier, and nothing else.** Supplier, customer,
   supplier+customer, and a gradient boosting model over all of them were tested
   on a time-based holdout. The supplier median, shrunk toward the overall
   median, won on mean absolute error and was the most stable. Adding more keys
   split the data into groups too small to be reliable and made it worse.

Shrinkage is what keeps a supplier with four deals from getting a confident band.
A group's own median is blended with the overall median in proportion to how much
history it has, so thin groups fall back to the house average automatically.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
import pandas as pd

# Product lines where the rate is negotiated deal by deal. The others are fixed
# by contract and are deliberately out of scope.
NEGOTIATED_DIVISIONS = ("knit", "garment")

DEFAULT_K = 30          # shrinkage strength, in "equivalent deals"
LOW_Q, HIGH_Q = 0.25, 0.75


@dataclass
class RateBenchmark:
    group_key: str = "supplier"
    k: int = DEFAULT_K
    low_q: float = LOW_Q
    high_q: float = HIGH_Q

    def fit(self, history: pd.DataFrame) -> "RateBenchmark":
        history = negotiated_only(history)
        if history.empty:
            raise ValueError("no negotiated deals in the history provided")

        rates = history.groupby(self.group_key).commission_pct
        table = pd.DataFrame({
            "support": rates.size(),
            "low": rates.quantile(self.low_q),
            "expected": rates.median(),
            "high": rates.quantile(self.high_q),
        })

        # House-wide fallback, used for groups with no history at all.
        self.house_ = {
            "support": 0,
            "low": history.commission_pct.quantile(self.low_q),
            "expected": history.commission_pct.median(),
            "high": history.commission_pct.quantile(self.high_q),
        }

        # Shrink each group toward the house numbers by its own sample size.
        weight = table.support / (table.support + self.k)
        for col in ("low", "expected", "high"):
            table[col] = weight * table[col] + (1 - weight) * self.house_[col]

        self.table_ = table
        return self

    def lookup(self, group_value) -> dict:
        """Band for one supplier. Falls back to the house band if unseen."""
        if group_value in self.table_.index:
            row = self.table_.loc[group_value]
            return {"low": row.low, "expected": row.expected,
                    "high": row.high, "support": int(row.support),
                    "basis": self.group_key}
        return {**self.house_, "basis": "house average"}

    def score(self, deals: pd.DataFrame) -> pd.DataFrame:
        """Attach the band, the position in it, and the money gap, per deal.

        Rows outside the negotiated product lines are dropped rather than scored.
        Use `in_scope` first if you need to know whether a specific deal will
        survive that filter.
        """
        out = negotiated_only(deals).copy()
        for col in ("low", "expected", "high"):
            out[f"bench_{col}"] = (out[self.group_key].map(self.table_[col])
                                   .fillna(self.house_[col]))
        out["bench_support"] = out[self.group_key].map(self.table_.support).fillna(0).astype(int)

        out["position"] = np.select(
            [out.commission_pct < out.bench_low, out.commission_pct > out.bench_high],
            ["below band", "above band"], default="within band")
        out["gap_pp"] = (out.bench_expected - out.commission_pct) * 100

        # Money is only counted for deals that fall below the band. A deal inside
        # the band is normal variation, not a loss.
        out["leakage_usd"] = np.where(
            out.position == "below band",
            (out.bench_expected - out.commission_pct) * out.deal_value_usd,
            0.0).clip(min=0)
        return out


def negotiated_only(deals: pd.DataFrame) -> pd.DataFrame:
    return deals[deals.product_division.isin(NEGOTIATED_DIVISIONS)]


def in_scope(customer: str) -> bool:
    """Whether a customer name belongs to a product line the benchmark covers.

    The product line is encoded as a suffix on the customer name ('Takko Fab'),
    so a caller with only a customer string can still check scope before asking
    for a band it should not act on.
    """
    return division_of(customer) in NEGOTIATED_DIVISIONS


def division_of(customer: str) -> str:
    match = re.search(
        r"[\s\-]+(fab|fabric|knit|knits|denim|woven|garment|garments|apparel)s?$",
        str(customer), re.IGNORECASE)
    return match.group(1).lower() if match else "garment"


def walk_forward(deals: pd.DataFrame, min_history: int = 200, **kwargs) -> pd.DataFrame:
    """Score every deal using only the deals that came before its year.

    Scoring a deal with a benchmark built from its own year would be judging the
    past with knowledge of the future, and the leakage figure would be
    meaningless. The benchmark is therefore rebuilt at the start of each year
    from earlier years only, which is also how a company would actually run it.
    """
    deals = negotiated_only(deals)
    scored = []
    for year in sorted(deals.contract_year.unique()):
        history = deals[deals.contract_year < year]
        if len(history) < min_history:
            continue                      # not enough past to judge this year
        model = RateBenchmark(**kwargs).fit(history)
        scored.append(model.score(deals[deals.contract_year == year]))
    if not scored:
        raise ValueError("not enough history to score any year")
    return pd.concat(scored).sort_values("contract_date")
