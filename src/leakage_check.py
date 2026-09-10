"""
A reusable check for the mistake that produced this project's original 98% model.

Target leakage is when a feature contains the answer. It does not announce
itself — the model simply reports an excellent score, and the score is real for
the test set and worthless everywhere else. The usual sign is a single feature
with implausible importance, or a feature that is arithmetically derivable from
the target.

`audit` runs three cheap tests over the candidate features and returns anything
that should be looked at before a model is trusted.
"""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd


def audit(features: pd.DataFrame, target: pd.Series,
          corr_threshold: float = 0.90,
          ratio_tolerance: float = 0.01,
          ratio_share: float = 0.90) -> pd.DataFrame:
    """Return suspicious features, worst first.

    Three tests, each catching a different disguise:

    1. **Near-perfect correlation** — the feature moves with the target almost
       exactly, so it is probably the target in different units. The default
       threshold is deliberately permissive: this is a screen that hands you a
       shortlist to check, not a filter that drops columns on its own.
    2. **Constant ratio** — feature / target is the same number on most rows,
       which means one is a rescaling of the other (a currency conversion, a
       tax deduction, a percentage share).
    3. **Constant product ratio** — a *pair* of features multiplies to the
       target. This is the one that hides: neither column looks suspicious on
       its own.
    """
    numeric = _coerce(features)
    usable = target.notna() & (target.abs() > 1e-9)
    findings: list[dict] = []

    for col in numeric.columns:
        values, actual = numeric.loc[usable, col], target[usable]
        pair = values.notna()
        if pair.sum() < 30:
            continue

        correlation = abs(values[pair].corr(actual[pair]))
        if correlation >= corr_threshold:
            findings.append({"feature": col, "test": "near-perfect correlation",
                             "evidence": f"|r| = {correlation:.4f}", "score": correlation})

        ratio = (values[pair] / actual[pair]).replace([np.inf, -np.inf], np.nan).dropna()
        # A column that is almost always zero (or otherwise constant) has a
        # constant ratio for a trivial reason. That is not leakage.
        if len(ratio) >= 30 and not _is_degenerate(values[pair]):
            share = (ratio.sub(ratio.median()).abs() <= ratio_tolerance * ratio.median().__abs__()).mean()
            if share >= ratio_share:
                findings.append({"feature": col, "test": "constant ratio to target",
                                 "evidence": f"{share:.1%} of rows at ratio {ratio.median():.4g}",
                                 "score": share})

    for left, right in itertools.combinations(numeric.columns, 2):
        product = (numeric[left] * numeric[right])[usable]
        pair = product.notna()
        if pair.sum() < 30:
            continue
        ratio = (product[pair] / target[usable][pair]).replace([np.inf, -np.inf], np.nan).dropna()
        if len(ratio) < 30:
            continue
        share = (ratio.sub(1).abs() <= ratio_tolerance).mean()
        if share >= ratio_share:
            findings.append({"feature": f"{left} x {right}", "test": "pair multiplies to target",
                             "evidence": f"{share:.1%} of rows reproduce the target", "score": share})

    if not findings:
        return pd.DataFrame(columns=["feature", "test", "evidence"])
    return (pd.DataFrame(findings).sort_values("score", ascending=False)
            .drop(columns="score").reset_index(drop=True))


def _coerce(features: pd.DataFrame) -> pd.DataFrame:
    """Numeric view of the features.

    Finance exports store numbers as text ('1,234', '$ -'), and those columns are
    exactly the ones that leak. Selecting only the already-numeric dtypes would
    skip them, so we convert first and keep whatever survives.
    """
    out = {}
    for col in features.columns:
        values = features[col]
        if pd.api.types.is_numeric_dtype(values):
            out[col] = values
            continue
        converted = pd.to_numeric(
            values.astype(str).str.replace(r"[,$\s]", "", regex=True)
                  .replace({"-": np.nan, "": np.nan, "nan": np.nan}),
            errors="coerce")
        if converted.notna().mean() > 0.5:      # genuinely a number column
            out[col] = converted
    return pd.DataFrame(out, index=features.index)


def _is_degenerate(values: pd.Series, dominant_share: float = 0.90) -> bool:
    """True if one value covers most of the column, so ratio tests are vacuous."""
    if values.nunique() <= 1:
        return True
    return values.value_counts(normalize=True).iloc[0] >= dominant_share
