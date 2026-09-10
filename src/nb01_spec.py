"""Content for notebook 01: understanding the data and cleaning it."""

BLOCKS = [
("md", r"""# 01 — Understanding the data, and cleaning it

## What this project is about

Softwood Pvt Ltd is a textile **sourcing company** based in Pakistan. It does not
own a factory and does not manufacture anything. It sits in the middle of two
groups:

- On one side, Pakistani mills that make fabric and garments — Rajby, Samad
  Rubber, Interloop, Pak Denim, and about a hundred others.
- On the other side, European clothing brands that want to buy — Takko,
  Tiffosi, Jennyfer, Vilanova, Minoti, and others.

Softwood introduces the two, manages the order, and takes a **commission** on
the value of the deal. That commission is the company's entire income.

The company shared six years of its deal records with us. This notebook is the
first step: understand exactly what is in that file, find what is wrong with it,
and produce a clean version we can trust.

## What one row means

The file has one row per deal. Read a row like a short story:

> On 11 June 2019 a deal was signed. A Pakistani mill called Pak Denim made
> 6,392 units of a cotton-polyester denim fabric for a European buyer called
> Tiffosi. Each unit cost \$2.07, so the deal was worth \$13,232. Softwood took
> a 2% commission, which came to \$264. The goods shipped on 8 January 2020 and
> the buyer had 30 days to pay.

There are about 4,700 such stories in the file, from June 2019 to April 2025.

## What this notebook does

1. Look at the raw file as it actually is
2. Find every problem in it, with evidence
3. Fix what can be fixed, flag what cannot, and record every decision
4. Save a clean dataset the rest of the project can build on
"""),

("code", r"""import sys
sys.path.append("../src")

import numpy as np
import pandas as pd

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 40)

raw = pd.read_csv("../data/raw/textile_sales.csv")
raw.columns = [c.strip() for c in raw.columns]

print(f"Rows:    {len(raw):,}")
print(f"Columns: {raw.shape[1]}")
print(f"Period:  {pd.to_datetime(raw.Contract_Date).min():%b %Y} "
      f"to {pd.to_datetime(raw.Contract_Date).max():%b %Y}")"""),

("code", r"""raw.head(3)"""),

("md", r"""## The 26 columns, grouped by what they tell us

Twenty-six columns looks like a lot, but they fall into five simple groups.

| Group | Columns | What it tells us |
|---|---|---|
| Who | `Team_Head`, `Supplier`, `Customer` | Who handled the deal, who made the goods, who bought them |
| What | `Style_Name`, `Contract_Qty.`, `Shipped_Quantity`, `Bal` | Which fabric, how much was ordered, how much actually shipped |
| When | `Contract_Date`, `Dispatch_Date`, `Payment_Days`, `Expected_Payment` | Signed, shipped, and when payment is due |
| How much | `Rate`, `Dollar_Rate`, `Sales_Volume`, `Exchange`, `Dollar_Exchange_Rate` | Unit price, total deal value, and the USD/PKR rate that day |
| What we earned | the 10 remaining commission and tax columns | Softwood's cut, before and after tax |

The last group is where the money is, so it is worth looking at closely.
"""),

("code", r"""money_cols = ["Sales_Volume", "Commission_Percentage", "Commission_Dollar_Value",
              "Exchange", "Commission_PKR_Value", "Tax_%", "Received",
              "Commission_SW_Percentage", "SW_Commission_PKR", "SW_Rcvd_after_Tax"]
raw[money_cols].head(3)"""),

("md", r"""### These ten columns are really one number and nine calculations

Look at the first row above and follow the arithmetic:

```
deal value                $13,232      (Sales_Volume)
x commission rate           2%         (Commission_Percentage)   <-- the only real decision
= commission              $264.64      (Commission_Dollar_Value)
x USD/PKR rate              157
= commission            Rs 41,548      (Commission_PKR_Value)
- tax                      10%
= net commission        Rs 37,393      (Received)
```

Only one of these is a business decision: **the commission rate**. Everything
else follows automatically once the rate is set. Let us check whether that is
true for the whole file, not just this one row.
"""),

("code", r"""def as_number(s):
    # Spreadsheet text -> numbers. Handles commas, $ signs and '-' placeholders.
    return pd.to_numeric(
        s.astype(str).str.replace(r"[,$\s]", "", regex=True)
         .replace({"-": np.nan, "": np.nan, "nan": np.nan}),
        errors="coerce")


def agrees(calculated, stated, tolerance=0.01):
    # Share of rows where a formula reproduces the stated value.
    usable = calculated.notna() & stated.notna() & (stated.abs() > 1e-9)
    return ((calculated[usable] - stated[usable]).abs() / stated[usable].abs() < tolerance).mean()


value = as_number(raw.Sales_Volume)
commission = as_number(raw.Commission_Dollar_Value)
rate = as_number(raw.Commission_Percentage)
net_pkr = as_number(raw.Received)

checks = {
    "deal value  = quantity x unit price": agrees(raw["Contract_Qty."] * raw.Dollar_Rate, value),
    "commission  = deal value x rate":      agrees(value * rate, commission),
    "commission in PKR = commission x FX":  agrees(commission * raw.Exchange, as_number(raw.Commission_PKR_Value)),
    "net = commission in PKR - tax":        agrees(as_number(raw.Commission_PKR_Value) * (1 - as_number(raw["Tax_%"])), net_pkr),
}
for label, share in checks.items():
    print(f"{share:6.1%}  of rows follow:  {label}")"""),

("md", r"""**This confirms it.** The formulas hold on almost every row. So of the ten
money columns, nine are bookkeeping. The business only decides one thing per
deal: what commission rate to charge.

This matters more than it looks, and we come back to it in notebook 04. An
earlier version of this project built a machine learning model to *predict*
commission — while giving the model the after-tax commission as an input. That
is like predicting someone's age after being shown their ID card. We will show
that properly later.

For now the useful conclusion is: **the commission rate is the thing worth
studying.**

---

## Problem 1: numbers stored as text

Excel writes a dash into empty accounting cells. When pandas reads that, the
whole column becomes text, and text cannot be added up or averaged.
"""),

("code", r"""text_but_numeric = ["Sales_Volume", "SW_Commission_PKR", "Tax_%", "Received",
                    "Payment_Days", "Rate"]
report = []
for col in text_but_numeric:
    values = raw[col].astype(str)
    unparseable = values[as_number(values).isna()]
    report.append({
        "column": col,
        "stored as": raw[col].dtype.name,
        "bad cells": len(unparseable),
        "example": unparseable.iloc[0] if len(unparseable) else "-",
    })
pd.DataFrame(report)"""),

("md", r"""Six columns are affected. Four of them carry Excel's `-` / `$ -` placeholder for
an empty accounting cell; `Rate` is the worst, with 132 rows written as text like
`Rs685`; and one `Payment_Days` cell holds a date where a number belongs.

The counts are small, but until this is fixed, anything that sums or averages
these columns is quietly wrong — and `Rate` is a price column.

---

## Problem 2: the same company written several different ways

If "Tiffosi Fab" and "Tiffosi fab" are treated as two different buyers, then
every per-customer number is split in half. We looked for names that differ only
by capitalisation or an obvious typo.
"""),

("code", r"""import difflib

def near_duplicates(names, cutoff=0.82):
    unique, seen, pairs = sorted(set(names)), set(), []
    for a in unique:
        if a in seen:
            continue
        similar = [b for b in unique
                   if b != a and difflib.SequenceMatcher(None, a.lower(), b.lower()).ratio() > cutoff]
        if similar:
            counts = names.value_counts()
            pairs.append({"name": f"{a} ({counts[a]})",
                          "looks like": ", ".join(f"{s} ({counts[s]})" for s in similar)})
            seen.update(similar + [a])
    return pd.DataFrame(pairs)

print("CUSTOMERS")
display(near_duplicates(raw.Customer))
print("SUPPLIERS")
display(near_duplicates(raw.Supplier))"""),

("md", r"""Two different things are showing up in that list, and telling them apart matters.

**Real duplicates — same company, typed differently.** These get merged:

- `Tiffosi fab` / `Tifossi Fab` -> `Tiffosi Fab`
- `Tiffosi knit` -> `Tiffosi Knit`, `Get over` -> `Get Over`
- `Bizbee` -> `Bizzbee`, `CDRL Fab` -> `CRDL Fab`
- `indigo` -> `Indigo`, `Safa tex` -> `Safa Tex`, `Shorts quality` -> `Shorts Quality`

**Not duplicates — genuinely different records.** `Takko` and `Takko Fab` are
*not* a typo of each other. `Authentic Style Denim` and `Authentic Style knits`
are not either. Merging these would destroy real information, and the next
section explains why.

---

## Problem 3: there is a hidden column inside the customer name

Notice the pattern: `Takko`, `Takko Fab`, `Takko Knit`. Same brand, different
suffix. The suffix is not decoration — it says which **product line** the deal
belongs to. Fabric is a different business from finished garments.

If that is true, the commission rate should behave differently across the
suffixes. Let us check.
"""),

("code", r"""import re

SUFFIX = re.compile(r"[\s\-]+(fab|fabric|knit|knits|denim|woven|garment|garments|apparel)s?$", re.I)

work = raw.copy()
work["rate"] = as_number(work.Commission_Percentage)
work = work[(work.rate > 0) & (work.rate < 1)]
work["division"] = work.Customer.str.extract(SUFFIX, expand=False).str.lower().fillna("garment")

summary = work.groupby("division").rate.agg(
    deals="size", p10=lambda s: s.quantile(.10), median="median",
    p90=lambda s: s.quantile(.90))
summary["at exactly 2%"] = work.assign(two=work.rate.round(4).eq(0.02)).groupby("division").two.mean()
summary.round(4)"""),

("md", r"""**This is the most important finding in the notebook.**

Deals in the **fab** (fabric) line are at exactly 2.00% commission in **98.9%**
of cases. The 10th and 90th percentile are both 2.00% — there is no spread at
all. It is a fixed contractual rate.

Deals in the **knit** (garment) line are almost never at 2%. Their median is 5%
and they spread widely, because each one is negotiated.

So Softwood is really running two different businesses with two different
pricing models, and the only place that is recorded is a suffix inside a text
column. Let us confirm it holds inside a single brand, where nothing else
changes.
"""),

("code", r"""takko = work[work.Customer.str.lower().str.startswith("takko")]
takko.groupby("Customer").rate.agg(
    deals="size", median="median",
    share_at_2pct=lambda s: s.round(4).eq(0.02).mean()).round(3)"""),

("md", r"""Same brand, same buyer relationship — and `Takko Fab` sits at 2% while `Takko`
and `Takko Knit` sit at 5%. The product line, not the customer, sets the rate.

Every rate comparison from here on has to respect this split. Comparing a fabric
deal against a garment deal is comparing two unrelated things. We therefore
split `Customer` into two proper columns: `customer_brand` and
`product_division`.

---

## Problem 4: the Team_Head column holds two different kinds of thing

`Team_Head` should be the Softwood staff member who handled the deal. Most
values are first names. But some are company names.
"""),

("code", r"""mills = set(raw.Supplier.unique())
looks_like_company = raw.Team_Head.isin(mills)

print(f"Rows where Team_Head is a known mill name: {looks_like_company.sum()}")
print(f"Of those, Team_Head equals the Supplier on the same row: "
      f"{(raw.Team_Head == raw.Supplier)[looks_like_company].sum()}")
print()
# The individual staff names are replaced with labels here. They are personal
# data about identifiable employees, they are not needed for the argument, and
# this notebook is published. The deal counts are what matter and are unchanged.
people = raw.Team_Head[~looks_like_company].value_counts()
people.index = [f"staff_{i+1}" for i in range(len(people))]

print("Value counts:")
pd.DataFrame({"staff (anonymised)": people.head(8)}).join(
    pd.DataFrame({"company names": raw.Team_Head[looks_like_company].value_counts().head(8)}),
    how="outer").fillna("")"""),

("md", r"""It is not a simple copy-paste error: in **none** of those rows does `Team_Head`
match the `Supplier` on the same row. So we cannot say the column was
accidentally filled from the supplier column, and we cannot recover who the
actual staff member was.

The honest response is to label the column rather than guess or delete. We add
`team_head_type` with values `person` and `organization`, keep the original
value, and let each later analysis decide whether to include organizations.

One note for context: `SWU` accounts for most of the company-name rows and is
almost certainly Softwood itself, since the company's own initials appear
throughout the commission columns. We do not rely on that assumption anywhere.

The staff names are shown as `staff_1 … staff_n` above. They are personal data
about identifiable employees, the analysis never uses them, and this notebook is
public. The company names are kept because the whole point of this section is
that they are the wrong kind of value for the column.

---

## Problem 5: rows that break business rules

A few rows describe things that cannot happen.
"""),

("code", r"""contract_on = pd.to_datetime(raw.Contract_Date, errors="coerce")
dispatch_on = pd.to_datetime(raw.Dispatch_Date, errors="coerce")
lead = (dispatch_on - contract_on).dt.days

rules = {
    "commission rate is zero":                 (as_number(raw.Commission_Percentage) == 0),
    "commission rate is 100% or more":          (as_number(raw.Commission_Percentage) >= 1),
    "contract quantity is zero or negative":    (raw["Contract_Qty."] <= 0),
    "USD/PKR rate outside 100-350":             ~raw.Exchange.between(100, 350),
    "shipped more than was ordered":            (raw.Shipped_Quantity > raw["Contract_Qty."]),
    "goods dispatched before contract signed":  (lead < 0),
    "lead time longer than two years":          (lead > 730),
}
pd.DataFrame([{"rule broken": k, "rows": int(v.sum()),
               "share": f"{v.mean():.2%}"} for k, v in rules.items()])"""),

("md", r"""These split into two kinds.

**Unusable — the row is dropped.** A deal with a zero or 100%+ commission rate,
or zero quantity, has nothing to analyse. An impossible exchange rate makes
every derived figure wrong. Together this is about 2.9% of the file.

**Usable but suspicious — the row is kept and flagged.** Shipping slightly more
than ordered is normal in textiles (mills over-run). A dispatch date before the
contract date is a typing error in the date, but the money on that row is still
valid — so we keep the row, blank out the lead time, and flag it.

The principle: never delete a row for being inconvenient. Delete it only when it
cannot answer the question, and count every deletion.

---

## Running the cleaner

All of the above is implemented in `src/clean_data.py`, so it runs the same way
every time and nothing is done by hand.
"""),

("code", r"""from clean_data import run

result = run("../data/raw/textile_sales.csv",
             "../data/processed/deals_clean.csv",
             "../reports/data_quality_audit.csv")

result.audit_frame"""),

("md", r"""Every row of that table is a decision, with a count and a reason. Anyone can
check our work against it.

## What we ended up with
"""),

("code", r"""deals = result.data
kept = len(deals) / len(raw)

print(f"Raw rows:    {len(raw):,}")
print(f"Clean rows:  {len(deals):,}   ({kept:.1%} of the original kept)")
print(f"Period:      {deals.contract_date.min():%b %Y} to {deals.contract_date.max():%b %Y}")
print(f"Suppliers:   {deals.supplier.nunique()}")
print(f"Customers:   {deals.customer.nunique()}  ({deals.customer_brand.nunique()} brands x product lines)")
print(f"Fabrics:     {deals.style_name.nunique():,} distinct style names")
print()
print("Deals by product line:")
print(deals.product_division.value_counts().to_string())
print()
print("Commission rate, after cleaning:")
print(deals.commission_pct.describe([.05, .25, .5, .75, .95]).round(4).to_string())"""),

("md", r"""The commission rate now tops out at 33% instead of 100%, and the impossible
values are gone.

## Where this leaves us

We started with a finance export and turned it into a dataset we can reason
about. Along the way we learned three things that shape the whole project:

1. **Nine of the ten money columns are formulas.** The only real decision in the
   data is the commission rate.
2. **The business has two pricing models, not one.** Fabric deals are fixed at
   2%; garment deals are negotiated around 5%. This was hidden in a text suffix.
3. **The commission rate varies a lot inside the negotiated side.** That is
   where the money is being won or lost.

Next, notebook 02 looks at what the business actually did over these six years —
and finds the problem the rest of the project exists to solve.
"""),
]
