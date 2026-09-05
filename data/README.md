# `data/` — not distributed

**The dataset is not in this repository and will not be.**

The records analysed here are the real deal book of **Softwood Pvt Ltd**, a Pakistani
textile sourcing agency. They were shared with the author under a confidentiality
understanding. Every row contains a named supplier mill, a named European buyer, a
contract number, a quantity, a unit price and a commission rate that is, the company's
customer list and its pricing, together. None of it is published.

If you are reading this to evaluate the work: the analysis, its outputs and its charts are
all committed. `notebooks/01` … `notebooks/05` are stored **executed**, with results
intact, so every number in the documentation can be read without running anything. What you
cannot do is re-run them, because the input is missing.

## Shape

| File                                | Rows                   | Columns |
| ----------------------------------- | ---------------------- | ------- |
| `data/raw/textile_sales.csv`      | 4,725                  | 26      |
| `data/processed/deals_clean.csv`  | 4,590 (97.1% retained) | 38      |
| `data/processed/deals_scored.csv` | 3,421                  | 45      |

One row is one deal: a contract between one mill and one buyer for one style, with the
commission Softwood earned on it. Contract dates run 2019-06-11 to 2025-04-11. The clean
file holds 100 suppliers, 73 customer strings (53 brands once the product-line suffix is
split off), and 2,401 distinct style names.

## Raw schema — what `data/raw/textile_sales.csv` must contain

These 26 headers are renamed by `src/clean_data.py`. Spelling is important because the mapping uses a literal dictionary (`COLUMN_MAP`) rather than fuzzy matching. Any trailing spaces in the headers are removed before the mapping is applied.

| Raw header                   | Renamed to                    | What it holds                                                               |
| ---------------------------- | ----------------------------- | --------------------------------------------------------------------------- |
| `Team_Head`                | `team_head`                 | who handled the deal —**mixed**: staff first names and company names |
| `Supplier`                 | `supplier`                  | the mill                                                                    |
| `Customer`                 | `customer`                  | the buyer, with a product-line suffix (`Takko Fab`, `Takko Knit`)       |
| `Contract_Date`            | `contract_date`             | date signed                                                                 |
| `Dispatch_Date`            | `dispatch_date`             | date shipped                                                                |
| `Payment_Days`             | `payment_days`              | agreed credit period                                                        |
| `Expected_Payment`         | `expected_payment_date`     | date payment is due                                                         |
| `Style_Name`               | `style_name`                | free-text style / article name                                              |
| `Rate`                     | `unit_rate_pkr`             | unit price in PKR                                                           |
| `Dollar_Exchange_Rate`     | `pkr_to_usd`                | PKR→USD factor used on the row                                             |
| `Dollar_Rate`              | `unit_rate_usd`             | unit price in USD                                                           |
| `Contract_#`               | `contract_no`               | the company's contract reference                                            |
| `Contract_Qty.`            | `contract_qty`              | units contracted                                                            |
| `Shipped_Quantity`         | `shipped_qty`               | units actually shipped                                                      |
| `Bal`                      | `balance_qty`               | units outstanding                                                           |
| `Sales_Volume`             | `deal_value_usd`            | deal value, USD                                                             |
| `Exchange`                 | `usd_to_pkr`                | USD→PKR rate                                                               |
| `Commission_Dollar_Value`  | `commission_usd`            | commission earned, USD                                                      |
| `Commission_Percentage`    | `commission_pct`            | **the commission rate, as a fraction** (0.05 = 5%)                    |
| `Commission_PKR_Value`     | `commission_pkr`            | commission in PKR                                                           |
| `Commission_SW_Percentage` | `sw_pct`                    | Softwood's share                                                            |
| `SW_Commission_PKR`        | `sw_commission_pkr`         | Softwood's share in PKR                                                     |
| `Tax_%`                    | `tax_pct`                   | withholding rate                                                            |
| `SW_After_Tax_Value`       | `sw_after_tax_pkr`          | Softwood's share after tax                                                  |
| `Received`                 | `commission_after_tax_pkr`  | commission received after tax                                               |
| `SW_Rcvd_after_Tax`        | `sw_received_after_tax_pkr` | Softwood's share received after tax                                         |

Nine of these ten money columns are  **arithmetic restatements of each other** . For example, deal value is equal to quantity × unit price in 99.3% of the rows, commission is equal to deal value × rate in 98.2% of the rows, and the same pattern continues for the other columns. Notebook 01 proves these relationships, while Notebook 04 shows what happens when these columns are given to a model. The only genuinely per-deal decision in the file is `commission_pct`.

The raw file is a spreadsheet export, so it contains common data quality issues. The `src/clean_data.py` script is designed to handle these issues, including numbers stored as text, `-` and `$ -` used as accounting placeholders, values such as `Rs 437` in numeric columns, a date stored in an integer column, and entity names entered in different ways.

## Cleaned schema — what `notebooks/02`–`05`, the service and the tests read

`data/processed/deals_clean.csv`, 4,590 × 38: the 26 renamed columns above plus 12 derived
ones.

| Derived column                    | Type  | How it is made                                                                    |
| --------------------------------- | ----- | --------------------------------------------------------------------------------- |
| `customer_brand`                | text  | `customer` with the product-line suffix stripped (`Takko Fab` → `Takko`)   |
| `product_division`              | text  | that suffix, lowercased:`fab` \| `knit` \| `denim` \| `garment` (default) |
| `team_head_type`                | text  | `organization` if `team_head` is in `ORG_TEAM_HEADS`, else `person`       |
| `lead_days`                     | float | `dispatch_date − contract_date`; set missing when negative                     |
| `credit_days`                   | float | `expected_payment_date − dispatch_date`                                        |
| `contract_year`                 | int   | from`contract_date`                                                             |
| `contract_month`                | int   | from`contract_date`                                                             |
| `contract_quarter`              | int   | from`contract_date`                                                             |
| `flag_overshipped`              | bool  | `shipped_qty > contract_qty`                                                    |
| `flag_dispatch_before_contract` | bool  | `lead_days < 0`                                                                 |
| `flag_long_lead`                | bool  | `lead_days > 730`                                                               |
| `flag_high_rate`                | bool  | `commission_pct > 0.20`                                                         |

Rows are sorted by `contract_date` with the index reset. The three date columns are written
as ISO strings; the notebooks re-parse `contract_date` on read.

### The minimum a substitute dataset needs

If you want to run this against your own commission book, `data/processed/deals_clean.csv`
is the only file that really matters. The benchmark, the service and the tests use just
these columns:

- `supplier` — the peer group the whole benchmark is built on
- `customer` — used for comparables and for the `customer + supplier` variant
- `product_division` — decides scope; only `knit` and `garment` are benchmarked
- `commission_pct` — the target, as a **fraction** strictly between 0 and 1
- `contract_date` / `contract_year` — the time-based split and the walk-forward
- `contract_qty`, `unit_rate_usd`, `deal_value_usd` — deal size, and the dollar gap
- `commission_usd` — used in the leakage totals
- `team_head` — a feature in the gradient-boosting comparison only

Everything else in the 38 columns is carried for the audit trail and for notebook 01's
data-quality argument, not because the model needs it.

Two important structural assumptions should be stated clearly. If a substitute dataset does not follow these assumptions, the output may look correct but will not have the intended meaning.

1. **The product-line suffix is meaningful.** `Takko`, `Takko Fab`, and `Takko Knit` represent three different pricing patterns for the same brand. Their median rates are 5.0%, 2.0%, and 5.0%. They are not three different spellings of the same customer. If the customer column in your dataset does not contain these suffixes, all records will be treated as `garment`, and the scope filter will no longer have any effect.
2. **Some product lines use contract-based pricing instead of negotiation.** Fab and denim have an interquartile rate range of exactly zero in this dataset, which is why they are excluded. If this is not true for your dataset, check `NEGOTIATED_DIVISIONS` in `src/benchmark.py` again instead of assuming that the same exclusion should be used.
