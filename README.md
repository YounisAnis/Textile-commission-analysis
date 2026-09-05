# Commission Intelligence Analysis for a Textile Sourcing Agency

Six years of deal records from a Pakistani textile sourcing company are used to construct a commission-rate benchmark that enables a sourcing manager to assess whether the commission rate proposed for a current deal is consistent with comparable historical deals.

**Data:** 4,725 contracts, June 2019–April 2025, from Softwood Pvt Ltd.
**Deliverable:** a supplier-level commission-rate benchmark, validated using a time-based holdout and deployed as a FastAPI service with 19 tests and a Docker image.

> **Confidentiality and reproducibility.** The dataset is not included in this repository and will not be distributed. It was shared under a confidentiality understanding, and each row contains a named mill, a named European buyer, a contract number, and a price. The published materials therefore consist of the code, the five executed notebooks with their outputs intact, and the figures. Consequently, every reported number can be read and checked without access to the underlying data, although the complete pipeline cannot be re-executed.

## 1. Data Description

Softwood Pvt Ltd. is a **sourcing agent**, rather than a manufacturer. It operates between Pakistani mills that manufacture fabric and garments, including Rajby, Samad Rubber, Interloop, Gul Ahmed, Pak Denim, and approximately one hundred other suppliers, and European clothing brands such as Takko, Tiffosi, Jennyfer, Vilanova, and Minoti. The company introduces the two parties, manages the order, and receives a commission on the value of each deal. This commission constitutes its entire income.

Each row represents one deal and records the person who handled it, the mill that produced the goods, the brand that purchased them, the style, the quantity, the unit price, the commission rate, and the contract, dispatch, and expected-payment dates. After cleaning, the dataset contains 100 suppliers, 73 customer strings that resolve to 53 brands, and 2,401 distinct style names.

## 2. Business Problem

**Commission income decreased by 51.6% between 2021 and 2024**, from $1.96M to $0.95M.

Because commission income is the product of exactly three factors, all three were measured:

|                                  |    2021 |    2024 |            Change |
| -------------------------------- | ------: | ------: | ----------------: |
| Number of deals                  |     911 |     855 |            −6.1% |
| Average deal size                | $52,591 | $26,865 | **−48.9%** |
| Commission rate (value-weighted) |   4.08% |   4.13% |             +1.0% |

The commission rate in this table is value-weighted, calculated as total commission divided by total deal value. This definition is used because the three factors then multiply back to the total commission income exactly. The unweighted mean rate per deal increased further, from 3.51% to 4.42%. Thus, the observed decline in commission income cannot be attributed to a decline in commission rates.

Deal count and commission rates remained broadly stable, whereas deal **size** decreased by approximately half. Moreover, this reduction was associated with volume rather than price: total units declined from 9.7M to 5.1M, while the median unit price increased from $5.75 to $6.60.

The concentration of the decline matters. **Takko alone accounts for −$1.12M of the company's −$1.01M decline.** All other customers combined contributed an increase of $0.11M. Takko represented 62–88% of income in every year of the record.

> Softwood did not lose its business as a whole; rather, it lost most of one customer relationship, and that customer represented most of its business.

Separately, among the deals that remain, commission rates are determined on a deal-by-deal basis without a quantitative reference point. Within a single customer, supplier, and year, observed rates range from 0.8% to 11.7%. **This is the problem addressed by the project's tool, and it is distinct from the historical decline in commission income.** The two problems are therefore analyzed separately rather than conflated.

## 3. Data Quality Assessment and Structural Findings

All data-quality issues described below were measured.

| Issue                                          |    Extent | Action                                                 |
| ---------------------------------------------- | --------: | ------------------------------------------------------ |
| Numbers stored as text (`$ -`, `-`)        | 4 columns | Parsed to numeric                                      |
| `Rate` written as `Rs 685`                 |  132 rows | Set to missing; column unused downstream               |
| One`Payment_Days` cell containing a date     |     1 row | Set to missing                                         |
| Same company spelled in several ways           |   43 rows | Merged case and typo variants only                     |
| Impossible commission rates (0%, ≥100%)       |   56 rows | Dropped                                                |
| Zero or negative contract quantity             |   65 rows | Dropped                                                |
| USD/PKR rate outside 100–350                  |   12 rows | Dropped                                                |
| Dispatch date before contract date             |   16 rows | Kept; lead time blanked and row flagged                |
| Shipped quantity greater than ordered quantity |   39 rows | Kept and flagged; this can occur in textile operations |
| `Team_Head` containing company names         |  372 rows | Labelled`organization` and retained                  |
| Exact duplicate rows                           |    2 rows | Dropped                                                |

The retained dataset contains **4,590 rows (97.1%)**, calculated as 4,725 − 56 − 65 − 12 − 2. The 56 impossible commission rates comprise 12 zero rates, 11 negative rates, and 33 rates at 100% or higher. Every deletion is counted in the audit log, and a test verifies that the log accounts for every row removed.

The governing rule is stated in the pipeline's own docstring: a row is not deleted merely because it is inconvenient; deletion is permitted only when the row cannot answer the question being analyzed, and every deletion must be counted.

Two structural findings were more consequential than the routine cleaning steps.

### 3.1 Product-Line Information Hidden in the Customer Name

A product-line distinction is embedded in the customer-name field. `Takko`, `Takko Fab`, and `Takko Knit` are not simple spelling variants: the suffix identifies the product line. Fabric deals have an exactly 2.00% rate in **98.9% of cases**, with an interquartile range of zero. Knit deals have a 5.0% median rate and a 3.7-fold difference between the 10th and 90th percentiles. Denim also has a zero interquartile range, but only 76% of denim deals have an exactly 2% rate, and there are only 58 such deals. Denim is therefore excluded from benchmarking because the sample is too small, rather than because the rate can be shown to be fixed.

These findings indicate that the company operates two pricing models, although this distinction is not recorded in a dedicated field and is visible only through the customer-name suffix.

### 3.2 Formula-Derived Monetary Columns

Nine of the ten monetary columns are formulas. Deal value equals quantity multiplied by unit price on 99.3% of rows; commission equals deal value multiplied by rate on 98.2%; and commission in PKR equals commission multiplied by the foreign-exchange rate on 99.9%. Therefore, the only genuine business decision at the individual-deal level is the **commission rate**.

## 4. Analytical Workflow

The analysis is organized into five executed notebooks. Their sequence and purpose are as follows:

| Notebook                                                                                    | Function                                                                                        | Rationale                                                                                                                |
| ------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------ |
| [`01_data_understanding_and_cleaning`](notebooks/01_data_understanding_and_cleaning.ipynb) | Profiles the raw export, identifies data-quality issues, and produces the clean dataset         | Downstream analysis cannot be considered reliable without first establishing data quality                                |
| [`02_business_analysis`](notebooks/02_business_analysis.ipynb)                             | Decomposes the revenue decline and tests whether commission-rate variation exists               | The analysis first identifies the underlying business problem before developing a solution                               |
| [`03_rate_benchmark_and_leakage`](notebooks/03_rate_benchmark_and_leakage.ipynb)           | Constructs the benchmark and scores historical deals                                            | This notebook contains the primary deliverable                                                                           |
| [`04_the_98_percent_illusion`](notebooks/04_the_98_percent_illusion.ipynb)                 | Examines the project's earlier 98% model and identifies the reason for its inflated performance | The earlier result was incorrect because of leakage; documenting the correction makes the result independently checkable |
| [`05_what_did_not_work`](notebooks/05_what_did_not_work.ipynb)                             | Reproduces the two abandoned approaches and documents the evidence against them                 | The rejected alternatives are therefore evaluated rather than merely asserted to be unsuitable                           |

## 5. Proposed Benchmarking Method

The proposed method groups historical deals by supplier, calculates the supplier's median commission rate, shrinks that value toward the house-wide median according to the supplier's historical support, and reports the 25th to 75th percentile as the acceptable range:

$$
\text{benchmark}_s = w_s \cdot \text{median}_s + (1 - w_s)\cdot\text{median}_{\text{house}},
\qquad w_s = \frac{n_s}{n_s + k}, \quad k = 30
$$

Shrinkage reduces overconfidence for suppliers with limited historical evidence. A supplier with 300 deals retains 91% of its own median, whereas a supplier with 4 deals retains 12%. A single expected rate would not provide sufficient information for negotiation, so the system returns a range and flags a deal only when its proposed rate falls below that range. Consequently, the tool is intentionally conservative in its alerting behavior: ordinary variation remains unflagged, so that a flag identifies a case that merits review.

**Scope is enforced in code at both layers.** `NEGOTIATED_DIVISIONS = ("knit", "garment")` which restricts batch scoring to negotiated divisions, while the API returns an explicit `out_of_scope` response rather than silently returning a benchmark that does not apply. Fabric is excluded because its rate is fixed by contract, as indicated by its zero interquartile range and 98.9% of observations at exactly 2%. Denim is excluded because only 58 deals are available for benchmarking. These exclusions are represented as explicit constants rather than runtime inferences, so the benchmark scope cannot change silently when the data changes.

The benchmark therefore covers **3,802 negotiated deals, representing 83% of the cleaned dataset**.

## 6. Validation and Model Selection

The method was trained on 2019–2023 (2,997 deals) and evaluated on 2024–2025 (805 deals), which are deals not observed during model construction. It was also compared with a deliberately simple baseline.

| Method                            |              MAE | R² (vs. median baseline) |
| --------------------------------- | ---------------: | ------------------------: |
| **Supplier median, shrunk** | **1.76pp** |                     13.8% |
| Supplier median, raw              |           1.82pp |                     11.6% |
| Customer + supplier median        |           1.91pp |                     10.6% |
| Customer median                   |           2.05pp |                     10.8% |
| Gradient boosting on everything   |           2.15pp |                     15.7% |
| House median for all deals        |           2.34pp |                      3.4% |

**The simple shrinkage formula outperformed gradient boosting in MAE and was 25% better in MAE than quoting the house median to every deal.**

Three points about this comparison are important.

First, R² is measured against a *median* baseline rather than the conventional mean baseline because the rate distribution is skewed and the median is the quantity against which the benchmark competes. This is a non-standard choice. Therefore, the notebook also reports the conventional R² values: 10.7%, 8.4%, 7.5%, 7.7%, 12.7%, and −0.05%, respectively. The ranking of the methods is unchanged.

Second, gradient boosting achieves the highest R² but has a higher MAE. This occurs because R² rewards fitting the larger deals, whereas MAE gives each deal equal weight. For a tool intended for routine negotiations, such as a $40,000 order, MAE is the more relevant metric, and this choice is explicitly justified rather than hidden.

Third, the result is not highly sensitive to the shrinkage constant. Sweeping the constant gives an MAE of 1.773pp at k=10, 1.756pp at k=30, 1.756pp at k=40, and 1.759pp at k=80. The performance therefore remains essentially flat across this range, indicating that k=30 is a choice within a performance plateau rather than a fitted parameter.

Finally, historical scoring follows a **walk-forward** rule. At the start of each year, the benchmark is rebuilt using earlier years only. Scoring 2021 with a benchmark that had already incorporated 2021 would evaluate the model using information that was not available at the time. A test enforces this temporal separation by truncating later years and verifying that earlier years' benchmark bands do not change.

## 7. Benchmarking Results

**8.7%** of the 3,421 deals that could be scored by the benchmark were priced below their supplier's normal range. The phrase "could be scored" excludes 2019–2020, which serve as historical support only. Of the scored deals, 2,062 (60.3%) fell within the benchmark band and 1,063 (31.1%) fell above it. The 296 deals below the band represent approximately **$393,000**, corresponding to 6.0% of the $6.51M in commission earned on the scored deals. The three largest suppliers account for 60% of this amount.

The largest individual case was a $308K Takko order through Samad Rubber that was signed at 2.0%, while the supplier's historical deals were running at 5.0%. The resulting gap was $9,200 on that contract.

### Interpretation of the Flagged Amount

The **$393,000 figure is neither recoverable cash nor an accusation**. Some flagged deals may have been intentionally discounted, for example, to win a first order, keep a mill busy, or support a multi-contract package. The dataset contains no field describing the intent behind a negotiated rate. Therefore, the appropriate interpretation is that **$393,000 of pricing lies outside the company's observed normal range, while the proportion that was deliberate remains unknown**.

The tool converts this uncertainty into a reviewable list. Thus, the figure represents the size of the question that requires review, rather than the size of an assured financial opportunity.

It is also worth noting that the figure may be an underestimate. Shrinkage pulls each supplier's benchmark toward a house median that is below most suppliers' own medians. Consequently, the resulting band floors are conservative and the method under-flags by construction.

## 8. Limitations

- **The benchmark explains 13.8% of out-of-sample rate variation.** Section 9 reports 31.8% for the same `supplier` column; that value is an in-sample result across all 3,802 deals and is used only to rank candidate features against one another. The 13.8% value is the relevant out-of-sample result because it measures performance on deals that the shipped model had not seen. Much of what determines a commission rate including relationship history, package deals, competitive pressure, and verbal agreements which is not represented in the dataset. Therefore, a 1.76pp benchmark band should be treated as negotiation guidance, not as a price.
- **The method cannot distinguish deliberate discounts from errors.** No intent field exists in the dataset, so every flagged deal requires human confirmation.
- **The data cover one company, one country, and one period.** The results should not be generalized to other sourcing agents without refitting the method.
- **The benchmark does not address the revenue decline.** The decline is primarily a volume and concentration problem, while pricing remained broadly stable. Notebook 02 documents this distinction explicitly.
- **Style names do not provide usable predictive signal.** There are 2,401 free-text style names, and only 5.0% contain any fabric specification. These features were tested and dropped; see Section 9.
- **2025 contains only four months of data** and is therefore excluded from year-on-year comparisons.
- **`Team_Head` is unreliable in the raw export for 372 rows**, of which 365 remain in the cleaned dataset, because the field contains a company name rather than a person's name. Consequently, no per-person conclusions are drawn in the project.
- **The reason the supplier is a better peer group than the customer has not been established.** One plausible explanation is that the rate reflects what the mill will concede rather than what the buyer will pay. However, the data cannot confirm this explanation, and the project does not present it as an established finding.

## 9. Approaches Evaluated and Abandoned

A complete methodology should document both successful and unsuccessful approaches. Accordingly, every result below is reproduced by an executed cell in [`notebooks/05_what_did_not_work.ipynb`](notebooks/05_what_did_not_work.ipynb).

### 9.1 Product Families from Style Names

The first alternative was to normalize the 2,401 free-text style names into fabric families and construct benchmarks within those families. The evidence did not support this approach.

Only **5.0%** of style names contain a fabric specification. The remaining names are largely buyer codes, such as `2068727/0`, `Tyler_143`, and `AMY_49`. Moreover, a careless `\d\s*/\s*\d` regular expression also matches buyer codes such as `2063808/15` and reports 22.6%, which is four and a half times the true proportion. This false positive would have made the approach appear substantially more promising than it actually was.

TF-IDF character n-gram clustering over a 2,401 × 7,618 matrix produced silhouette scores of **0.048–0.160** across 10 to 200 clusters, indicating no cluster structure that was sufficiently useful for the intended task. Inspection of the clusters further showed that they grouped code prefixes such as `Tyler_*` and `John_*`, which act as proxies for customer information already available in another column.

The garment segment explained **−9.2%** of rate variance, which was worse than quoting the overall median. Garment type explained 2.4%, whereas the plain `supplier` column explained 31.8% in-sample. The evidence therefore did not justify using style-name-derived product families.

### 9.2 SARIMA Forecasting

The second alternative was SARIMA forecasting. The original project forecast commission for 12 months and published four months of *negative* commission, which is not possible for an agent whose income consists of received commission. When backtested properly on a 12-month holdout of the cleaned series, SARIMA produced an MAE of **$63,593**, compared with **$59,732** for a naive last-value forecast. The seasonal-naive and 12-month-mean approaches produced MAEs of $75,744 and $73,027, respectively, while Holt-Winters performed worse again at $81,167. Thus, every sophisticated method tested was worse than the simple rule of using the last month's value.

The 70 monthly observations do not provide strong support for an annual seasonal pattern, particularly while the underlying business was approximately halving. The forecasting approach was therefore dropped.

The negative values did not reproduce on the cleaned series. They are consequently attributed to the original data preparation rather than to SARIMA itself. The remaining descriptive time-series analysis in the archive—including seasonal decomposition, ADF/KPSS tests, and ACF/PACF analysis—was methodologically sound and is not withdrawn.

### 9.3 Other Abandoned Alternatives

Two smaller alternatives were also rejected. First, merging `Takko`, `Takko Fab`, and `Takko Knit` as fuzzy-matched name variants was rejected once the suffix was shown to identify the product line. Second, benchmarking the fabric and denim lines was rejected because both have an interquartile rate range of exactly zero.

The common principle across these decisions is the project's stated method:

> **Build the simple alternative first, and make the sophisticated method beat it.**

## 10. Deployment as a Service

The benchmark is deployed as a service. `POST /score-deal` accepts the deal being negotiated and returns the benchmark band, the dollar gap, and, importantly, the specific historical contracts that support the result.

```json
POST /score-deal
{"supplier": "Samad Rubber", "customer": "Takko",
 "quantity": 20000, "unit_price_usd": 5.50, "proposed_rate_pct": 3.2}

{"deal_value_usd": 110000.0,
 "benchmark": {"low_pct": 3.73, "expected_pct": 4.94, "high_pct": 5.0,
               "based_on": "supplier", "past_deals": 362, "confidence": "high"},
 "expected_commission_usd": 5430.97,
 "assessment": {"position": "below band", "gap_to_expected_pp": 1.74,
                "gap_to_expected_usd": 1910.97,
                "verdict": "3.20% is below the 3.73% floor for this supplier.
                            Worth confirming the discount is intentional."},
 "comparable_deals": [...]}
```

The service exposes four routes: `GET /health`, `GET /suppliers`, `POST /score-deal`, and FastAPI's `/docs` interface. The benchmark is fitted once during application startup from the cleaned dataset. Therefore, no training occurs at request time and no model artifact is stored on disk.

Support of 100 or more historical deals is reported as **high** confidence, support of 30 or more as **medium**, and lower support as **low**. An unknown supplier falls back to the house average and reports this fallback in a warning.

Two implementation details are important. First, the support value of 362 shown in the example is larger than the 315 reported in notebook 03 because the service fits the benchmark on the complete cleaned history, whereas the notebook tables are fitted only on the 2019–2023 training slice. Second, a `Takko Fab` request returns `out_of_scope: true` with an explanation rather than a benchmark band because fabric rates are not negotiated.

The service is designed to remain auditable. Each comparable observation identifies a real historical contract that a user can locate and either confirm or reject. Consequently, users do not have to treat the benchmark as an opaque model in order to use it.

### 10.1 Next Development Priorities

The next developments, in order of expected value, are: first, a review interface for flagged deals so that negotiation decisions can be recorded, which would also create the intent labels currently absent from the dataset; second, a concentration monitor for the Takko risk; and third, quarterly refitting once enough recorded decisions have accumulated to evaluate whether flagged deals are subsequently repriced.

### Repository Structure

```text
notebooks/           01–05, stored executed and intended to be read in order
figures/             12 figures
```

The tests are not designed primarily for code coverage. As stated in their module docstring, they protect against **the specific mistakes that this project has already made once**. In particular, they verify that cleaning does not silently lose rows, that a fixed-rate product line cannot receive a negotiated benchmark band, that walk-forward scoring does not use future information, that leakage is counted only below the band, and that the leakage audit can identify a derived column without incorrectly flagging a clean one.

## Licence

Code is released under the [MIT Licence](LICENSE). The underlying commercial data is not covered by the licence and is not distributed.
