# Comparison with an independent Isolation Forest analysis

An independent data-science review scored the same 50 cases with an unsupervised model
(`Fraud_Detection_Algo.ipynb`, output `final_output.csv`). This page compares it with the cockpit's triage,
case by case, and records what was adopted.

## The two approaches

| | Independent analysis | Cockpit |
|---|---|---|
| Primary model | Isolation Forest (11 features), 30 seeds averaged | Transparent rules: capped signal groups, combination rules, strong indicators |
| Score | Percentile rank of the anomaly score (0-100) | Absolute points out of 100 |
| Tiers | Fixed percentile bands: Low < 35, Medium < 65, High < 85, Critical | Thresholds on points: red ≥ 60 or strong indicator, yellow ≥ 20 |
| Explanation | Per-feature percentile of the top 3 features | The findings that produced the points, with source columns |
| Stability | Bootstrap over seeds (score range) | Confidence: data quality, peer size, rules/anomaly disagreement, ±20% weight perturbation |
| Other models | None | Seed-averaged Isolation Forest + LOF + PCA as a cross-check on the real cases. No supervised model is used in the product (no outcome labels); a supervised pipeline exists for development only |
| Data quality | Duplicate claim number found in exploration | Duplicate claim number routes the case to Data review |

The inputs are identical (all 50 rows match the supplied file).

## How closely they agree

- Rank correlation of the two scores: **0.86**.
- Top 10 cases: **8 of 10** shared.
- Every cockpit high-priority case is Critical or High in the independent analysis, and no cockpit
  lower-priority case is Critical.

| Independent tier ↓ / Cockpit level → | High priority | Needs attention | Data review | Lower priority |
|---|---|---|---|---|
| Critical | 7 | 1 | 0 | 0 |
| High | 3 | 6 | 0 | 1 |
| Medium | 0 | 0 | 1 | 14 |
| Low | 0 | 0 | 0 | 17 |

## Where they differ, and why

| Case | Independent | Cockpit | Explanation |
|---|---|---|---|
| C1023 | Critical (96) | High priority (62.9) | Shared contact + service overlap + policy change, no duplicate billing. The cockpit had no rule for this pattern and left it just below the red line (57.9). **Adopted a new combination rule** (below). |
| C1011 | Critical (90) | Needs attention (27.6) | Several moderately elevated signals (9 visits/week, 69 miles, +45% vs peer, policy change) and no billing-integrity flag. Isolation Forest reacts strongly to multi-dimensional oddness; the rules treat it as worth a look, not top priority. Both methods mark it unusual. |
| C1001 | Medium (38) | Data review | Its claim number is also used by C1031. The independent analysis found this in exploration but the score does not use it. |
| C1013, C1029, C1004 | Medium (40-58) | Lower priority (0) | Small, quiet claims (for example C1013: $858, 2 visits/week, every signal at or below the norm). Isolation Forest is two-sided: unusually low values are "anomalous" too. The explanations list median values as drivers ("top 45% of queue"), because they are per-feature percentiles rather than the model's reasons. |
| 15 cases | Medium | Lower priority | Percentile banding always places about 30% of any queue in Medium and 15% in Critical, however risky the queue really is. The cockpit's points are absolute: a queue of benign cases stays green. |

## What was adopted

1. **Shared contact + service overlap combination (rules-v1.1, +5 points).** Suggested by the exploration
   finding that the two flags co-occur (9 of the 13 cases with either flag). C1023 moves to high priority with a
   specific next step: check whether the overlapping provider is linked to the member. As an offline
   development check against the synthetic set, the change was neutral (PR-AUC 0.733 before and after).
2. **Seed-averaged Isolation Forest** in the anomaly cross-check: 10 forests of 100 trees (3 forests for very
   large queues), 1,000 trees averaged instead of one 300-tree forest. Individual forests disagreed by up to
   30 percentile points on some cases; the case view now shows the average and the spread.
   The averaged cross-check marks C1011, C1019, C1023 and C1050 unusual, four of the independent top six.
3. **Direction check for anomalies.** Learning from the small-claims effect above, an anomaly only raises a
   lower-priority case to Needs attention when the case is also unusual in a suspicious direction (a flag is set
   or a signal is above the queue norm).

4. **The second opinion is visible.** The Overview has a "Rules vs Isolation Forest" chart (rules score against
   the seed-averaged anomaly percentile, disagreements shaded and labelled), and a case where the methods disagree
   shows a "Second opinion" note explaining why. On the supplied data the disagreements are C1011 and C1019.

## What was deliberately not adopted

- **Percentile-rank scores and tiers.** They make the score relative to today's queue: the same claim can be
  "Critical" one day and "Medium" the next, and a fixed share of every queue is always Critical.
- **Claim amount as a scoring feature.** The cockpit treats amount as financial exposure (shown separately and
  used as a tie-breaker), not as evidence, as the product spec requires.
- **An unsupervised model as the primary score.** With no outcome labels, a transparent rule set is easier for
  investigators to challenge (reject a finding and see the score change) and for the business to govern.
  Isolation Forest stays as the cross-check, where its strength (spotting unusual combinations) is most useful.
