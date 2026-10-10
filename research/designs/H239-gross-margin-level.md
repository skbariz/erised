# H239: GrossMarginLevel — Persistent GP/Revenue Level

## Hypothesis

High gross margin is a structural competitive advantage — it reflects pricing power, a wide economic moat, or a differentiated product that commands a premium. Unlike change-based signals (H214: gross margin momentum gate), the level itself should predict: companies with persistently high GP/revenue have durable cost advantages that compound over time.

This is the level-based counterpart to H214 (gate) and H229 (change). Testing whether persistent quality as of the current period — without requiring recent improvement — predicts future returns.

Expected regime behavior:
- **2000–2002 tech crash**: High-margin software/IP companies may underperform early but recover faster.
- **2008–2009 GFC**: High-margin firms retain profitability when revenue drops — safety in downturns.
- **2020 COVID**: High-margin firms (tech, pharma) outperform dramatically; low-margin firms (retail, airlines) crushed.
- **2013–2017 low-vol**: Persistent quality premium should be stable and positive throughout.

Success criteria: OOS Sharpe > 0.51, CI lower bound > 0. Expect moderate H001 corr (quality factors are price-independent).
Falsification: If the level has no predictive power beyond sector membership (all high-margin firms are in the same sector), the sector-neutral ranking would eliminate the signal.

## Results

| Metric | Value |
|--------|-------|
| OOS Sharpe | 0.861 |
| 95% CI | [0.457, 1.269] |
| CAGR | 10.78% |
| Max DD | −34.59% |
| H001 corr | −0.012 |
| Decision | **KEEP** |

**Sensitivity (within_sector_quantile):** 10% → 0.862/MaxDD −30.49%, 20% → 0.861, 30% → 0.856. Remarkably flat — signal is robust to portfolio concentration. Top 10% quantile achieves best MaxDD (−30.49%) with equivalent Sharpe.

**Folds 6–11 Sharpe:** 0.827, 2.218, 1.053, 1.957, 1.216, 1.037. Fold 5 shows 0.432 (partial EDGAR coverage). Fold 7 at 2.218 is the highest in the batch.

**Batch star** — best Sharpe (0.861), best MaxDD (−34.59%), and highest fold 7 Sharpe (2.218) in Batch 8. Level-based gross margin signal proves superior to change-based variants: persistent pricing power is a more stable predictor than recent margin improvement. Confirms Novy-Marx (2013) finding that GP/TA (H227) generalizes: denominator matters less than gross profit signal itself.
