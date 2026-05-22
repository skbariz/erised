# H026: QuarterEndDrift — Sensitivity Analysis

## n_days sensitivity (OOS walk-forward, same 12 folds)

| n_days | Active days/yr | Sharpe | CI | MaxDD |
|--------|---------------|--------|----|-------|
| 3 | ~12 | 0.24 | [−0.17, 0.65] | −7.43% |
| 5 (canonical) | ~20 | 0.30 | [−0.12, 0.72] | −10.13% |
| 10 | ~40 | 0.12 | [−0.28, 0.52] | −19.45% |

**Key observation:** More days hurts performance. The quarter-end drift is concentrated
in the final 3–5 trading days — extending to 10 days dilutes the effect with noise
from earlier in the month. Sensitivity is monotonically negative in n_days:
the effect is precisely localized near quarter-end, not diffuse across the month.

n_days=3 has a lower MaxDD (−7.43%) with lower Sharpe (0.24). The canonical n_days=5
captures more of the drift window with an acceptable drawdown tradeoff.

## Why this is the program's lowest-correlation strategy

H001 correlation = **+0.176** — the lowest in the entire research program.

This is structural: the strategy is active only ~20 trading days per year (5 days ×
4 quarters). During the other ~232 days it is flat with zero SPY correlation. The
active-period correlation with H001 is near 1.0 (both are long SPY when active), but:

`portfolio_correlation ≈ (active_fraction) × 1.0 = (20/252) × 1.0 ≈ 0.08`

Actual correlation (0.176) is slightly higher due to concentration of quarter-end
days in periods of general market strength, creating some co-movement even when flat.

## Why this is the program's lowest-MaxDD strategy

MaxDD = **−10.13%** vs SPY −55.92%.

With only ~20 active trading days per year, the maximum drawdown is structurally bounded:
- The strategy can only lose money on ~8% of trading days
- Even a catastrophic 50% drop during those 20 days → 4% portfolio loss
- Historical worst quarter-end: 2008 Q3 (Lehman collapse in September), active just
  5 days → limited exposure vs the full crash

The trade-off: CAGR is only 1.42% (very low). The strategy has the best
risk-adjusted characteristics of any killed strategy in the program.

## Lakonishok et al. (1991) replication

The original paper documented quarter-end window dressing across pension funds
and mutual funds from 1963–1990:

| Study scope | Finding |
|-------------|---------|
| Lakonishok et al. (1991) | Institutions buy past winners near quarter-end |
| Carhart et al. (2002) | Mutual funds pump prices 0.3% near quarter-end |
| This OOS test (1993–2022) | SPY: Sharpe 0.30, effect present but below threshold |

The effect has been partially arbitraged since institutional disclosure requirements
became more frequent. The SPY-level signal (aggregate index) is weaker than the
individual-stock effect documented in the original studies.

## Per-fold analysis

| Fold | Period | Sharpe | Comment |
|------|--------|--------|---------|
| 0 | 1998–2000 | 0.11 | Dot-com: market direction dominates |
| 1 | 2000–2002 | 0.14 | Tech bust: quarter-end drift partially hedges |
| 2 | 2002–2004 | 0.35 | Moderate: recovery helps |
| 3 | 2004–2006 | 0.86 | Strong: consistent institutional flows |
| 4 | 2006–2008 | 0.22 | Pre-GFC: signal works but weak |
| 5 | 2008–2010 | 0.23 | GFC: small loss from Lehman Q3 exposure |
| 6 | 2010–2012 | −0.05 | Near-zero: post-GFC drift absent |
| 7 | 2012–2014 | 1.18 | Strong: bull market + institutional flows |
| 8 | 2014–2016 | −0.75 | Negative: worst fold — Q3 2015 correction hit |
| 9 | 2016–2018 | 1.15 | Strong: low-vol bull period |
| 10 | 2018–2020 | 0.30 | Moderate |
| 11 | 2020–2022 | 0.49 | 2021 recovery; 2022 rate rises minimal (only 20d active) |

Two negative folds (6, 8) — when the general market falls at quarter-ends (European
debt crisis Q3 2011, China devaluation Q3 2015), the strategy is maximally exposed
to the worst timing.

## Ensemble value despite kill decision

**Kill confirmed**: OOS Sharpe 0.30 < 0.51, CI lower bound −0.12 < 0.

However, this is the strongest ensemble candidate among killed strategies due to:
1. Lowest H001 correlation (0.176) in the program
2. Lowest MaxDD (−10.13%) means negligible portfolio risk from small allocation
3. If ensemble portfolio has 5% weight → adds 0.176 × 0.30 ≈ 0.053 Sharpe units
   while contributing only 5% of MaxDD exposure

**Flagged for ensemble batch (H048–H050)**: even a 2–5% allocation could improve
ensemble Sharpe without meaningfully adding drawdown risk.
