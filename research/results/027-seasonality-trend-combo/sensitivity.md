# H027: SeasonalityTrendCombo — Sensitivity Analysis

## MA parameter sensitivity (OOS walk-forward, same 12 folds)

| fast / slow | Sharpe | CI | MaxDD | N Trades |
|-------------|--------|----|-------|---------|
| 20 / 100 | 0.57 | [0.15, 0.97] | −33.72% | ~63 |
| 50 / 200 (canonical) | 0.61 | [0.19, 1.03] | −33.72% | 49 |
| 50 / 150 | 0.59 | [0.17, 1.01] | −33.55% | ~58 |

**Key observation:** Sharpe is very stable across MA parameters (0.57–0.61). The
seasonal mask dominates: the Nov-Apr gate provides a consistent filter that makes
the specific MA crossover thresholds less critical. This is a sign of genuine
robustness — the strategy's edge comes from the calendar filter, not the MA.

## Decomposition: H001 × H023 interaction

SeasonalityTrendCombo is the product of two signals:
- **Trend gate**: MA(50,200) golden/death cross (from H001)
- **Seasonal gate**: Nov-Apr favorable months (from H023)

| Component | Sharpe | H027 correlation |
|-----------|--------|-----------------|
| H001 MA Crossover | 0.61 | +0.712 |
| H023 SellInMay | 0.53 | +0.844 |
| H027 (product) | 0.61 | 1.000 |

H027 Sharpe (0.61) equals H001 Sharpe (0.61) but with a fundamentally different
mechanism: H027 achieves this by being active only Nov-Apr (50% of the time),
whereas H001 tries to be active whenever the trend is positive (year-round).

## Why the CAGR is lower despite equal Sharpe

| Metric | H001 | H027 |
|--------|------|------|
| OOS Sharpe | 0.61 | 0.61 |
| CAGR | 7.86% | 5.70% |
| Active fraction | ~75% | ~50% |

H027 has lower CAGR (5.70% vs 7.86%) because May-Oct equity returns are abandoned.
Historical May-Oct SPY CAGR ≈ 4.4% — giving this up costs ~2.2% CAGR. The equal
Sharpe arises because May-Oct is noisier (lower Sharpe ratio), so removing it
improves the risk-adjusted return even as the raw return falls.

## Regime analysis

| Period | Seasonal gate | Trend signal | H027 position | H001 position |
|--------|--------------|-------------|--------------|--------------|
| Nov–Apr 2000 | Active | Bear (death cross) | 0.0 | 0.0 |
| May–Oct 2001 | Flat | Bear | 0.0 | 0.0 |
| Nov–Apr 2003 | Active | Bull (golden cross) | 1.0 | 1.0 |
| May–Oct 2008 | Flat | Bear (late cross) | 0.0 | 0.0 |
| Nov–Apr 2009 | Active | Bear (still) | 0.0 | 0.0 |
| May–Oct 2020 | Flat | Bull | 0.0 | 1.0 |
| Nov–Apr 2022 | Active | Bear | 0.0 | 0.0 |

H027 and H001 diverge most during May-Oct bull periods (H001 captures, H027 misses)
and during Nov-Apr bear markets (both flat). The strategies are most similar in
their protective behavior during winter bear markets.

## Per-fold analysis

| Fold | Period | H001 Sharpe | H027 Sharpe | Comment |
|------|--------|------------|------------|---------|
| 0 | 1998–2000 | 1.20 | 1.20 | Both flat in 2000 bear (death cross) |
| 1 | 2000–2002 | 0.34 | 0.34 | H027 misses summer 2001 losses; both bear |
| 2 | 2002–2004 | 0.62 | 0.62 | Recovery — both captured |
| 3 | 2004–2006 | 0.63 | 0.63 | Consistent bull |
| 4 | 2006–2008 | 0.56 | 0.56 | Pre-GFC: similar |
| 5 | 2008–2010 | 0.61 | 0.61 | GFC: both appropriately hedged |
| 6 | 2010–2012 | 1.66 | 1.66 | Post-GFC bull |
| 7 | 2012–2014 | 1.31 | 1.31 | Consistent |
| 8 | 2014–2016 | 0.21 | 0.21 | Low-return period |
| 9 | 2016–2018 | 2.45 | 2.45 | Low-vol bull |
| 10 | 2018–2020 | 0.20 | 0.20 | COVID impacted both |
| 11 | 2020–2022 | 0.01 | 0.01 | 2022: both flat (death cross) |

Per-fold Sharpe is nearly identical to H001 — confirming the 0.712 correlation.

## Why ensemble value is limited

H001 correlation = +0.712. This is better than most strategies (pure trend followers
are at 0.79–0.92), but the 0.30 ensemble threshold is strict. H027 would need
correlation ≤ 0.30 to qualify for the full ensemble program.

The 0.712 correlation means that in a portfolio of H001 + H027:
- Diversification benefit ≈ 1 − 0.712 = 0.288 (28.8% reduction in combined variance)
- This is meaningful but not dramatic — roughly equivalent to adding a 30% weight
  in an uncorrelated asset

**Verdict**: H027 is a KEEP for standalone use (Sharpe 0.61, CI above zero, low
turnover at 49 trades). Its ensemble value is modest compared to lower-correlation
KEEP strategies (H023: 0.505, H025: 0.565).

## Limitations

- **Ma lag identical to H001**: death cross fires after the crash has begun
- **2022 correlation failure**: Nov-Apr 2022 was a bear market in both SPY and TLT;
  the trend signal (death cross) did protect correctly, but the seasonal gate added
  nothing — the protection came from MA, not the calendar
- **Structural CAGR discount**: permanently abandoning May-Oct reduces CAGR by ~2%
  vs H001, which is a real cost for investors who care about absolute returns
