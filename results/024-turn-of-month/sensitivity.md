# H024: TurnOfMonth — Sensitivity Analysis

## n_days sensitivity (OOS walk-forward, same 12 folds)

| n_days | Active days/yr | Sharpe | CI | MaxDD |
|--------|---------------|--------|----|-------|
| 2 | ~48 | 0.32 | [−0.09, 0.73] | −25.71% |
| 3 (canonical) | ~72 | 0.41 | [0.02, 0.82] | −34.41% |
| 5 | ~120 | 0.54 | [0.13, 0.94] | −38.95% |

**Key observation:** n_days=5 achieves Sharpe 0.54 (above the 0.51 benchmark threshold)
but at the cost of a substantially wider drawdown (−38.95%). The n_days=3 canonical
misses the threshold. The effect is real but concentrated near month-end: adding more
days dilutes the signal with mid-month noise while increasing drawdown exposure.

The canonical n_days=3 is killed strictly on the 0.51 threshold. A pragmatic
re-evaluation with n_days=5 would produce a KEEP decision.

## Academic evidence

Ariel (1987) showed that virtually all US stock market gains from 1963–1981 occurred
in the last day of a month and the first four days of the following month. Since then:

| Study | Market | Finding |
|-------|--------|---------|
| Ariel (1987) | US | Pre-holiday bias at month turn |
| Lakonishok & Smidt (1988) | US | 90+ years of confirmation |
| Cadsby & Ratner (1992) | 10 countries | International confirmation |
| This OOS test (1993–2022) | SPY | n_days=3 Sharpe 0.41; n_days=5 Sharpe 0.54 |

The effect has weakened post-publication but not eliminated — consistent with partial
arbitrage by institutional traders aware of the anomaly.

## Per-fold analysis

| Fold | Period | Sharpe | Comment |
|------|--------|--------|---------|
| 0 | 1998–2000 | 0.20 | Effect muted in late 1990s bull |
| 1 | 2000–2002 | 1.08 | Tech bust: month-turn rallies captured |
| 2 | 2002–2004 | 0.35 | Recovery — inconsistent |
| 3 | 2004–2006 | 1.44 | Strong: near-threshold consistency |
| 4 | 2006–2008 | 0.47 | Below benchmark |
| 5 | 2008–2010 | −0.41 | GFC: month-turn structure broke |
| 6 | 2010–2012 | −0.15 | Post-GFC: negative |
| 7 | 2012–2014 | 0.96 | Recovery of pattern |
| 8 | 2014–2016 | −0.29 | Pattern absent |
| 9 | 2016–2018 | 1.05 | Strong bull — pattern resurgent |
| 10 | 2018–2020 | 0.68 | Moderate performance |
| 11 | 2020–2022 | 1.18 | COVID era: month-turn pattern strong |

**Three negative folds** (5, 6, 8) — the signal is regime-dependent and fails in
volatile, trend-less environments (2008–2012, 2014–2016).

## Why the effect exists mechanically

Institutional cash flows concentrate at month boundaries:
1. **Pension contributions**: monthly contributions invested in first days
2. **Dividend reinvestment**: many ETFs/funds reinvest at month-start
3. **Window dressing reversal**: managers sell losers at month-end then buy back
4. **Options expiration**: gamma hedging flows at contract rollovers

These mechanical flows create predictable buy pressure at exact calendar points,
independent of market trends — explaining the low H001 correlation (0.363).

## Turnover note

576 trades over ~12 OOS years = ~48 trades/year = switching on/off every ~5 days.
At 1.5 bps/trade, annual turnover cost ≈ 86 bps, which substantially erodes the gross
TOM premium, especially for the n_days=3 version.

## Kill re-evaluation threshold

Kill confirmed: Sharpe 0.41 < 0.51. Candidate for revisit with n_days=5 in the
ensemble batch (H048–H050) where its low H001 correlation (0.363) could add value
as a small-weight component.
