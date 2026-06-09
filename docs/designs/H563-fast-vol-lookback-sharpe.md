# H563: Fast Vol Lookback Leveraged Sharpe (target_vol=6.5%, cap=2.0×, lookback=5d)

## Signal
Lev-Sharpe SLSQP on 55-strategy pool with fast volatility lookback (5d vs H559's 20d).
Same pool, target_vol, and cap as H559; only lookback changes.

## Why should it work?
H559 used a 20d rolling vol estimate to set leverage. A 5d lookback reacts faster to vol spikes,
potentially reducing "caught with high leverage at peak vol" events. In crash regimes where vol
spikes are sudden (March 2020, Oct 2008), a 5d estimate de-levers within the same week rather
than lagging 4 weeks. This should reduce MaxDD while keeping CAGR similar to H559.

## Expected regime behavior
- 2000-2002 tech crash: gradual decay → 5d and 20d estimates converge; minimal difference
- 2008-2009 GFC: rapid vol spikes → 5d de-levers much faster, potentially saving drawdown
- 2020 COVID: 5d reacts within days of February sell-off → lower peak leverage at peak loss
- 2013-2017 low vol: 5d noisier → more leverage oscillation, potential slight Sharpe drag

## Risk
5d vol estimates are noisier day-to-day, leading to higher leverage oscillation and increased
turnover costs (borrow cost applies to positive leverage above 1.0). If oscillation dominates,
Sharpe degrades vs H559.

## Success criteria
Sh_lev > 1.879 (H559) OR MaxDD < −9.73% (H559) at Sh_lev > 1.7.

## Falsification
5d lookback introduces too much noise — Sharpe degrades below H559 AND MaxDD worsens
(worst of both worlds: high turnover without drawdown improvement).

## Reference
H559: Sh_lev=1.879, CAGR=13.08%, MaxDD=−9.73%, at 6.5%/2.0x/20d.
