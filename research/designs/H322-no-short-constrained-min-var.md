# H322: NoShortConstrainedMinVar -- Minimum variance portfolio over H001-H100 strategies

## Hypothesis
MDR (H095) maximizes diversification ratio which implicitly rewards low-vol, high-vol-dispersion strategies. Minimum variance instead directly minimizes portfolio variance -- it should produce lower realized volatility and potentially lower MaxDD at the cost of some expected return. Minimum variance portfolios are known to deliver competitive Sharpe ratios in equity markets (Clarke, de Silva & Thorley 2011) due to the low-volatility anomaly. Applied to strategy return streams, minimum variance weights the most stable, consistent strategies most heavily. The no-short constraint (w >= 0) prevents the optimizer from shorting strategies (which would be mechanically shorting ETF-based portfolios and adds leverage/cost risk). The 25% per-strategy cap prevents extreme concentration; at N=35 strategies the unconstrained min-var often degenerates to 2-3 strategies. This is a direct alternative to H095 that optimizes a different objective (minimize risk rather than maximize diversification ratio).

**Expected behavior:**
- vs H095 (Sharpe 1.133, MaxDD -7.16%): expect lower MaxDD (pure variance minimization should produce the smoothest equity curve) but potentially lower Sharpe if reducing variance also reduces returns
- vs H048 equal-weight (Sharpe 0.800): expect meaningful Sharpe improvement via concentration on stable strategies
- 2000-2002 tech crash: min-var heavily weights H043/H078 (lowest vol strategies); expect strong protection
- 2008-2009 GFC: same low-vol weighting benefits in crisis; expect similar behavior to H095
- 2013-2017 low vol: stable return period -- min-var should perform well since stable strategies dominate
- 2020 COVID: brief spike -- min-var weighting toward low-vol strategies means lower initial drawdown

**Success criteria:** OOS Sharpe >= 1.00; MaxDD < -10%; CI lower > 0. MaxDD specifically should beat H095 (-7.16%) or come close, since min-var directly minimizes variance.
**Falsification:** MaxDD worse than H095 (would suggest min-var is overfitting to the OOS covariance structure). Sharpe < 0.90 would suggest min-var sacrifices too much expected return.

## Results
(to be filled after running)
