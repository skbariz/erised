# H344: VolTargetedMultiSleeve — Four-sleeve inverse-vol weighted blend (H021 / H086 / H078 / H089) quarterly rebalanced

## Hypothesis
Risk parity applied at the strategy-blend level rather than the asset level. Instead of allocating 1/σ to SPY vs TLT, allocate 1/σ to four strategy sleeves: H021 (RiskParity), H086 (LowVolTargetPortfolio), H078 (SeasonalRiskParity), H089 (H043H021RegimeBlend). Each sleeve's trailing 252d realized volatility determines its inverse-vol weight. Sleeves with lower realized volatility receive more capital — this naturally increases H078 and H089 weight when they are in calm TLT positioning, and increases H086 weight when vol-targeting has reduced H086's realized vol below H021. Quarterly rebalancing (first trading day of Jan, Apr, Jul, Oct) prevents transaction cost drag from daily recalculation while capturing meaningful vol-regime shifts. This is motivated by H090 (monthly rebalance parity) which showed rebalance frequency matters less than expected — quarterly is sufficient and more cost-efficient than daily. The four components span two uncorrelated families (momentum-based risk parity, vol-scaled parity, seasonal parity, calendar-enhanced parity), which should produce stable inverse-vol weights.

**Expected behavior:**
- 2000-2002 tech crash: H021 and H086 may have elevated vol from equity exposure; H078 (in TLT Oct-May) has low vol; H078 receives higher weight, increasing defensive positioning automatically
- 2008-2009 GFC: H021 vol spikes; H086's vol-targeting reduces its own realized vol (mechanism reduces exposure in high-vol); H086 and H078 receive higher inverse-vol weights; natural de-risking
- 2020 COVID: brief spike then normalization; all four strategies experience vol elevation; weights remain approximately equal; smooth recovery
- 2013-2017 low vol: all four strategies show low vol; weights converge toward equal; behavior similar to equal-weighted blend; no harm from the mechanism

**Success criteria:** OOS Sharpe >= 0.95; MaxDD < -15%; CI lower > 0; the quarterly rebalance provides enough vol signal without excessive turnover.
**Falsification:** Inverse-vol weights at the strategy level are too noisy (252d window too short for quarterly strategy-level vol estimation); weights are unstable and produce worse risk-adjusted returns than equal weighting.
