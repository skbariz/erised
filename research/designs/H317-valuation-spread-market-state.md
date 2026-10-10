# H317: ValuationSpreadMarketState — Growth/value ETF spread as daily risk-on/risk-off regime signal

## Hypothesis
The IVW (S&P 500 Growth) vs IVE (S&P 500 Value) 252-day return spread is a real-time proxy for risk appetite within equities. When growth outperforms value, investors are extending duration and paying up for future earnings — a behaviorally risk-on signal that has been linked to both momentum and the Fed model (Asness 2003; Fama-French 1992). We apply this signal to H021's exposure: growth-on regime (IVW > IVE) → 1.10x; value-on regime (IVE > IVW) → 0.90x. Unlike term-spread or VIX overlays, this signal is always available at market close, never gaps, and is not derived from SPY itself, keeping lookahead risk low.

**Expected behavior:**
- 2000-2002 tech crash: IVE massively outperforms IVW (value decade begins); persistent 0.90x scale reduces SPY/TLT H021 exposure and dampens drawdown
- 2008-2009 GFC: both ETFs fall, but financials (value-heavy) lead declines; spread ambiguous early, then value underperforms in deepest phase — signal less clean than term spread
- 2020 COVID: rapid rotation to growth (tech/QQQ surge); IVW crushes IVE from Q3 2020; scale lifts to 1.10x, capturing recovery
- 2013-2017 low vol: growth steadily outperforms; persistent 1.10x scale boosts CAGR modestly

**Success criteria:** OOS Sharpe >= 0.85; MaxDD < -28%; CI lower > 0.
**Falsification:** Value/growth spread has no predictive power for the portfolio; sensitivity flat across growth_scale 1.05-1.15. Or: 0.90x during value regimes hurts Sharpe because H021 already reduces SPY vol naturally.
