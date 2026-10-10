# H226 — BookToMarketMomentum

## Hypothesis

Value-momentum composite: z(book_equity/price) + z(12-1m momentum). Combines the Fama-French HML value factor with price momentum, which have low correlation to each other (value tends to be contrarian, momentum is trend-following). The composite should capture both the value premium (cheap stocks outperform) and the momentum premium (recent winners continue) without having price or fundamental data dominate.

**B/M proxy**: book_equity / adj_close_price. This is per-share equivalent since both numerator and denominator are on the same stock — it is a price-normalized measure that is comparable cross-sectionally.

**Mechanics**: bm = book_equity / price. price_ret = 12-1m return. score = z(bm) + z(price_ret). Sector-neutral top 20%. Monthly rebalance.

## Expected behavior by regime

- **2000-2002 tech crash**: High B/M (value) stocks avoided the tech crash; momentum also turned negative for tech after the peak → both signals agree to avoid high-price growth stocks.
- **2008-2009 GFC**: Value stocks (high B/M) appeared cheap during the crisis but continued to decline. Momentum also turned negative for financials. Mixed signal.
- **2020 COVID**: Momentum component captures the rapid recovery leaders; value component adds diversification.
- **Low-vol 2013-2017**: Both momentum and value premium are documented in this regime; composite should perform well.

## Success criteria

- OOS Sharpe > 0.51 with CI lower bound > 0.
- H001 corr slightly above zero (momentum component partially correlated with trend-following).
- Better Max DD than pure momentum strategies (value component provides quality anchor).

## Falsification

If Sharpe < H163 (residual momentum, ~0.96) substantially, value dilutes the pure momentum signal. If CI lower ≤ 0, the B/M proxy (using price as M) is not informative.

## Results (2026-05-25)

**Decision: KEEP**

| Metric | Value |
|---|---|
| OOS Sharpe | 0.766 |
| 95% CI | [0.364, 1.176] |
| CAGR | 9.77% |
| Max DD | -38.18% |
| H001 corr | -0.015 |

**Per-fold Sharpes**: 0.000, 0.000, 0.000, 0.000, 0.000, 0.582, 0.651, 2.286, 0.918, 1.672, 0.953, 0.920

Fold 5 active (0.582) due to partial B/M coverage (book_equity has some XBRL data pre-2010). Folds 6-11 average 1.233. H001 corr -0.015 — the momentum component did not reintroduce equity trend correlation. Max DD -38.18% is better than pure momentum strategies. 504d momentum window is marginally stronger (0.768 vs 0.766).
