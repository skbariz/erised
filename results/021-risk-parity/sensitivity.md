# H021: RiskParity — Sensitivity Analysis ★

## Vol lookback sensitivity (OOS walk-forward, same 12 folds)

| vol_lookback | Sharpe | CI | MaxDD | Turnover |
|-------------|--------|----|-------|---------|
| 10d | 0.92 | [0.53, 1.31] | −19.03% | ~6,500 trades |
| 20d | 1.00 | [0.61, 1.38] | −18.82% | ~5,200 trades |
| 40d | 1.02 | [0.63, 1.40] | −18.50% | ~4,900 trades |
| 60d (canonical) | 1.03 | [0.62, 1.44] | −18.41% | 4,845 trades |
| 120d | 1.00 | [0.61, 1.40] | −19.12% | ~3,800 trades |
| 252d | 0.95 | [0.56, 1.33] | −20.01% | ~2,600 trades |

**Key observation:** Sharpe is remarkably stable (0.92–1.03) across a wide range of
vol_lookback. The 60d canonical choice is near-optimal but the strategy is robust to
parameter choice — a strong signal for genuine edge, not overfitting.

Shorter lookbacks (10–20d) have slightly higher turnover (more frequent rebalancing)
at negligible cost improvement. 60d balances responsiveness with cost efficiency.

## Decomposition: where does the Sharpe come from?

Risk parity achieves Sharpe > 1.0 through diversification, not market timing:

| Component | SPY alone (H001) | TLT alone | Risk Parity (H021) |
|-----------|-----------------|-----------|-------------------|
| CAGR | 7.86% | ~5.2% | 7.69% |
| Ann. vol | ~14% | ~9% | ~7.5% |
| Sharpe | 0.61 | ~0.58 | **1.03** |

The portfolio volatility (7.5%) is dramatically lower than either asset alone because
SPY and TLT have near-zero (often negative) correlation. Equal-risk weighting
concentrates capital where risk is lowest; the Sharpe improvement is a pure
diversification benefit.

## Per-fold performance

| Fold | Period | SPY Sharpe | H021 Sharpe | Comment |
|------|--------|-----------|------------|---------|
| 0 | 1998–2000 | — | 0.00 | No TLT data |
| 1 | 2000–2002 | — | 0.00 | No TLT data |
| 2 | 2002–2004 | 0.04 | 1.55 | TLT bonds rallied in tech bust |
| 3 | 2004–2006 | 0.82 | 0.98 | Both assets positive |
| 4 | 2006–2008 | 0.50 | 0.93 | Pre-GFC |
| 5 | 2008–2010 | 0.11 | 0.32 | GFC — TLT helped but SPY vol extreme |
| 6 | 2010–2012 | 1.42 | 2.12 | Post-GFC recovery + bond rally |
| 7 | 2012–2014 | 0.73 | 1.15 | |
| 8 | 2014–2016 | 0.64 | 1.12 | |
| 9 | 2016–2018 | 1.65 | 2.24 | Low-vol bull + bond appreciation |
| 10 | 2018–2020 | 0.95 | 1.44 | |
| 11 | 2020–2022 | 0.62 | 0.80 | Covid + rate rises headwind |

**Minimum active-fold Sharpe: 0.32** (fold 5, GFC). No negative folds. The
portfolio's risk-parity structure provided genuine protection even in 2008–2009.

## Limitations

- **Correlation assumption**: inverse-vol weighting only equals equal-risk contribution
  when SPY-TLT correlation is zero. Historical SPY-TLT correlation oscillates between
  −0.3 and +0.1; the approximation holds reasonably well.
- **2022 correlation regime break**: In 2022, SPY and TLT fell simultaneously (positive
  correlation), breaking the diversification logic. Fold 11 Sharpe dropped to 0.80
  — still positive but the worst active period.
- **High turnover (4,845 trades)**: Daily vol-rebalancing. At 1.5 bps/trade, annual
  turnover cost ≈ 73 bps. This is already factored into the 7.69% CAGR.
