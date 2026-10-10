# H257: CHSMarketDistress — Market+Accounting Failure Probability

## Hypothesis

Campbell, Hilscher, and Szilagyi (2008) demonstrate that market-based inputs (equity value,
recent returns, volatility) combined with accounting inputs (leverage, profitability) predict
failure probability more accurately than pure accounting scores.

Signal: CHS-style distress composite using:
- market_cap proxy = price × shares_outstanding (log-size signal)
- leverage = (total_assets - book_equity) / total_assets
- roa = net_income / total_assets
- 3-month return (price momentum)
- 12-month realized volatility
- cash_ratio = operating_cf / total_assets

Z-score each; sum with published-sign constraints (leverage+, roa-, return-, vol+, cash_ratio-,
log_size-). Long bottom-quintile (low estimated failure probability). Sector-neutral, monthly.

Market-based inputs react faster than accounting; the combination is more predictive than
either alone (H256 is pure accounting; H257 adds market signals).

Expected regime behavior:
- **2008–2009**: Strongest — both market and accounting signals align.
- **2000–2002**: Strong — equity values collapse for distressed tech.
- **2013–2017**: Weak — spreads compressed across the board.
- **2020**: Mixed — sudden shock; market signals react; accounting lags.

Success criteria: OOS Sharpe ≥ 0.70, Max DD < −45%, CI lower bound > 0.
Falsification: H257–H256 correlation > 0.85 (not additive to pure accounting signal).

## Results

| Metric | Value |
|--------|-------|
| OOS Sharpe | 0.712 |
| 95% CI | [0.314, 1.114] |
| CAGR | 8.24% |
| Max DD | −34.46% |
| H001 corr | −0.017 |
| Decision | **KEEP** |

**Sensitivity (vol_window):** 126d → 0.720 | 252d → 0.712 | 504d → 0.710. Nearly flat — market vol signal is robust across lookback windows.

**H257 (0.712) underperforms H256 (0.766):** Surprising — pure accounting O-score beats combined market+accounting signal. Market inputs (recent return, vol) may add noise in cross-sectional S&P 500 universe. Weakest in batch 2. Both signals have near-zero H001 corr, confirming orthogonality to trend-following.
