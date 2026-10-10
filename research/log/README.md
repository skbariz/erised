# Research log

Every hypothesis we test gets a numbered markdown file in this directory.
Use `_TEMPLATE.md` as the starting point.

## Convention

- `001-short-name.md` — first hypothesis tested
- `002-short-name.md` — second
- ...

Killed hypotheses are kept, not deleted. The graveyard is the point.

## Index

| # | Title | Family | Status | Sharpe (OOS) | Notes |
|---|-------|--------|--------|--------------|-------|
| 000 | sanity: buy-and-hold | benchmark | reference | (n/a) | benchmark used everywhere |
| 001 | sanity: MA(50,200) crossover | trend | reference | (n/a) | sanity test of pipeline |
| 002 | mean reversion after 2σ down moves | mean reversion | killed | 0.24 | weak signal (CI contains zero); returns corr +0.55 with H001 |
| 003 | short-hold mean reversion (2d) | mean reversion | killed | 0.01 | near-zero Sharpe; 1-day hold variant (0.45) noted |
| 004 | RSI oversold reversion | mean reversion | killed | 0.06 | CI contains zero; sparse signal |
| 005 | consecutive down days | mean reversion | killed | 0.44 | CI positive but corr +0.40 fails ensemble test |
| 006 | VIX-filtered reversion | mean reversion | killed | 0.01 | VIX filter adds no value |
| 007 | 12-1 month time-series momentum | trend | kept | 0.62 | matches H001; high trend corr confirms same signal family |
| 008 | absolute dual momentum (12m) | trend | kept | 0.62 | nearly identical to H001 and H007 |
| 009 | MACD crossover (12,26,9) | trend | killed | 0.37 | CI lower bound negative; dominated by H001 |
| 010 | fast MA crossover (10/50) | trend | killed | 0.45 | real signal but dominated by H001 on Sharpe and turnover |
| 011 | ADX-filtered MA crossover | trend | killed | 0.27 | CI contains zero; ADX filter adds lag without value |
| 012 | realized vol regime | volatility | killed | 0.33 | CI contains zero; MaxDD near SPY level |
