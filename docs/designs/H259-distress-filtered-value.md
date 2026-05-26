# H259: DistressFilteredValue — Value With Distress Guard

## Hypothesis

Cheap stocks can be value traps. Campbell, Hilscher, and Szilagyi (2008) show that
accounting-cheap stocks are often distressed, and low-distress cheap stocks earn the
full value premium while high-distress cheap stocks do not.

Signal: Book-to-market value score (book_equity / (price × shares_outstanding)) combined with
CHS-style distress screen. Include only stocks in:
- Top-quintile book-to-market (cheap on B/M) AND
- Bottom-half CHS distress composite (H257 signal ≤ 50th percentile)

Equal-weight qualifying names. Sector-neutral, monthly rebalance.

The distress filter removes value traps while preserving genuinely cheap firms with
manageable balance-sheet health.

Expected regime behavior:
- **2000–2002**: Strong — value premium plus distress screen should avoid telecom blowups.
- **2008–2009**: Strong in books; bank/fin names may fail distress screen and be correctly excluded.
- **2013–2017**: Moderate — value spreads compressed; distress filter provides modest benefit.
- **2020**: Mixed — sudden shock distorts both signals simultaneously.

Success criteria: OOS Sharpe ≥ 0.75, Max DD < −45%, CI lower > 0, Sharpe > raw B/M factor.
Falsification: Distress filter excludes so many value names that N < 20 per rebalance on average.

## Results

| Metric | Value |
|--------|-------|
| OOS Sharpe | |
| 95% CI | |
| CAGR | |
| Max DD | |
| H001 corr | |
| Decision | |
