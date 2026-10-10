# H331: MacroConfirmedSeasonal -- H078 + VRP/term-spread macro gate in Nov-Apr

## Hypothesis
H078 (SeasonalRiskParity) is the program's best diversifier (H001 corr +0.008) but applies H021 weights unconditionally in the Nov-Apr window. In winters where the macro environment is hostile — VRP is negative (realized vol exceeds implied) AND the yield curve is inverted (SHY outperforms TLT on 252d returns) — committing to full risk-parity exposure may be suboptimal. This strategy adds a macro gate: if VRP > 0 OR the term spread is positive, H021 weights apply in Nov-Apr (same as H078); if both signals fail simultaneously, fall back to a conservative 60% SPY / 40% TLT instead of full H021. May-Oct stays 100% TLT (unchanged from H078). The gate is an OR not an AND, meaning only when both macro signals are hostile does the fallback activate — preventing over-conservatism.

**Expected behavior:**
- 2000-2002: Negative VRP common, term spread likely inverted → gate fires some winters → partial protection in 2001-2002 winter
- 2008-2009: Deep crisis, both signals hostile → winters in 2008 and 2009 use 60/40 fallback instead of H021 → better protection
- 2013-2017 low vol: Positive VRP throughout → gate rarely fires → near-identical to H078
- 2020 COVID: Brief crisis; gate may fire for a few months → modest protection benefit

**Success criteria:** OOS Sharpe ≥ H078 (only need to beat 0.51 keep bar); MaxDD ≤ -18% (H078 was ~-18%); CI lower > 0; H001 corr near H078's +0.008 (expect +0.05 to +0.20 range).
**Falsification:** Gate fires too rarely to differentiate from H078; or gate fires during false positives (good winters) and removes returns without reducing drawdown.

## Results
*[To be filled after backtest]*
Decision:
