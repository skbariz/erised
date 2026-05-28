# H336: VRPRefinedH060 -- CarryMomentumSPYTLT (H060) + VRP overlay on SPY leg

## Hypothesis
H060 (CarryMomentumSPYTLT) produces three states: 100% SPY (curve steep + SPY trend up), 100% TLT (curve inverted), or flat. The 100% SPY state is the only one where additional risk management is warranted — when the curve is inverted (already in TLT) or flat (already in cash), there is no additional equity risk to trim. This strategy adds a single VRP signal only when H060 would be 100% SPY: if VRP < 0 at that moment, reduce to 70% SPY + 30% TLT instead. VRP < 0 means realized vol has been exceeding implied vol — the market is not pricing risk correctly. This is a mild bearish indicator that justifies trimming the full equity bet, without overriding the H060 logic entirely.

**Expected behavior:**
- 2000-2002: H060 likely in TLT (inverted curve); VRP overlay has no effect (already defensive)
- 2008-2009: H060 in TLT; overlay irrelevant; protection comes from H060 itself
- 2013-2017 low vol: H060 mostly in SPY; VRP consistently positive → overlay never fires → identical to H060; no cost
- 2020 COVID: H060 may be in TLT; VRP turns very negative → if H060 is in SPY, trimmed to 70/30

**Success criteria:** OOS Sharpe > H060 (likely 0.6-0.7); MaxDD improvement; CI lower > 0.
**Falsification:** VRP < 0 rarely co-occurs with H060's full-SPY state (H060 already has momentum + curve filters that exclude most stress periods), making the overlay irrelevant.

## Results
*[To be filled after backtest]*
Decision:
