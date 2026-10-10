# H318: RegimeClassifier — 4-state macro regime (Growth/Recovery/Caution/Stress) with VIX + SPY momentum

## Hypothesis
Hamilton (1989) and subsequent regime-switching literature show equity returns cluster into distinct macro states. Rather than a continuous signal, this strategy discretizes the macro environment into four states using SPY 252d return (annualized momentum) and VIX level (fear gauge), and applies a fixed exposure multiplier per state. Growth (SPY up, VIX < 20): full risk expression at 1.15x. Recovery (SPY up, VIX 20-30): moderate caution, 1.00x. Caution (all else): mild reduction, 0.90x. Stress (SPY down OR VIX > 30): defensive, 0.70x. The discrete scaling is deliberately simple to avoid overfitting — each state has a clear economic justification rather than a data-mined threshold.

**Expected behavior:**
- 2000-2002 tech crash: SPY 252d goes negative and VIX spikes; Stress state for majority of the crash; 0.70x scale substantially reduces drawdown
- 2008-2009 GFC: deepest Stress regime; VIX > 30 sustained for months; 0.70x is maximum protection; expect materially better MaxDD than H021
- 2020 COVID: brief spike to Stress (March-April 2020), rapid recovery through Recovery, then Growth; scale trajectory mirrors VIX normalization
- 2013-2017 low vol: VIX mostly < 20, SPY up; persistent Growth state; 1.15x adds modest return at minimal risk

**Success criteria:** OOS Sharpe >= 0.95; MaxDD < -20%; CI lower > 0.
**Falsification:** Stress regime 0.70x captures the same drawdown reduction as H086's vol-targeting at the cost of Sharpe (misses rallies during high-VIX recovery). If Sharpe < H021 (1.029) with comparable MaxDD, the discrete state machine adds noise over continuous vol-scaling.
