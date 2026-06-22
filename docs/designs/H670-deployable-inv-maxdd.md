# H670 — DeployableInvMaxDD670

**Signal:** A deployable risk-measure baseline that allocates the pool **inversely to each sleeve's standalone
realized maximum drawdown**: `w_i ∝ 1 / |MaxDD_i|`, normalized. A path-dependent downside-risk parity — the
drawdown-depth analog of inverse-vol (H653), which uses point-in-time dispersion. The same 1-day-lagged
interior-grid leverage as H650 is then applied (threshold {15,16,17} × window {20}, plain + asymmetric).

**Why it should work / what it tests:** inverse-vol (H653) and ERC (H659) size sleeves by *volatility*, a
symmetric per-day measure that is blind to how losses compound. Two sleeves with identical daily vol can have
very different worst peak-to-trough paths if one's bad days cluster. MaxDD captures that clustering directly,
so inverse-MaxDD parity down-weights the sleeves that have historically dug the deepest holes — the property a
deployable, drawdown-averse book actually cares about — while leaving the smooth low-drawdown sleeves at full
size. It rounds out the Set14/15 risk-measure battery: vol (H653), variance-contribution (H659), CVaR (H656),
MAD (H658), and now drawdown depth as the sizing statistic.

**Expected behavior:** because drawdown depth and volatility are strongly rank-correlated, weights should
resemble inverse-vol (H653) but tilt extra weight toward the smoothest-equity sleeves, plausibly improving
MaxDD/Calmar at a similar Sharpe. Note this measure is mildly return-aware (a high-drift sleeve has shallower
drawdowns), so it is not a pure covariance baseline.

**Success:** deployable Sharpe ≥ the inverse-vol baseline H653 (1.303) and clearing the keep bar, with a valid
simplex weight vector that down-weights the deepest-drawdown sleeves. **Falsify:** Sharpe below inverse-vol,
or no keep → drawdown-depth sizing adds nothing over plain volatility sizing on this pool.

**Result (KEEP).** Config t15w20-asym. Sharpe **1.311** [0.92, 1.70], Calmar 0.879, Sortino 1.400, CAGR 7.06%,
MaxDD −8.03%, H001 corr +0.686, folds 0.51–2.05. Sizing by inverse standalone-MaxDD essentially **ties naive
inverse-vol** (H653, 1.303) at +0.008 and lands below ERC (H659, 1.414) by −0.103 — confirming the design's
prior that drawdown depth and daily volatility are strongly rank-correlated, so the drawdown-depth statistic
carries almost no information beyond vol on this pool. The asymmetric leverage config wins here (the only
batch member where asym beats plain), but the **allocation** itself adds nothing over plain inverse-vol. Kept
as the risk-measure baseline that closes the Set14/15 sizing-statistic battery: vol ≈ MaxDD < CVaR/MAD <
variance-contribution (ERC) < return-aware (H650).
