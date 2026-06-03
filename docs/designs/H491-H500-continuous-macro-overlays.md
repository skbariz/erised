# H491–H500 Design Docs — Set 7 Batch 5: Continuous Macro Overlays & Regime Strategies

**Date drafted:** 2026-06-03  
**Batch thesis:** H301–H350 Set 4 found that continuous regime scalers beat binary flags. All prior
AA implementations used binary thresholds. Batch 5 tests fully continuous quantile-based scaling
and novel regime combinations not previously tested, plus regime-adaptive CS signal selection.

---

## H491 — ContinuousYieldCurveScaler

**Signal:** TLT-SHY trailing 252d return spread, z-scored over rolling 252d window. H021 SPY weight
scaled linearly in [0.6, 1.4]: high z (steep yield curve) → equity tilt; low z (flat/inverted) →
defensive. Continuous analog of H311's binary yield-curve signal (Sharpe 0.608).

**Why novel:** H311 used a binary TLT>SHY return flag (steep=1, inverted=0). H491 converts the
same signal to a continuous z-score that avoids threshold-crossing whipsaw. The smooth scale
eliminates the daily flipping that degraded H311's Sharpe.

**Expected behavior:** 2000-2002: yield curve flattening/inversion → defensive. 2008-2009: curve
steepened sharply as Fed cut rates → equity tilt at exactly the wrong time — this is the key risk.
2020 COVID: brief inversion then steep recovery → signal should rebound.

**Success criteria:** Sharpe > H311 (0.608). MaxDD improvement over H021.
**Falsified if:** Continuous scaling doesn't improve over binary H311.

**Results:** TBD

---

## H492 — VIXPercentileScalerPort

**Signal:** VIX 252d trailing percentile rank (0=lowest VIX in past year, 1=highest). SPY scale =
1.0 + (0.5 - pct) × 0.8 → ranges [0.6 at max VIX, 1.4 at min VIX]. Continuous vs H066's
binary VIX>35 spike override and H070's vol-target approach.

**Why novel:** H066 (Sharpe 0.990) used binary VIX>35 crisis gate. H070 (Sharpe 1.066) used
portfolio vol-target. H492 uses VIX percentile as a continuous scaler — always active, smoothly
reducing exposure as VIX rises. No threshold required; no lag from hysteresis.

**Expected behavior:** Low VIX environments (2013-2017 calm) → systematic equity boost increases
CAGR. VIX spikes (2020, 2008-2009) → smooth reduction avoids worst drawdowns. Better than binary
because it starts scaling down before VIX crosses a threshold.

**Success criteria:** Sharpe > H066 (0.990) or H070 (1.066). MaxDD < H021 (−18.41%).
**Falsified if:** Continuous VIX scaling doesn't improve over binary VIX threshold.

**Results:** TBD

---

## H493 — CreditSpreadContinuousPort

**Signal:** HYG 252d return minus LQD 252d return, z-scored over rolling 252d window. Positive z
(HYG outperforms = tight spreads) → equity tilt. Negative z (wide spreads) → defensive. Scale
in [0.6, 1.4] on H021 SPY weight. HYG from 2007; early folds fall back to H021 (scale=1.0).

**Why novel:** H308/H312 used binary credit spread gates (kill). H493 is the first continuous
credit spread version. The 252d return spread is a leading indicator for credit conditions —
HYG underperforming LQD signals rising default risk before it hits equity markets.

**Expected behavior:** 2008-2009: HYG collapsed vs LQD → deeply negative z → defensive at the
right time. 2020 COVID: brief credit stress then recovery. Challenge: HYG only from 2007 means
folds 0-5 (1993-2010) have no credit signal — significant data gap risk.

**Success criteria:** Sharpe > H308 (binary credit: 0.684). MaxDD < H021.
**Falsified if:** HYG data gap causes folds 0-5 to drag performance below keep bar.

**Results:** TBD

---

## H494 — MultiplexMacroComposite

**Signal:** Average of 3 continuous scalers: (1) VIX percentile [0.6, 1.4], (2) yield curve z-score
[0.6, 1.4], (3) credit spread z-score [0.6, 1.4]. Average applied to H021 SPY weight. Credit
falls back to 1.0 when HYG/LQD unavailable (pre-2007).

**Why novel:** H314 (binary+binary: 0.641) and H316 (QuarterlyMacroStateOverlay: 0.722) used
binary composite macro signals. H494 makes ALL inputs continuous, eliminates threshold whipsaw,
and averages rather than gating. The averaging smooths individual signal noise.

**Expected behavior:** When all 3 signals agree (e.g., 2008-2009: high VIX + inverted curve +
wide spreads) → strongly defensive. When signals disagree → moderate response. In pre-HYG period,
effectively a 2-signal composite (VIX + yield curve).

**Success criteria:** Sharpe > H316 QuarterlyMacroStateOverlay (0.722).
**Falsified if:** Composite blending dilutes each signal's individual edge.

**Results:** TBD

---

## H495 — GLDMomentumOverlay

**Signal:** SPY/TLT/GLD 3-asset inverse-vol parity base (H076 style). GLD 6m return > 0 → GLD
weight × 1.2, renormalize. GLD 6m return ≤ 0 → GLD weight × 0.5, floor at 10% of portfolio.
Tests whether GLD trend signal improves H076 (Sharpe 0.666, MaxDD −43.71%).

**Why novel:** H489 (ThreeAssetMomentumBridge) applied a 3-asset momentum multiplier to all
assets simultaneously. H495 isolates GLD momentum as the sole overlay variable, leaving SPY and
TLT at their parity weights. Tests GLD's independent trend contribution.

**Expected behavior:** 2008-2009: GLD surged → overlay increases GLD weight = extra diversification
benefit. 2013-2015: GLD bear market → reduces GLD to floor, preventing drag. 2020: GLD momentum
strong → boosts returns during equity stress.

**Success criteria:** Sharpe > H076 (0.666). MaxDD < H076 (−43.71%).
**Falsified if:** GLD trend signal adds noise over static 3-asset parity weights.

**Results:** TBD

---

## H496 — VIXMeanReversionPort

**Signal:** VIX / VIX_252d_MA ratio. Ratio ≥ 1.5 (extreme fear) → SPY weight × 1.2 (contrarian
buy). Ratio < 0.7 (extreme calm) → SPY × 1.0 (normal H021). Linear interpolation between.
Opposite of H066/H492 which penalize high VIX.

**Why novel:** All prior VIX overlays (H066, H070, H492) penalize high VIX → reduce equity.
H496 is CONTRARIAN: extreme VIX spikes are historically followed by mean reversion → opportunity.
Tests whether the VIX premium is better harvested by leaning IN to fear vs leaning out.

**Expected behavior:** 2008-2009: VIX reached 80 (5× typical) → strong contrarian buy signal.
In hindsight, buying equity during VIX>50 periods has been historically profitable at 1-year horizon.
Risk: VIX can stay elevated for months (2008-2009 lasted 18 months).

**Success criteria:** Sharpe > 0.51. MaxDD < H021 (benefit from mean reversion timing).
**Falsified if:** Contrarian VIX overlay increases MaxDD beyond H021 (sustained high VIX = sustained drawdown).

**Results:** TBD

---

## H497 — RegimeAdaptiveCSSignal

**Signal:** VIX-conditional CS signal selection. VIX < 20 → 6m price momentum (top 20%,
sector-neutral). VIX ≥ 20 → GP/TA quality (top 20%, sector-neutral). Monthly rebalance.

**Why novel:** All prior regime-adaptive strategies (H316, H318, H342) applied regime conditioning
to AA weights. H497 tests regime conditioning applied to CS SIGNAL SELECTION. Academic hypothesis:
momentum works better in calm regimes (Barroso & Santa-Clara 2015 show momentum crashes in
high-vol); quality outperforms in stressed regimes (Asness 2014 quality-at-a-reasonable-price).

**Expected behavior:** In calm markets (low VIX, 2013-2017): momentum selected → captures trend
alpha. In stress (2008-2009, 2020): quality selected → avoids momentum crash, benefits from
quality flight-to-safety. Key weakness: GP/TA has EDGAR gap in folds 0-5; quality mode in early
folds will have sparse coverage.

**Success criteria:** Sharpe > max(H150 momentum 0.981, H227 quality 0.879) in respective regimes.
**Falsified if:** Signal-switching creates whipsaw at VIX=20 boundary (worse than either standalone).

**Results:** TBD

---

## H498 — CoreDiversifierSleeve

**Signal:** Equal blend of 3 lowest H001-corr actively-traded strategies: 33% H078
(SeasonalRiskParity, H001 corr +0.008), 33% H086 (LowVolTargetPortfolio, H001 corr +0.413),
33% H473 (LongShortPCAReversal, H001 corr +0.029).

**Why novel:** H094 (H049+H078: 1.084) showed adding H078's zero-corr seasonal to an ensemble
sharply reduces MaxDD. H498 extends this by blending THREE program-defined low-corr strategies:
seasonal timing (H078), vol-control (H086), and market-neutral equity alpha (H473). The 3 strategies
have very different return generation mechanisms with minimal pairwise correlation.

**Expected behavior:** H078 earns only Nov-Apr, is flat May-Oct. H086 de-risks in high-vol
environments. H473 earns weekly L/S reversal alpha regardless of market direction. Blend should
have MaxDD much lower than any component — May-Oct H078 flat + H086 de-risked + H473 earning.

**Success criteria:** Sharpe > min(H078 ~0.65, H086 1.109, H473 0.964). MaxDD < worst component.
**Falsified if:** Blend Sharpe < standalone H086 (the dominant component).

**Results:** TBD

---

## H499 — VolAccelerationCSSwitch

**Signal:** SPY 21d realized vol > SPY 63d realized vol × 1.2 (vol accelerating) → GP/TA quality
signal. Otherwise (vol normal/decelerating) → 6m price momentum signal. Sector-neutral, monthly.

**Why novel:** H327 (VolAccelerationDetector: Sharpe 0.726) applied vol acceleration to AA portfolio
EXPOSURE (reduce position size). H499 applies the same vol-acceleration concept to CS SIGNAL
SELECTION (switch between momentum and quality signals). Tests whether CS signal selection benefits
from the same acceleration indicator that improved AA strategies.

**Expected behavior:** 2008-2009 vol acceleration: switch to quality → avoids momentum crash.
2017-2018 vol collapse: use momentum → captures trend alpha. Expected to be similar to H497 but
using realized price vol instead of VIX level.

**Success criteria:** Sharpe > H150 (momentum: 0.981). MaxDD < H150 (−52.12%).
**Falsified if:** Vol acceleration switching creates more whipsaw than VIX-based switching.

**Results:** TBD

---

## H500 — AdaptiveFrequencyQualityPort

**Signal:** GP/TA quality, sector-neutral top 20%. Rebalance MONTHLY (21 days) when VIX ≥ 25
(high vol = signal becoming stale faster). Rebalance QUARTERLY (63 days) when VIX < 25.

**Why novel:** H292 (quarterly GP/TA: Sharpe 0.884) > H227 (monthly GP/TA: 0.852). Quarterly
beats monthly for GP/TA in normal environments. H500 tests whether HIGH-VOL regimes require
faster refresh — in stressed markets, company fundamentals can deteriorate rapidly, making
quarterly stale. The adaptive frequency should improve on both fixed-frequency alternatives.

**Expected behavior:** Normal markets (VIX < 25, ~75% of time): quarterly rebalance → lower
turnover → higher net returns. Stress periods (VIX ≥ 25): monthly refresh → captures quality
deterioration faster. Expected Sharpe between H227 (0.852) and H292 (0.884) or slightly above.

**Success criteria:** Sharpe > H292 (quarterly GP/TA: 0.884). MaxDD ≤ H292.
**Falsified if:** Adaptive frequency produces neither the cost savings of quarterly nor the timeliness
of monthly — Sharpe below both fixed-frequency alternatives.

**Results:** TBD
