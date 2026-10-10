# Writing strategies

Erised has three strategy interfaces. Pick the one that matches what your strategy outputs.

| Interface | Implement | Output | Validate with |
|---|---|---|---|
| `Strategy` | `generate_positions(prices)` | `Series` of SPY positions in `[-1, 1]` | `walk_forward` |
| `MultiAssetStrategy` | `generate_weights(prices)` | `DataFrame` of weights, one column per asset | `walk_forward_multi` |
| `CrossSectionalStrategy` | `generate_weights(prices_panel, membership)` | `DataFrame` of weights, one column per stock | `walk_forward_cross_sectional` |

All three follow the timing rule in [Methodology](methodology.md): output the target for
each date using data through that date, and let the engine apply the one-bar lag.

## Single-asset

```python
import pandas as pd
from erised.strategies import Strategy


class Breakout(Strategy):
    def __init__(self, lookback: int = 100):
        self.lookback = lookback
        self.name = f"breakout_{lookback}"

    def generate_positions(self, prices: pd.DataFrame) -> pd.Series:
        px = prices["adj_close"]
        prior_high = px.rolling(self.lookback).max().shift(1)
        return (px > prior_high).astype(float).rename("position")
```

`prices` follows the schema in `erised.data.schema`: a tz-naive `DatetimeIndex` and columns
`open, high, low, close, volume, adj_close`. Use `adj_close` for returns. To feed in other
series, join them as extra columns (the research runners use `tlt_close`, `gld_close`,
`vix_close`, and so on).

## Multi-asset

```python
import pandas as pd
from erised.strategies import MultiAssetStrategy


class SixtyForty(MultiAssetStrategy):
    name = "sixty_forty"

    def generate_weights(self, prices: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame({"SPY": 0.6, "TLT": 0.4}, index=prices.index)
```

`run_multiasset_backtest` maps the `SPY` column to `adj_close` and any other asset name to
`<name>_close` (`TLT` to `tlt_close`, `XLU` to `xlu_close`), so join those price columns onto
`prices` before running. Each row's absolute weights must sum to at most 1; the remainder is
cash.

## Cross-sectional

```python
import pandas as pd
from erised.strategies.cross_sectional import CrossSectionalStrategy


class LowVol(CrossSectionalStrategy):
    name = "low_vol"

    def generate_weights(self, prices_panel: pd.DataFrame, membership: pd.DataFrame) -> pd.DataFrame:
        vol = prices_panel.pct_change().rolling(252).std()
        eligible = vol.where(membership)
        picks = eligible.rank(axis=1, pct=True) <= 0.2
        return picks.div(picks.sum(axis=1), axis=0).fillna(0.0)
```

`prices_panel` is adjusted closes (dates by tickers) and `membership` is a boolean panel of
point-in-time S&P 500 membership. Load them with `erised.data.stocks.load_stocks_panel` and
`load_sp500_members`. Long-only rows should sum to at most 1; market-neutral gross exposure
to at most 2.

## Test it

Every strategy in the repository has at least a lookahead test. Add one for yours:

```python
from erised.backtest.engine import assert_no_lookahead
from erised.data.synthetic import synthetic_bars


def test_breakout_no_lookahead():
    assert_no_lookahead(Breakout().generate_positions, synthetic_bars())
```

For multi-asset strategies use `assert_no_lookahead_multi`. Then add behavioral tests: flat
before enough history, outputs in range, the regime switch fires when it should.

## Evaluate it

Use `examples/02_custom_strategy.py` as a template: lookahead check, walk-forward, bootstrap
confidence interval, then a keep/kill decision you wrote down before running it. The
`research/` archive shows the full workflow the project used: a design doc, a runner, a
`summary.json`, and a leaderboard entry for every hypothesis.

## Contributing a strategy

Library strategies live in `src/erised/strategies/base.py` (single- and multi-asset) and
`src/erised/strategies/cross_sectional.py`. Follow the style of nearby classes, export new
public classes from `erised/strategies/__init__.py` if they're broadly useful, and include
tests. See [CONTRIBUTING.md](../CONTRIBUTING.md).
