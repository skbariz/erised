"""Command-line interface: ``erised <command>``.

erised data fetch [SYMBOL ...]       download ETF/index bars (default: full universe)
erised data seal [--force]           split research/holdout files and record hashes
erised data synthetic                write synthetic bars for every universe symbol
erised data sp500                    reconstruct point-in-time S&P 500 membership
erised data stocks                   download constituent prices
erised data fundamentals             download SEC EDGAR fundamentals
erised strategies [--cross-sectional]
erised backtest STRATEGY [--assets TLT,GLD] [--param name=value ...]
"""

from __future__ import annotations

import argparse
import ast
import inspect
import logging
import sys
from importlib.metadata import PackageNotFoundError, version

import pandas as pd


def _version() -> str:
    try:
        return version("erised")
    except PackageNotFoundError:
        return "unknown"


def _data(args: argparse.Namespace) -> int:
    if args.data_command == "fetch":
        from .data.sources.prices import download_universe

        for r in download_universe(args.symbols or None):
            print(f"{r['symbol']}: {r['rows']} rows {r['start']}..{r['end']}")
    elif args.data_command == "seal":
        from .data.holdout import seal_holdout

        for r in seal_holdout(args.symbols or None, force=args.force):
            print(f"{r['symbol']}: {r['status']}")
    elif args.data_command == "synthetic":
        from .data.synthetic import write_synthetic_universe

        paths = write_synthetic_universe()
        print(f"Wrote {len(paths)} synthetic files. Do not use them to evaluate strategies.")
    elif args.data_command == "sp500":
        from .data.sources.sp500 import download_sp500_membership

        members = download_sp500_membership()
        print(f"Membership: {members.shape[0]} days x {members.shape[1]} tickers")
    elif args.data_command == "stocks":
        from .data.sources.prices import download_constituent_prices

        print(download_constituent_prices())
    elif args.data_command == "fundamentals":
        from .data.sources.edgar import download_fundamentals

        print(download_fundamentals(user_agent=args.user_agent))
    return 0


def _strategy_classes(cross_sectional: bool) -> dict[str, type]:
    if cross_sectional:
        from .strategies import cross_sectional as module
        from .strategies.cross_sectional import CrossSectionalStrategy

        bases: tuple[type, ...] = (CrossSectionalStrategy,)
    else:
        from . import strategies as module
        from .strategies import MultiAssetStrategy, Strategy

        bases: tuple[type, ...] = (Strategy, MultiAssetStrategy)
    return {
        name: obj
        for name in module.__all__
        if inspect.isclass(obj := getattr(module, name))
        and issubclass(obj, bases)
        and not inspect.isabstract(obj)
    }


def _strategies(args: argparse.Namespace) -> int:
    for name, cls in sorted(_strategy_classes(args.cross_sectional).items()):
        summary = (inspect.getdoc(cls) or "").split("\n")[0]
        print(f"{name:<36} {summary}")
    return 0


def _parse_params(pairs: list[str]) -> dict:
    params = {}
    for pair in pairs:
        key, sep, raw = pair.partition("=")
        if not sep:
            raise SystemExit(f"--param expects name=value, got {pair!r}")
        try:
            params[key] = ast.literal_eval(raw)
        except (ValueError, SyntaxError):
            params[key] = raw
    return params


def _backtest(args: argparse.Namespace) -> int:
    from .data.loader import load_research
    from .metrics.performance import bootstrap_sharpe_ci, report
    from .strategies import MultiAssetStrategy
    from .validation.walk_forward import make_folds, walk_forward, walk_forward_multi

    classes = _strategy_classes(cross_sectional=False)
    if args.strategy not in classes:
        raise SystemExit(
            f"Unknown strategy {args.strategy!r}. Run `erised strategies` to list them."
        )
    cls = classes[args.strategy]
    params = _parse_params(args.param)

    prices = load_research("SPY")
    for asset in filter(None, (a.strip().upper() for a in args.assets.split(","))):
        col = f"{asset.lower()}_close"
        extra = load_research(asset)[["adj_close"]].rename(columns={"adj_close": col})
        prices = prices.join(extra, how="left")
        prices[col] = prices[col].ffill()

    folds = make_folds(prices, train_years=args.train_years, test_years=args.test_years)
    run = walk_forward_multi if issubclass(cls, MultiAssetStrategy) else walk_forward
    try:
        wf = run(
            prices,
            lambda _train: cls(**params),
            folds=folds,
            commission_bps=args.commission_bps,
            slippage_bps=args.slippage_bps,
        )
    except KeyError as e:
        raise SystemExit(
            f"{args.strategy} needs price column {e}. Pass the asset with --assets, "
            "for example --assets TLT,VIX."
        ) from e

    lo, hi = bootstrap_sharpe_ci(wf.combined_returns)
    bench = report(pd.concat([f.bench_returns for f in wf.fold_results]).sort_index())
    print(
        f"{args.strategy}  {len(folds)} folds, {args.train_years:g}y train / {args.test_years:g}y test"
    )
    for fold, result in zip(wf.folds, wf.fold_results, strict=True):
        print(f"  {fold}  Sharpe {report(result.net_returns).sharpe:+.2f}")
    print()
    print(wf.combined_report)
    print(f"\nSharpe 95% CI [{lo:.2f}, {hi:.2f}]")
    print(
        f"SPY buy-and-hold, same windows: Sharpe {bench.sharpe:.2f}, max DD {bench.max_drawdown:.1%}"
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="erised", description="Walk-forward backtesting for daily-bar strategies."
    )
    parser.add_argument("--version", action="version", version=f"erised {_version()}")
    parser.add_argument("-v", "--verbose", action="store_true", help="show debug logging")
    sub = parser.add_subparsers(dest="command", required=True)

    data = sub.add_parser("data", help="download, seal, or synthesize market data")
    data_sub = data.add_subparsers(dest="data_command", required=True)
    fetch = data_sub.add_parser("fetch", help="download ETF and index bars")
    fetch.add_argument("symbols", nargs="*", help="subset of the universe (default: all)")
    seal = data_sub.add_parser("seal", help="split research/holdout files and record hashes")
    seal.add_argument("symbols", nargs="*", help="subset of the universe (default: all)")
    seal.add_argument("--force", action="store_true", help="re-seal existing holdouts")
    data_sub.add_parser("synthetic", help="write synthetic bars for offline testing")
    data_sub.add_parser("sp500", help="reconstruct point-in-time S&P 500 membership")
    data_sub.add_parser("stocks", help="download S&P 500 constituent prices")
    fundamentals = data_sub.add_parser("fundamentals", help="download SEC EDGAR fundamentals")
    fundamentals.add_argument(
        "--user-agent", help="SEC contact string (default: $SEC_EDGAR_USER_AGENT)"
    )
    data.set_defaults(func=_data)

    strategies = sub.add_parser("strategies", help="list available strategies")
    strategies.add_argument(
        "--cross-sectional", action="store_true", help="list stock-selection strategies"
    )
    strategies.set_defaults(func=_strategies)

    bt = sub.add_parser("backtest", help="walk-forward backtest of a strategy on research data")
    bt.add_argument("strategy", help="strategy class name, e.g. MovingAverageCrossover")
    bt.add_argument("--assets", default="", help="extra price columns to join, e.g. TLT,GLD,VIX")
    bt.add_argument(
        "--param",
        action="append",
        default=[],
        metavar="NAME=VALUE",
        help="strategy constructor argument (repeatable)",
    )
    bt.add_argument("--train-years", type=float, default=5.0)
    bt.add_argument("--test-years", type=float, default=2.0)
    bt.add_argument("--commission-bps", type=float, default=0.5)
    bt.add_argument("--slippage-bps", type=float, default=1.0)
    bt.set_defaults(func=_backtest)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    try:
        return args.func(args)
    except (FileNotFoundError, RuntimeError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
