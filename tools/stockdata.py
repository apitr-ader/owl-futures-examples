"""Loader for the Taiwan common-stock dataset in data/stocks (built by tools/fetch_stocks.py).

    import sys; sys.path.insert(0, "/workspace/fubon-threads/tools"); import stockdata as sd
    sd.load_raw("2330")             # raw OHLCV/turnover (unadjusted), DatetimeIndex
    sd.load_adjusted("2330")        # total-return adjusted OHLC, factor, ret (dividends reinvested)
    sd.load_corporate_actions()     # one row per event (optionally codes=[...], types=[...])
    sd.load_adj_factors("2330")     # event factors (date, event_factor, cum_factor_before)
    sd.universe(as_of="2015-06-30") # stocks trading on that date (incl. later-delisted ones)
    sd.panel("close", adjusted=True, codes=None, start="2010-01-01")  # wide date x code frame
"""
from functools import lru_cache
from pathlib import Path
import pandas as pd

ROOT = Path("/workspace/fubon-threads/data/stocks")


def load_raw(code):
    return pd.read_parquet(ROOT / "raw" / f"{code}.parquet")


def load_adjusted(code):
    return pd.read_parquet(ROOT / "adj" / f"{code}.parquet")


@lru_cache(None)
def _ca():
    return pd.read_parquet(ROOT / "corporate_actions.parquet")


def load_corporate_actions(codes=None, types=None, applied_only=False):
    d = _ca()
    if codes is not None:
        d = d[d.code.isin([str(c) for c in codes])]
    if types is not None:
        d = d[d.type.str.contains("|".join(types))]
    if applied_only:
        d = d[d.applied]
    return d.copy()


def load_adj_factors(code=None):
    d = pd.read_parquet(ROOT / "adj_factors.parquet")
    return d if code is None else d[d.code == str(code)].reset_index(drop=True)


@lru_cache(None)
def _universe():
    return pd.read_csv(ROOT / "universe.csv", dtype={"code": str}, parse_dates=["first_date", "last_date", "delist_date"])


def universe(as_of=None, market=None, include_delisted=True):
    """Metadata table. as_of: keep stocks whose first_date <= as_of <= last_date (point-in-time,
    survivorship-free within the data period)."""
    u = _universe()
    if as_of is not None:
        t = pd.Timestamp(as_of)
        u = u[(u.first_date <= t) & (u.last_date >= t)]
    if market:
        u = u[u.market == market]
    if not include_delisted:
        u = u[u.status == "listed"]
    return u.reset_index(drop=True)


def panel(field="close", adjusted=True, codes=None, start=None, end=None):
    codes = list(universe().code) if codes is None else [str(c) for c in codes]
    f = load_adjusted if adjusted else load_raw
    out = {}
    for c in codes:
        try:
            out[c] = f(c)[field]
        except FileNotFoundError:
            pass
    return pd.DataFrame(out).sort_index().loc[start:end]
