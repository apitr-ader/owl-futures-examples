"""Loader for TAIEX option (TXO) daily bars built by tools/opt_convert.py from TAIFEX RPT ticks.

Data lives outside the repo (TAIFEX licensing): data/options_daily/daily/year=YYYY.parquet.

    from tools.optdaily import load_txo_daily, expiries, near_expiry, chain
    d = load_txo_daily('2024-01-01', '2024-03-31', expiry='monthly', cp='C')
    e = near_expiry('2024-08-05')            # nearest series (weekly or monthly) still trading
    c = chain('2024-08-05', e)               # that day's call/put chain for one expiry

expiry strings: 'YYYYMM' (monthly, 3rd Wednesday), 'YYYYMMWn' (Wednesday weekly of week n),
'YYYYMMFn' (Friday weekly, from 2025). volume = contracts (RPT both-sides count / 2).
"""
import calendar, datetime as dt, glob, os, re
from functools import lru_cache
import pandas as pd
import pyarrow.dataset as ds

ROOT = os.environ.get('OPT_DAILY_DIR', '/workspace/fubon-threads/data/options_daily')
ERE = re.compile(r'^(\d{4})(\d{2})(?:([WF])(\d))?$')


def _files(start, end):
    fs = sorted(glob.glob(os.path.join(ROOT, 'daily', 'year=*.parquet')))
    y0 = pd.Timestamp(start).year if start else 0
    y1 = pd.Timestamp(end).year if end else 9999
    return [f for f in fs if y0 <= int(re.search(r'year=(\d{4})', f).group(1)) <= y1]


def kind(expiry):
    m = ERE.match(expiry)
    return 'monthly' if m and not m.group(3) else ('weekly' if m else 'other')


def load_txo_daily(start=None, end=None, expiry=None, cp=None, strike=None, columns=None):
    """expiry: exact string, list, 'monthly' or 'weekly'; cp: 'C'/'P'; strike: number or (lo, hi)."""
    fs = _files(start, end)
    if not fs:
        return pd.DataFrame()
    f = None
    def add(x):
        nonlocal f
        f = x if f is None else f & x
    if start: add(ds.field('date') >= pd.Timestamp(start))
    if end: add(ds.field('date') <= pd.Timestamp(end))
    if cp: add(ds.field('cp') == cp)
    if strike is not None:
        if isinstance(strike, (tuple, list)):
            add(ds.field('strike') >= float(strike[0])); add(ds.field('strike') <= float(strike[1]))
        else:
            add(ds.field('strike') == float(strike))
    if isinstance(expiry, str) and expiry not in ('monthly', 'weekly'):
        add(ds.field('expiry') == expiry)
    elif isinstance(expiry, (list, tuple)):
        add(ds.field('expiry').isin(list(expiry)))
    d = ds.dataset(fs, format='parquet').to_table(filter=f, columns=columns).to_pandas()
    if expiry in ('monthly', 'weekly'):
        d = d[d.expiry.map(kind) == expiry]
    if 'cp' in d:
        d['cp'] = d.cp.astype(str)
    return d.reset_index(drop=True)


def _nominal(expiry):
    """Nominal expiry date from the code (holidays not applied)."""
    m = ERE.match(expiry)
    y, mo = int(m.group(1)), int(m.group(2))
    wd, n = (calendar.WEDNESDAY, 3) if not m.group(3) else \
        ((calendar.WEDNESDAY if m.group(3) == 'W' else calendar.FRIDAY), int(m.group(4)))
    days = [d for d in range(1, calendar.monthrange(y, mo)[1] + 1) if dt.date(y, mo, d).weekday() == wd]
    return pd.Timestamp(y, mo, days[min(n, len(days)) - 1])


@lru_cache(maxsize=1)
def expiries():
    """Table of series: expiry, kind, first_date, last_date (last day traded), expiry_date
    (= last traded date once expired, else nominal date from the code)."""
    d = load_txo_daily(columns=['date', 'expiry'])
    g = d.groupby('expiry').date.agg(first_date='min', last_date='max').reset_index()
    g = g[g.expiry.str.match(ERE)]
    end = d.date.max()
    g['kind'] = g.expiry.map(kind)
    g['nominal'] = g.expiry.map(_nominal)
    g['expiry_date'] = g.last_date.where((g.last_date < end) | (g.nominal <= end), g.nominal)
    return g.sort_values('expiry_date').reset_index(drop=True)


def near_expiry(date, which='any', n=0):
    """n-th nearest series not yet expired on `date` (date <= expiry_date). which: any/monthly/weekly."""
    e = expiries()
    t = pd.Timestamp(date)
    e = e[(e.first_date <= t) & (e.expiry_date >= t)]
    if which != 'any':
        e = e[e.kind == which]
    return e.expiry.iloc[n] if len(e) > n else None


def chain(date, expiry):
    return load_txo_daily(date, date, expiry=expiry).sort_values(['cp', 'strike']).reset_index(drop=True)
