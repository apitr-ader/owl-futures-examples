"""Stock-futures daily K bars rebuilt from the TAIFEX tick archive (tools/tickdata.py layout).

build_contracts(): day-session, non-spread ticks -> per product per (trade_date, contract) OHLCV
    -> OUT/contracts/<product>.parquet  (volume = RPT B+S / 2, matches Fubon convention)
expiry_calendar(): last trade date of each monthly contract, taken from TX (same 3rd-Wednesday
    expiry as stock futures, holiday shifts included).
build_continuous(): front-month series (front = nearest monthly contract whose last trade date
    >= day; rolls the day after expiry, like the Fubon 1! series), raw + additive back-adjusted
    (gap = new contract open - previous front close, same as data/stock_futures/daily/*_adj).
    Close = last trade (settlement price is not in the RPT files).
    Known archive gap: 2010-04-12..14,16,19..22 and 2010-10-11..13 have no tick files (10 trading days).
Loaders: load_contracts(p), load_front(p, adjusted=False), products().
"""
import os, re, sys, glob
import numpy as np, pandas as pd, pyarrow.parquet as pq

HERE = os.path.dirname(os.path.abspath(__file__))
TICKS = os.path.join(HERE, '..', '..', 'data', 'futures_ticks', 'ticks')
OUT = os.path.join(HERE, '..', '..', 'data', 'stock_futures_daily')
INDEX_LIKE = {'TX', 'MTX', 'TE', 'TF', 'TMF', 'TXF', 'MXF', 'EXF', 'FXF'}
COLS = ['ts', 'trade_date', 'contract', 'price', 'volume', 'is_spread', 'session']


def candidates():
    ps = sorted(d[8:] for d in os.listdir(TICKS) if d.startswith('product='))
    return [p for p in ps if re.fullmatch(r'[A-Z0-9]{3}', p) and p not in INDEX_LIKE]


def daily_from_ticks(product):
    out = []
    for f in sorted(glob.glob(os.path.join(TICKS, f'product={product}', 'month=*.parquet'))):
        t = pq.read_table(f, columns=COLS).to_pandas()
        t = t[(t.session == 'day') & ~t.is_spread & t.contract.str.fullmatch(r'\d{6}')]
        if t.empty:
            continue
        g = t.groupby(['trade_date', 'contract'], sort=True)
        out.append(pd.DataFrame({'open': g.price.first(), 'high': g.price.max(), 'low': g.price.min(),
                                 'close': g.price.last(), 'volume': g.volume.sum() // 2,
                                 'ticks': g.price.size()}).reset_index())
    if not out:
        return pd.DataFrame()
    d = pd.concat(out, ignore_index=True)
    return d.groupby(['trade_date', 'contract'], as_index=False).agg(
        open=('open', 'first'), high=('high', 'max'), low=('low', 'min'), close=('close', 'last'),
        volume=('volume', 'sum'), ticks=('ticks', 'sum'))


def build_contracts(products=None):
    os.makedirs(os.path.join(OUT, 'contracts'), exist_ok=True)
    for i, p in enumerate(products or candidates() + ['TX']):
        d = daily_from_ticks(p)
        if len(d):
            d.to_parquet(os.path.join(OUT, 'contracts', f'{p}.parquet'), compression='zstd', index=False)
        if i % 50 == 0:
            print(i, p, len(d), flush=True)


def expiry_calendar():
    tx = load_contracts('TX')
    e = tx.groupby('contract').trade_date.max().sort_index()
    days = np.sort(tx.trade_date.unique())
    # archive gap (2010-04-12..22, 2010-10-11..13 missing): if the 3rd Wednesday falls inside a gap, use it
    for c, last in e.items():
        w = pd.Timestamp(int(c[:4]), int(c[4:]), 1)
        w = w + pd.Timedelta(days=(2 - w.weekday()) % 7 + 14)
        i = np.searchsorted(days, np.datetime64(last), side='right')
        if last < w and i < len(days) and days[i] > np.datetime64(w):
            e[c] = w
    return e


def build_continuous(d, exp):
    """d: per-contract daily bars of one product. Returns front-month raw frame with adj column."""
    if d.empty:
        return d
    d = d[d.contract.isin(exp.index)].copy()
    d['last'] = d.contract.map(exp)
    d = d[d.last >= d.trade_date]
    # front = nearest month still alive on that day (whether or not it traded)
    days = np.sort(d.trade_date.unique())
    em = exp
    ce = np.array(em.values, dtype='datetime64[ns]'); cm = np.array(em.index)
    order = np.argsort(ce, kind='stable'); ce, cm = ce[order], cm[order]
    front = cm[np.searchsorted(ce, days.astype('datetime64[ns]'))]
    f = pd.DataFrame({'trade_date': days, 'contract': front})
    r = f.merge(d, on=['trade_date', 'contract']).sort_values('trade_date').reset_index(drop=True)
    r = r.rename(columns={'contract': 'contract_month'})
    prev = r.contract_month.shift()
    r['is_roll'] = r.contract_month.ne(prev) & prev.notna()
    r['change'] = r.close.diff().where(~r.is_roll, r.close - r.open)
    gap = (r.open - r.close.shift()).where(r.is_roll, 0.0).fillna(0.0)
    r['adj'] = gap[::-1].cumsum()[::-1].shift(-1, fill_value=0.0)
    r['time'] = pd.to_datetime(r.trade_date).dt.tz_localize('Asia/Taipei')
    r['session'] = 'regular'
    return r[['time', 'open', 'high', 'low', 'close', 'volume', 'change', 'contract_month', 'session', 'is_roll', 'adj']]


def load_contracts(p):
    return pd.read_parquet(os.path.join(OUT, 'contracts', f'{p}.parquet'))


def load_front(p, adjusted=False):
    d = pd.read_parquet(os.path.join(OUT, 'daily', f'{p}_{"adj" if adjusted else "raw"}.parquet'))
    return d


def products():
    return pd.read_csv(os.path.join(OUT, 'products.csv'), dtype={'underlying_code': str})


def build_daily(map_csv=None):
    """products_map.csv (symbol, underlying_code, name, underlying_type S/E/A/I, contract_size, mini)
    -> OUT/products.csv + OUT/daily/<sym>_raw.parquet, _adj.parquet in the data/stock_futures/daily layout."""
    m = pd.read_csv(map_csv or os.path.join(OUT, 'products_map.csv'), dtype={'underlying_code': str})
    m = m[m.underlying_type.isin(['S', 'E', 'A'])]
    os.makedirs(os.path.join(OUT, 'daily'), exist_ok=True)
    exp = expiry_calendar()
    for s in m.symbol:
        r = build_continuous(load_contracts(s), exp)
        if len(r):
            r.drop(columns='adj').to_parquet(os.path.join(OUT, 'daily', f'{s}_raw.parquet'), index=False)
            r.to_parquet(os.path.join(OUT, 'daily', f'{s}_adj.parquet'), index=False)
    m.to_csv(os.path.join(OUT, 'products.csv'), index=False)


if __name__ == '__main__':
    if sys.argv[1:2] == ['daily']:
        build_daily()
    if sys.argv[1:2] == ['contracts']:
        build_contracts(sys.argv[2:] or None)
