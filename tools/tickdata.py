"""Loader for TAIFEX futures tick parquet (built by tools/tick_convert.py).

Layout: ROOT/product=XX/month=YYYY-MM.parquet
Columns: ts (datetime64[ms, Asia/Taipei], calendar time), trade_date (trading day; night
session rows belong to the NEXT trading day), contract (str, e.g. '202610', weekly '202610W2',
spread '202608/202609'), price, volume (int, B+S as in RPT), is_spread, near_price, far_price
(spread legs only), session ('day'|'night'), open_auction (bool, '*' flag in newer files).

    from tools.tickdata import load_ticks, list_products, to_bars
    t = load_ticks('TX', '2024-01-02', '2024-01-05', contract='near')
    bars = to_bars(t, '1min')
"""
import os, glob
import pandas as pd
import pyarrow.parquet as pq

ROOT = os.environ.get('TICK_ROOT', os.path.join(os.path.dirname(os.path.abspath(__file__)),
                      '..', '..', 'data', 'futures_ticks', 'ticks'))


def list_products(root=ROOT):
    return sorted(d.split('=', 1)[1] for d in os.listdir(root) if d.startswith('product='))


def list_months(product, root=ROOT):
    return sorted(os.path.basename(f)[6:13] for f in glob.glob(os.path.join(root, f'product={product}', 'month=*.parquet')))


def load_ticks(product, start, end, contract=None, include_spread=False, session=None, root=ROOT):
    """Ticks for trade_date in [start, end] (inclusive).
    contract: None = all; 'YYYYMM' or exact code; 'near' = per trade_date the monthly
    (non-weekly) outright contract with the most volume (simple continuous series).
    Note: volume is RPT 'B+S' (both sides counted); divide by 2 for contracts traded."""
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    # night rows of trade_date `start` are in the previous calendar month possibly
    m0 = (start - pd.Timedelta(days=10)).strftime('%Y-%m'); m1 = end.strftime('%Y-%m')
    files = [os.path.join(root, f'product={product}', f'month={m}.parquet')
             for m in list_months(product, root) if m0 <= m <= m1]
    if not files:
        return pd.DataFrame()
    df = pd.concat([pq.read_table(f).to_pandas() for f in files], ignore_index=True)
    df = df[(df.trade_date >= start) & (df.trade_date <= end)]
    if not include_spread:
        df = df[~df.is_spread]
    if session:
        df = df[df.session == session]
    if contract == 'near':
        mo = df[df.contract.str.fullmatch(r'\d{6}')]
        vol = mo.groupby(['trade_date', 'contract']).volume.sum().reset_index()
        pick = vol.sort_values(['trade_date', 'volume'], ascending=[True, False]).drop_duplicates('trade_date')
        df = df.merge(pick[['trade_date', 'contract']], on=['trade_date', 'contract'])
    elif contract:
        df = df[df.contract == str(contract)]
    return df.sort_values('ts', kind='stable').reset_index(drop=True)


def to_bars(ticks, freq='1min', label='left'):
    """OHLCV bars from ticks (single contract recommended). Bar time = interval start (left)."""
    if ticks.empty:
        return pd.DataFrame()
    t = ticks.set_index('ts')
    g = t.groupby([pd.Grouper(freq=freq, label=label, closed='left'), 'trade_date', 'session'], observed=True)
    b = g.agg(open=('price', 'first'), high=('price', 'max'), low=('price', 'min'),
              close=('price', 'last'), volume=('volume', 'sum'), ticks=('price', 'size'))
    return b.reset_index().rename(columns={'ts': 'time'})
