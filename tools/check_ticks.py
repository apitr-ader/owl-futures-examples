"""Checks for converted TAIFEX tick parquet.
1) row counts vs RPT data lines (from _convert_stats.jsonl, and parquet totals per month)
2) sample-day sanity (TX: times within sessions, price range, no negative volume)
3) TX 1m bars from ticks vs Fubon 1m bars (data/TXF_1c_1m_both_*.parquet) for a few days
Usage: python tools/check_ticks.py [day1 day2 ...]
"""
import os, sys, json, glob
import numpy as np, pandas as pd, pyarrow.parquet as pq
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools.tickdata import ROOT, load_ticks, to_bars, list_products

BASE = os.path.join(os.path.dirname(ROOT))


def rowcounts():
    st = [json.loads(l) for l in open(os.path.join(ROOT, '_convert_stats.jsonl'))]
    s = pd.DataFrame(st).drop_duplicates('file_date', keep='last')
    s['month'] = s.file_date.str[:7]
    per = s.groupby('month')[['lines', 'rows', 'bad']].sum()
    pqn = {}
    for f in glob.glob(os.path.join(ROOT, 'product=*', 'month=*.parquet')):
        m = os.path.basename(f)[6:13]
        pqn[m] = pqn.get(m, 0) + pq.ParquetFile(f).metadata.num_rows
    per['parquet_rows'] = pd.Series(pqn)
    per['ok'] = (per.lines == per.parquet_rows + per.bad) & (per.rows == per.parquet_rows)
    print('days', len(s), s.file_date.min(), s.file_date.max(), 'RPT lines', per.lines.sum(),
          'parquet rows', per.parquet_rows.sum(), 'bad', per.bad.sum(), 'months not ok', (~per.ok).sum())
    if (~per.ok).any():
        print(per[~per.ok])
    return s


def sanity(day):
    t = load_ticks('TX', day, day, include_spread=True)
    if t.empty:
        print(day, 'no TX'); return
    h = t.ts.dt.hour * 100 + t.ts.dt.minute
    d = t[t.session == 'day']; n = t[t.session == 'night']
    print(day, 'TX rows', len(t), 'day', len(d), 'night', len(n), 'spread', int(t.is_spread.sum()),
          'day time', d.ts.min().time() if len(d) else '-', d.ts.max().time() if len(d) else '-',
          'night', n.ts.min() if len(n) else '-', n.ts.max() if len(n) else '-',
          'px', t.price.min(), t.price.max(), 'vol<=0', int((t.volume <= 0).sum()),
          'contracts', sorted(t.contract.unique())[:6])


def vs_fubon(days):
    f = glob.glob(os.path.join(BASE, '..', 'TXF_1c_1m_both_*.parquet'))[0]
    fb = pd.read_parquet(f)
    fb = fb.rename(columns={'date': 'time'})
    for day in days:
        t = load_ticks('TX', day, day)
        if t.empty:
            print(day, 'no ticks'); continue
        res = []
        for lab, sh in (('start', 0), ('end', 1)):
            out = []
            for c, g in t.groupby('contract'):
                bb = to_bars(g, '1min'); bb['contract'] = c; out.append(bb)
            b = pd.concat(out)
            b['time'] = b.time + pd.Timedelta(minutes=sh)
            td = pd.Timestamp(day, tz='Asia/Taipei')
            fd = fb[(fb.time >= td - pd.Timedelta(hours=10)) & (fb.time < td + pd.Timedelta(hours=14))]
            fd2 = fd.assign(contract=fd.contractMonth.astype(str).str.replace('-', ''))
            m = fd2.merge(b, on=['time', 'contract'], suffixes=('_f', '_t'))
            if m.empty:
                res.append((lab, 0)); continue
            ex = lambda a, b_: float(np.mean(np.isclose(m[a], m[b_])))
            res.append((lab, len(m), len(fd2), ex('close_f', 'close_t'), ex('open_f', 'open_t'),
                        ex('high_f', 'high_t'), ex('low_f', 'low_t'),
                        float(np.mean(m.volume_f == m.volume_t // 2)), float(np.mean(m.volume_f == m.volume_t))))
        print(day, 'fubon contracts', sorted(fd2.contract.unique()), 'tick contracts', sorted(t.contract.unique())[:4])
        for r in res:
            print('  label', r[0], 'matched', r[1:3], 'close/open/high/low match', [round(x, 3) for x in r[3:7]] if len(r) > 2 else '',
                  'vol==B+S/2', round(r[7], 3) if len(r) > 7 else '', 'vol==B+S', round(r[8], 3) if len(r) > 8 else '')


def night_coverage():
    """TX trade days since night session launch (2017-05-15) with no night rows, by year/weekday."""
    from tools.tickdata import list_months
    out = []
    for m in list_months('TX'):
        if m < '2017-05':
            continue
        d = pq.read_table(os.path.join(ROOT, 'product=TX', f'month={m}.parquet'), columns=['trade_date', 'session']).to_pandas()
        out.append(d.groupby('trade_date').session.apply(lambda s: (s == 'night').sum()))
    s = pd.concat(out); s = s[s.index >= '2017-05-16']
    miss = s[s == 0]
    print('TX night-session coverage: trade days', len(s), 'without night rows', len(miss))
    print(pd.crosstab(miss.index.year, miss.index.day_name()))
    return miss


if __name__ == '__main__':
    days = sys.argv[1:] or ['2017-06-05', '2020-03-19', '2024-08-05', '2026-08-18']
    print('products', len(list_products()))
    rowcounts()
    for d in days:
        sanity(d)
    vs_fubon(days)
    night_coverage()
