"""Checks for TXO daily bars (data/options_daily): rows per day, calendar gaps vs the futures
tick calendar, and a spot-check vs TAIFEX published daily data (optDataDown), per session.

Usage: python tools/check_opt.py [--per-year 2] [--dates 2024-08-05,...] [--no-taifex]
Writes data/options_daily/_check_days.csv and _check_taifex.csv (outside the repo).
"""
import argparse, io, json, os, sys, time
import numpy as np, pandas as pd, requests
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools.optdaily import load_txo_daily, ROOT

FUT_STATS = '/workspace/fubon-threads/data/futures_ticks/ticks/_convert_stats.jsonl'


def taifex(date):
    d = pd.Timestamp(date).strftime('%Y/%m/%d')
    r = requests.post('https://www.taifex.com.tw/cht/3/optDataDown', timeout=60,
                      data=dict(down_type=1, commodity_id='TXO', commodity_id2='', queryStartDate=d, queryEndDate=d))
    t = pd.read_csv(io.StringIO(r.content.decode('big5', 'replace')), index_col=False)
    t.columns = [c.strip() for c in t.columns]
    if '交易時段' not in t:
        t['交易時段'] = '一般'
    t = pd.DataFrame({'expiry': t['到期月份(週別)'].astype(str).str.strip(), 'strike': t['履約價'].astype(float),
                      'cp': t['買賣權'].map({'買權': 'C', '賣權': 'P'}),
                      'session': np.where(t['交易時段'].astype(str).str.strip().eq('盤後'), 'night', 'day'),
                      **{k: pd.to_numeric(t[c], errors='coerce') for k, c in
                         (('open', '開盤價'), ('high', '最高價'), ('low', '最低價'), ('close', '收盤價'), ('volume', '成交量'))}})
    return t[t.volume > 0]


def compare(date):
    o = load_txo_daily(date, date)
    t = taifex(date)
    res = []
    for s in ('day', 'night'):
        a = o[o[f'{s}_volume'] > 0][['expiry', 'strike', 'cp'] + [f'{s}_{c}' for c in ('open', 'high', 'low', 'close', 'volume')]]
        a.columns = ['expiry', 'strike', 'cp', 'open', 'high', 'low', 'close', 'volume']
        b = t[t.session == s]
        m = a.merge(b, on=['expiry', 'strike', 'cp'], how='outer', suffixes=('', '_tx'), indicator=True)
        both = m[m._merge == 'both']
        res.append(dict(date=date, session=s, ours=len(a), taifex=len(b), matched=len(both),
                        only_ours=int((m._merge == 'left_only').sum()), only_taifex=int((m._merge == 'right_only').sum()),
                        vol_ours=int(a.volume.sum()), vol_taifex=int(b.volume.sum()),
                        **{f'{c}_eq': round(float((both[c] == both[c + '_tx']).mean()), 4) if len(both) else np.nan
                           for c in ('open', 'high', 'low', 'close', 'volume')}))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--per-year', type=int, default=2); ap.add_argument('--dates', default='')
    ap.add_argument('--no-taifex', action='store_true')
    a = ap.parse_args()
    d = load_txo_daily(columns=['date', 'expiry', 'volume', 'day_volume', 'night_volume'])
    days = d.groupby('date').agg(rows=('expiry', 'size'), series=('expiry', 'nunique'), volume=('volume', 'sum'),
                                 night_volume=('night_volume', 'sum')).reset_index()
    st = pd.read_json(os.path.join(ROOT, '_convert_stats.jsonl'), lines=True)
    print(f"days {len(days)}  {days.date.min().date()} .. {days.date.max().date()}  rows {len(d):,}")
    print('stats: files', len(st), 'errors', int(st.get('errors', pd.Series(dtype=object)).notna().sum()),
          'bad lines', int(st.bad.sum()), 'zero-vol rows', int(st.get('zero_vol', pd.Series([0])).sum()))
    if os.path.exists(FUT_STATS):
        fut = set(pd.to_datetime(pd.read_json(FUT_STATS, lines=True).file_date))
        od = set(days.date)
        lo, hi = days.date.min(), min(days.date.max(), max(fut))
        miss = sorted(x for x in fut if lo <= x <= hi and x not in od)
        extra = sorted(x for x in od if lo <= x <= hi and x not in fut)
        print('futures-calendar days missing in options:', [str(x.date()) for x in miss][:50], len(miss))
        print('options days not in futures calendar:', [str(x.date()) for x in extra][:50], len(extra))
    small = days[days.rows < days.rows.rolling(60, min_periods=10).median() * 0.3]
    print('days with unusually few rows:', [(str(r.date.date()), r.rows) for r in small.itertuples()][:30])
    days.to_csv(os.path.join(ROOT, '_check_days.csv'), index=False)
    if a.no_taifex:
        return
    if a.dates:
        sample = [pd.Timestamp(x) for x in a.dates.split(',')]
    else:
        rng = np.random.default_rng(7)
        sample = [pd.Timestamp(rng.choice(g.date.values)) for _, g in days.groupby(days.date.dt.year)
                  for _ in range(a.per_year)]
    out = []
    for x in sorted(set(sample)):
        try:
            r = compare(x.strftime('%Y-%m-%d'))
            out += r
            for z in r:
                print(z, flush=True)
        except Exception as e:
            print(x.date(), 'ERR', e, flush=True)
        time.sleep(3)
    pd.DataFrame(out).to_csv(os.path.join(ROOT, '_check_taifex.csv'), index=False)


if __name__ == '__main__':
    main()
