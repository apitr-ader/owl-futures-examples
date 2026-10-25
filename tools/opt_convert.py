"""Convert TAIFEX options tick RPT files (OptionsDaily_YYYY_MM_DD, zip) to DAILY bars for TAIEX
options (product TXO, incl. weekly series 'YYYYMMWn' / Friday weeklies 'YYYYMMFn'). Ticks are not kept.

Sources (ZIPDIR, searched recursively): OptionsDaily_*.zip (one .rpt each), incl. the zips extracted
from the 2002to2009 / OPT_2010to2015 RAR archives. For each date the candidate with the largest
uncompressed RPT is used; on a read/CRC error the next candidate is tried.

RPT formats: comma-separated (2010s+) or whitespace fixed-width (2000s). Columns: trade date,
product, strike, expiry(week), C/P, time (HHMMSS or HHMMSScc), price, volume, [open-auction '*'].
Volume: each trade is reported for both buyer and seller sides -> contracts = sum / 2
(verified: ratio exactly 2.0 vs TAIFEX published daily volume, day and night session).
Trading day = file date; night-session rows (calendar date before file date, or before 08:00)
belong to that trading day (same as TAIFEX's 盤後 rows dated with the next trading day).

Columns: date, expiry, strike, cp, open, high, low, close, ticks (RPT records, both sides), vwap,
volume (contracts), day_*/night_* (open/high/low/close/volume/ticks; NaN/0 if no trade in session).
Output: OUT/daily/year=YYYY.parquet (zstd), OUT/_convert_stats.jsonl (per-date line counts/source).
Usage: python tools/opt_convert.py ZIPDIR OUTDIR [--years 2024,2025] [--jobs 2] [--skip-done]
"""
import io, os, re, sys, json, zipfile, argparse
from concurrent.futures import ProcessPoolExecutor
import numpy as np, pandas as pd, pyarrow as pa, pyarrow.parquet as pq

DRE = re.compile(r'OptionsDaily_(\d{4})_(\d{2})_(\d{2})\.zip$', re.I)
PRODUCT = b'TXO'


def index_sources(zipdir):
    cand = {}
    for root, _, files in os.walk(zipdir):
        for fn in files:
            m = DRE.search(fn)
            if not m:
                continue
            p = os.path.join(root, fn)
            try:
                zi = zipfile.ZipFile(p).infolist()
                size = max(z.file_size for z in zi)
            except Exception as e:
                size = -1
            cand.setdefault('%s-%s-%s' % m.groups(), []).append((size, p))
    return {d: sorted(v, reverse=True) for d, v in cand.items()}


def read_rpt(refs):
    errs = []
    for size, p in refs:
        try:
            z = zipfile.ZipFile(p)
            mem = max(z.infolist(), key=lambda i: i.file_size)
            return z.read(mem.filename), p, errs
        except Exception as e:
            errs.append(f'{os.path.basename(p)}: {e}')
    raise RuntimeError('; '.join(errs))


def parse(b, file_date):
    """Return (tick DataFrame for TXO, n_lines_kept, n_bad)."""
    lines = [l for l in b.split(b'\n') if PRODUCT in l]
    rows, bad = [], 0
    for l in lines:
        s = l.decode('big5', 'replace')
        f = [x.strip() for x in s.split(',')] if ',' in s else s.split()
        if len(f) < 8 or not f[0][:1].isdigit() or f[1] != 'TXO':
            bad += 1
            continue
        rows.append(f[:9] + [''] * (9 - len(f[:9])))
    if not rows:
        return None, 0, bad, 0
    a = np.array(rows, dtype=object)
    df = pd.DataFrame({'cal': a[:, 0], 'strike': a[:, 2], 'expiry': a[:, 3], 'cp': a[:, 4],
                       't': a[:, 5], 'price': a[:, 6], 'vol': a[:, 7]})
    num = lambda s: pd.to_numeric(s, errors='coerce')
    df['strike'] = num(df.strike); df['price'] = num(df.price); df['vol'] = num(df.vol)
    t = df.t.str.zfill(6)
    long = df.t.str.len() > 6           # old HHMMSScc
    t = np.where(long, df.t.str.zfill(8), t)
    t = pd.Series(t)
    hh = num(t.str[0:2]); mm = num(t.str[2:4]); ss = num(t.str[4:6])
    df['sec'] = hh * 3600 + mm * 60 + ss
    good = df.price.notna() & df.vol.notna() & df.strike.notna() & df.sec.notna() \
        & df.cp.isin(['C', 'P']) & (df.price >= 0)
    bad += int((~good).sum())
    zero = int((good & (df.vol <= 0)).sum())  # 2000s files list zero-volume (price 0) non-trade rows
    good &= df.vol > 0
    df = df[good.values].copy()
    fd = file_date.replace('-', '')
    df['night'] = (df.cal < fd) | (hh[good.values].values < 8)
    df['order'] = np.arange(len(df))
    df = df.sort_values(['night', 'cal', 'sec', 'order'], ascending=[False, True, True, True], kind='stable')
    return df, len(lines), bad, zero


def _agg(g):
    g = g.assign(pv=g.price * g.vol)
    r = g.groupby(['expiry', 'strike', 'cp'], sort=False).agg(
        open=('price', 'first'), high=('price', 'max'), low=('price', 'min'), close=('price', 'last'),
        vol2=('vol', 'sum'), ticks=('vol', 'size'), pv=('pv', 'sum'))
    r['vwap'] = r.pv / r.vol2
    r['volume'] = (r.vol2 // 2).astype('int64')
    return r.drop(columns=['pv', 'vol2'])


def daily_bars(df, d):
    allb = _agg(df)  # df is ordered night-first then day -> first/last are session-correct
    out = allb
    for name, sub in (('day', df[~df.night]), ('night', df[df.night])):
        cols = ['open', 'high', 'low', 'close', 'volume', 'ticks']
        if len(sub):
            out = out.join(_agg(sub)[cols].add_prefix(name + '_'), how='left')
        else:
            for c in cols:
                out[name + '_' + c] = np.nan
    out = out.reset_index()
    out.insert(0, 'date', pd.Timestamp(d))
    for c in ('day_volume', 'night_volume', 'day_ticks', 'night_ticks'):
        out[c] = out[c].fillna(0).astype('int64')
    out['ticks'] = out.ticks.astype('int64')
    out['strike'] = out.strike.astype('float64')
    return out


def do_year(args):
    outdir, year, items = args
    parts, stats = [], []
    for d, refs in items:
        try:
            b, used, errs = read_rpt(refs)
        except RuntimeError as e:
            stats.append({'file_date': d, 'lines': 0, 'bad': 0, 'rows': 0, 'source': None, 'errors': str(e)})
            continue
        df, n, bad, zero = parse(b, d)
        del b
        bars = daily_bars(df, d) if df is not None and len(df) else None
        st = {'file_date': d, 'lines': n, 'bad': bad, 'zero_vol': zero, 'ticks': 0 if df is None else len(df),
              'rows': 0 if bars is None else len(bars), 'night_ticks': 0 if df is None else int(df.night.sum()),
              'source': os.path.relpath(used, os.path.dirname(os.path.dirname(used))), 'n_candidates': len(refs)}
        if errs:
            st['errors'] = '; '.join(errs)
        stats.append(st)
        if bars is not None:
            parts.append(bars)
    if parts:
        y = pd.concat(parts, ignore_index=True).sort_values(['date', 'expiry', 'cp', 'strike'])
        y['cp'] = y.cp.astype('category'); y['date'] = y.date.astype('datetime64[s]')
        os.makedirs(os.path.join(outdir, 'daily'), exist_ok=True)
        p = os.path.join(outdir, 'daily', f'year={year}.parquet')
        pq.write_table(pa.Table.from_pandas(y, preserve_index=False), p + '.tmp', compression='zstd')
        os.replace(p + '.tmp', p)
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('zipdir'); ap.add_argument('outdir')
    ap.add_argument('--years', default=''); ap.add_argument('--jobs', type=int, default=2)
    ap.add_argument('--skip-done', action='store_true')
    a = ap.parse_args()
    src = index_sources(a.zipdir)
    byyear = {}
    for d in sorted(src):
        byyear.setdefault(d[:4], []).append((d, src[d]))
    years = a.years.split(',') if a.years else sorted(byyear)
    os.makedirs(a.outdir, exist_ok=True)
    statf = os.path.join(a.outdir, '_convert_stats.jsonl')
    done = set()
    if a.skip_done:
        done = {f[5:9] for f in os.listdir(os.path.join(a.outdir, 'daily'))} if os.path.isdir(os.path.join(a.outdir, 'daily')) else set()
    old = [json.loads(l) for l in open(statf)] if os.path.exists(statf) else []
    jobs = [(a.outdir, y, byyear[y]) for y in years if y in byyear and y not in done]
    with ProcessPoolExecutor(a.jobs) as ex:
        for st in ex.map(do_year, jobs):
            yr = st[0]['file_date'][:4]
            old = [s for s in old if s['file_date'][:4] != yr] + st
            with open(statf + '.tmp', 'w') as f:
                for s in sorted(old, key=lambda s: s['file_date']):
                    f.write(json.dumps(s) + '\n')
            os.replace(statf + '.tmp', statf)
            print(yr, 'days', len(st), 'ticks', sum(s.get('ticks', 0) for s in st), 'rows', sum(s['rows'] for s in st),
                  'bad', sum(s['bad'] for s in st), 'errors', sum('errors' in s for s in st), flush=True)


if __name__ == '__main__':
    main()
