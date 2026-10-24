"""Convert TAIFEX daily futures tick RPT files (zip) to parquet.

Sources (in ZIPDIR): Daily_YYYY_MM_DD.zip (one .rpt), 2000_2010.zip (YYYY/Daily_*.rpt),
2011_2018.zip (YYYY/Daily_*.zip nested). For each date the source with the largest
uncompressed RPT is used (2010 standalone dailies are partial; yearly archive is complete).

Output: OUT/product=XX/month=YYYY-MM.parquet (zstd), one file per product per month
(per-day files would be ~1M tiny files because of ~200+ stock-futures codes).
Usage: python tools/tick_convert.py ZIPDIR OUTDIR [--months 2024-01,2024-02] [--jobs 6]
"""
import io, os, re, sys, zipfile, json, argparse, datetime as dt
from concurrent.futures import ProcessPoolExecutor
import numpy as np, pandas as pd, pyarrow as pa, pyarrow.parquet as pq

DRE = re.compile(r'Daily_(\d{4})_(\d{2})_(\d{2})\.(rpt|zip)$', re.I)


def index_sources(zipdir):
    """date -> (size, kind, zipname, member)"""
    cand = {}
    def put(d, size, ref):
        cand.setdefault(d, []).append((size,) + ref)
    for fn in sorted(os.listdir(zipdir)):
        p = os.path.join(zipdir, fn)
        if not fn.endswith('.zip'):
            continue
        z = zipfile.ZipFile(p)
        for zi in z.infolist():
            m = DRE.search(zi.filename)
            if not m:
                continue
            d = '%s-%s-%s' % m.group(1, 2, 3)
            if m.group(4).lower() == 'rpt':
                put(d, zi.file_size, ('rpt', fn, zi.filename))
            else:  # nested zip: open to get rpt size
                inner = zipfile.ZipFile(io.BytesIO(z.read(zi.filename)))
                ii = inner.infolist()[0]
                put(d, ii.file_size, ('nested', fn, zi.filename))
    # largest first; standalone daily preferred on ties (archive copies can be corrupt)
    return {d: sorted(v, key=lambda r: (-r[0], r[1] != 'Daily_' + d.replace('-', '_') + '.zip'))
            for d, v in cand.items()}


def read_rpt(zipdir, refs):
    """Try candidates in order; return (bytes, ref used, errors)."""
    errs = []
    for ref in refs:
        try:
            return _read(zipdir, ref), ref, errs
        except Exception as e:  # bad CRC etc.
            errs.append(f'{ref[2]}: {e}')
    raise RuntimeError('; '.join(errs))


def _read(zipdir, ref):
    _, kind, fn, mem = ref
    z = zipfile.ZipFile(os.path.join(zipdir, fn))
    b = z.read(mem)
    if kind == 'nested':
        inner = zipfile.ZipFile(io.BytesIO(b))
        b = inner.read(inner.namelist()[0])
    return b


def parse(b, file_date):
    """Return (DataFrame, n_data_lines, n_bad)."""
    txt = b.decode('big5', errors='replace')
    lines = txt.splitlines()
    rows = []
    bad = 0
    for l in lines[1:]:
        f = [x.strip() for x in l.split(',')]
        if not f or not f[0][:1].isdigit():
            if l.strip(' \t\x1a') and not set(l.strip()) <= set('- '):
                bad += 1
            continue
        rows.append(f[:9] + [''] * (9 - len(f)))
    n = len(rows)
    if not n:
        return None, 0, bad
    a = np.array(rows, dtype=object)
    df = pd.DataFrame({'cal_date': a[:, 0], 'product': a[:, 1], 'contract': a[:, 2],
                       't': a[:, 3], 'price': a[:, 4], 'volume': a[:, 5],
                       'near_price': a[:, 6], 'far_price': a[:, 7], 'auc': a[:, 8]})
    num = lambda s: pd.to_numeric(s.replace({'-': None, '': None}), errors='coerce')
    df['price'] = num(df.price).astype('float64')
    df['near_price'] = num(df.near_price).astype('float64')
    df['far_price'] = num(df.far_price).astype('float64')
    vol = num(df.volume)
    t = df.t.str.zfill(6)
    hh = t.str[0:2].astype(int); mm = t.str[2:4].astype(int); ss = t.str[4:6].astype(int)
    frac = t.str[6:].replace('', '0').astype(int)  # 8-digit format: last 2 = 1/100 s
    ms = ((hh * 3600 + mm * 60 + ss) * 1000 + np.where(t.str.len() > 6, frac * 10, 0))
    cal = pd.to_datetime(df.cal_date, format='%Y%m%d')
    ts = cal + pd.to_timedelta(ms, unit='ms')
    good = df.price.notna() & vol.notna() & cal.notna()
    bad += int((~good).sum())
    # Daily_X.rpt holds the night session that ends on X (calendar date = previous trading
    # evening / X-morning) followed by X's day session. Some day-session products trade past
    # 15:00 on X itself, so session is decided by calendar date, not clock time.
    fd = np.datetime64(file_date)
    c = cal.values.astype('datetime64[D]')
    night = (c < fd) | (hh.values < 8)
    trade = np.full(len(df), fd, dtype='datetime64[D]')
    out = pd.DataFrame({
        'ts': ts.dt.tz_localize('Asia/Taipei').astype('datetime64[ms, Asia/Taipei]'),
        'trade_date': trade.astype('datetime64[s]'),
        'product': df['product'],
        'contract': df.contract.astype(str),
        'price': df.price, 'volume': vol.fillna(0).astype('int32'),
        'is_spread': df.contract.str.contains('/'),
        'near_price': df.near_price, 'far_price': df.far_price,
        'session': np.where(night, 'night', 'day'),
        'open_auction': df.auc.eq('*'),
    })[good.values]
    return out, n, bad


def do_month(args):
    zipdir, outdir, month, refs, trading_days = args
    parts, stats = [], []
    for d, ref in refs:
        try:
            b, used, errs = read_rpt(zipdir, ref)
        except RuntimeError as e:
            stats.append({'file_date': d, 'lines': 0, 'bad': 0, 'rows': 0, 'source': None, 'errors': str(e)})
            continue
        df, n, bad = parse(b, d)
        stats.append({'file_date': d, 'lines': n, 'bad': bad, 'rows': 0 if df is None else len(df),
                      'source': used[1] + ':' + used[2], **({'errors': '; '.join(errs)} if errs else {})})
        if df is not None:
            parts.append(df)
    if not parts:
        return stats
    m = pd.concat(parts, ignore_index=True)
    m['session'] = m.session.astype('category')
    for prod, g in m.groupby('product', sort=True):
        d = os.path.join(outdir, f'product={prod}')
        os.makedirs(d, exist_ok=True)
        g = g.drop(columns='product').sort_values('ts', kind='stable').reset_index(drop=True)
        tbl = pa.Table.from_pandas(g, preserve_index=False)
        pq.write_table(tbl, os.path.join(d, f'month={month}.parquet.tmp'), compression='zstd')
        os.replace(os.path.join(d, f'month={month}.parquet.tmp'), os.path.join(d, f'month={month}.parquet'))
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('zipdir'); ap.add_argument('outdir')
    ap.add_argument('--months', default=''); ap.add_argument('--jobs', type=int, default=6)
    ap.add_argument('--skip-done', action='store_true')
    a = ap.parse_args()
    src = index_sources(a.zipdir)
    days = sorted(src)
    # file dates are trading days (night rows' calendar date is the previous evening)
    bymonth = {}
    for d in days:
        bymonth.setdefault(d[:7], []).append((d, src[d]))
    months = a.months.split(',') if a.months else sorted(bymonth)
    statf = os.path.join(a.outdir, '_convert_stats.jsonl')
    done = set()
    if a.skip_done and os.path.exists(statf):
        done = {json.loads(l)['file_date'][:7] for l in open(statf)}
    os.makedirs(a.outdir, exist_ok=True)
    jobs = [(a.zipdir, a.outdir, m, bymonth[m], days) for m in months if m in bymonth and m not in done]
    with ProcessPoolExecutor(a.jobs) as ex, open(statf, 'a') as sf:
        for st in ex.map(do_month, jobs):
            for s in st:
                sf.write(json.dumps(s) + '\n')
            sf.flush()
            print(st[0]['file_date'][:7], sum(s['rows'] for s in st), 'bad', sum(s['bad'] for s in st), flush=True)


if __name__ == '__main__':
    main()
