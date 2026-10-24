#!/usr/bin/env python
"""Download Taiwan listed (TWSE) + OTC (TPEx) common-stock daily data with corporate actions and
build total-return (dividend-reinvested) adjusted prices.  Output: /workspace/fubon-threads/data/stocks/

Sources (all public, no credentials):
  daily prices  TWSE MI_INDEX?type=ALLBUT0999 (whole market per date) / TPEx afterTrading/otc?type=EW
                Whole-market-per-date snapshots include every stock that traded that day, so delisted
                stocks are covered automatically (survivorship-free within the period).
  ex-rights/div TWSE TWT49U, TPEx exDailyQ_result.php (除權除息計算結果表, with reference prices)
  cap reduction TWSE TWTAUU (from 2011); TPEx: FinMind TaiwanStockCapitalReductionReferencePrice per
                candidate stock (free tier needs data_id) - candidates = halt gap + issued-shares drop;
                if FinMind has nothing the ratio is inferred from TPEx issued shares (flag inferred=1)
  par change    FinMind TaiwanStockParValueChange (both markets)
  metadata      isin.twse.com.tw (listed/OTC, industry, list date), TWSE suspendListing,
                FinMind TaiwanStockDelisting (delist dates)
Fubon Neo SDK is NOT used: no delisted names, 1-year/request with 60 req/min shared with the running
stock-futures 1m download.

Usage:  python fetch_stocks.py all            # meta + actions + daily (resumable) + build
        python fetch_stocks.py daily|actions|meta|crfix|build
Incremental: daily skips dates already cached; actions re-pull only the last 2 years (--full for all);
build re-derives everything from caches (no network).  Every date is cached in cache/<mkt>/<yyyymmdd>.json.gz (empty marker on non-trading days);
progress in _state.json.  Rate limits: TWSE 1 req / 3.5 s, TPEx 1 req / 2.5 s, FinMind 1 req / 13 s.

Adjustment: on each ex/resumption date e, f_e = ref_price / prev_close (TWT49U/exDailyQ 除權息參考價 /
除權息前收盤價; TWTAUU 恢復買賣參考價 / 停止買賣前收盤價; par change after/before).  Backward factor
F(t) = prod_{e>t} f_e; adj_* = raw * F, so the latest prices equal real prices and
adj_close[e]/adj_close[e-1] = close[e]/ref_e  (cash dividends reinvested at the reference price).
"""
import argparse, datetime as dt, gzip, json, re, sys, threading, time
from pathlib import Path
import pandas as pd
import numpy as np
import requests

OUT = Path("/workspace/fubon-threads/data/stocks")
CACHE = OUT / "cache"
START = dt.date(2008, 1, 1)
UA = {"User-Agent": "Mozilla/5.0 (research; polite rate-limited)"}
SLEEP = {"twse": 3.5, "tpex": 2.5, "fm": 13.0}
_last = {}


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def get(kind, url, params=None, tries=6):
    for k in range(tries):
        w = SLEEP[kind] - (time.time() - _last.get(kind, 0))
        if w > 0:
            time.sleep(w)
        _last[kind] = time.time()
        try:
            r = requests.get(url, params=params, headers=UA, timeout=40, allow_redirects=False)
            if r.status_code == 200 and r.text.lstrip()[:1] in "{[":
                return r.json()
            log(f"{kind} HTTP {r.status_code} non-json, backoff")
        except Exception as e:
            log(f"{kind} error {type(e).__name__}, backoff")
        time.sleep(min(600, 30 * 2 ** k))
    raise RuntimeError(f"{kind} failed: {url} {params}")


def num(x):
    if x is None:
        return np.nan
    s = re.sub(r"<[^>]+>", "", str(x)).replace(",", "").strip()
    try:
        return float(s)
    except ValueError:
        return np.nan


def is_common(code):  # 4-digit, not ETF (0xxx) / TDR (91xx)
    return bool(re.fullmatch(r"[1-9]\d{3}", code)) and not code.startswith("91")


def roc(s):  # '97年01月02日' / '097/01/02' / '100/01/25' -> date
    y, m, d = map(int, re.findall(r"\d+", s)[:3])
    return dt.date(y + 1911 if y < 1911 else y, m, d)


# ---------------- daily ----------------
def fetch_day(mkt, d):
    if mkt == "twse":
        j = get("twse", "https://www.twse.com.tw/rwd/zh/afterTrading/MI_INDEX",
                dict(date=d.strftime("%Y%m%d"), type="ALLBUT0999", response="json"))
        rows = []
        for t in j.get("tables", []) or []:
            f = t.get("fields") or []
            if "證券代號" in f and "收盤價" in f:
                ix = {k: f.index(k) for k in ("證券代號", "證券名稱", "成交股數", "成交筆數", "成交金額", "開盤價", "最高價", "最低價", "收盤價")}
                for r in t["data"]:
                    c = r[ix["證券代號"]].strip()
                    if is_common(c):
                        rows.append([c, r[ix["證券名稱"]].strip()] + [num(r[ix[k]]) for k in ("開盤價", "最高價", "最低價", "收盤價", "成交股數", "成交金額", "成交筆數")] + [np.nan])
        return rows
    j = get("tpex", "https://www.tpex.org.tw/www/zh-tw/afterTrading/otc",
            dict(date=d.strftime("%Y/%m/%d"), type="EW", response="json"))
    rows = []
    for t in j.get("tables", []) or []:
        f = [x.strip() for x in t.get("fields") or []]
        if "代號" not in f:
            continue
        g = lambda *ks: next(f.index(k) for k in ks if k in f)
        ix = dict(c=g("代號"), n=g("名稱"), o=g("開盤"), h=g("最高"), l=g("最低"), cl=g("收盤"), v=g("成交股數"), t=g("成交金額(元)", "成交金額"), k=g("成交筆數"), s=g("發行股數"))
        for r in t.get("data") or []:
            c = r[ix["c"]].strip()
            if is_common(c):
                rows.append([c, r[ix["n"]].strip()] + [num(r[ix[k]]) for k in ("o", "h", "l", "cl", "v", "t", "k", "s")])
    return rows


def daily_worker(mkt, end, state, lock):
    d, n = START, 0
    while d <= end:
        p = CACHE / mkt / f"{d:%Y%m%d}.json.gz"
        if d.weekday() < 5 and not p.exists():
            rows = fetch_day(mkt, d)
            p.parent.mkdir(parents=True, exist_ok=True)
            tmp = p.with_suffix(".tmp")
            with gzip.open(tmp, "wt") as fh:
                json.dump(rows, fh)
            tmp.rename(p)
            n += 1
            with lock:
                state[mkt] = dict(last=str(d), fetched=n, updated=time.strftime("%Y-%m-%d %H:%M:%S"))
                (OUT / "_state.json").write_text(json.dumps(state, indent=1))
            if n % 50 == 0:
                log(f"{mkt} {d} ({len(rows)} stocks) fetched {n}")
        d += dt.timedelta(days=1)
    log(f"{mkt} daily done ({n} new dates)")


def run_daily(end):
    st = json.loads((OUT / "_state.json").read_text()) if (OUT / "_state.json").exists() else {}
    lock = threading.Lock()
    th = [threading.Thread(target=daily_worker, args=(m, end, st, lock)) for m in ("twse", "tpex")]
    [t.start() for t in th]
    [t.join() for t in th]


# ---------------- corporate actions ----------------
def run_actions(end, full=False):
    rows, prev = [], None
    y0 = START.year
    if not full and (OUT / "_actions_official.csv").exists():
        prev = pd.read_csv(OUT / "_actions_official.csv", dtype={"code": str}, parse_dates=["date"])
        y0 = max(START.year, prev.date.max().year - 1)     # re-pull last 2 years (late corrections)
        prev = prev[(prev.date.dt.year < y0) & (prev.action != "par_change")]
        log(f"actions incremental from {y0} (keeping {len(prev)} older rows)")
    for y in range(y0, end.year + 1):
        a, b = dt.date(y, 1, 1), min(dt.date(y, 12, 31), end)
        j = get("twse", "https://www.twse.com.tw/rwd/zh/exRight/TWT49U", dict(startDate=f"{a:%Y%m%d}", endDate=f"{b:%Y%m%d}", response="json"))
        f = j.get("fields") or []
        for r in j.get("data") or []:
            x = dict(zip(f, r))
            kind = str(x.get("權/息", "")).strip()
            tot = num(x.get("權值+息值"))
            cash = num(x["息值"]) if "息值" in x else (tot if kind == "息" else (0.0 if kind == "權" else np.nan))
            rv = num(x["權值"]) if "權值" in x else (tot if kind == "權" else (0.0 if kind == "息" else np.nan))
            rows.append(dict(market="twse", code=x["股票代號"].strip(), name=x["股票名稱"].strip(), date=roc(x["資料日期"]), action="exright_" + kind,
                             prev_close=num(x["除權息前收盤價"]), ref_price=num(x["除權息參考價"]), rights_value=rv, cash_dividend=cash,
                             div_plus_rights=tot, source="TWSE TWT49U"))
        j = get("tpex", "https://www.tpex.org.tw/web/stock/exright/dailyquo/exDailyQ_result.php", dict(l="zh-tw", d=f"{a:%Y/%m/%d}", ed=f"{b:%Y/%m/%d}", o="json"))
        for t in j.get("tables") or []:
            for r in t.get("data") or []:
                rows.append(dict(market="tpex", code=r[1].strip(), name=r[2].strip(), date=roc(r[0]), action="exright_" + r[8].strip().replace("除", ""), div_plus_rights=num(r[7]),
                                 prev_close=num(r[3]), ref_price=num(r[4]), rights_value=num(r[5]), cash_dividend=num(r[13]),
                                 stock_div_per_1000=num(r[14]), cash_issue_shares=num(r[16]), cash_issue_price=num(r[17]), source="TPEx exDailyQ"))
        if y >= 2011:
            j = get("twse", "https://www.twse.com.tw/rwd/zh/reducation/TWTAUU", dict(startDate=f"{a:%Y%m%d}", endDate=f"{b:%Y%m%d}", response="json"))
            for r in j.get("data") or []:
                rows.append(dict(market="twse", code=r[1].strip(), name=r[2].strip(), date=roc(r[0]), action="cap_reduction",
                                 prev_close=num(r[3]), ref_price=num(r[4]), exright_ref_after=num(r[8]), reason=r[9].strip(), source="TWSE TWTAUU"))
        log(f"actions {y}: total {len(rows)}")
    j = get("fm", "https://api.finmindtrade.com/api/v4/data", dict(dataset="TaiwanStockParValueChange", start_date=str(START)))
    for r in j.get("data") or []:
        rows.append(dict(market="", code=r["stock_id"], name=r.get("stock_name", ""), date=pd.Timestamp(r["date"]).date(), action="par_change",
                         prev_close=r["before_close"], ref_price=r["after_ref_close"], source="FinMind TaiwanStockParValueChange"))
    ca = pd.DataFrame(rows)
    ca["date"] = pd.to_datetime(ca.date)
    if prev is not None:
        ca = pd.concat([prev, ca], ignore_index=True)
    ca = ca[ca.code.map(is_common)].drop_duplicates(["code", "date", "action", "ref_price"])
    ca.sort_values(["date", "code"]).to_csv(OUT / "_actions_official.csv", index=False)
    log(f"official actions: {len(ca)}")


# ---------------- metadata ----------------
def run_meta():
    rows = []
    for mode, mkt in ((2, "twse"), (4, "tpex")):
        r = requests.get(f"https://isin.twse.com.tw/isin/C_public.jsp?strMode={mode}", headers=UA, timeout=60)
        r.encoding = "big5"
        for tr in re.findall(r"<tr>(.*?)</tr>", r.text, re.S):
            td = [re.sub(r"<[^>]+>", "", x).strip() for x in re.findall(r"<td[^>]*>(.*?)</td>", tr, re.S)]
            if len(td) >= 6 and "\u3000" in td[0]:
                code, name = td[0].split("\u3000", 1)
                if is_common(code.strip()) and td[5].startswith("ES"):
                    rows.append(dict(code=code.strip(), name=name.strip(), market=mkt, industry=td[4], list_date=td[2].replace("/", "-"), status="listed"))
        time.sleep(3)
    cur = pd.DataFrame(rows).drop_duplicates("code")
    dl = []
    j = get("twse", "https://www.twse.com.tw/rwd/zh/company/suspendListing", dict(response="json"))
    for r in j.get("data") or []:
        dl.append(dict(code=r[2].strip(), name=r[1].strip(), delist_date=str(roc(r[0])), delist_src="TWSE suspendListing"))
    j = get("fm", "https://api.finmindtrade.com/api/v4/data", dict(dataset="TaiwanStockDelisting"))
    for r in j.get("data") or []:
        dl.append(dict(code=r["stock_id"], name=r["stock_name"], delist_date=r["date"], delist_src="FinMind TaiwanStockDelisting"))
    dl = pd.DataFrame(dl)
    dl = dl[dl.code.map(is_common)]
    cur.to_csv(OUT / "_meta_current.csv", index=False)
    dl.to_csv(OUT / "_delist_lists.csv", index=False)
    log(f"meta: current {len(cur)}, delist records {len(dl)}")


# ---------------- assemble ----------------
COLS = ["code", "name", "open", "high", "low", "close", "volume", "turnover", "trades", "shares"]


def load_daily():
    fr = []
    for mkt in ("twse", "tpex"):
        for p in sorted((CACHE / mkt).glob("*.json.gz")):
            with gzip.open(p, "rt") as fh:
                rows = json.load(fh)
            if rows:
                df = pd.DataFrame(rows, columns=COLS)
                df["date"], df["market"] = pd.Timestamp(p.name[:8]), mkt
                fr.append(df)
    return pd.concat(fr, ignore_index=True)


def tpex_cr_candidates(px, ca):
    """TPEx stocks whose issued shares drop >5% across a >=3-business-day halt, with no official event."""
    out = []
    t = px[px.market == "tpex"].sort_values(["code", "date"])
    for c, g in t.groupby("code"):
        s, d = g.shares.values, g.date.values
        for k in range(1, len(g)):
            gap = np.busday_count(d[k - 1].astype("M8[D]"), d[k].astype("M8[D]"))
            if gap >= 3 and s[k - 1] > 0 and s[k] < 0.95 * s[k - 1]:
                out.append(dict(code=c, date=pd.Timestamp(d[k]).date(), prev_date=pd.Timestamp(d[k - 1]).date(), share_ratio=s[k] / s[k - 1],
                                prev_close=g.close.values[k - 1]))
    return pd.DataFrame(out)


def run_crfix(px=None):
    px = load_daily() if px is None else px
    cand = tpex_cr_candidates(px, None)
    cand.to_csv(OUT / "_tpex_cr_candidates.csv", index=False)
    cache = OUT / "_finmind_cr.json"
    got = json.loads(cache.read_text()) if cache.exists() else {}
    codes = sorted(set(cand.code))
    log(f"TPEx capital-reduction candidates: {len(cand)} events / {len(codes)} stocks; FinMind cached {len(got)}")
    for i, c in enumerate(codes):
        if c in got:
            continue
        j = get("fm", "https://api.finmindtrade.com/api/v4/data", dict(dataset="TaiwanStockCapitalReductionReferencePrice", data_id=c, start_date="2008-01-01"))
        if j.get("status") != 200:
            log(f"FinMind refused ({j.get('msg', '')[:60]}); stop, rerun later")
            break
        got[c] = j.get("data") or []
        cache.write_text(json.dumps(got))
        if i % 20 == 0:
            log(f"FinMind CR {i}/{len(codes)}")


def _type(e):
    a = str(e.action)
    if a in ("cap_reduction", "par_change"):
        return a
    k = a.replace("exright_", "")
    if e.market == "tpex" and k != "息":
        parts = (["cash_div"] if (e.cash_dividend or 0) > 0 else []) + (["stock_div"] if (e.stock_div_per_1000 or 0) > 0 else []) \
            + (["rights_issue"] if (e.cash_issue_shares or 0) > 0 else [])
        return "+".join(parts) or "stock_div_or_rights"
    return {"息": "cash_div", "權": "stock_div_or_rights", "權息": "cash_div+stock_div_or_rights"}.get(k, k)


def run_build():
    px = load_daily()
    px = px[px.close > 0].drop_duplicates(["code", "date"], keep="last")
    ca = pd.read_csv(OUT / "_actions_official.csv", dtype={"code": str}, parse_dates=["date"])
    # TPEx capital reductions: FinMind where available, otherwise inferred from issued shares
    extra = []
    fm = json.loads((OUT / "_finmind_cr.json").read_text()) if (OUT / "_finmind_cr.json").exists() else {}
    for c, lst in fm.items():
        for r in lst:
            extra.append(dict(market="tpex", code=c, date=pd.Timestamp(r["date"]), action="cap_reduction", prev_close=r["ClosingPriceonTheLastTradingDay"],
                              ref_price=r["PostReductionReferencePrice"], reason=r.get("ReasonforCapitalReduction", ""), source="FinMind CapitalReductionReferencePrice"))
    cand = tpex_cr_candidates(px, None)
    known = [(e["code"], e["date"]) for e in extra] + [(r.code, r.date) for r in ca[ca.action == "cap_reduction"].itertuples()]
    for r in cand.itertuples():
        if not any(k[0] == r.code and abs((k[1] - pd.Timestamp(r.date)).days) <= 10 for k in known):
            extra.append(dict(market="tpex", code=r.code, date=pd.Timestamp(r.date), action="cap_reduction", prev_close=r.prev_close,
                              ref_price=r.prev_close / r.share_ratio, reason="inferred from issued shares", inferred=1, source="TPEx issued shares (inferred)"))
    ca = pd.concat([ca, pd.DataFrame(extra)], ignore_index=True)
    for k in ("inferred", "cash_dividend", "stock_div_per_1000", "cash_issue_shares", "cash_issue_price", "rights_value", "div_plus_rights", "reason", "exright_ref_after"):
        if k not in ca:
            ca[k] = np.nan
    ca["inferred"] = ca.inferred.fillna(0).astype(int)
    mk = px.groupby("code").market.last()
    ca["market"] = ca.code.map(mk).fillna(ca.market)
    ca = ca[ca.code.isin(mk.index)].sort_values(["code", "date"]).reset_index(drop=True)
    rawdir, adjdir = OUT / "raw", OUT / "adj"
    rawdir.mkdir(exist_ok=True); adjdir.mkdir(exist_ok=True)
    ca["trade_date"], ca["adj_factor"], ca["prev_close_data"] = pd.NaT, np.nan, np.nan
    byc = {c: g for c, g in ca.groupby("code")}
    facs, inf = [], []
    for c, g in px.sort_values("date").groupby("code"):
        g = g.set_index("date")
        g[["market", "name", "open", "high", "low", "close", "volume", "turnover", "trades", "shares"]].to_parquet(rawdir / f"{c}.parquet")
        f = pd.Series(1.0, g.index)
        for ix, e in byc.get(c, pd.DataFrame()).iterrows():
            k = g.index.searchsorted(e.date)
            if k == 0 or k >= len(g) or not (e.ref_price > 0):
                continue
            pc = g.close.iloc[k - 1]
            base = e.prev_close if e.prev_close > 0 else pc
            fac = e.ref_price / base
            f.iloc[k] *= fac
            ca.loc[ix, ["trade_date", "adj_factor", "prev_close_data"]] = [g.index[k], fac, pc]
        # unexplained jump after a trading halt (e.g. TWSE capital reductions before 2011, missed events):
        # reference price unknown -> use the resumption-day open (error <= that day's opening gap); inferred=2
        d = g.index.values.astype("M8[D]")
        gap = np.r_[0, np.busday_count(d[:-1], d[1:])]
        jump = (g.close / g.close.shift() * f).sub(1).abs()
        lim = np.where(g.index < "2015-06-01", 0.07, 0.10) + 0.015
        for k in np.where((gap >= 3) & (jump.values > lim) & (f.values == 1.0))[0]:
            if g.open.iloc[k] > 0:
                fac = g.open.iloc[k] / g.close.iloc[k - 1]
                f.iloc[k] = fac
                inf.append(dict(code=c, market=g.market.iloc[-1], name=g.name.iloc[k], ex_date=g.index[k].date(), trade_date=g.index[k].date(),
                                type="halt_jump_inferred", ref_price=g.open.iloc[k], prev_close=g.close.iloc[k - 1], prev_close_data=g.close.iloc[k - 1],
                                adj_factor=fac, applied=True, mismatch=False, inferred=2,
                                reason=f"{gap[k]}-business-day halt, |ret| {jump.iloc[k]:.0%}; ref = resumption open", source="inferred from prices"))
        F = f.iloc[::-1].cumprod().iloc[::-1].shift(-1).fillna(1.0)
        a = pd.DataFrame({k: g[k] * F for k in ("open", "high", "low", "close")})
        a["volume"], a["turnover"], a["factor"] = g.volume / F, g.turnover, F
        a["ret"] = a.close.pct_change()
        a.to_parquet(adjdir / f"{c}.parquet")
        ch = f[f != 1.0]
        facs.append(pd.DataFrame(dict(code=c, date=ch.index, event_factor=ch.values, cum_factor_before=F.reindex(ch.index).values / ch.values)))
    std = pd.DataFrame(dict(code=ca.code, market=ca.market, name=ca.name, ex_date=pd.to_datetime(ca.date).dt.date, trade_date=pd.to_datetime(ca.trade_date).dt.date,
                            type=ca.apply(_type, axis=1), cash_div=ca.cash_dividend, stock_div=ca.stock_div_per_1000 / 1000, rights_value=ca.rights_value,
                            div_plus_rights=ca.div_plus_rights, cash_issue_shares=ca.cash_issue_shares, cash_issue_price=ca.cash_issue_price,
                            ref_price=ca.ref_price, prev_close=ca.prev_close, prev_close_data=ca.prev_close_data, adj_factor=ca.adj_factor,
                            applied=ca.adj_factor.notna(), mismatch=(ca.prev_close_data / ca.prev_close - 1).abs() > 0.01, inferred=ca.inferred,
                            reason=ca.reason, source=ca.source))
    std = pd.concat([std, pd.DataFrame(inf)], ignore_index=True).sort_values(["code", "ex_date"]).reset_index(drop=True)
    std.to_parquet(OUT / "corporate_actions.parquet", index=False)
    std.to_csv(OUT / "corporate_actions.csv", index=False)
    pd.concat(facs, ignore_index=True).to_parquet(OUT / "adj_factors.parquet", index=False)
    # universe / metadata
    cur = pd.read_csv(OUT / "_meta_current.csv", dtype={"code": str})
    dl = pd.read_csv(OUT / "_delist_lists.csv", dtype={"code": str}).sort_values("delist_date").drop_duplicates("code", keep="last")
    agg = px.groupby("code").agg(name_last=("name", "last"), market=("market", "last"), first_date=("date", "min"), last_date=("date", "max"), n_days=("date", "size"))
    u = agg.join(cur.set_index("code")[["name", "industry", "list_date", "status"]], how="left").join(dl.set_index("code")[["delist_date", "delist_src"]], how="left")
    last = px.date.max()
    u["status"] = np.where(u.status.eq("listed"), "listed", np.where((u.last_date < last - pd.Timedelta(days=30)) | u.delist_date.notna(), "delisted", "unknown"))
    m = u.status.eq("delisted") & u.delist_date.isna()
    u.loc[m, "delist_src"] = "inferred: last trading day + 1"
    u.loc[m, "delist_date"] = (u.loc[m, "last_date"] + pd.Timedelta(days=1)).dt.date.astype(str)
    u.loc[u.status.eq("listed"), ["delist_date", "delist_src"]] = np.nan
    u["name"] = u.name.fillna(u.name_last)
    u = u.drop(columns="name_last").reset_index()
    u.to_csv(OUT / "universe.csv", index=False)
    log(f"build: {len(u)} stocks, {len(std)} actions ({int(std.applied.sum())} applied), last date {last.date()}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["all", "meta", "actions", "daily", "crfix", "build"])
    ap.add_argument("--end", default=str(dt.date.today()))
    ap.add_argument("--full", action="store_true", help="re-pull all corporate-action years")
    a = ap.parse_args()
    end = dt.date.fromisoformat(a.end)
    OUT.mkdir(parents=True, exist_ok=True)
    if a.stage in ("all", "meta"):
        run_meta()
    if a.stage in ("all", "actions"):
        run_actions(end, a.full)
    if a.stage in ("all", "daily"):
        run_daily(end)
    if a.stage in ("all", "crfix"):
        run_crfix()
    if a.stage in ("all", "build"):
        run_build()


if __name__ == "__main__":
    main()
