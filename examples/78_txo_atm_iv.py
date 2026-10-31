"""78: TXO 每日價平隱含波動率（ATM IV）序列 + 30 天固定天期 + 官方 VIX 驗證。資料不在 repo（TAIFEX 授權）。

方法
  * 選擇權價格：options_daily 的日盤最後成交 day_close（≤13:45），day_volume ≥ MINVOL 才採用；
    違反內含價值（價格 < intrinsic*e^{-rT} - 1 tick）剔除。
  * F：TXF 近月 1 分鐘 K 棒在 --snap（預設 13:44）之前最後一根 close（2017-06 前無 1m → 日盤收盤）。
    若 TXF 合約月 ≠ 選擇權到期月（換月後近月期貨已是下一月、或遠月選擇權），改用 put-call parity
    在最接近的 3 檔履約價推隱含遠期 F_imp（flag fwd_src='parity'）；parity 失敗則用 TXF（'txf_mismatch'）。
  * Black-76，r = R（1.5%，常數；對 1 個月 ATM IV 影響 < 0.05 vol 點）。
  * T：分鐘數 至 到期日 13:30 / (365*1440)，起點為當日 13:45。
  * 每到期：F 上下兩檔履約價，取 OTM（K<F 用 put，K≥F 用 call；另存 call/put 平均）IV，在 K 線性內插到 F。
  * 30 天：選 DTE ≥ 7 的最近兩個月到期（VIX 慣例：近月 < 7 天換下兩個），以總變異 σ²T 對時間線性內插/外推至 30 天。
    --weekly 會把週選也納入到期池。
用法
  python examples/78_txo_atm_iv.py            # 增量：只算 parquet 最後日期之後
  python examples/78_txo_atm_iv.py --full     # 全部重算
  python examples/78_txo_atm_iv.py --snap 13:30 --weekly
  from examples._rqo import load_atm_iv; iv = load_atm_iv()   # 回測取用
輸出：data/txo_atm_iv_daily.parquet、data/taifex_vix_official.parquet（增量附加新月份）、results/atm_iv.png
"""
import sys, argparse, io, math, urllib.request
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import norm
from scipy.optimize import brentq
sys.path.insert(0, str(Path(__file__).resolve().parents[1])); sys.path.insert(0, str(Path(__file__).resolve().parent))
from tools.optdaily import load_txo_daily, expiries
from _rqo import ROOT, RES, plt_style, sty, foot, GOLD, BLUE, WH

OUT = ROOT / "data/txo_atm_iv_daily.parquet"; VIXF = ROOT / "data/taifex_vix_official.parquet"
F1M = ROOT / "data/TXF_1c_1m_both_2017-06-01_2026-10-22.parquet"; FD = ROOT / "data/TXF_1c_D_regular_2010-01-01_2026-10-22.parquet"
R, MINVOL = 0.015, 1
VIXURL = "https://www.taifex.com.tw/file/taifex/Dailydownload/vix/log2data/{}new.txt"


def b76(F, K, T, s, cp):
    d1 = (math.log(F / K) + .5 * s * s * T) / (s * math.sqrt(T)); d2 = d1 - s * math.sqrt(T); D = math.exp(-R * T)
    return D * (F * norm.cdf(d1) - K * norm.cdf(d2)) if cp == "C" else D * (K * norm.cdf(-d2) - F * norm.cdf(-d1))


def iv(p, F, K, T, cp):
    intr = math.exp(-R * T) * max(F - K if cp == "C" else K - F, 0)
    if not (p > intr + 1e-9) or T <= 0: return np.nan
    try: return brentq(lambda s: b76(F, K, T, s, cp) - p, 1e-4, 5.0)
    except ValueError: return np.nan


def fut_snap(snap):
    f = pd.read_parquet(F1M, columns=["date", "session", "contractMonth", "close"]); f = f[f.session == "regular"]
    f["t"] = f.date.dt.tz_localize(None); hh, mm = map(int, snap.split(":"))
    f = f[(f.t.dt.hour * 60 + f.t.dt.minute) <= hh * 60 + mm]
    g = f.groupby(f.t.dt.normalize()).last(); g = g[["contractMonth", "close", "t"]].rename(columns={"close": "F", "t": "ftime"})
    d = pd.read_parquet(FD); d.index = d.date.dt.tz_localize(None).dt.normalize()
    d = d[["contractMonth", "close"]].rename(columns={"close": "F"}); d["ftime"] = pd.NaT
    d = d[d.index < g.index.min()]
    out = pd.concat([d, g]); out["contractMonth"] = out.contractMonth.astype(str); return out


def day_rows(dte, o, ex, fx, weekly):
    Fx, cm = float(fx.F), fx.contractMonth
    t0 = dte + pd.Timedelta(hours=13, minutes=45); per = []
    for X, ch in o.groupby("expiry"):
        if X not in ex.index or (ex.loc[X, "kind"] != "monthly" and not weekly): continue
        T = ((ex.loc[X, "expiry_date"] + pd.Timedelta(hours=13, minutes=30)) - t0).total_seconds() / 60 / (365 * 1440)
        if T <= 0: continue
        pv = ch[ch.day_volume >= MINVOL].pivot_table(index="strike", columns="cp", values="day_close")
        src, F = "txf", Fx
        if X[:6] != cm:
            src = "txf_mismatch"
            if {"C", "P"} <= set(pv.columns):
                b = pv.dropna(); b = b.iloc[(b.index.to_series() - Fx).abs().argsort()[:3]]
                if len(b): F = float(np.median(b.index + (b.C - b.P) * math.exp(R * T))); src = "parity"
        ks = pv.index.values; lo, hi = ks[ks <= F], ks[ks > F]
        if not len(lo) or not len(hi): continue
        res = {}
        for tag, K in (("lo", lo.max()), ("hi", hi.min())):
            r = pv.loc[K]
            ivc = iv(r.get("C", np.nan), F, K, T, "C"); ivp = iv(r.get("P", np.nan), F, K, T, "P")
            otm = ivp if K < F else ivc
            if np.isnan(otm): otm = ivc if K < F else ivp
            res[tag] = (K, ivc, ivp, otm)
        (K1, c1, p1, o1), (K2, c2, p2, o2) = res["lo"], res["hi"]; w = (F - K1) / (K2 - K1)
        L = lambda a, b: a + w * (b - a) if not (np.isnan(a) or np.isnan(b)) else np.nan
        per.append(dict(expiry=X, T=T, F=F, fwd_src=src, atm=L(o1, o2), atm_c=L(c1, c2), atm_p=L(p1, p2), gap=K2 - K1))
    return per


def cm30(per):
    m = [p for p in per if not np.isnan(p["atm"])]; m.sort(key=lambda p: p["T"])
    m = [p for p in m if p["T"] * 365 >= 7]
    if len(m) < 2: return None
    a, b = m[0], m[1]; T30 = 30 / 365
    v = a["atm"] ** 2 * a["T"] + (b["atm"] ** 2 * b["T"] - a["atm"] ** 2 * a["T"]) * (T30 - a["T"]) / (b["T"] - a["T"])
    return a, b, math.sqrt(max(v, 0) / T30), T30 < a["T"]


def build(start, snap, weekly):
    fut = fut_snap(snap); ex = expiries().set_index("expiry")
    o = load_txo_daily(start, None, columns=["date", "expiry", "strike", "cp", "day_close", "day_volume"])
    o["date"] = pd.to_datetime(o.date).dt.normalize(); o = o[o.day_close.notna()]
    rows = []
    for dte, od in o.groupby("date"):
        if dte not in fut.index: continue
        fx = fut.loc[dte]; c = cm30(day_rows(dte, od, ex, fx, weekly))
        if c is None: rows.append(dict(date=dte, F=fx.F, ok=False)); continue
        a, b, s30, extrap = c
        rows.append(dict(date=dte, F=float(fx.F), fut_month=fx.contractMonth, near_expiry=a["expiry"], next_expiry=b["expiry"],
                         near_dte=a["T"] * 365, next_dte=b["T"] * 365, F_near=a["F"], F_next=b["F"], fwd_src_near=a["fwd_src"], fwd_src_next=b["fwd_src"],
                         atm_iv_near=a["atm"], atm_iv_next=b["atm"], atm_iv_30d=s30, call_iv_near=a["atm_c"], put_iv_near=a["atm_p"],
                         call_iv_next=b["atm_c"], put_iv_next=b["atm_p"], cp_gap_near=a["atm_c"] - a["atm_p"], extrapolated=extrap,
                         inverted_term=b["atm"] < a["atm"], ok=True))
    return pd.DataFrame(rows)


def vix_update():
    old = pd.read_parquet(VIXF) if VIXF.exists() else pd.DataFrame(columns=["date", "vix", "vix_1m_avg"])
    have = set(pd.to_datetime(old.date).dt.strftime("%Y%m")) if len(old) else set()
    months = pd.period_range("2026-07", pd.Timestamp.today(), freq="M").strftime("%Y%m")
    new = []
    for m in months:
        if m in have and m != months[-1]: continue
        try: raw = urllib.request.urlopen(VIXURL.format(m), timeout=30).read().decode("big5", "replace")
        except Exception as e: print("vix", m, "fail", e); continue
        for ln in raw.splitlines():
            p = ln.split()
            if len(p) >= 4 and p[0].isdigit(): new.append(dict(date=pd.Timestamp(p[0]), vix=float(p[2]), vix_1m_avg=float(p[3])))
    v = pd.concat([x for x in (old, pd.DataFrame(new)) if len(x)]).drop_duplicates("date", keep="last").sort_values("date").reset_index(drop=True)
    v["date"] = pd.to_datetime(v.date).astype("datetime64[ms]")
    v.to_parquet(VIXF); return v


def chart(d, v):
    plt = plt_style(); fig, (a1, a2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True, gridspec_kw=dict(height_ratios=[2, 1]), facecolor="#111111")
    a1.plot(d.date, d.atm_iv_30d * 100, color=GOLD, lw=.8, label="TXO 30 天 ATM IV（Black-76）")
    a1.plot(v.date, v.vix, color=BLUE, lw=1.2, label="TAIFEX VIX 官方")
    a1.set_title("台指選擇權 30 天價平隱含波動率"); a1.legend(facecolor="#111111", labelcolor=WH); a1.set_ylabel("%")
    a2.plot(d.date, d.F, color=WH, lw=.7); a2.set_ylabel("TXF")
    for a in (a1, a2): sty(a)
    foot(fig); fig.tight_layout(); fig.savefig(RES / "atm_iv.png", dpi=130, facecolor="#111111")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--full", action="store_true"); ap.add_argument("--snap", default="13:44")
    ap.add_argument("--weekly", action="store_true"); a = ap.parse_args()
    old = None if a.full or not OUT.exists() else pd.read_parquet(OUT)
    start = "2017-01-01" if old is None else str((old.date.max() + pd.Timedelta(days=1)).date())
    new = build(start, a.snap, a.weekly); print("new rows", len(new))
    d = pd.concat([old, new]) if old is not None else new
    d["date"] = pd.to_datetime(d.date).astype("datetime64[ms]")
    d = d.drop_duplicates("date", keep="last").sort_values("date").reset_index(drop=True); d.to_parquet(OUT)
    v = vix_update(); chart(d[d.ok], v)
    m = d[d.ok].merge(v, on="date")
    if len(m):
        x = m.atm_iv_30d * 100
        print(f"VIX val n={len(m)} corr={x.corr(m.vix):.3f} mean(ATM-VIX)={(x - m.vix).mean():.2f} sd={(x - m.vix).std():.2f}")


if __name__ == "__main__":
    main()
