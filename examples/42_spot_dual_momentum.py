"""42 現股雙動能（上市櫃普通股日K，含已下市，total-return 還原）。

訊號（同 27 pick 精神）：複合動能 = 21/63/126 日報酬 / (同期日報酬標準差*sqrt(L)) 的平均（波動調整）；
絕對動能：21/63/126 平均報酬 > rf(1.5%/年) 同期報酬；排名取前 10 檔、等權（各 1/10，不足則持現金）。
可選池（point-in-time）：當日有成交、>= 127 日歷史、20 日平均成交金額 >= LIQ（主設定 5,000 萬）。
每月最後交易日收盤換股；組合 60 日共變異（Ledoit-Wolf）估波動，目標 15%，縮放上限 1.0（現股不加槓桿）。
大盤濾網：(a) cmp = TXF 21/63/126 平均報酬 > 0；(b) ma250 = TXF 收盤 > 250 日均線；(c) none。濾網關 → 全現金（現金 0 報酬）。
TXF 日K 自 2010 起，濾網尚未可算的期間視為關（cmp 約 2010-07、ma250 約 2011-01 起才有效）。
成本：手續費 0.1425%/邊、證交稅 0.3%（賣）、滑價 1 tick/邊（台股升降單位，以未還原價計）。
下市：最後交易日後報酬記 0，下次換股時以最後價格出清（含成本）。
樣本：IS 2010-2019，OOS 2020-01 ~ 最新，OOS 除 2025-26。DSR：以 3 濾網 × 3 流動性門檻 = 9 試驗的 OOS 日 Sharpe 變異。
對照：41（重建股票期貨 27 pick，讀 results/rb_summary.csv）、0050（Yahoo adjclose，data/bench/0050.json）、TXF B&H、39 TXF SMA250。
"""
import importlib.util, json, math, sys
from pathlib import Path
import numpy as np, pandas as pd
from statistics import NormalDist

ROOT = Path("/workspace/fubon-threads")
sys.path.insert(0, str(ROOT / "tools"))
import stockdata as sd  # noqa: E402

_s = importlib.util.spec_from_file_location("ma39", Path(__file__).with_name("39_txf_ma_scaling.py"))
m39 = importlib.util.module_from_spec(_s); _s.loader.exec_module(m39)
OUT = ROOT / "results"
RF, N, VT, LEGS = 0.015, 10, 0.15, (21, 63, 126)
COMM, TAX = 0.001425, 0.003
LIQS = {"20M": 2e7, "50M": 5e7, "100M": 1e8}; LIQ0 = "50M"
IS0, IS1, EX1 = pd.Timestamp("2010-01-01"), pd.Timestamp("2020-01-01"), pd.Timestamp("2025-01-01")
PER = (("IS", IS0, IS1), ("OOS", IS1, None), ("OOS_ex2526", IS1, EX1))


def tick(p):
    return np.select([p < 10, p < 50, p < 100, p < 500, p < 1000], [0.01, 0.05, 0.1, 0.5, 1.0], 5.0)


def stats(r):
    if len(r) < 30 or r.std() == 0:
        return dict(CAGR=0.0, vol=0.0, sharpe=0.0, MDD=0.0)
    e = (1 + r).cumprod(); y = len(r) / 252
    return dict(CAGR=e.iloc[-1] ** (1 / y) - 1, vol=r.std() * math.sqrt(252), sharpe=r.mean() / r.std() * math.sqrt(252),
                MDD=(e / e.cummax() - 1).min())


def win(r, lo, hi):
    return r[(r.index >= lo) & ((r.index < hi) if hi is not None else True)]


def dsr(r, srs, n):
    x = r - RF / 252; sr = x.mean() / x.std(); T = len(x); nd = NormalDist(); g = 0.5772156649
    sr0 = math.sqrt(np.var(srs, ddof=1)) * ((1 - g) * nd.inv_cdf(1 - 1 / n) + g * nd.inv_cdf(1 - 1 / (n * math.e)))
    sk, ku = float(x.skew()), float(x.kurt() + 3)
    return nd.cdf((sr - sr0) * math.sqrt(T - 1) / math.sqrt(1 - sk * sr + (ku - 1) / 4 * sr ** 2))


def lw_cov(X):
    X = X - X.mean(0); n, p = X.shape; S = X.T @ X / n
    mu = np.trace(S) / p; F = mu * np.eye(p)
    d2 = ((S - F) ** 2).sum(); b2 = min(sum(((np.outer(x, x) - S) ** 2).sum() for x in X) / n ** 2, d2)
    a = b2 / d2 if d2 > 0 else 1.0
    return a * F + (1 - a) * S


def load():
    adj = sd.panel("close", adjusted=True, start="2008-01-01")
    raw = sd.panel("close", adjusted=False, start="2008-01-01").reindex_like(adj)
    tov = sd.panel("turnover", adjusted=False, start="2008-01-01").reindex_like(adj)
    X = m39.m37.load()
    tI = pd.Series((1 + X["r"]).cumprod().values, X.index) if "r" in X else None
    return adj, raw, tov, X, tI


def market_signals(tI, days):
    cmp = np.mean([(tI / tI.shift(L) - 1) for L in LEGS], 0)
    cmp = pd.Series(cmp, tI.index) > 0
    ma = tI.rolling(250).mean(); ma250 = (tI > ma) & ma.notna()
    f = lambda s: s.reindex(days).ffill().fillna(False).astype(bool).values
    return {"cmp": f(cmp & tI.shift(126).notna()), "ma250": f(ma250), "none": np.ones(len(days), bool)}


def simulate(adj, raw, tov, sig, liq):
    days = adj.index; A = adj.values; P = raw.values
    R = np.vstack([np.full(A.shape[1], np.nan), A[1:] / A[:-1] - 1])
    traded = ~np.isnan(P)
    R0 = np.nan_to_num(R)
    nhist = np.cumsum(traded, 0)
    t20 = pd.DataFrame(np.where(traded, tov.values, 0.0)).rolling(20).mean().values
    me = np.r_[days[1:].month != days[:-1].month, True]
    lastpx = pd.DataFrame(P).ffill().values
    thr = (1 + RF) ** (np.mean(LEGS) / 252) - 1
    w = np.zeros(A.shape[1]); out = np.zeros(len(days)); turn = cost = 0.0
    for t in range(len(days)):
        if t > 0:
            pr = w @ R0[t]; out[t] = pr
            g = 1 + pr
            w = w * (1 + R0[t]) / g if g > 0 else w * 0
        if not me[t] or t < 130 or days[t] < IS0 - pd.Timedelta(days=40):
            continue
        tgt = np.zeros_like(w)
        if sig[t]:
            ok = np.where(traded[t] & (nhist[t] >= 127) & (t20[t] >= liq))[0]
            raw_m = np.zeros(len(ok)); vs = np.zeros(len(ok))
            for L in LEGS:
                rL = A[t, ok] / A[t - L, ok] - 1
                v = np.nanstd(R[t - L + 1:t + 1, ok], 0) * math.sqrt(L)
                raw_m += rL; vs += rL / np.maximum(v, 1e-6)
            raw_m /= 3; vs /= 3
            good = np.isfinite(vs) & np.isfinite(raw_m) & (raw_m > thr)
            cand = ok[good][np.argsort(-vs[good])][:N]
            if len(cand):
                tgt[cand] = 1.0 / N
                C = lw_cov(np.nan_to_num(R[t - 59:t + 1, cand]))
                pv = math.sqrt(max(tgt[cand] @ C @ tgt[cand], 1e-12) * 252)
                tgt[cand] *= min(VT / pv, 1.0)
        d = tgt - w
        px = np.where(np.isnan(lastpx[t]), 1.0, lastpx[t])
        slip = tick(px) / px
        c = (np.clip(d, 0, None) * (COMM + slip)).sum() + (np.clip(-d, 0, None) * (COMM + TAX + slip)).sum()
        out[t] -= c; cost += c; turn += np.abs(d).sum() / 2; w = tgt
    r = pd.Series(out, days); r = r[r.index >= IS0]
    return r, turn, cost


def bench_0050():
    d = json.load(open(ROOT / "data/bench/0050.json"))["chart"]["result"][0]
    s = pd.Series(d["indicators"]["adjclose"][0]["adjclose"], pd.to_datetime(d["timestamp"], unit="s") + pd.Timedelta(hours=8)).dropna()
    s.index = s.index.normalize(); s = s[~s.index.duplicated(keep="last")]
    r = s.pct_change().dropna()
    # 2025-06 1拆4 若 adjclose 未處理會出現 -75% 跳空：保險起見剔除 |r|>40%
    return r[r.abs() < 0.4]


def main():
    pd.set_option("display.width", 250)
    adj, raw, tov, X, _ = load()
    tI = pd.Series((1 + m39.sim(X, np.ones(len(X)))["r"]).cumprod())
    sigs = market_signals(tI, adj.index)
    curves, rows, srs = {}, [], []
    for f in ("cmp", "ma250", "none"):
        for lk, lv in LIQS.items():
            r, turn, cost = simulate(adj, raw, tov, sigs[f], lv)
            yrs = len(r) / 252
            srs.append(win(r, IS1, None).pipe(lambda x: (x - RF / 252).mean() / (x - RF / 252).std()))
            key = f"spot {f} liq{lk}"; curves[key] = r
            for p, lo, hi in PER:
                rows.append(dict(name=key, period=p, **stats(win(r, lo, hi)), turnover_yr=turn / yrs, cost_pct_yr=cost / yrs * 100,
                                 expo=float((r != 0).mean())))
            print(key, "done", flush=True)
    for k in [k for k in curves if k.endswith(LIQ0)]:
        for row in rows:
            if row["name"] == k and row["period"] == "OOS":
                row["DSR"] = dsr(win(curves[k], IS1, None), srs, len(srs))
    end = adj.index[-1]
    refs = {"0050 B&H": bench_0050(), "TXF B&H": m39.sim(X, np.ones(len(X)))["r"],
            "39 TXF SMA250": m39.run(X, m39.state(X, 250), 1, 0)["r"]}
    for k, r in refs.items():
        r = win(r, IS0, end + pd.Timedelta(days=1)); curves[k] = r
        for p, lo, hi in PER:
            rows.append(dict(name=k, period=p, **stats(win(r, lo, hi))))
    rb = pd.read_csv(OUT / "rb_summary.csv")
    rb = rb[(rb.slip == 1.0) & (rb.start == "IS2010") & rb.data.str.startswith("new")]
    for x in rb.itertuples():
        rows.append(dict(name=f"41 rebuilt SF {x.filter} (to 2026-08-18)", period=x.period, CAGR=x.CAGR, sharpe=x.sharpe, MDD=x.MDD))
    T = pd.DataFrame(rows); T.to_csv(OUT / "spot_summary.csv", index=False)
    pt = T.pivot_table(index="name", columns="period", values=["CAGR", "sharpe", "MDD"], sort=False)
    print(pt.round(3).to_string())
    print(T[T.period == "OOS"][["name", "turnover_yr", "cost_pct_yr", "expo", "DSR"]].dropna(subset=["turnover_yr"]).round(3).to_string(index=False))
    print("data end", end.date())
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, axs = plt.subplots(2, 1, figsize=(13, 9), sharex=True)
    for k, r in curves.items():
        if "spot" in k and not k.endswith(LIQ0):
            continue
        e = (1 + r).cumprod(); st = dict(c="k", lw=2) if "TXF B&H" in k else dict(c="gray", lw=1.5) if "SMA" in k else dict(lw=1.2)
        axs[0].plot(e.index, e, label=k, **st); axs[1].plot(e.index, e / e.cummax() - 1, **st)
    axs[0].set_yscale("log"); axs[0].legend(fontsize=8); axs[1].set_ylabel("drawdown")
    axs[0].set_title("42 spot dual momentum (10 names, vt15%, liq 50M, full TW costs) vs benchmarks")
    for a in axs:
        a.axvline(IS1, c="gray", lw=0.8); a.axvline(EX1, c="gray", lw=0.6, ls=":")
    plt.tight_layout(); plt.savefig(OUT / "spot_equity_dd.png", dpi=110); plt.close()


if __name__ == "__main__":
    main()
