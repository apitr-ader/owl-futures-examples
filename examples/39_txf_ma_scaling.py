"""39 台指期均線加減碼（38 對照組 e 的正式回測）。
規則：收盤 > MA → L_high 倍，否則 L_low 倍；收盤訊號、隔日開盤成交、偏離 20% 才調整（與 38 同一模擬器：
小台整數口、1,000 萬、rf 1.5%、手續費 50/邊、稅 2e-5、滑價 1 tick、策略與 B&H 都計換月成本）。
價格用換月調整後的連續序列（累乘 TXF 日報酬）。MA：SMA 20/50/100/150/200/250（headline 另測 EMA）。
緩衝（遲滯）：突破需超過 MA·(1±b)（b=1%/2%）或連續 n 日（2/3 日）確認才切換。
多重檢定：報告網格試驗數並計算 Deflated Sharpe Ratio（Bailey & López de Prado 2014）。
"""
import importlib.util, itertools, math
from pathlib import Path
import numpy as np, pandas as pd
from statistics import NormalDist


class norm:
    cdf = staticmethod(NormalDist().cdf); ppf = staticmethod(NormalDist().inv_cdf)

_s = importlib.util.spec_from_file_location("ivt38", Path(__file__).with_name("38_txf_inverse_vol.py"))
m38 = importlib.util.module_from_spec(_s); _s.loader.exec_module(m38)
m37 = m38.m37
RF, PERIODS, OUT, stats, win, sim = m38.RF, m38.PERIODS, m38.OUT, m38.stats, m38.win, m38.sim
MAS, LH, LL = (20, 50, 100, 150, 200, 250), (1, 1.25, 1.5, 2), (0, 0.5, 0.75, 1)


def state(X, ma, kind="SMA", band=0.0, confirm=1):
    px = (1 + X.r).cumprod()
    m = px.rolling(ma).mean() if kind == "SMA" else px.ewm(span=ma, adjust=False, min_periods=ma).mean()
    up = (px > m * (1 + band)).values; dn = (px < m * (1 - band)).values
    if confirm > 1:
        up = pd.Series(up).rolling(confirm).sum().eq(confirm).values; dn = pd.Series(dn).rolling(confirm).sum().eq(confirm).values
    s = np.full(len(X), np.nan); cur = np.nan
    for t in range(len(X)):
        if np.isnan(m.iloc[t]):
            continue
        if up[t]:
            cur = 1.0
        elif dn[t]:
            cur = 0.0
        elif np.isnan(cur):
            cur = float(px.iloc[t] > m.iloc[t])
        s[t] = cur
    return s


def run(X, st, lh, ll):
    sig = np.where(np.isnan(st), 1.0, np.where(st == 1, lh, ll))
    o = sim(X, sig)
    sw = np.nansum(np.abs(np.diff(st)) > 0)
    # whipsaw：切換後 20 個交易日內又切回
    idx = np.where(np.abs(np.diff(np.nan_to_num(st, nan=-1))) > 0)[0]
    ws = int(np.sum(np.diff(idx) <= 20)) if len(idx) > 1 else 0
    o.update(switches=sw, whipsaw=ws, nontrade_cost=o["cost"] - o["rcost"])
    return o


def rows_for(name, o, extra=None):
    y = len(o["r"]) / 252
    out = []
    for per, lo, hi in PERIODS:
        rr = win(o["r"], lo, hi)
        out.append(dict(variant=name, period=per, **stats(rr, win(o["lev"], lo, hi)), **(extra or {}),
                        switches_yr=o.get("switches", 0) / y, whipsaw_yr=o.get("whipsaw", 0) / y,
                        cost_pct_yr=o["cost"] / y * 100, roll_cost_pct_yr=o["rcost"] / y * 100, trade_cost_pct_yr=(o["cost"] - o["rcost"]) / y * 100))
    return out


def dsr(r, srs, n_trials):
    """r: 日報酬（超額）；srs: 所有試驗的日 Sharpe（未年化）。"""
    x = r - RF / 252; sr = x.mean() / x.std(); T = len(x)
    g = 0.5772156649; V = np.var(srs, ddof=1)
    sr0 = math.sqrt(V) * ((1 - g) * norm.ppf(1 - 1 / n_trials) + g * norm.ppf(1 - 1 / (n_trials * math.e)))
    sk, ku = float(x.skew()), float(x.kurt() + 3)
    z = (sr - sr0) * math.sqrt(T - 1) / math.sqrt(1 - sk * sr + (ku - 1) / 4 * sr ** 2)
    return dict(SR_ann=sr * math.sqrt(252), SR0_ann=sr0 * math.sqrt(252), DSR=norm.cdf(z), PSR0=norm.cdf(sr * math.sqrt(T - 1) / math.sqrt(1 - sk * sr + (ku - 1) / 4 * sr ** 2)))


HEAD = None   # 由 IS 平台（鄰格平均 IS Sharpe 最高）自動決定；2026-10 執行結果為 SMA250 H1 L0（位於網格邊緣）


def main():
    X = m37.load(); X = X[X.index >= pd.Timestamp("2010-01-01")]
    pd.set_option("display.width", 260)
    ST = {ma: state(X, ma) for ma in MAS}
    grid, rets = [], {}
    for ma, lh, ll in itertools.product(MAS, LH, LL):
        if lh == 1 and ll == 1:
            continue
        o = run(X, ST[ma], lh, ll); rets[(ma, lh, ll)] = o["r"]
        grid += rows_for(f"SMA{ma} H{lh} L{ll}", o, dict(ma=ma, lh=lh, ll=ll))
    bh = sim(X, np.ones(len(X))); bh.update(switches=0, whipsaw=0)
    G = pd.DataFrame(grid); G.to_csv(OUT / "ma_grid.csv", index=False)
    bhs = {r["period"]: r for r in rows_for("B&H", bh)}
    for per in ("IS", "OOS", "OOS_ex2526"):
        print(f"{per} Sharpe (B&H {bhs[per]['sharpe']:.3f}); rows=MA, cols=(L_high,L_low)")
        print(G[G.period == per].pivot_table(index="ma", columns=["lh", "ll"], values="sharpe").round(2).to_string())
    # IS 平台：每格與其相鄰格（MA ±1 檔、L_high ±1、L_low ±1）IS Sharpe 平均
    P = G[G.period == "IS"].set_index(["ma", "lh", "ll"]).sharpe
    def nb(ma, lh, ll):
        i, j, k = MAS.index(ma), LH.index(lh), LL.index(ll); v = []
        for di, dj, dk in itertools.product((-1, 0, 1), repeat=3):
            a, b, c = i + di, j + dj, k + dk
            if 0 <= a < len(MAS) and 0 <= b < len(LH) and 0 <= c < len(LL) and (MAS[a], LH[b], LL[c]) in P.index:
                v.append(P[(MAS[a], LH[b], LL[c])])
        return np.mean(v)
    pl = pd.Series({k: nb(*k) for k in P.index}).sort_values(ascending=False)
    print("IS plateau (neighbour-avg Sharpe) top 8:"); print(pl.head(8).round(3).to_string())
    ma, lh, ll = HEAD or pl.index[0]
    print("HEADLINE", ma, lh, ll)
    # DSR
    n = len(rets)
    for per, lo, hi in (("IS", m37.IS0, m37.IS1), ("FULL", pd.Timestamp("2010-07-01"), m37.END)):
        srs = [(lambda x: x.mean() / x.std())(win(r, lo, hi) - RF / 252) for r in rets.values()]
        d = dsr(win(rets[(ma, lh, ll)], lo, hi), srs, n); print(per, "DSR", {k: round(v, 3) for k, v in d.items()}, "trials", n)
        pd.Series(d).to_csv(OUT / f"ma_dsr_{per}.csv")
    # 變體
    V = {"B&H 1x": bh, f"HEADLINE SMA{ma} H{lh} L{ll}": run(X, ST[ma], lh, ll),
         f"EMA{ma} H{lh} L{ll}": run(X, state(X, ma, "EMA"), lh, ll)}
    for b in (0.01, 0.02):
        V[f"SMA{ma} band {int(b*100)}%"] = run(X, state(X, ma, band=b), lh, ll)
    for c in (2, 3):
        V[f"SMA{ma} confirm {c}d"] = run(X, state(X, ma, confirm=c), lh, ll)
    V["SMA200 H1.5 L1 (38 e)"] = run(X, ST[200], 1.5, 1)
    hl = V[f"HEADLINE SMA{ma} H{lh} L{ll}"]
    rows = []
    for k, o in V.items():
        rows += rows_for(k, o)
    b = bh["r"] - RF / 252
    for per, lo, hi in PERIODS:
        bb, ss, lv = win(b, lo, hi), win(hl["r"], lo, hi), win(hl["lev"], lo, hi).mean() / win(bh["lev"], lo, hi).mean()
        kv = (ss - RF / 252).std() / bb.std()
        rows.append(dict(variant="B&H 縮放至同波動", period=per, scale=kv, **stats(kv * bb + RF / 252)))
        rows.append(dict(variant="B&H 縮放至同平均槓桿", period=per, scale=lv, **stats(lv * bb + RF / 252)))
    T = pd.DataFrame(rows); T.to_csv(OUT / "ma_summary.csv", index=False)
    print(T.pivot_table(index="variant", columns="period", values=["CAGR", "sharpe", "MDD", "vol", "avg_lev"], sort=False).round(3).to_string())
    print(T[T.period == "FULL"][["variant", "switches_yr", "whipsaw_yr", "cost_pct_yr", "roll_cost_pct_yr", "trade_cost_pct_yr", "scale"]].round(3).to_string(index=False))
    print(T[T.variant.str.startswith("B&H 縮放")][["variant", "period", "scale"]].round(3).to_string(index=False))
    # 逐年
    Y = pd.DataFrame({k: (1 + V[k]["r"]).groupby(V[k]["r"].index.year).prod() - 1 for k in ("B&H 1x", f"HEADLINE SMA{ma} H{lh} L{ll}")})
    sc = T[(T.variant == "B&H 縮放至同平均槓桿") & (T.period == "FULL")].scale.iloc[0]
    Y["B&H x同平均槓桿"] = (1 + sc * b + RF / 252).groupby(b.index.year).prod() - 1
    Y["diff vs scaled"] = Y.iloc[:, 1] - Y.iloc[:, 2]; Y = Y[Y.index >= 2011]
    Y.to_csv(OUT / "ma_yearly.csv"); print(Y.round(3).to_string()); print("years beating scaled B&H:", int((Y["diff vs scaled"] > 0).sum()), "/", len(Y))
    W = []
    for lab, kd, key in (("2015", "Y", 2015), ("2015/08", "M", "2015-08"), ("2018/10", "M", "2018-10"), ("2020/03", "M", "2020-03"), ("2022", "Y", 2022), ("2025/04", "M", "2025-04")):
        row = dict(window=lab)
        for k in ("B&H 1x", f"HEADLINE SMA{ma} H{lh} L{ll}"):
            r = V[k]["r"]; g = (1 + r).groupby(r.index.year if kd == "Y" else r.index.to_period("M")).prod() - 1
            row[k] = g[key] if kd == "Y" else g[pd.Period(key)]
        W.append(row)
    Wd = pd.DataFrame(W); Wd.to_csv(OUT / "ma_bad_periods.csv", index=False); print(Wd.round(3).to_string(index=False))
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, axs = plt.subplots(3, 1, figsize=(12, 11), sharex=True, gridspec_kw=dict(height_ratios=[3, 1.5, 1]))
    curves = {"B&H 1x": bh["r"], f"SMA{ma} H{lh} L{ll}": hl["r"], f"B&H x{sc:.2f} (same avg lev)": sc * b + RF / 252}
    for k, r in curves.items():
        r = win(r, pd.Timestamp("2010-07-01"), m37.END); e = (1 + r).cumprod()
        axs[0].plot(e.index, e, label=k, lw=1.2); axs[1].plot(e.index, e / e.cummax() - 1, lw=1)
    axs[2].plot(hl["lev"].index, hl["lev"], c="tab:orange", lw=0.8); axs[2].set_ylabel("leverage")
    axs[0].set_yscale("log"); axs[0].legend(fontsize=9); axs[1].set_ylabel("drawdown")
    for a in axs:
        a.axvline(m37.IS0, c="gray", lw=0.6); a.axvline(m37.IS1, c="gray", lw=0.8)
    plt.tight_layout(); plt.savefig(OUT / "ma_equity_dd.png", dpi=110); plt.close()


if __name__ == "__main__":
    main()
