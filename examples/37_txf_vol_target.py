"""37 台指期（TXF）買進持有 + 波動率目標（日K）。
部位（名目/權益）L = 目標波動 / 已實現波動，上限 max_lev。已實現波動：過去 20 日、60 日標準差，或 EWMA（半衰期 20 日）。
收盤算訊號、隔日開盤成交。再平衡：每日檢查，偏離 > 門檻（10%/20%）才交易，或每週一次（週一開盤）。
整數口數：起始 1,000 萬元，以小台（MTX，每點 50 元）為單位；另報大台（TXF，每點 200 元）整數版與零股（連續）近似。
成本：手續費 50 元/口/邊、交易稅 2e-5、滑價 1 tick；換月時持有口數兩邊成本（與 TXF B&H 報酬的換月處理相同：換月日只算新合約開到收）。
現金（全部權益）賺 rf=1.5%/年；Sharpe 以超額報酬（扣 rf）計算，與原 TXF B&H（不含 rf、不含成本）定義一致。
期間：完整 2010-（暖身後）、IS 2014-2019、OOS 2020-2026/10、OOS 不含 2025-26。
"""
import itertools, math
from pathlib import Path
import numpy as np, pandas as pd

DATA = "data/TXF_1c_D_regular_2010-01-01_2026-10-22.parquet"
OUT = Path("results")
RF, TAX, COMM, SLIP = 0.015, 2e-5, 50.0, 1.0
EQ0 = 10_000_000
IS0, IS1, EX1, END = map(pd.Timestamp, ("2014-01-01", "2020-01-01", "2025-01-01", "2026-11-01"))
PERIODS = (("FULL", pd.Timestamp("2010-07-01"), END), ("IS", IS0, IS1), ("OOS", IS1, END), ("OOS_ex2526", IS1, EX1))


def load():
    d = pd.read_parquet(DATA)
    idx = d["date"].dt.tz_localize(None).dt.normalize()
    c, o = d.close.values.astype(float), d.open.values.astype(float)
    cm = d.contractMonth.astype(str)
    roll = (cm.ne(cm.shift()) & cm.shift().notna()).values
    pc = np.r_[np.nan, c[:-1]]
    r = np.where(roll, (c - o) / pc, (c - pc) / pc)            # 同 TXF B&H
    gap = np.where(roll, 0.0, (o - pc) / pc)                   # 開盤前（舊部位）
    intr = (c - o) / pc                                        # 開到收（新部位）
    return pd.DataFrame(dict(o=o, c=c, r=r, gap=gap, intr=intr, roll=roll), idx).iloc[1:]


def vol_est(r, kind):
    if kind == "sd20":
        v = r.rolling(20).std()
    elif kind == "sd60":
        v = r.rolling(60).std()
    else:
        v = np.sqrt((r ** 2).ewm(halflife=20, min_periods=20).mean())
    return v * math.sqrt(252)


def simulate(X, tv, lev, est="ewm20", reb="thr10", unit="MTX", slip=SLIP):
    mult = {"MTX": 50, "TXF": 200, "FRAC": 50}[unit]
    sig = (tv / vol_est(X.r, est)).clip(upper=lev).fillna(0.0).values
    o, c, gap, intr, roll = X.o.values, X.c.values, X.gap.values, X.intr.values, X.roll.values
    wk = X.index.to_series().dt.isocalendar().week.values
    n = len(X); eq = EQ0; q = 0.0; rets = np.zeros(n); levs = np.zeros(n); turn = cost = 0.0; ntr = 0
    thr = {"thr10": 0.10, "thr20": 0.20, "daily": 0.0, "weekly": None}[reb]
    rfd = RF / 252
    for t in range(n):
        e0 = eq
        pc = c[t - 1] if t else o[t]
        pnl = q * mult * pc * gap[t]                          # 舊部位：前收→開
        if roll[t] and q:
            cc = abs(q) * 2 * (COMM + mult * (TAX * o[t] + slip)); pnl -= cc; cost += cc / e0; turn += 2 * abs(q) * mult * o[t] / e0
        # 開盤依前一日收盤訊號調整
        if t:
            tgt = sig[t - 1] * (e0 + pnl) / (o[t] * mult)
            tgt = tgt if unit == "FRAC" else float(np.round(tgt))
            first_week_day = wk[t] != wk[t - 1]
            dev = abs(tgt - q) / max(abs(q), 1e-9) if q else (1.0 if tgt else 0.0)
            do = (first_week_day if thr is None else dev > thr) and tgt != q
            if do:
                dq = abs(tgt - q); cc = dq * (COMM + mult * (TAX * o[t] + slip))
                pnl -= cc; cost += cc / e0; turn += dq * mult * o[t] / e0; q = tgt; ntr += 1
        pnl += q * mult * pc * intr[t] + e0 * rfd             # 新部位：開→收；現金利息
        eq = e0 + pnl; rets[t] = pnl / e0; levs[t] = q * mult * c[t] / eq
        if eq <= 0:
            rets[t + 1:] = 0; break
    return dict(r=pd.Series(rets, X.index), lev=pd.Series(levs, X.index), turn=turn, cost=cost, ntr=ntr)


def stats(r, lev=None):
    if len(r) < 30:
        return {}
    e = (1 + r).cumprod(); y = len(r) / 252; x = r - RF / 252
    out = dict(CAGR=e.iloc[-1] ** (1 / y) - 1, vol=r.std() * math.sqrt(252), sharpe=x.mean() / x.std() * math.sqrt(252), MDD=(e / e.cummax() - 1).min())
    if lev is not None:
        out["avg_lev"] = lev.mean()
    return out


def win(s, lo, hi):
    return s[(s.index >= lo) & (s.index < hi)]


def rows_for(name, o, X):
    out = []
    for per, lo, hi in PERIODS:
        rr = win(o["r"], lo, hi); y = len(rr) / 252
        st = stats(rr, win(o["lev"], lo, hi) if "lev" in o else None)
        # 期間內的成本/換手：以整段年化近似
        out.append(dict(variant=name, period=per, **st, turnover_yr=o.get("turn", np.nan) / (len(o["r"]) / 252),
                        cost_pct_yr=o.get("cost", np.nan) / (len(o["r"]) / 252) * 100, trades_yr=o.get("ntr", np.nan) / (len(o["r"]) / 252)))
    return out


def bh(X):
    r = X.r + RF / 252
    return dict(r=r, lev=pd.Series(1.0, X.index))


def bh_scaled(X, target_r):
    # 事後把 B&H 縮放到與策略相同的已實現波動（各期間分別縮放）→ 用來公平比較報酬
    out = {}
    for per, lo, hi in PERIODS:
        b = win(X.r, lo, hi); s = win(target_r, lo, hi)
        k = s.std() / b.std(); out[per] = (k, stats(k * b + RF / 252))
    return out


def main():
    X = load(); X = X[X.index >= pd.Timestamp("2010-01-01")]
    pd.set_option("display.width", 250)
    grid = []
    for est, reb, tv, lev in itertools.product(("sd20", "sd60", "ewm20"), ("thr10", "thr20", "weekly"), (0.10, 0.15, 0.20, 0.25), (1, 1.5, 2, 3)):
        o = simulate(X, tv, lev, est, reb)
        for row in rows_for(f"{est} {reb} tv{int(tv*100)} L{lev}", o, X):
            grid.append(dict(est=est, reb=reb, tv=tv, lev=lev, **row))
    G = pd.DataFrame(grid); G.to_csv(OUT / "vt_grid.csv", index=False)
    for est in ("sd20", "sd60", "ewm20"):
        for per in ("IS", "OOS"):
            print(est, "thr10", per, "sharpe"); print(G[(G.est == est) & (G.reb == "thr10") & (G.period == per)].pivot_table(index="tv", columns="lev", values="sharpe").round(3).to_string())
    print("IS sharpe by est x reb (avg over tv/lev grid):"); print(G[G.period == "IS"].pivot_table(index="est", columns="reb", values="sharpe").round(3).to_string())
    print("OOS sharpe by est x reb:"); print(G[G.period == "OOS"].pivot_table(index="est", columns="reb", values="sharpe").round(3).to_string())
    return X, G


def report(X, G, H):
    est, reb, tv, lev = H
    V = {}
    V["TXF B&H (+rf, 無成本)"] = bh(X)
    V[f"HEADLINE {est} {reb} tv{int(tv*100)} L{lev} MTX"] = simulate(X, tv, lev, est, reb)
    V[f"same, TXF 大台整數"] = simulate(X, tv, lev, est, reb, unit="TXF")
    V[f"same, 連續（零股近似）"] = simulate(X, tv, lev, est, reb, unit="FRAC")
    V[f"same, 每日再平衡"] = simulate(X, tv, lev, est, reb="daily")
    V[f"same, 每週再平衡"] = simulate(X, tv, lev, est, reb="weekly")
    V[f"same, 0.5 tick"] = simulate(X, tv, lev, est, reb, slip=0.5)
    for e2 in ("sd20", "sd60", "ewm20"):
        if e2 != est:
            V[f"same, vol={e2}"] = simulate(X, tv, lev, e2, reb)
    rows = []
    for k, o in V.items():
        rows += rows_for(k, o, X)
    hl = V[f"HEADLINE {est} {reb} tv{int(tv*100)} L{lev} MTX"]
    for per, (k, st) in bh_scaled(X, hl["r"]).items():
        rows.append(dict(variant=f"B&H 縮放至同波動 (x{k:.2f})", period=per, **st, avg_lev=k))
    T = pd.DataFrame(rows); T.to_csv(OUT / "vt_summary.csv", index=False)
    print(T.pivot_table(index="variant", columns="period", values=["CAGR", "sharpe", "MDD", "vol", "avg_lev"], sort=False).round(3).to_string())
    print(T[T.period == "FULL"][["variant", "turnover_yr", "cost_pct_yr", "trades_yr"]].round(2).to_string(index=False))
    # 壞月份
    mon = {k: (1 + o["r"]).groupby(o["r"].index.to_period("M")).prod() - 1 for k, o in V.items() if k.startswith(("TXF", "HEADLINE"))}
    yr = {k: (1 + o["r"]).groupby(o["r"].index.year).prod() - 1 for k, o in V.items() if k.startswith(("TXF", "HEADLINE"))}
    W = []
    for lab, sel in (("2015 年", ("Y", 2015)), ("2018 年", ("Y", 2018)), ("2020/03", ("M", "2020-03")), ("2022 年", ("Y", 2022)), ("2025/04", ("M", "2025-04")),
                     ("2018/10", ("M", "2018-10")), ("2015/08", ("M", "2015-08"))):
        row = dict(window=lab)
        for k in mon:
            row[k[:12]] = yr[k][sel[1]] if sel[0] == "Y" else mon[k][pd.Period(sel[1])]
        W.append(row)
    for k in mon:
        m = mon[k][mon[k].index >= pd.Period("2010-07")]
        W.append(dict(window=f"worst month {k[:12]}", **{kk[:12]: np.nan for kk in mon}, note=f"{m.idxmin()} {m.min():.3f}"))
    Wd = pd.DataFrame(W); Wd.to_csv(OUT / "vt_worst.csv", index=False); print(Wd.round(3).to_string(index=False))
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, axs = plt.subplots(3, 1, figsize=(12, 11), sharex=True, gridspec_kw=dict(height_ratios=[3, 1.5, 1]))
    sc = bh_scaled(X, hl["r"])["FULL"][0]
    curves = {"TXF B&H": V["TXF B&H (+rf, 無成本)"]["r"], "Vol target (headline)": hl["r"], f"B&H x{sc:.2f} (same vol)": sc * X.r + RF / 252}
    for k, r in curves.items():
        r = win(r, pd.Timestamp("2010-07-01"), END); e = (1 + r).cumprod()
        axs[0].plot(e.index, e, label=k, lw=1.2); axs[1].plot(e.index, e / e.cummax() - 1, lw=1)
    axs[2].plot(hl["lev"].index, hl["lev"], c="tab:orange", lw=0.8); axs[2].set_ylabel("leverage")
    axs[0].set_yscale("log"); axs[0].legend(fontsize=9); axs[1].set_ylabel("drawdown")
    axs[0].set_title(f"TXF vol target {est} {reb} tv{int(tv*100)}% maxL{lev} (MTX units, 10M start)")
    for a in axs:
        a.axvline(IS0, c="gray", lw=0.6); a.axvline(IS1, c="gray", lw=0.8)
    plt.tight_layout(); plt.savefig(OUT / "vt_equity_dd.png", dpi=110); plt.close()


if __name__ == "__main__":
    import sys
    X, G = main()
    # headline 依 IS 平台：sd60 + thr20（IS 平均 Sharpe 最高的估計法/再平衡組），tv15% / maxL1.5 位於平台中段（非 OOS 最佳）
    e, rb, tv, lv = sys.argv[1:5] if len(sys.argv) > 4 else ("sd60", "thr20", "0.15", "1.5")
    if True:
        report(X, G, (e, rb, float(tv), float(lv) if "." in lv else int(lv)))
