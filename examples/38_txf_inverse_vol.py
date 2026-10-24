"""38 台指期「反向」波動率部位：波動上升加碼、下降減碼（37 的反向）。日K、收盤訊號、隔日開盤成交、偏離 20% 才調整。
小台整數口、起始 1,000 萬、rf 1.5%、手續費 50/邊、稅 2e-5、滑價 1 tick；換月成本「策略與 B&H 都計」（B&H = 固定 1 倍、同一模擬器）。
變體（v = 過去 lb 日已實現波動，lb=20/60）：
  a  ratio  : L = v / 過去 3 年（至少 1 年）v 的中位數，限制於 [0, maxL]
  b  pctile : L = 0.5 + rank·(maxL-0.5)，rank = v 在過去 2 年的百分位
  c  chg    : L = v / v120（短波動/120 日波動），限制於 [0, maxL]
  d  trend  : 波動上升（v/v120 > 1）且收盤 > 200 日均線 → maxL，否則 1 倍
  e  trend-only 對照：收盤 > 200 日均線 → maxL，否則 1 倍
期間：FULL 2010.7-、IS 2014-2019、OOS 2020-2026/10、OOS 不含 2025-26。
"""
import importlib.util, itertools, math
from pathlib import Path
import numpy as np, pandas as pd

_s = importlib.util.spec_from_file_location("vt37", Path(__file__).with_name("37_txf_vol_target.py"))
m37 = importlib.util.module_from_spec(_s); _s.loader.exec_module(m37)
RF, TAX, COMM, SLIP, EQ0, PERIODS, OUT = m37.RF, m37.TAX, m37.COMM, m37.SLIP, m37.EQ0, m37.PERIODS, m37.OUT
stats, win = m37.stats, m37.win


def sim(X, sig, thr=0.20, mult=50, slip=SLIP):
    """sig: 每日收盤決定的目標倍數（隔日開盤執行）。與 37 的 simulate 同一邏輯，訊號外部給定。"""
    sig = np.nan_to_num(np.asarray(sig, float), nan=0.0)
    o, c, gap, intr, roll = X.o.values, X.c.values, X.gap.values, X.intr.values, X.roll.values
    n = len(X); eq = EQ0; q = 0.0; rets = np.zeros(n); levs = np.zeros(n); cost = turn = 0.0; ntr = 0; rcost = 0.0
    for t in range(n):
        e0 = eq; pc = c[t - 1] if t else o[t]
        pnl = q * mult * pc * gap[t]
        if roll[t] and q:
            cc = abs(q) * 2 * (COMM + mult * (TAX * o[t] + slip)); pnl -= cc; cost += cc / e0; rcost += cc / e0; turn += 2 * abs(q) * mult * o[t] / e0
        if t:
            tgt = float(np.round(sig[t - 1] * (e0 + pnl) / (o[t] * mult)))
            dev = abs(tgt - q) / abs(q) if q else (1.0 if tgt else 0.0)
            if dev > thr and tgt != q:
                dq = abs(tgt - q); cc = dq * (COMM + mult * (TAX * o[t] + slip))
                pnl -= cc; cost += cc / e0; turn += dq * mult * o[t] / e0; q = tgt; ntr += 1
        pnl += q * mult * pc * intr[t] + e0 * RF / 252
        eq = e0 + pnl; rets[t] = pnl / e0; levs[t] = q * mult * c[t] / eq
    return dict(r=pd.Series(rets, X.index), lev=pd.Series(levs, X.index), turn=turn, cost=cost, rcost=rcost, ntr=ntr)


def signals(X, kind, lb, L):
    r = X.r; v = r.rolling(lb).std() * math.sqrt(252); v120 = r.rolling(120).std() * math.sqrt(252)
    px = (1 + r).cumprod(); ma = px.rolling(200).mean()
    if kind == "a_ratio":
        med = v.rolling(756, min_periods=252).median(); s = (v / med).clip(0, L)
    elif kind == "b_pctile":
        rk = v.rolling(504, min_periods=252).rank(pct=True); s = 0.5 + rk * (L - 0.5)
    elif kind == "c_chg":
        s = (v / v120).clip(0, L)
    elif kind == "d_volup_trend":
        s = pd.Series(np.where((v / v120 > 1) & (px > ma), L, 1.0), X.index).where(v120.notna() & ma.notna())
    elif kind == "e_trend_only":
        s = pd.Series(np.where(px > ma, L, 1.0), X.index).where(ma.notna())
    return s.fillna(1.0)       # 暖身期以 1 倍持有


def rows_for(name, o):
    y = len(o["r"]) / 252
    return [dict(variant=name, period=per, **stats(win(o["r"], lo, hi), win(o["lev"], lo, hi)),
                 turnover_yr=o["turn"] / y, cost_pct_yr=o["cost"] / y * 100, roll_cost_pct_yr=o["rcost"] / y * 100, trades_yr=o["ntr"] / y)
            for per, lo, hi in PERIODS]


def main(H=None):
    X = m37.load(); X = X[X.index >= pd.Timestamp("2010-01-01")]
    pd.set_option("display.width", 260)
    KINDS = ("a_ratio", "b_pctile", "c_chg", "d_volup_trend", "e_trend_only")
    grid = []
    for kind, lb, L in itertools.product(KINDS, (20, 60), (1, 1.5, 2)):
        if kind == "e_trend_only" and lb == 60:
            continue
        o = sim(X, signals(X, kind, lb, L).values)
        for rw in rows_for(f"{kind} lb{lb} L{L}", o):
            grid.append(dict(kind=kind, lb=lb, L=L, **rw))
    bhrows = rows_for("B&H", sim(X, np.ones(len(X))))
    G = pd.DataFrame(grid + [dict(kind="B&H", lb=0, L=1, **r) for r in bhrows]); G.to_csv(OUT / "ivt_grid.csv", index=False)
    for per in ("IS", "OOS", "OOS_ex2526", "FULL"):
        print(per, "sharpe (B&H:", round([r for r in bhrows if r["period"] == per][0]["sharpe"], 3), ")")
        print(G[(G.period == per) & (G.kind != "B&H")].pivot_table(index=["kind", "lb"], columns="L", values="sharpe", sort=False).round(3).to_string())
    # headline：IS 平台（見輸出）
    kind, lb, L = H or HEAD
    V = {"B&H 1x (含換月成本)": sim(X, np.ones(len(X))),
         f"HEADLINE {kind} lb{lb} L{L}": sim(X, signals(X, kind, lb, L).values),
         "37 vol target sd60 tv15 L1.5": sim(X, (0.15 / (X.r.rolling(60).std() * math.sqrt(252))).clip(upper=1.5).fillna(0).values)}
    for k2 in KINDS:
        if k2 != kind:
            V[f"{k2} lb{lb if k2 != 'e_trend_only' else 20} L{L}"] = sim(X, signals(X, k2, lb if k2 != "e_trend_only" else 20, L).values)
    rows = []
    for k, o in V.items():
        rows += rows_for(k, o)
    hl = V[f"HEADLINE {kind} lb{lb} L{L}"]; b = V["B&H 1x (含換月成本)"]["r"]
    for per, lo, hi in PERIODS:
        bb, ss = win(b, lo, hi) - RF / 252, win(hl["r"], lo, hi)
        k = ss.std() / bb.std(); rows.append(dict(variant="B&H 縮放至主版本同波動", period=per, scale=k, **stats(k * bb + RF / 252)))
    T = pd.DataFrame(rows); T.to_csv(OUT / "ivt_summary.csv", index=False)
    print(T.pivot_table(index="variant", columns="period", values=["CAGR", "sharpe", "MDD", "vol", "avg_lev"], sort=False).round(3).to_string())
    print(T[T.period == "FULL"][["variant", "turnover_yr", "cost_pct_yr", "roll_cost_pct_yr", "trades_yr", "scale"]].round(3).to_string(index=False))
    sel = {k: o for k, o in V.items() if k.startswith(("B&H", "HEADLINE", "37"))}
    W = []
    for lab, kind_, key in (("2015", "Y", 2015), ("2015/08", "M", "2015-08"), ("2018/10", "M", "2018-10"), ("2020/03", "M", "2020-03"),
                            ("2022", "Y", 2022), ("2025/04", "M", "2025-04")):
        row = dict(window=lab)
        for k, o in sel.items():
            g = (1 + o["r"]).groupby(o["r"].index.year if kind_ == "Y" else o["r"].index.to_period("M")).prod() - 1
            row[k] = g[key] if kind_ == "Y" else g[pd.Period(key)]
        W.append(row)
    Wd = pd.DataFrame(W); Wd.to_csv(OUT / "ivt_bad_periods.csv", index=False); print(Wd.round(3).to_string(index=False))
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, axs = plt.subplots(3, 1, figsize=(12, 11), sharex=True, gridspec_kw=dict(height_ratios=[3, 1.5, 1]))
    for k, o in sel.items():
        r = win(o["r"], pd.Timestamp("2010-07-01"), m37.END); e = (1 + r).cumprod()
        axs[0].plot(e.index, e, label=k, lw=1.2); axs[1].plot(e.index, e / e.cummax() - 1, lw=1)
    axs[2].plot(hl["lev"].index, hl["lev"], c="tab:orange", lw=0.8); axs[2].set_ylabel("headline leverage")
    axs[0].set_yscale("log"); axs[0].legend(fontsize=9); axs[1].set_ylabel("drawdown")
    for a in axs:
        a.axvline(m37.IS0, c="gray", lw=0.6); a.axvline(m37.IS1, c="gray", lw=0.8)
    plt.tight_layout(); plt.savefig(OUT / "ivt_equity_dd.png", dpi=110); plt.close()


HEAD = ("c_chg", 20, 1.5)   # IS 平台：純波動變體中 c_chg lb20 的 IS Sharpe 在 L1–2 皆最高（0.78–0.83），取中間 L1.5（非 OOS 最佳）

if __name__ == "__main__":
    main()
