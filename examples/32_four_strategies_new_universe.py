"""32 四個策略在新可選池（31 的條件）下重測：任何上市交易中的股票期貨（換倉日有成交、>=22 日歷史，
無量/252日/換月跳空門檻；排除 ETF 與 '1' 結尾），持有 10 檔，取第 1-10 名（過絕對濾網）。

(a) 原設定（只改可選池、N=10）；(b) 樣本內平台輕量重選：目標波動 none/15%/25% × 大盤濾網 on/off。
  27 pick : VS 複合（1/3/6m，波動調整）、>rf、每到期、無緩衝、大盤(TXF 1/3/6m 平均)、vt 15%、EW
  28 s126 : 複合/126日波動、>rf、每到期、無緩衝、大盤(TXF 1/3/6m)、vt 15%、EW
  26 lb20 : 20日報酬>0、每到期、緩衝（續抱排名 <= 2N=20）、大盤(TXF 252日)、vt 15%、EW
  26 DM   : 252日報酬略過最近 21 日 >0、每季（每 3 個到期）、無緩衝、無大盤、vt 15%、EW
新上市：複合分數用可用的腿（21 必需）；s126 的波動用可用的日數（至少 21）；DM 需 253 日歷史，不足者不列入排名。
成本：手續費 50，稅 2e-5，滑價 1 / 0.5 tick，無換月成本。存活者偏差：products.csv 只含現存商品。
"""
import importlib.util
import itertools
import math
from pathlib import Path

import numpy as np
import pandas as pd

_s = importlib.util.spec_from_file_location("rb31", Path(__file__).with_name("31_rank_band.py"))
m31 = importlib.util.module_from_spec(_s); _s.loader.exec_module(m31)
m24, o22, dm, D = m31.m24, m31.o22, m31.dm, m31.D
OUT = Path("results")
NH = 10
IS0, IS1, EX1, END, PERIODS = m31.IS0, m31.IS1, m31.EX1, m31.END, m31.PERIODS
STRATS = {
    "27 pick": dict(score="VS", cad=1, buf=0, mkt=1, msig="cmp", vt=0.15),
    "28 s126": dict(score="S126", cad=1, buf=0, mkt=1, msig="cmp", vt=0.15),
    "26 lb20": dict(score="LB20", cad=1, buf=1, mkt=1, msig="252", vt=0.15),
    "26 DM": dict(score="DM252", cad=3, buf=0, mkt=0, msig="252", vt=0.15),
}
OLD_MDD = {"27 pick": (-0.163, -0.237), "28 s126": (-0.236, -0.232), "26 lb20": (-0.24, -0.22), "26 DM": (-0.30, -0.31)}
COST = dict(slip=1.0)


def setup():
    m31.setup()
    M = D["M"]; nh = M["nhist"].values; I, ret = D["I"], D["ret"]
    thr = (1 + dm.RF) ** (np.mean(m31.LEGS) / 252) - 1
    D["sig"] = {"cmp": D["mktsig"], "252": D["txf_mom"]}
    for x in D["X"]:
        U = D["univ"][x]; h = nh[x, U]
        raw = D["scores"][x]["RAW"][0]
        n = np.minimum(h - 1, 126)
        v = np.array([ret[x - k + 1:x + 1, u].std() for u, k in zip(U, n)]) * math.sqrt(252)
        D["scores"][x]["S126"] = (raw / np.maximum(v, 1e-4), raw > thr)
        okh = h >= 253
        m = np.where(okh, I[x - 21, U] / I[max(x - 252, 0), U] - 1, -np.inf)
        D["scores"][x]["DM252"] = (m, okh & (m > 0))


def simulate(P, lo=None, hi=None, trace=False):
    days, mult, A, O, R, RO = D["days"], D["mult"], D["A"], D["O"], D["R"], D["RO"]
    V = D["M"]["V"].values; slip = COST["slip"]; comm = 50.0
    lo_, hi_ = (IS0 if lo is None else lo), (END if hi is None else hi)
    X = [x for x in D["X"] if lo_ - pd.Timedelta(days=40) <= days[x] < hi_]
    nd = len(days); eq = 1.0; vals = np.full(nd, np.nan)
    turn = c_open = c_set = 0.0
    held, w_held, gross, recs = [], None, [], []
    for n, x in enumerate(X):
        k0 = x + 1
        if k0 >= nd:
            break
        k1 = X[n + 1] + 1 if n + 1 < len(X) else nd
        if n % P["cad"] == 0:
            U = D["univ"][x]; sel = []
            if len(U) >= dm.MIN_UNIVERSE and not (P["mkt"] and not D["sig"][P["msig"]][x] > 0):
                sc, ok = D["scores"][x][P["score"]]
                o = np.argsort(-sc); rk = U[o]; okr = ok[o]
                pos = {int(s): r for r, s in enumerate(rk)}; okd = {int(s): bool(v) for s, v in zip(rk, okr)}
                keep = [s for s in held if s in pos and pos[s] < 2 * NH and okd[s]][:NH] if P["buf"] else []
                sel = keep + [int(s) for s, v in zip(rk[:NH], okr[:NH]) if v and int(s) not in keep][:NH - len(keep)]
            sel = sorted(sel)
            w_held = m31.weights(x, np.array(sel), "EW", P["vt"]) if sel else np.zeros(0)
            held = sel
        okf = [i for i, s in enumerate(held) if not np.isnan(RO[k0, s])]
        if okf:
            s_ = np.array([held[i] for i in okf]); w_ = np.asarray(w_held)[okf]
            gross.append(w_.sum())
            q = w_ * eq / (RO[k0, s_] * mult[s_]); px = RO[k0, s_]
            c = q @ (comm + mult[s_] * (dm.TAX * px + slip * o22.tick_arr(px)))
            turn += (q * mult[s_] * px).sum() / eq
            eq -= c; c_open += c / eq
            vals[k0:k1] = eq + (A[k0:k1, s_] - O[k0, s_]) @ (q * mult[s_])
            eq = vals[k1 - 1]
            if n + 1 < len(X):
                sp = np.nan_to_num(R[k1 - 1, s_], nan=np.nan_to_num(D["M"]["Rf"].values[k1 - 1, s_]))
                c = q @ (comm + mult[s_] * dm.TAX * sp)
                eq -= c; c_set += c / eq; vals[k1 - 1] = eq
            if trace:
                vol = np.nanmean(V[max(x - 19, 0):x + 1, s_], 0)
                recs += [dict(date=days[x], und=D["und"][s], w=float(w), adv20=float(v)) for s, w, v in zip(s_, w_, vol)]
        else:
            gross.append(0.0); vals[k0:k1] = eq
        if n == 0 and k0 > 0:
            vals[k0 - 1] = 1.0
    r = pd.Series(vals, days).dropna().pct_change().fillna(0.0)
    if lo is not None:
        r = r[(r.index >= lo) & (r.index < hi)]
    return dict(r=r, turn=turn, cost=c_open + c_set, yrs=len(r) / 252, gross=float(np.mean(gross)) if gross else 0.0, recs=recs)


def main():
    setup()
    W = m31.winners()
    # (b) 平台重選：每個策略 6 組（vt × mkt），鄰域 = 同 vt 或同 mkt 的其他組
    grid = []
    for name, P0 in STRATS.items():
        for vt, mkt in itertools.product([0.0, 0.15, 0.25], [0, 1]):
            P = {**P0, "vt": vt, "mkt": mkt}
            r = simulate(P)["r"]
            grid.append(dict(strategy=name, vt=vt, mkt=mkt, **{f"{nm}_sharpe": m24.stats(m24.win(r, lo, hi))["sharpe"] for nm, lo, hi in PERIODS}))
    g = pd.DataFrame(grid)
    g["plateau"] = [g[(g.strategy == r.strategy) & ((g.vt == r.vt) | (g.mkt == r.mkt))].IS_sharpe.median() for r in g.itertuples()]
    g.to_csv(OUT / "nu4_grid.csv", index=False)
    sr0 = dm.expected_max_sr(len(g), float(g.IS_sharpe.var()))
    rows, curves = [], {}
    for name, P0 in STRATS.items():
        b = g[g.strategy == name].sort_values(["plateau", "IS_sharpe"], ascending=False).iloc[0]
        for var, P in (("a original", P0), ("b re-pick", {**P0, "vt": float(b.vt), "mkt": int(b.mkt)})):
            for slip in (1.0, 0.5):
                COST["slip"] = slip
                full = simulate(P)
                if slip == 1.0:
                    curves[f"{name} {var}"] = full["r"]
                for per, lo, hi in PERIODS:
                    o = simulate(P, lo, hi, trace=True); s = m24.stats(m24.win(full["r"], lo, hi)); y = max(o["yrs"], 1e-9)
                    rec = pd.DataFrame(o["recs"])
                    rows.append(dict(strategy=name, variant=var, slip=slip, period=per, vt=P["vt"], mkt=P["mkt"],
                                     CAGR=s["CAGR"], sharpe=s["sharpe"], MDD=s["mdd"], calmar=s["CAGR"] / abs(s["mdd"]) if s["mdd"] else np.nan,
                                     turnover_yr=o["turn"] / y, cost_pct_yr=o["cost"] / y * 100, gross=o["gross"],
                                     winner_capture=m31.capture(o["recs"], W, lo, hi),
                                     DSR=dm.dsr(s["sharpe"], s["n"], s["skew"], s["kurt"], sr0) if per == "OOS" else np.nan,
                                     lt50_share=(rec.adv20 < 50).mean() if len(rec) else np.nan,
                                     old_MDD={"IS": OLD_MDD[name][0], "OOS": OLD_MDD[name][1]}.get(per, np.nan)))
        COST["slip"] = 1.0
    for k, r in {"TXF B&H": D["txr"]}.items():
        curves[k] = r
        for per, lo, hi in PERIODS:
            s = m24.stats(m24.win(r, lo, hi))
            rows.append(dict(strategy=k, variant="-", period=per, CAGR=s["CAGR"], sharpe=s["sharpe"], MDD=s["mdd"], calmar=s["CAGR"] / abs(s["mdd"]), gross=1.0))
    T = pd.DataFrame(rows); T.to_csv(OUT / "nu4_summary.csv", index=False)
    pd.set_option("display.width", 250)
    print(f"SR0 {sr0:.3f}\n", g.round(3).to_string(index=False))
    t = T[(T.slip != 0.5)]
    w = t.pivot_table(index=["strategy", "variant"], columns="period",
                      values=["CAGR", "sharpe", "MDD", "calmar"], sort=False)
    print(w.round(3).to_string())
    o = t[t.period == "OOS"][["strategy", "variant", "vt", "mkt", "turnover_yr", "cost_pct_yr", "gross", "winner_capture", "DSR", "lt50_share", "old_MDD"]]
    print(o.round(3).to_string(index=False))
    print(T[(T.slip == 0.5) & (T.period == "OOS")][["strategy", "variant", "CAGR", "sharpe", "MDD"]].round(3).to_string(index=False))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(2, 1, figsize=(12, 9), sharex=True)
    for k, r in curves.items():
        r = m24.win(r, IS0, END); e = (1 + r).cumprod()
        st = dict(c="k", lw=2) if k == "TXF B&H" else dict(lw=1.1, ls="-" if "original" in k else "--")
        axs[0].plot(e.index, e, label=k, **st); axs[1].plot(e.index, e / e.cummax() - 1, **st)
    axs[0].set_yscale("log"); axs[0].legend(fontsize=7, ncol=2); axs[0].set_title("Four strategies, new universe (any listed, 10 names), 1 tick")
    axs[1].set_ylabel("drawdown")
    for a in axs:
        a.axvline(IS1, c="gray", lw=0.8)
    plt.tight_layout(); plt.savefig(OUT / "nu4_equity_dd.png", dpi=110); plt.close()
    m = t[t.period.isin(["IS", "OOS"])].copy(); m["key"] = m.strategy + " " + m.variant
    pv = m.pivot_table(index="key", columns="period", values="MDD", sort=False)
    ax = pv.plot.barh(figsize=(9, 5)); ax.set_xlabel("max drawdown"); plt.tight_layout(); plt.savefig(OUT / "nu4_mdd.png", dpi=110); plt.close()


if __name__ == "__main__":
    main()
