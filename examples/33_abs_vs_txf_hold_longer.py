"""33 雙動能改良：(1) 絕對動能改比 TXF；(2) 強勢股續抱（賣出規則）；(3) 兩者合併。
基礎：32 的新可選池（任何上市交易中、>=22 日歷史、排除 ETF/'1' 結尾）、10 檔、每到期換倉、大盤濾網(TXF 1/3/6m)、vt 15%、EW。
分數：VS（27 pick）、S126（28）。
絕對濾網 abs（進場；股票原始 1/3/6m 平均報酬 raw 與同腿 TXF 調整連續報酬 txf 比較）：
  rf  : raw > rf 門檻（原設定）     txf : raw > txf     txf1 : raw > txf + 1%     max : raw > max(txf, rf)
續抱 K（None = 原本每期重選前 10）：持有中的標的續抱，除非 (a) 動能轉弱：abs=rf 時 raw < 0；
  其他 abs（合併測試）時不再通過同一 abs 規則；或 (b) 排名掉到 K 名之外。空出的位置才由前排名、過 abs 的新標的補上，上限 10 檔。
最短持有 mh（0/2/3 個到期）：未滿 mh 期的持股不因 (a)(b) 賣出（大盤濾網轉空仍全數出場）。
存活者偏差：products.csv 只含現存商品。
"""
import importlib.util
import itertools
from pathlib import Path

import numpy as np
import pandas as pd

_s = importlib.util.spec_from_file_location("nu32", Path(__file__).with_name("32_four_strategies_new_universe.py"))
m32 = importlib.util.module_from_spec(_s); _s.loader.exec_module(m32)
m31, m24, o22, dm, D = m32.m31, m32.m24, m32.o22, m32.dm, m32.D
OUT = Path("results")
NH = 10
IS0, IS1, EX1, END, PERIODS = m32.IS0, m32.IS1, m32.EX1, m32.END, m32.PERIODS
THR = (1 + dm.RF) ** (np.mean(m31.LEGS) / 252) - 1
G = [dict(score=sc, abs=a, K=K, mh=mh) for sc in ("VS", "S126") for a in ("rf", "txf", "txf1", "max")
     for K, mh in [(0, 0)] + list(itertools.product((15, 20, 30), (0, 2, 3)))]
COST = dict(slip=1.0)


def absok(a, raw, x):
    t = D["mktsig"][x]
    return raw > {"rf": THR, "txf": t, "txf1": t + 0.01, "max": max(t, THR)}[a]


def simulate(P, lo=None, hi=None, trace=False):
    days, mult, A, O, R, RO = D["days"], D["mult"], D["A"], D["O"], D["R"], D["RO"]
    V = D["M"]["V"].values; slip = COST["slip"]; comm = 50.0
    lo_, hi_ = (IS0 if lo is None else lo), (END if hi is None else hi)
    X = [x for x in D["X"] if lo_ - pd.Timedelta(days=40) <= days[x] < hi_]
    nd = len(days); eq = 1.0; vals = np.full(nd, np.nan)
    turn = cost = 0.0
    held, age, gross, recs, exits, spells = [], {}, [], [], [], []
    for n, x in enumerate(X):
        k0 = x + 1
        if k0 >= nd:
            break
        k1 = X[n + 1] + 1 if n + 1 < len(X) else nd
        U = D["univ"][x]; sel = []
        why = {}
        if len(U) >= dm.MIN_UNIVERSE and D["mktsig"][x] > 0:
            sc, _ = D["scores"][x][P["score"]]
            raw = D["scores"][x]["RAW"][0]
            o = np.argsort(-sc); rk = U[o]
            ent = absok(P["abs"], raw[o], x)
            brk = (raw[o] < 0) if P["abs"] == "rf" else ~ent
            pos = {int(s): r + 1 for r, s in enumerate(rk)}
            bk = {int(s): bool(b) for s, b in zip(rk, brk)}
            if P["K"]:
                keep = []
                for s in held:
                    if s not in pos:
                        why[s] = "not tradable"
                    elif age[s] < P["mh"] or (pos[s] <= P["K"] and not bk[s]):
                        keep.append(s)
                    else:
                        why[s] = "momentum break" if bk[s] else "rank>K"
                keep = sorted(keep, key=lambda s: pos[s])[:NH]
                sel = keep + [int(s) for s, e in zip(rk[:NH], ent[:NH]) if e and int(s) not in keep][:NH - len(keep)]
            else:
                sel = [int(s) for s, e in zip(rk[:NH], ent[:NH]) if e]
                for s in held:
                    if s not in sel:
                        why[s] = "not tradable" if s not in pos else ("momentum break" if bk[s] or not ent[pos[s] - 1] else "displaced (rank>10)")
        else:
            for s in held:
                why[s] = "market filter"
        sel = sorted(s for s in sel if not np.isnan(RO[k0, s]))
        for s in held:
            if s not in sel:
                exits.append(why.get(s, "not tradable")); spells.append(age[s])
        age = {s: age.get(s, 0) + 1 for s in sel}
        held = sel
        if sel:
            s_ = np.array(sel); w_ = m31.weights(x, s_, "EW", 0.15)
            gross.append(w_.sum())
            q = w_ * eq / (RO[k0, s_] * mult[s_]); px = RO[k0, s_]
            # 續抱的部位不重複收開倉成本？維持 26 架構：每到期結算後重開（股票期貨到期必換月）
            c = q @ (comm + mult[s_] * (dm.TAX * px + slip * o22.tick_arr(px)))
            turn += (q * mult[s_] * px).sum() / eq; eq -= c; cost += c / eq
            vals[k0:k1] = eq + (A[k0:k1, s_] - O[k0, s_]) @ (q * mult[s_]); eq = vals[k1 - 1]
            if n + 1 < len(X):
                sp = np.nan_to_num(R[k1 - 1, s_], nan=np.nan_to_num(D["M"]["Rf"].values[k1 - 1, s_]))
                c = q @ (comm + mult[s_] * dm.TAX * sp); eq -= c; cost += c / eq; vals[k1 - 1] = eq
            if trace:
                vol = np.nanmean(V[max(x - 19, 0):x + 1, s_], 0)
                recs += [dict(date=days[x], und=D["und"][s], w=float(w), adv20=float(v)) for s, w, v in zip(s_, w_, vol)]
        else:
            gross.append(0.0); vals[k0:k1] = eq
        if n == 0 and k0 > 0:
            vals[k0 - 1] = 1.0
    spells += list(age.values())
    r = pd.Series(vals, days).dropna().pct_change().fillna(0.0)
    if lo is not None:
        r = r[(r.index >= lo) & (r.index < hi)]
    return dict(r=r, turn=turn, cost=cost, yrs=len(r) / 252, gross=float(np.mean(gross)) if gross else 0.0, recs=recs,
                exits=pd.Series(exits, dtype=object), spells=spells)


def group(P):
    if P["abs"] == "rf":
        return "0 baseline" if not P["K"] else "2 hold longer"
    return "1 abs vs TXF" if not P["K"] else "3 combined"


def main():
    m32.setup()
    W = m31.winners()
    rows = []
    for P in G:
        r = simulate(P)["r"]
        rows.append(dict(**P, group=group(P), **{f"{nm}_sharpe": m24.stats(m24.win(r, lo, hi))["sharpe"] for nm, lo, hi in PERIODS}))
    g = pd.DataFrame(rows)
    pl = []
    for r in g.itertuples():
        nb = g[(g.score == r.score) & (g.group == r.group) & (((g["abs"] == r.abs) & (g.K == r.K)) | ((g["abs"] == r.abs) & (g.mh == r.mh)) | ((g.K == r.K) & (g.mh == r.mh)))]
        pl.append(nb.IS_sharpe.median())
    g["plateau"] = pl
    g.to_csv(OUT / "avt_grid.csv", index=False)
    ntr = len(g); sr0 = dm.expected_max_sr(ntr, float(g.IS_sharpe.var()))
    pd.set_option("display.width", 250)
    print(f"trials {ntr} SR0 {sr0:.3f} Spearman {g[['IS_sharpe','OOS_sharpe']].rank().corr().iloc[0,1]:.3f}")
    print(g.groupby(["score", "group"])[["IS_sharpe", "OOS_sharpe", "OOS_ex2526_sharpe"]].median().round(3).to_string())
    print(g.groupby(["score", "abs"])[["IS_sharpe", "OOS_sharpe", "OOS_ex2526_sharpe"]].median().round(3).to_string())
    print(g[g.K > 0].groupby(["score", "K", "mh"])[["IS_sharpe", "OOS_sharpe", "OOS_ex2526_sharpe"]].median().round(3).to_string())
    T, curves = [], {}
    for sc in ("VS", "S126"):
        for grp in sorted(g.group.unique()):
            b = g[(g.score == sc) & (g.group == grp)].sort_values(["plateau", "IS_sharpe"], ascending=False).iloc[0]
            P = dict(score=sc, abs=b["abs"], K=int(b.K), mh=int(b.mh))
            name = f"{'27' if sc == 'VS' else '28'} {grp}"
            for slip in (1.0, 0.5):
                COST["slip"] = slip
                full = simulate(P)
                if slip == 1.0:
                    curves[name] = full["r"]
                for per, lo, hi in PERIODS:
                    o = simulate(P, lo, hi, trace=True); s = m24.stats(m24.win(full["r"], lo, hi)); y = max(o["yrs"], 1e-9)
                    rec = pd.DataFrame(o["recs"]); ex = o["exits"]
                    T.append(dict(variant=name, cfg=f"abs={P['abs']} K={P['K'] or '-'} mh={P['mh']}", slip=slip, period=per,
                                  CAGR=s["CAGR"], sharpe=s["sharpe"], MDD=s["mdd"], calmar=s["CAGR"] / abs(s["mdd"]),
                                  turnover_yr=o["turn"] / y, cost_pct_yr=o["cost"] / y * 100, gross=o["gross"],
                                  winner_capture=m31.capture(o["recs"], W, lo, hi), avg_hold_exp=np.mean(o["spells"]) if o["spells"] else np.nan,
                                  exit_displaced=ex.str.startswith(("rank", "displaced")).mean(), exit_mom_break=(ex == "momentum break").mean(),
                                  exit_mkt_filter=(ex == "market filter").mean(), DSR=dm.dsr(s["sharpe"], s["n"], s["skew"], s["kurt"], sr0) if per == "OOS" else np.nan,
                                  trials=ntr, lt50_share=(rec.adv20 < 50).mean() if len(rec) else np.nan))
            COST["slip"] = 1.0
    for per, lo, hi in PERIODS:
        s = m24.stats(m24.win(D["txr"], lo, hi))
        T.append(dict(variant="TXF B&H", period=per, CAGR=s["CAGR"], sharpe=s["sharpe"], MDD=s["mdd"], calmar=s["CAGR"] / abs(s["mdd"]), gross=1.0))
    curves["TXF B&H"] = D["txr"]
    T = pd.DataFrame(T); T.to_csv(OUT / "avt_summary.csv", index=False)
    t = T[T.slip != 0.5]
    print(t.pivot_table(index="variant", columns="period", values=["CAGR", "sharpe", "MDD", "calmar"], sort=False).round(3).to_string())
    print(t[t.period == "OOS"][["variant", "cfg", "turnover_yr", "cost_pct_yr", "gross", "winner_capture", "avg_hold_exp", "exit_displaced",
                                "exit_mom_break", "exit_mkt_filter", "DSR", "lt50_share"]].round(3).to_string(index=False))
    print(T[(T.slip == 0.5) & (T.period == "OOS")][["variant", "CAGR", "sharpe", "MDD"]].round(3).to_string(index=False))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(2, 1, figsize=(12, 9), sharex=True)
    for k, r in curves.items():
        r = m24.win(r, IS0, END); e = (1 + r).cumprod()
        st = dict(c="k", lw=2) if k == "TXF B&H" else dict(lw=1.1, ls="-" if k.startswith("27") else "--")
        axs[0].plot(e.index, e, label=k, **st); axs[1].plot(e.index, e / e.cummax() - 1, **st)
    axs[0].set_yscale("log"); axs[0].legend(fontsize=7, ncol=2); axs[1].set_ylabel("drawdown")
    for a in axs:
        a.axvline(IS1, c="gray", lw=0.8)
    plt.tight_layout(); plt.savefig(OUT / "avt_equity_dd.png", dpi=110); plt.close()
    fig, ax = plt.subplots(figsize=(7, 5))
    for grp, gg in g.groupby("group"):
        ax.scatter(gg.IS_sharpe, gg.OOS_sharpe, s=14, label=grp)
    ax.set_xlabel("IS Sharpe"); ax.set_ylabel("OOS Sharpe"); ax.legend(); plt.tight_layout(); plt.savefig(OUT / "avt_is_vs_oos.png", dpi=110); plt.close()


if __name__ == "__main__":
    main()
