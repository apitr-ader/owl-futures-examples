"""72 Reddit #7：TXO 結算前日曆價差 + forward factor（FF = (IV1 − σfwd)/σfwd，σfwd² = (IV2²T2 − IV1²T1)/(T2−T1)）。

訊號：前檔（最近到期，週/月）到期前 k 個交易日收盤，以 TXF 收盤為遠期、Black-76（r=0）算 ATM 履約價（最接近 TXF）
  前檔與次檔的 IV（call/put 平均，用 day_close）。FF > thr → 次日日盤開盤做「多日曆」（賣前檔 ATM straddle、買次檔
  同履約價 straddle，各 1 口）；持有到前檔結算：前檔以 TXF 結算日收盤算內含價值，次檔以結算日 day_close 賣出。
  另測 FF < −thr 做空日曆（反向）。thr=None 為無條件基準。
成本：每腿每邊 30 元 + 0.1% 權利金稅 + 1 tick；前檔結算有價內值收 30 元 + 0.1%。
MDD 盤中最差：每日以前檔 day_high、次檔 day_low 標記（無成交用 day_close/內含價值）。
IS 2013–2021、OOS 2022+；Spearman（FF vs 單筆損益、IS/OOS 參數）、隨機進場基準、逐年 WF（前 5 年）。
"""
import itertools, math
import numpy as np, pandas as pd
from scipy.stats import norm, spearmanr
from _rqo import *

END = pd.Timestamp("2026-10-21"); START = pd.Timestamp("2013-01-01")
rng = np.random.default_rng(11)


def b76(F, K, T, s, cp):
    d1 = (np.log(F / K) + .5 * s * s * T) / (s * np.sqrt(T)); d2 = d1 - s * np.sqrt(T)
    return np.where(cp == "C", F * norm.cdf(d1) - K * norm.cdf(d2), K * norm.cdf(-d2) - F * norm.cdf(-d1))


def iv(p, F, K, T, cp):
    lo, hi = np.full_like(p, 1e-3), np.full_like(p, 3.0)
    for _ in range(60):
        m = (lo + hi) / 2; v = b76(F, K, T, m, cp); lo = np.where(v < p, m, lo); hi = np.where(v >= p, m, hi)
    return (lo + hi) / 2


def px(g, d, K, cp, col):
    try: r = g.loc[(K, cp, d)]
    except KeyError: return np.nan
    return r[col]


def build(F, O, E, k):
    days = F.index; out = []
    E = E[(E.expiry_date >= START) & (E.expiry_date <= END)].reset_index(drop=True)
    for i in range(len(E) - 1):
        f, xd = E.expiry[i], E.expiry_date[i]
        if xd not in days or f not in O: continue
        p = days.searchsorted(xd) - k
        if p < 1: continue
        s = days[p]; d1 = days[p + 1]
        nxt = E[(E.expiry_date > xd) & (E.first_date <= s)]
        if nxt.empty: continue
        b, bxd = nxt.expiry.iloc[0], nxt.expiry_date.iloc[0]
        if b not in O: continue
        gf, gb = O[f], O[b]; Fs = F.close[s]
        try: ch = gf.xs(s, level="date").day_close.unstack("cp").dropna()
        except KeyError: continue
        if ch.empty: continue
        K = ch.index[np.abs(ch.index - Fs).argmin()]
        T1 = ((xd - s).days + .5) / 365; T2 = ((bxd - s).days + .5) / 365
        q = {}
        for nm, g, T in (("f", gf, T1), ("b", gb, T2)):
            v = [px(g, s, K, c, "day_close") for c in "CP"]
            if any(pd.isna(x) or x <= 0 for x in v): q = None; break
            q[nm] = np.mean([iv(np.array([v[j]]), Fs, K, T, np.array([c]))[0] for j, c in enumerate("CP")])
        if q is None: continue
        fv = (q["b"] ** 2 * T2 - q["f"] ** 2 * T1) / (T2 - T1)
        if fv <= 0: continue
        FF = (q["f"] - math.sqrt(fv)) / math.sqrt(fv)
        legs = {}
        ok = True
        for nm, g in (("f", gf), ("b", gb)):
            for c in "CP":
                o_ = px(g, d1, K, c, "day_open"); o_ = px(g, d1, K, c, "day_close") if pd.isna(o_) else o_
                if pd.isna(o_): ok = False
                legs[nm + c] = o_
        if not ok: continue
        S = F.close[xd]; fset = {"C": max(S - K, 0), "P": max(K - S, 0)}
        bx = {c: px(gb, xd, K, c, "day_close") for c in "CP"}
        if any(pd.isna(v) for v in bx.values()): continue
        # 多日曆損益（點）：+前檔權利金 − 結算值 − 次檔成本 + 次檔賣出價；成本全扣
        gross = sum(legs["f" + c] - fset[c] + bx[c] - legs["b" + c] for c in "CP")
        cost = sum(opt_cost_pts(legs[x]) for x in legs) + sum(opt_cost_pts(bx[c]) for c in "CP") + sum(OCOMM / MULT + OTAX * fset[c] for c in "CP" if fset[c] > 0)
        hold = days[(days >= d1) & (days <= xd)]
        worst = []
        for d in hold:   # 盤中最差：前檔 high（空）、次檔 low（多）
            fw = sum(np.nanmax([px(gf, d, K, c, "day_high"), fset[c] if d == xd else 0]) if not pd.isna(px(gf, d, K, c, "day_high")) else (fset[c] if d == xd else legs["f" + c]) for c in "CP")
            bw = sum(px(gb, d, K, c, "day_low") if not pd.isna(px(gb, d, K, c, "day_low")) else legs["b" + c] for c in "CP")
            worst.append(sum(legs["f" + c] - legs["b" + c] for c in "CP") - fw + bw)
        out.append(dict(sig=s, d1=d1, xd=xd, front=f, back=b, K=K, IV1=q["f"], IV2=q["b"], FF=FF, gross=gross, cost=cost,
                        long=gross - cost, short=-gross - cost, worst_long=min(worst) - cost / 2, kind=E.kind[i]))
    return pd.DataFrame(out)


def evalf(T, side, thr, a, b):
    t = T[(T.sig >= a) & (T.sig < b)]
    if thr is not None: t = t[t.FF > thr] if side == "long" else t[t.FF < -thr]
    pnl = t[side] * MULT
    if len(t) < 5: return dict(n=len(t), net=pnl.sum(), Sharpe=np.nan, win=np.nan, MDD=np.nan)
    eq = pnl.cumsum().values; w = (t.worst_long * MULT).values if side == "long" else pnl.values
    mddw = float(np.min(np.r_[0, eq[:-1]] + np.minimum(w, pnl.values) - np.maximum.accumulate(np.r_[0, eq[:-1]])))
    yrs = (b - a).days / 365.25
    return dict(n=len(t), net=pnl.sum(), Sharpe=pnl.mean() / pnl.std() * math.sqrt(len(t) / yrs), win=(pnl > 0).mean(), MDD=min(mddw, mdd(eq)))


def main():
    F = load_fut("2012-01-01"); E = expiries()
    O0 = load_opt("2012-10-01", END)
    O = {k: g.set_index(["strike", "cp", "date"]).sort_index() for k, g in O0.groupby("expiry")}
    rows = []; TT = {}
    for k in (1, 2, 3, 5):
        T = build(F, O, E, k); TT[k] = T
        print(f"k={k} trades {len(T)}  Spearman(FF, long pnl) IS {spearmanr(T[T.sig<IS_END].FF, T[T.sig<IS_END].long)[0]:.3f} OOS {spearmanr(T[T.sig>=IS_END].FF, T[T.sig>=IS_END].long)[0]:.3f}")
        for side, thr, kind in itertools.product(("long", "short"), (None, 0.0, 0.1, 0.2), ("all", "weekly", "monthly")):
            Tk = T if kind == "all" else T[T.kind == kind]
            r = dict(k=k, side=side, thr=thr, kind=kind)
            for nm, a, b in (("IS", START, IS_END), ("OOS", IS_END, END)):
                r.update({f"{nm}_{x}": v for x, v in evalf(Tk, side, thr, a, b).items()})
            rows.append(r)
    R = pd.DataFrame(rows); R.to_csv(RES / "rqo_72_grid.csv", index=False); pd.set_option("display.width", 250)
    print(R.round(3).to_string(index=False))
    g = R.dropna(subset=["IS_Sharpe", "OOS_Sharpe"]); g = g[g.IS_n >= 30]
    print("Spearman IS/OOS Sharpe:", round(g.IS_Sharpe.rank().corr(g.OOS_Sharpe.rank()), 3))
    best = g.loc[g.IS_Sharpe.idxmax()]; print("best IS:\n", best.to_string())
    T = TT[int(best.k)]; T = T if best.kind == "all" else T[T.kind == best.kind]
    thr = None if pd.isna(best.thr) else best.thr
    sel = T if thr is None else (T[T.FF > thr] if best.side == "long" else T[T.FF < -thr])
    oo = T[T.sig >= IS_END]; so = sel[sel.sig >= IS_END]
    rnd = [oo.sample(len(so), random_state=int(rng.integers(1e9)))[best.side].sum() * MULT for _ in range(500)]
    print(f"隨機抽同筆數 OOS：實際 {so[best.side].sum()*MULT:.0f} 百分位 {(np.array(rnd) < so[best.side].sum()*MULT).mean():.2f} 中位數 {np.median(rnd):.0f}")
    # WF
    wf = []
    for Y in range(2018, 2027):
        a, b = pd.Timestamp(f"{Y-5}-01-01"), pd.Timestamp(f"{Y}-01-01"); c = pd.Timestamp(f"{Y+1}-01-01")
        best_s, best_c = -9, None
        for (k, side, thr_, kind) in itertools.product((1, 2, 3, 5), ("long", "short"), (None, 0.0, 0.1, 0.2), ("all", "weekly", "monthly")):
            Tk = TT[k] if kind == "all" else TT[k][TT[k].kind == kind]
            s = evalf(Tk, side, thr_, a, b)
            if s["n"] >= 20 and s["Sharpe"] > best_s: best_s, best_c = s["Sharpe"], (k, side, thr_, kind)
        k, side, thr_, kind = best_c; Tk = TT[k] if kind == "all" else TT[k][TT[k].kind == kind]
        t = Tk[(Tk.sig >= b) & (Tk.sig < c)]
        if thr_ is not None: t = t[t.FF > thr_] if side == "long" else t[t.FF < -thr_]
        wf.append(pd.Series(t[side].values * MULT, t.xd.values)); print("WF", Y, best_c, round(best_s, 2), round(t[side].sum() * MULT))
    wf = pd.concat(wf); wo = wf[wf.index >= IS_END]
    print(f"WF 2022+ net {wo.sum():.0f} trades {len(wo)} per-trade Sharpe(ann) {wo.mean()/wo.std()*math.sqrt(len(wo)/4.8):.2f}")
    plt = plt_style(); fig, axs = plt.subplots(2, 1, figsize=(12, 9), facecolor=BG)
    ax = axs[0]; sty(ax)
    base = T.sort_values("xd"); ax.plot(base.xd, (base[best.side] * MULT).cumsum() / 1e4, color=GR, label=f"無條件 {best.side}（k={int(best.k)}, {best.kind}）")
    ss = sel.sort_values("xd"); ax.plot(ss.xd, (ss[best.side] * MULT).cumsum() / 1e4, color=GOLD, lw=2, label=f"IS 最佳 FF 門檻 {thr}")
    ax.plot(wf.index, wf.cumsum() / 1e4, color=BLUE, label="walk-forward"); ax.axvline(IS_END, color="white", ls="--")
    ax.set_title(f"TXO 結算前日曆價差 + forward factor｜IS Sharpe {best.IS_Sharpe:.2f} → OOS {best.OOS_Sharpe:.2f}"); ax.set_ylabel("累積淨損益（萬元）")
    ax.legend(facecolor=BG, labelcolor=WH, edgecolor=GR, fontsize=9)
    ax = axs[1]; sty(ax); A = TT[int(best.k)]
    ax.scatter(A.FF, A.long * MULT / 1e4, s=6, color=GOLD, alpha=.5); ax.set_xlabel("forward factor"); ax.set_ylabel("多日曆單筆淨損益（萬元）")
    ax.set_title(f"FF vs 多日曆損益（Spearman 全期 {spearmanr(A.FF, A.long)[0]:.2f}）"); ax.set_xlim(-1, 1.5)
    foot(fig); fig.tight_layout(rect=[0, .03, 1, 1]); fig.savefig(RES / "rqo_72_calendar_ff.png", dpi=110, facecolor=BG)


if __name__ == "__main__":
    main()
