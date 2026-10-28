"""70 Reddit #3：系統化賣 TXO 價外 put + 兩道 kill switch（相關結構 Frobenius 距離、GARCH 波動百分位）。

進場：結算日 E 收盤訊號（TXF 收盤 × (1-x) 定履約價、當日濾網），E+1 日盤開盤賣出下一檔到期 put
  M=月選（下一個月選）、W=週選（最快到期；含 2025 起週五週選）。持有到結算，結算價≈結算日 TXF 日盤收盤。
成本：30 元/邊 + 0.1% 權利金稅 + 1 tick 滑價（進場）；結算若有價內值再收 30 元 + 0.1%×結算值。1 口，50 元/點。
濾網（E 日已知資料）：
  Frob：30 檔大型股（固定名單，有倖存者偏差）日報酬 21d vs 252d 相關矩陣差的 Frobenius 範數，取其近 252 日百分位 ≤ q 才進場。
  GARCH：TXF 日報酬 GARCH(1,1)（參數只用 IS 估）條件波動，近 252 日百分位在 [lo,hi] 才進場。
MDD 盤中最差：持倉期間 put 以當日 day_high 標記（無成交以 intrinsic of TXF low 估）。
IS 2010(月)/2013(週)–2021、OOS 2022+；Spearman、隨機 gate 基準、逐年 walk-forward（前 5 年重選）。
"""
import itertools
import numpy as np, pandas as pd
from _rqo import *

STK = "2330 2317 2454 2412 2308 2882 2881 2891 2886 2303 1301 1303 2002 2884 2892 1216 2885 2357 2382 2207 5880 2880 2883 2912 1326 2395 3008 2408 2105 1101".split()
END = pd.Timestamp("2026-10-21")
rng = np.random.default_rng(7)


def frob_pct():
    R = pd.concat({c: pd.read_parquet(ROOT / f"data/stocks/adj/{c}.parquet").ret for c in STK}, axis=1)
    R.index = pd.to_datetime(R.index); R = R[R.index >= "2008-01-01"]
    v = R.values; out = {}
    for i in range(252, len(R)):
        a = pd.DataFrame(v[i - 20:i + 1]).corr().values; b = pd.DataFrame(v[i - 251:i + 1]).corr().values
        m = ~np.isnan(a - b); out[R.index[i]] = np.sqrt(np.nansum((a - b)[m] ** 2))
    s = pd.Series(out)
    return s, s.rolling(252).apply(lambda x: (x[:-1] < x[-1]).mean(), raw=True)


def garch_pct(F):
    r = fut_ret(F).dropna(); r = r - r[r.index < IS_END].mean()
    p, _ = garch11(r[r.index < IS_END].values * 100); _, h = garch11(r.values * 100, p)
    s = pd.Series(np.sqrt(h), r.index)
    return p, s, s.rolling(252).apply(lambda x: (x[:-1] < x[-1]).mean(), raw=True)


def trades(F, O, E, mode, x):
    days = F.index; out = []
    exps = E[E.kind == "monthly"] if mode == "M" else E
    sigs = [d for d in E[E.kind == "monthly"].expiry_date] if mode == "M" else list(E.expiry_date)
    sigs = sorted(set(d for d in sigs if d in days and d <= END))
    for s in sigs:
        p = days.searchsorted(s) + 1
        if p >= len(days): continue
        d1 = days[p]; nx = exps[exps.expiry_date > s]
        if nx.empty or nx.expiry_date.iloc[0] > END: continue
        ex, xd = nx.expiry.iloc[0], nx.expiry_date.iloc[0]
        if ex not in O or xd not in days: continue
        g = O[ex]; k0 = F.close[s] * (1 - x)
        try: today = g.xs(d1, level="date")
        except KeyError: continue
        today = today[today.day_open.notna() | today.day_close.notna()]
        today = today[(today.index <= F.close[s])]
        if today.empty: continue
        K = today.index[np.abs(today.index - k0).argmin()]
        prem = today.day_open.get(K); prem = today.day_close[K] if pd.isna(prem) else prem
        if prem < 0.5: continue
        S = F.close[xd]; intr = max(K - S, 0.0)
        net = prem - opt_cost_pts(prem) - intr - (OCOMM / MULT + OTAX * intr if intr > 0 else 0)
        hold = days[(days >= d1) & (days <= xd)]
        hs = g.xs(K, level="strike") if K in g.index.get_level_values(0) else pd.DataFrame()
        cl, hi = [], []
        last = prem
        for d in hold:
            row = hs.loc[d] if d in hs.index else None
            iv_lo = max(K - F.low[d], 0); iv_c = max(K - F.close[d], 0)
            c = row.day_close if row is not None and pd.notna(row.day_close) else max(last, iv_c) if d != xd else intr
            h = row.day_high if row is not None and pd.notna(row.day_high) else max(c, iv_lo)
            if d == xd: c = intr; h = max(h, intr)
            last = c; cl.append(c); hi.append(max(h, c))
        out.append(dict(sig=s, d1=d1, expiry=ex, xd=xd, K=K, F=F.close[s], prem=prem, settle=S, net=net,
                        hold=hold, mtm_c=prem - np.array(cl), mtm_w=prem - np.array(hi)))
    return out


def series(T, mask, days):
    """日損益（收盤標記，元）與盤中最差權益。"""
    pc = pd.Series(0.0, days); w = pd.Series(0.0, days); closed = pd.Series(0.0, days)
    for t, m in zip(T, mask):
        if not m: continue
        cost_in = opt_cost_pts(t["prem"])
        mc = t["mtm_c"] - cost_in; mc[-1] = t["net"]; mw = t["mtm_w"] - cost_in
        d = np.diff(np.r_[0, mc]); pc[t["hold"]] += d * MULT
        w[t["hold"]] += (mw - mc) * MULT
    eqc = pc.cumsum(); eqw = eqc + w
    return pc, eqc, float((eqw - eqc.cummax().clip(lower=0)).min())


def stats(pc, mddw, a, b, n):
    s = pc[(pc.index >= a) & (pc.index < b)]
    return dict(n=n, net=s.sum(), ann=s.sum() / (len(s) / 252), Sharpe=sharpe(s), MDDw=mddw)


def main():
    F = load_fut("2009-01-01"); days = F.index[F.index <= END]
    E = expiries(); E = E[E.expiry_date <= END + pd.Timedelta(days=60)]
    O0 = load_opt("2009-10-01", END, cp="P")
    O = {k: g.set_index(["strike", "date"]).sort_index() for k, g in O0.groupby("expiry")}
    fd, fp = frob_pct(); gp_par, gv, gp = garch_pct(F)
    print("GARCH params (IS):", np.round(gp_par, 4))
    rows, keep = [], {}
    GRID = dict(M=[.03, .05, .07], W=[.02, .03, .04])
    FQ = [None, .8, .9]; GB = [None, (.1, .9), (0, .9), (.1, 1.0)]
    for mode in "MW":
        st = pd.Timestamp("2010-01-01" if mode == "M" else "2013-01-01")
        for x in GRID[mode]:
            T = [t for t in trades(F, O, E, mode, x) if t["sig"] >= st]
            sig = pd.DatetimeIndex([t["sig"] for t in T])
            fpv = fp.reindex(sig, method="ffill").values; gpv = gp.reindex(sig, method="ffill").values
            for fq, gb in itertools.product(FQ, GB):
                m = np.ones(len(T), bool)
                if fq is not None: m &= np.nan_to_num(fpv, nan=0) <= fq
                if gb is not None: m &= (np.nan_to_num(gpv, nan=.5) >= gb[0]) & (np.nan_to_num(gpv, nan=.5) <= gb[1])
                pc, eqc, mw = series(T, m, days)
                pc = pc[pc.index >= st]
                key = (mode, x, fq, gb); keep[key] = (T, m, pc)
                r = dict(mode=mode, x=x, frob_q=fq, garch=gb, pass_rate=m.mean())
                for nm, a, b in (("IS", st, IS_END), ("OOS", IS_END, END + pd.Timedelta(days=1))):
                    sel = m & (sig >= a) & (sig < b)
                    pcs = pc[(pc.index >= a) & (pc.index < b)]
                    _, _, mws = series([t for t, s_ in zip(T, sel) if s_], np.ones(sel.sum(), bool), days) if sel.sum() else (0, 0, 0)
                    r.update({f"{nm}_n": int(sel.sum()), f"{nm}_net": pcs.sum(), f"{nm}_Sharpe": sharpe(pcs), f"{nm}_MDDw": mws,
                              f"{nm}_win": np.mean([t["net"] > 0 for t, s_ in zip(T, sel) if s_]) if sel.sum() else np.nan})
                rows.append(r)
    R = pd.DataFrame(rows); pd.set_option("display.width", 250)
    R.to_csv(RES / "rqo_70_grid.csv", index=False)
    print(R.round(3).to_string(index=False))
    summ = {}
    for mode in "MW":
        g = R[R["mode"] == mode]
        sp = g.IS_Sharpe.rank().corr(g.OOS_Sharpe.rank())
        best = g.loc[g.IS_Sharpe.idxmax()]; base = g[(g.x == best.x) & g.frob_q.isna() & g.garch.isna()].iloc[0]
        key = (mode, best.x, None if pd.isna(best.frob_q) else best.frob_q, best.garch if isinstance(best.garch, tuple) else None)
        T, m, pc = keep[key]
        # 隨機 gate：同通過率，500 次
        sig = pd.DatetimeIndex([t["sig"] for t in T]); oos = sig >= IS_END
        per = np.array([t["net"] for t in T]) * MULT
        rnd = [per[oos & (rng.random(len(T)) < m.mean())].sum() for _ in range(500)]
        rk = (np.array(rnd) < per[oos & m].sum()).mean()
        # WF：每年用前 5 年選最佳 (x, filt)
        wf = []; picks = []
        for Y in range(2015 if mode == "M" else 2018, 2027):
            a, b = pd.Timestamp(f"{Y-5}-01-01"), pd.Timestamp(f"{Y}-01-01")
            sc = {k: sharpe(v[2][(v[2].index >= a) & (v[2].index < b)]) for k, v in keep.items() if k[0] == mode}
            kb = max(sc, key=sc.get); s = keep[kb][2]; wf.append(s[(s.index >= b) & (s.index < pd.Timestamp(f"{Y+1}-01-01"))]); picks.append((Y, kb[1:], round(sc[kb], 2)))
        wf = pd.concat(wf); wfo = wf[wf.index >= IS_END]
        summ[mode] = dict(base_pc=keep[(mode, best.x, None, None)][2], best=key, best_row=best, base=base, spearman=sp, rnd_pct=rk, rnd_med=np.median(rnd), wf=wf, wf_sh=sharpe(wfo), wf_net=wfo.sum(), picks=picks, pc=pc, T=T, m=m)
        print(f"\n== {mode} == best IS {key} | IS Sh {best.IS_Sharpe:.2f} OOS Sh {best.OOS_Sharpe:.2f} OOS net {best.OOS_net:.0f} | 無濾網同 x: IS {base.IS_Sharpe:.2f} OOS {base.OOS_Sharpe:.2f} net {base.OOS_net:.0f}")
        print(f"Spearman IS/OOS {sp:.2f}; 隨機 gate OOS 百分位 {rk:.2f} (中位數 {np.median(rnd):.0f})；WF 2022+ Sharpe {sharpe(wfo):.2f} net {wfo.sum():.0f}")
        print("WF picks:", picks)
    # 濾網平均效果：所有 x 下，濾網 vs 無濾網 OOS Sharpe 差
    R["filt"] = R.frob_q.astype(str) + "|" + R.garch.astype(str)
    print("\n濾網組合平均 (IS/OOS Sharpe):"); print(R.groupby(["mode", "filt"])[["IS_Sharpe", "OOS_Sharpe", "OOS_MDDw", "pass_rate"]].mean().round(3).to_string())
    chart(summ, fd, fp, gv, gp)


def chart(S, fd, fp, gv, gp):
    plt = plt_style()
    fig, axs = plt.subplots(3, 1, figsize=(12, 11), facecolor=BG, gridspec_kw=dict(height_ratios=[2, 2, 1.3]))
    for ax, mode in zip(axs[:2], "MW"):
        sty(ax); s = S[mode]; nm = "月選" if mode == "M" else "週選"
        base_key = (mode, s["best"][1], None, None)
        ax.plot(s["pc"].cumsum() / 1e4, color=GOLD, lw=2, label=f"IS 最佳 x={s['best'][1]:.0%} Frob≤{s['best'][2]} GARCH{s['best'][3]}")
        ax.plot(S[mode]["base_pc"].cumsum() / 1e4 if "base_pc" in s else s["pc"].cumsum() * 0, color=GR, lw=1.2, label="同 x 無濾網")
        ax.plot(s["wf"].cumsum() / 1e4, color=BLUE, lw=1.3, label="walk-forward")
        ax.axvline(IS_END, color="white", ls="--", lw=1)
        ax.set_title(f"賣 TXO {nm}價外 put｜IS Sharpe {s['best_row'].IS_Sharpe:.2f} → OOS {s['best_row'].OOS_Sharpe:.2f}（無濾網 OOS {s['base'].OOS_Sharpe:.2f}）")
        ax.set_ylabel("累積淨損益（萬元，1 口）"); ax.legend(facecolor=BG, labelcolor=WH, edgecolor=GR, fontsize=9, loc="lower left")
    ax = axs[2]; sty(ax); ax.plot(fp.index, fp.values, color=GOLD, lw=.7, label="Frobenius 距離百分位"); ax.plot(gp.index, gp.values, color=BLUE, lw=.7, alpha=.7, label="GARCH 波動百分位")
    ax.set_xlim(pd.Timestamp("2010-01-01"), END); ax.legend(facecolor=BG, labelcolor=WH, edgecolor=GR, fontsize=9); ax.set_title("kill switch 指標")
    foot(fig); fig.tight_layout(rect=[0, .03, 1, 1]); fig.savefig(RES / "rqo_70_put_selling.png", dpi=110, facecolor=BG); plt.close(fig)


if __name__ == "__main__":
    main()
