"""60 r/quant「Tested 80+ hypotheses」主要假說在台指期 TXF 1 分 K 的檢驗（research/reddit_quant_ideas.md #8）。
假說（皆日盤 08:45-13:45；夜盤歸屬次一交易日；交易不跨 contractMonth）：
 H4 gap 跳空回補：gap=日盤開−前日盤收，|gap|>k×ATR14(日盤高低) 時反向；第 2 根 1 分 K 開盤進場；停利=前收（限價），停損 none/1×gap；收盤平倉。
 H8 隔夜溢酬：做多 夜盤(開→收) / 日盤(開→收) / 收→開（日盤收→次日開，含夜盤+跳空）。
 H6 月底效應：月最後交易日前 n 日開盤買進，持有到次月第 m 個交易日收盤。
 SETT 結算日（第 3 個週三）扭曲反轉：結算日日盤（或結算週 5 日）方向，次日開盤 fade/follow，持有 h 日。
 SWP 前日高低掃單反轉：5 分 K 高點 > 前日日盤高+buf 且收回其下 → 下一根開盤放空（低點對稱）；停損=該根極值+1；出場 eod/前日中點/2R。12:30 後不進場。
 NOISE Zarattini 開盤波動突破：上下緣 = max/min(開, 前收) × (1 ± k×σ_t)，σ_t = 過去 14 日同時點 |收/開−1| 平均；
       每 f 分檢查收盤，突破下一根開盤進場；出場 = 回到帶內（反向帶 or 開盤價）或收盤。
 H1 日內動能：前 w 分報酬（自前收 or 自開盤）方向 → 最後 L 分做同向。
成本/成交/MDD 見 _rq.py。IS 2017-06~2021，OOS 2022+；WF 3 年滾動選參數；隨機方向基準。
"""
import sys, itertools
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent)); import _rq as R

IS_END = pd.Timestamp("2022-01-01")


def day_bars(d, rule):
    x = d[~d.night]
    g = x.groupby(["tdate", x.ts.dt.floor(rule)])
    b = g.agg(o=("open", "first"), h=("high", "max"), l=("low", "min"), c=("close", "last"), cm=("cm", "last")).reset_index()
    b["k"] = b.groupby("tdate").cumcount()
    return b


def main():
    d = R.load_1m(); D, N = R.sessions(d)
    o, h, l, c = (d[k].values for k in ("open", "high", "low", "close"))
    days = D.index; alld = days
    pc = D.c.shift(); pcm = D.cm1.shift(); atr = (D.h - D.l).rolling(14).mean().shift()
    same = (D.cm0 == pcm)
    res = {}; summary = []

    def finish(name, results, min_tr=30, rnd=True):
        G, DP, sp = R.grid_eval(results, alld, IS_END, min_tr)
        ok = G[G.IS_trades >= min_tr]; bk = ok.sort_values("IS_Sharpe", ascending=False).key.iloc[0]
        wfp, wfs = R.walk_forward(G, DP, alld, 2020, min_tr=min_tr)
        bt = results[bk]; rd = R.rand_dir(bt, alld, IS_END) if rnd else None
        s = R.report(name, G, sp, bk, bt, alld, IS_END, wfp, wfs, rd)
        s["rand_OOS_p95"] = np.percentile(rd, 95) if rnd else np.nan
        summary.append(s); res[name] = (G, bt, wfs)
        G.to_csv(R.OUT / f"rq60_{name}_grid.csv", index=False)

    # ---- H4 gap fade
    out = {}
    for k, ex, st in itertools.product((0.0, 0.1, 0.2, 0.3, 0.5), ("fill", "close"), ("none", "gap")):
        rows = []
        for t in days:
            r = D.loc[t]
            if not same[t] or not atr[t] == atr[t]: continue
            gap = r.o - pc[t]
            if abs(gap) <= k * atr[t] or gap == 0: continue
            dr = -np.sign(gap); j0 = r.i0 + 1; ep = o[j0] + dr
            tg = pc[t] if ex == "fill" else np.nan; sp_ = (r.o + np.sign(gap) * abs(gap)) if st == "gap" else np.nan
            xp, _, mae, _ = R.manage(o, h, l, c, j0, r.i1, dr, ep, sp_, tg)
            rows.append((t, dr, ep, xp, mae))
        out[(k, ex, st)] = R.mk_trades(rows)
    finish("H4_gapfade", out)

    # ---- H8 overnight premium（固定做多，不最佳化；4 種持有區段）
    out = {}
    nn = N.reindex(days)
    for seg in ("night", "day", "close2open", "night_short"):
        rows = []
        for i, t in enumerate(days):
            r = D.loc[t]
            if seg == "day": rows.append((t, 1, r.o + 1, r.c - 1, r.l - r.o - 1))
            elif seg in ("night", "night_short"):
                n = nn.loc[t]
                if n.isna().o or n.cm0 != n.cm1: continue
                dr = 1 if seg == "night" else -1
                rows.append((t, dr, n.o + dr, n.c - dr, (n.l - n.o if dr > 0 else n.o - n.h) - 1))
            else:
                if i == 0 or not same[t]: continue
                lo = min(r.o, nn.loc[t].l) if nn.loc[t].l == nn.loc[t].l else r.o
                rows.append((t, 1, pc[t] + 1, r.o - 1, lo - pc[t] - 1))
        out[seg] = R.mk_trades(rows)
    finish("H8_overnight", out, rnd=False)

    # ---- H6 month-end
    out = {}
    mo = pd.Series(days.to_period("M"), index=days); last = mo.ne(mo.shift(-1)).values
    li = np.where(last)[0]
    for n_in, m in itertools.product((0, 1, 2, 3, 4), (1, 2, 3)):
        rows = []
        for e in li:
            a, b = e - n_in, e + m
            if a < 1 or b >= len(days): continue
            seg = D.iloc[a:b + 1]
            if seg.cm0.iloc[0] != seg.cm1.iloc[-1]: continue
            ep = seg.o.iloc[0] + 1; rows.append((days[b], 1, ep, seg.c.iloc[-1] - 1, seg.l.min() - ep))
        out[(n_in, m)] = R.mk_trades(rows)
    finish("H6_monthend", out, min_tr=20)

    # ---- settlement day (3rd Wed; 以 contractMonth 變更確認)
    out = {}
    sett = [i for i in range(len(days) - 1) if D.cm1.iloc[i] != D.cm0.iloc[i + 1]]
    for sig, mode, hh in itertools.product(("sday", "week"), ("fade", "follow"), (1, 2, 3)):
        rows = []
        for i in sett:
            if i < 5 or i + hh >= len(days): continue
            mv = (D.c.iloc[i] - D.o.iloc[i]) if sig == "sday" else (D.c.iloc[i] - D.c.iloc[i - 5])
            if mv == 0: continue
            dr = -np.sign(mv) if mode == "fade" else np.sign(mv)
            seg = D.iloc[i + 1:i + 1 + hh]
            if seg.cm0.iloc[0] != seg.cm1.iloc[-1]: continue
            ep = seg.o.iloc[0] + dr; xp = seg.c.iloc[-1] - dr
            mae = (seg.l.min() - ep) if dr > 0 else (ep - seg.h.max())
            rows.append((days[i + hh], dr, ep, xp, mae))
        out[(sig, mode, hh)] = R.mk_trades(rows)
    finish("SETT_reversal", out, min_tr=20)

    # ---- sweep reversal on 5m
    b = day_bars(d, "5min"); bo, bh, bl, bc = (b[k].values for k in ("o", "h", "l", "c"))
    st = b.groupby("tdate").apply(lambda x: (x.index[0], x.index[-1]))
    PH, PL = D.h.shift(), D.l.shift(); cut = b.ts.dt.hour * 60 + b.ts.dt.minute
    cutv = cut.values
    out = {}
    for buf, tg in itertools.product((0.0, 0.05, 0.1, 0.2), ("eod", "mid", "2R")):
        rows = []
        for t in days[1:]:
            if not same[t] or not atr[t] == atr[t]: continue
            s0, s1 = st[t]; ph, pl = PH[t], PL[t]; bf = buf * atr[t]; j = s0; done = {1: False, -1: False}
            while j < s1:
                if cutv[j] > 12 * 60 + 30: break
                dr = 0
                if not done[-1] and bh[j] > ph + bf and bc[j] < ph: dr = -1; ext = bh[j]
                elif not done[1] and bl[j] < pl - bf and bc[j] > pl: dr = 1; ext = bl[j]
                if dr == 0: j += 1; continue
                done[dr] = True; j0 = j + 1; ep = bo[j0] + dr; stp = ext - dr * 1.0; rk = abs(ep - stp)
                if rk <= 0: j += 1; continue
                tgt = {"eod": np.nan, "mid": (ph + pl) / 2, "2R": ep + dr * 2 * rk}[tg]
                if tg == "mid" and dr * (tgt - ep) <= 0: tgt = np.nan
                xp, _, mae, xj = R.manage(bo, bh, bl, bc, j0, s1, dr, ep, stp, tgt)
                rows.append((t, dr, ep, xp, mae)); j = xj + 1
        out[(buf, tg)] = R.mk_trades(rows)
    finish("SWP_sweep", out)

    # ---- noise area (Zarattini) on 5m
    B = b.pivot(index="tdate", columns="k", values="c").reindex(days)
    O0 = D.o.values[:, None]; mv = (B.values / O0 - 1).__abs__()
    sig_ = pd.DataFrame(mv, index=days).rolling(14, min_periods=10).mean().shift().values
    idx = b.pivot(index="tdate", columns="k", values=b.index.name or "index") if False else None
    b["gi"] = np.arange(len(b)); GI = b.pivot(index="tdate", columns="k", values="gi").reindex(days).values
    out = {}
    for kk, f, ex in itertools.product((1.0, 1.5, 2.0), (15, 30), ("band", "open")):
        step = f // 5; rows = []
        for i, t in enumerate(days):
            if i == 0 or not same[t]: continue
            up0 = max(D.o.iloc[i], pc.iloc[i]); dn0 = min(D.o.iloc[i], pc.iloc[i]); pos = 0; ep = 0.0; mae = 0.0; ndx = GI[i]
            nk = int(np.sum(~np.isnan(ndx)))
            for k in range(step - 1, nk - 1, step):
                s = sig_[i, k]
                if s != s: continue
                up = up0 * (1 + kk * s); dn = dn0 * (1 - kk * s); cc = B.values[i, k]; jn = int(ndx[k + 1])
                if pos != 0:
                    lvl = (up if pos > 0 else dn) if ex == "band" else D.o.iloc[i]
                    if (pos > 0 and cc < lvl) or (pos < 0 and cc > lvl):
                        xp = bo[jn] - pos; rows.append((t, pos, ep, xp, mae)); pos = 0
                if pos == 0:
                    nd = 1 if cc > up else (-1 if cc < dn else 0)
                    if nd: pos = nd; ep = bo[jn] + nd; mae = 0.0; j_ent = jn
                if pos != 0:
                    seg_l = bl[j_ent:int(ndx[min(k + step, nk - 1)]) + 1].min(); seg_h = bh[j_ent:int(ndx[min(k + step, nk - 1)]) + 1].max()
                    mae = min(mae, (seg_l - ep) if pos > 0 else (ep - seg_h))
            if pos != 0: rows.append((t, pos, ep, bc[int(ndx[nk - 1])] - pos, mae))
        out[(kk, f, ex)] = R.mk_trades(rows)
    finish("NOISE_breakout", out)

    # ---- H1 intraday momentum
    out = {}
    for w, ref, L in itertools.product((30, 60), ("prevclose", "open"), (15, 30)):
        rows = []
        for i, t in enumerate(days):
            if i == 0 or not same[t]: continue
            r = D.loc[t]; jw = r.i0 + w - 1; jl = r.i1 - L + 1
            if jw >= jl or d.tdate.values[jl] != np.datetime64(t): continue
            mv = c[jw] - (pc.iloc[i] if ref == "prevclose" else r.o)
            if mv == 0: continue
            dr = np.sign(mv); ep = o[jl] + dr; xp, _, mae, _ = R.manage(o, h, l, c, jl, r.i1, dr, ep, np.nan, np.nan)
            rows.append((t, dr, ep, xp, mae))
        out[(w, ref, L)] = R.mk_trades(rows)
    finish("H1_intraday_mom", out)

    S = pd.DataFrame(summary); S.to_csv(R.OUT / "rq60_summary.csv", index=False); print(S.round(3).to_string())
    plt = R.style(); fig, ax = plt.subplots(len(res) // 2 + len(res) % 2, 2, figsize=(16, 4 * ((len(res) + 1) // 2)))
    for a, (nm, (G, bt, wfs)) in zip(ax.flat, res.items()):
        dp, _ = R.daily_from_trades(bt, alld); a.plot(alld, dp.cumsum() / 1e4, label="IS 最佳參數")
        if len(wfs): a.plot(wfs.index, wfs.cumsum() / 1e4, label="WF")
        a.axvline(IS_END, c=R.GRAY, ls="--"); a.set_title(nm); a.grid(alpha=.3); a.legend(fontsize=8); a.set_ylabel("萬元/1口")
    fig.suptitle("r/quant 80+ 假說 × 台指期（扣成本）", color=R.GOLD); fig.tight_layout(); fig.savefig(R.OUT / "rq_60_hypotheses.png", dpi=100)
    print("chart results/rq_60_hypotheses.png")


if __name__ == "__main__":
    main()
