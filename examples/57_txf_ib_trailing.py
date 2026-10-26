"""57 IB 突破（56）出場改良：保本、ATR 移動停損、N 根 K 吊燈停損、組合、時間出場。

基準：56 的 IS 最佳 = nus（美股開盤後 1 小時 IB）、停損單 +1 tick、寬 IB（>過去 20 盤中位數）、無 bias、
初始停損 IB 另一側、每方向 1 筆、抱到收盤（05:00）。成本、成交同 55/56。
停損觸發：以 min(停損價, 開盤)（多單）成交再扣 1 點滑價；移動停損於 K 棒收完後更新，下一根才生效（無偷看）。
成交當根只檢查初始停損。同根停損 + 停利 → 停損（只有固定停利的 top10 參數會遇到）。
出場變體：
  be{k}           ：有利幅度（MFE）達 k R 後停損移到進場價。
  atr{m}_a{0|1}   ：1 分 K ATR14 × m 的吊燈停損（自進場後最高點回落）；a1 = 達 1R 才啟動。
  datr{f}_a{0|1}  ：日 ATR（前 14 個日盤高低差平均）× f。
  nb{N}_a{0|1}    ：前 N 根 1 分 K 最低點（空單最高點）；nb5m{N} = N 根 5 分 K ≈ 5N 根 1 分 K（近似，未對齊 5 分邊界）。
  be1+atr{m} / be1+datr{f}：保本 1R + 移動停損。
  t0100 / t0300   ：改在 01:00 / 03:00 平倉。
參數池以本檔核心重跑 56（成交當根停損改為 min(開盤, 停損) − 1，與 56 有極小差異）。
IS 2017-06~2021 選出場；OOS 2022+ 報告。walk-forward：56 的 10,800 組 + 本檔變體（nus / d60+nus、抱到收盤的組合）一起選。
"""
import sys, math, importlib.util
from pathlib import Path
import numpy as np, pandas as pd
from numba import njit

HERE = Path(__file__).resolve().parent
def _imp(n, f):
    sp = importlib.util.spec_from_file_location(n, HERE / f); m = importlib.util.module_from_spec(sp); sp.loader.exec_module(m); return m
ib = _imp("ib56", "56_txf_ib_breakout.py")
OUT = Path("results"); PV = 200.0; CAP = 1_000_000; IS_END = pd.Timestamp("2022-01-01")
BASE = (("stop", 0.0), "gt50_20", "none", "opp", 0.0, "dir")

VARIANTS = {"base": (0, 0, 0, 0, 0)}  # name: (be_k, ttype, tpar, tact, fe_mode)  ttype 1 atr1m, 2 datr, 3 nbar; fe_mode 0 close,1 01:00,2 03:00
for k in (0.5, 1, 1.5, 2): VARIANTS[f"be{k}"] = (k, 0, 0, 0, 0)
for a in (0, 1):
    for m in (5, 10, 20): VARIANTS[f"atr{m}_a{a}"] = (0, 1, m, a, 0)
    for f in (0.25, 0.5, 1.0): VARIANTS[f"datr{f}_a{a}"] = (0, 2, f, a, 0)
    for N in (5, 15, 30, 60): VARIANTS[f"nb{N}_a{a}"] = (0, 3, N, a, 0)
    for N in (5, 15, 30, 60): VARIANTS[f"nb5m{N}_a{a}"] = (0, 3, 5 * N, a, 0)
for m in (5, 10, 20): VARIANTS[f"be1+atr{m}"] = (1, 1, m, 0, 0)
for f in (0.25, 0.5): VARIANTS[f"be1+datr{f}"] = (1, 2, f, 0, 0)
VARIANTS["t0100"] = (0, 0, 0, 0, 1); VARIANTS["t0300"] = (0, 0, 0, 0, 2)


@njit(cache=True)
def manage2(o, h, l, c, A, j0, fe, dr, ep, stp, tg, risk, bek, tt, tp, ta, DA):
    hh = h[j0] if dr > 0 else l[j0]
    for j in range(j0 + 1, fe):
        if dr > 0:
            hs = l[j] <= stp; ht = h[j] >= tg
            if hs: return min(o[j], stp) - 1.0, 1, 1 if ht else 0, j
            if ht: return max(o[j], tg), 2, 0, j
            hh = max(hh, h[j]); mfe = hh - ep
        else:
            hs = h[j] >= stp; ht = l[j] <= tg
            if hs: return max(o[j], stp) + 1.0, 1, 1 if ht else 0, j
            if ht: return min(o[j], tg), 2, 0, j
            hh = min(hh, l[j]); mfe = ep - hh
        ns = stp
        if bek > 0 and mfe >= bek * risk: ns = ep
        if tt > 0 and (ta == 0 or mfe >= ta * risk):
            if tt == 1: v = hh - dr * tp * A[j]
            elif tt == 2: v = hh - dr * tp * DA
            else:
                n = int(tp); v = l[j] if dr > 0 else h[j]
                for q in range(max(0, j - n + 1), j + 1):
                    v = min(v, l[q]) if dr > 0 else max(v, h[q])
            ns = max(ns, v) if dr > 0 else min(ns, v)
        stp = max(stp, ns) if dr > 0 else min(stp, ns)
    return c[fe - 1] - dr * 1.0, 3, 0, fe - 1


@njit(cache=True)
def run(o, h, l, c, atr, E, CE, FE, IBH, IBL, ALLOW, BD, DAT, mode, kbuf, stopm, tm, one, bek, tt, tp, ta):
    n = len(E); out = np.full((n, 2, 7), np.nan)
    for k in range(n):
        if not ALLOW[k]: continue
        hi = IBH[k]; lo = IBL[k]; W = hi - lo
        if W <= 0: continue
        usedL = False; usedS = False; armL = False; armS = False
        j = E[k]; ce = min(CE[k], FE[k])
        while j < ce:
            if one and (usedL or usedS): break
            canL = (not usedL) and BD[k] >= 0; canS = (not usedS) and BD[k] <= 0
            if not canL and not canS: break
            dr = 0; ep = 0.0
            if mode == 0:
                buf = 1.0 + kbuf * atr[j - 1]; tL = hi + buf; tS = lo - buf
                hl = canL and h[j] >= tL; hs = canS and l[j] <= tS
                if hl and hs:
                    if o[j] >= tL: hs = False
                    elif o[j] <= tS: hl = False
                    else: hl = False; hs = False
                if hl: dr = 1; ep = max(o[j], tL) + 1.0
                elif hs: dr = -1; ep = min(o[j], tS) - 1.0
            else:
                if armL and canL and l[j] <= hi - 1.0: dr = 1; ep = min(o[j], hi)
                elif armS and canS and h[j] >= lo + 1.0: dr = -1; ep = max(o[j], lo)
                if c[j] > hi: armL = True
                if c[j] < lo: armS = True
            if dr == 0:
                j += 1; continue
            if stopm == 0: stp = lo if dr > 0 else hi
            elif stopm == 1: stp = (hi + lo) / 2
            else: stp = ep - dr * 0.25 * W
            risk = dr * (ep - stp)
            if risk <= 0: stp = ep - dr * 0.25 * W; risk = 0.25 * W
            tg = ep + dr * tm * W if tm > 0 else dr * 1e9
            slot = 0 if dr > 0 else 1
            if (dr > 0 and l[j] <= stp) or (dr < 0 and h[j] >= stp):
                xp = (min(o[j], stp) - 1.0) if dr > 0 else (max(o[j], stp) + 1.0); kind = 1; amb = 0; xj = j
            else:
                xp, kind, amb, xj = manage2(o, h, l, c, atr, j, FE[k], dr, ep, stp, tg, risk, bek, tt, tp, ta, DAT[k])
            out[k, slot, 0] = dr; out[k, slot, 1] = ep; out[k, slot, 2] = stp; out[k, slot, 3] = xp
            out[k, slot, 4] = kind; out[k, slot, 5] = amb; out[k, slot, 6] = risk
            if dr > 0: usedL = True
            else: usedS = True
            armL = False; armS = False; j = xj + 1
    return out


def prep(d, I):
    ts = d.ts.values.astype("datetime64[ns]").astype(np.int64)
    reg = d[~d.night]; dr = reg.groupby("skey").agg(h=("high", "max"), l=("low", "min"))
    datr = (dr.h - dr.l).rolling(14).mean().shift(1)  # 前 14 個日盤
    # 夜盤 tdate = 次一交易日 → 用 tdate 前的日盤 ATR（含 tdate 前一日，即當晚之前的日盤）
    I["datr"] = datr.reindex(I.tdate.values, method="ffill").values
    I.loc[I.typ.str.startswith("d"), "datr"] = datr.reindex(I.tdate[I.typ.str.startswith("d")].values).values
    fe1, fe3 = [], []
    for e, fe in zip(I.e.values, I.fe.values):
        t0 = pd.Timestamp(d.ts.values[e]); base = t0.normalize() + (pd.Timedelta(days=1) if t0.hour >= 12 else pd.Timedelta(0))
        if I_is_day(t0): fe1.append(fe); fe3.append(fe); continue
        for H, lst in ((1, fe1), (3, fe3)):
            x = np.searchsorted(ts, np.datetime64(base + pd.Timedelta(hours=H), "ns").astype(np.int64)); lst.append(int(min(max(x, e + 1), fe)))
    I["fe1"] = fe1; I["fe3"] = fe3
    return I


def I_is_day(t0): return 8 <= t0.hour < 14


def sim(I, arr, typ, p, v):
    (em, kb), wf, bi, sm, tm, mx = p; bek, tt, tp, ta, fm = VARIANTS[v]
    J = I[I.typ == typ]; allow = np.ones(len(J), bool)
    if wf != "none":
        side, L_ = wf.split("_"); pc = J[f"pct{L_}"].values; allow = ((pc < .5) if side == "lt50" else (pc > .5)) & ~np.isnan(pc)
    bd = np.zeros(len(J)) if bi == "none" else J["b_" + bi].values.astype(float)
    if bi != "none": allow &= bd != 0
    if tt == 2: allow &= ~np.isnan(J.datr.values)
    FE = J[("fe", "fe1", "fe3")[fm]].values
    o, h, l, c, a = arr
    out = run(o, h, l, c, a, J.e.values, J.ce.values, FE, J.ibh.values, J.ibl.values, allow, bd, np.nan_to_num(J.datr.values),
              0 if em == "stop" else 1, kb, ib.STOPM.index(sm), tm, mx == "one", float(bek), tt, float(tp), float(ta))
    return ib.to_trades(out, J)


def simw(I, arr, win, p, v):
    return pd.concat([sim(I, arr, ty, p, v) for ty in ib.COMBOS.get(win, (win,))])


def full_metrics(t, alld, lo, hi):
    t = t[(t.tdate >= lo) & (t.tdate < hi)]; ad = alld[(alld >= lo) & (alld < hi)]
    dp = t.groupby("tdate").net.sum().reindex(ad, fill_value=0) * PV; eq = dp.cumsum(); pk = eq.cummax()
    w = t.net[t.net > 0].sum(); ls = -t.net[t.net < 0].sum()
    # 最長回撤期間（日曆天）：自高點到回到高點（或期末）
    under = (eq < pk).values; longest = 0; start = None
    for i, u in enumerate(under):
        if u and start is None: start = i - 1 if i > 0 else 0
        if (not u or i == len(under) - 1) and start is not None:
            longest = max(longest, (ad[i] - ad[start]).days); start = None
    tot = t.net.sum()
    return dict(trades=len(t), win=(t.net > 0).mean(), avgR=t.R.mean(), PF=w / ls if ls > 0 else np.nan, net_twd=tot * PV,
                Sharpe=dp.mean() / dp.std() * math.sqrt(252) if dp.std() > 0 else np.nan, MDD_twd=(eq - pk).min(),
                top10_share=t.net.nlargest(10).sum() / tot if tot > 0 else np.nan, longestDD_days=longest)


def main():
    d = ib.load_bars(); I = prep(d, ib.build(d)); arr = tuple(d[k].values for k in ("open", "high", "low", "close", "atr"))
    alld = pd.DatetimeIndex(sorted(I.tdate.unique())); nd = len(alld); dpos = pd.Series(np.arange(nd), index=alld)
    end = alld[-1] + pd.Timedelta(days=1); isd = alld < IS_END
    per = [("IS", alld[0], IS_END), ("OOS", IS_END, end), ("FULL", alld[0], end)]
    pd.set_option("display.width", 300); pd.set_option("display.max_columns", 50); pd.set_option("display.max_rows", 300)
    # 1) 基準組合 × 全部變體
    rows = []; TT = {}
    for v in VARIANTS:
        t = simw(I, arr, "nus", BASE, v); TT[v] = t
        r = {"variant": v}
        for pn, lo, hi in per: r.update({f"{pn}_{k}": x for k, x in full_metrics(t, alld, lo, hi).items()})
        x25 = t[(t.tdate >= IS_END) & (~t.tdate.dt.year.isin([2025, 2026]))]; x5 = t[(t.tdate >= IS_END) & (t.tdate.dt.year != 2025)]
        r["OOS_ex25_twd"] = x5.net.sum() * PV; r["OOS_ex25_26_twd"] = x25.net.sum() * PV; r["amb"] = int(t.amb.sum())
        rows.append(r)
    V = pd.DataFrame(rows).set_index("variant"); V.to_csv(OUT / "ibt_variants.csv")
    c1 = ["IS_trades", "IS_win", "IS_avgR", "IS_PF", "IS_net_twd", "IS_Sharpe", "IS_MDD_twd", "IS_top10_share", "IS_longestDD_days",
          "OOS_trades", "OOS_win", "OOS_avgR", "OOS_PF", "OOS_net_twd", "OOS_Sharpe", "OOS_MDD_twd", "OOS_top10_share", "OOS_longestDD_days", "OOS_ex25_twd", "OOS_ex25_26_twd", "amb"]
    print("== base config × exit variants (sorted by IS Sharpe) ==")
    print(V.sort_values("IS_Sharpe", ascending=False)[c1].round(3).to_string())
    sel = V.IS_Sharpe.idxmax(); print("\nIS-selected exit:", sel)
    selmdd = V[V.IS_Sharpe >= V.loc["base", "IS_Sharpe"] * 0.9].IS_MDD_twd.idxmax(); print("IS-selected min-MDD with IS Sharpe ≥ 90% base:", selmdd)
    yr = pd.DataFrame({v: TT[v].groupby(TT[v].tdate.dt.year).net.sum() * PV for v in dict.fromkeys(["base", sel, selmdd, "be1", "atr10_a1", "nb60_a1", "t0300"])}).round(0)
    print("\nyearly TWD:"); print(yr.to_string())
    # 2) 一致性：IS top10 家族、nus / d60+nus 抱到收盤組合
    G = pd.read_csv(OUT / "ib_grid.csv") if (OUT / "ib_grid.csv").exists() else None
    def pkey(r):
        e = ("retest", 0.0) if r.entry == "retest" else ("stop", float(r.entry.split("_k")[1]))
        return (e, r.wfil, r.bias, r.stop, float(r.tgt), r.maxt)
    fam = []
    top10 = G[G.IS_trades >= 150].sort_values("IS_Sharpe", ascending=False).head(10)
    subset = G[G.win.isin(["nus", "d60+nus"]) & (G.tgt == 0)]
    print(f"\nconsistency: top10 family + {len(subset)} nus/d60+nus hold-to-close configs × {len(VARIANTS)} variants")
    cons = []
    for src, df in (("top10", top10), ("subset", subset)):
        for _, r in df.iterrows():
            p = pkey(r)
            for v in VARIANTS:
                if src == "subset" and v != "base" and False: pass
                t = simw(I, arr, r.win, p, v); t["_di"] = dpos.reindex(t.tdate).values
                dv = np.bincount(t._di.values, weights=t.net.values * PV, minlength=nd)
                eqi = dv[isd].cumsum(); eqo = dv[~isd].cumsum()
                cons.append(dict(src=src, win=r.win, p=str(p), variant=v, IS_Sharpe=ib.sharpe(dv[isd]), OOS_Sharpe=ib.sharpe(dv[~isd]),
                                 IS_MDD=(eqi - np.maximum.accumulate(eqi)).min(), OOS_MDD=(eqo - np.maximum.accumulate(eqo)).min(),
                                 OOS_top10=t[t.tdate >= IS_END].net.nlargest(10).sum() / max(t[t.tdate >= IS_END].net.sum(), 1e-9),
                                 IS_trades=int((t.tdate < IS_END).sum()), dv=dv.astype(np.float32)))
    C = pd.DataFrame(cons)
    base_c = C[C.variant == "base"].set_index(["src", "win", "p"])
    C = C.join(base_c[["IS_Sharpe", "OOS_Sharpe", "IS_MDD", "OOS_MDD"]], on=["src", "win", "p"], rsuffix="_b")
    for k in ("IS_Sharpe", "OOS_Sharpe"): C[f"d{k}"] = C[k] - C[f"{k}_b"]
    C["dOOS_MDD"] = C.OOS_MDD - C.OOS_MDD_b  # >0 = 回撤變小
    agg = C[C.variant != "base"].groupby(["src", "variant"]).agg(dIS=("dIS_Sharpe", "mean"), dOOS=("dOOS_Sharpe", "mean"),
            IS_better=("dIS_Sharpe", lambda x: (x > 0).mean()), OOS_better=("dOOS_Sharpe", lambda x: (x > 0).mean()),
            OOS_MDD_smaller=("dOOS_MDD", lambda x: (x > 0).mean()), OOS_top10=("OOS_top10", "median")).round(3)
    print(agg.to_string())
    # 子集內每組自選 IS 最佳出場，OOS 表現 vs base
    sub = C[C.src == "subset"]
    pick = sub.loc[sub.groupby(["win", "p"]).IS_Sharpe.idxmax()]
    print("\nsubset: per-config IS-chosen exit → OOS Sharpe mean %.3f vs base %.3f; better in %.2f; chosen exits:" % (pick.OOS_Sharpe.mean(), pick.OOS_Sharpe_b.mean(), (pick.dOOS_Sharpe > 0).mean()))
    print(pick.variant.value_counts().head(10).to_string())
    # 3) walk-forward：56 全部組合 + 子集變體
    print("\nrebuilding 56 pool for walk-forward ...")
    pool_dv, pool_lab, pool_n = [], [], []
    T56 = {}
    for typ in ib.TYPES:
        for p in ib.param_iter():
            t = sim(I, arr, typ, p, "base"); t["_di"] = dpos.reindex(t.tdate).values; T56[(typ, p)] = t
    def addp(lab, ts_):
        t = pd.concat(ts_); dv = np.bincount(t._di.values, weights=t.net.values * PV, minlength=nd)
        pool_dv.append(dv.astype(np.float32)); pool_lab.append(lab); pool_n.append(np.bincount(t._di.values, minlength=nd))
    for typ in ib.TYPES:
        for p in ib.param_iter(): addp((typ, str(p), "56"), [T56[(typ, p)]])
    for cn, (a, b) in ib.COMBOS.items():
        for p in ib.param_iter(): addp((cn, str(p), "56"), [T56[(a, p)], T56[(b, p)]])
    n56 = len(pool_dv)
    for _, r in sub[sub.variant != "base"].iterrows():
        pool_dv.append(r.dv); pool_lab.append((r.win, r.p, r.variant)); pool_n.append(None)
    DP = np.vstack(pool_dv); print("pool", DP.shape, "of which 56:", n56)
    # 交易數門檻：以 IS 交易頻率估計
    freq = np.array([nn[isd].sum() / isd.sum() if nn is not None else np.nan for nn in pool_n])
    isfreq = dict(zip(zip(sub.win, sub.p, sub.variant), sub.IS_trades / isd.sum()))
    for i in range(n56, len(pool_lab)): freq[i] = isfreq[pool_lab[i]]
    def wf(mask_pool, lab):
        picks, out = [], []
        for Y in range(2020, alld.year.max() + 1):
            trm = (alld.year >= Y - 3) & (alld.year < Y); tsm = alld.year == Y
            X = DP[:, trm]; sh = X.mean(1) / X.std(1) * math.sqrt(252); sh[~mask_pool | (freq * trm.sum() < 90)] = np.nan
            b = int(np.nanargmax(sh)); y = DP[b, tsm].astype(float); out.append(pd.Series(y, index=alld[tsm]))
            picks.append(dict(year=Y, pick=pool_lab[b], train=round(sh[b], 3), test=round(ib.sharpe(y), 3), test_twd=round(y.sum())))
        s = pd.concat(out); eq = s.cumsum()
        print(f"\n== WF {lab} =="); print(pd.DataFrame(picks).to_string(index=False))
        ex = s[~s.index.year.isin([2025])]; ex2 = s[~s.index.year.isin([2025, 2026])]
        print("stitched net %.0f Sharpe %.3f MDD %.0f | ex2025 Sharpe %.3f net %.0f | ex2025-26 Sharpe %.3f net %.0f" % (s.sum(), ib.sharpe(s.values), (eq - eq.cummax()).min(), ib.sharpe(ex.values), ex.sum(), ib.sharpe(ex2.values), ex2.sum()))
        return s
    m56 = np.zeros(len(DP), bool); m56[:n56] = True
    w56 = wf(m56, "56 pool only"); wall = wf(np.ones(len(DP), bool), "56 pool + exit variants")
    chart(TT, V, sel, selmdd, alld, agg, w56, wall)


def chart(TT, V, sel, selmdd, alld, agg, w56, wall):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = "Noto Sans CJK TC"; plt.rcParams["axes.unicode_minus"] = False
    fig, ax = plt.subplots(1, 2, figsize=(17, 6)); a = ax[0]
    for v in dict.fromkeys(["base", sel, selmdd, "be1", "atr10_a1", "nb60_a1", "t0300"]):
        t = TT[v]; a.plot(alld, (t.groupby("tdate").net.sum().reindex(alld, fill_value=0) * PV).cumsum() / 1e4, label=v, lw=1.6 if v in ("base", sel) else 1)
    a.axvline(IS_END, c="gray", ls="--"); a.legend(); a.grid(alpha=.3); a.set_ylabel("累積淨損益（萬元，1 口）"); a.set_title("nus IB 突破：出場變體")
    a = ax[1]; a.plot(w56.index, w56.cumsum() / 1e4, label="WF：56 參數池"); a.plot(wall.index, wall.cumsum() / 1e4, label="WF：56 + 出場變體")
    a.legend(); a.grid(alpha=.3); a.set_title("Walk-forward（3 年滾動）"); a.set_ylabel("萬元")
    fig.tight_layout(); fig.savefig(OUT / "ibt_equity.png", dpi=100); plt.close(fig)
    fig, ax = plt.subplots(1, 3, figsize=(20, 9)); Vs = V.sort_values("IS_Sharpe")
    y = np.arange(len(Vs))
    ax[0].barh(y - .2, Vs.IS_Sharpe, .4, label="IS"); ax[0].barh(y + .2, Vs.OOS_Sharpe, .4, label="OOS"); ax[0].set_title("Sharpe"); ax[0].legend()
    ax[1].barh(y - .2, Vs.IS_MDD_twd / 1e4, .4, label="IS"); ax[1].barh(y + .2, Vs.OOS_MDD_twd / 1e4, .4, label="OOS"); ax[1].set_title("MDD（萬元）")
    ax[2].barh(y, Vs.OOS_top10_share, .6, color="C2"); ax[2].set_title("OOS 前 10 大單占淨利比例")
    for a in ax: a.set_yticks(y); a.set_yticklabels(Vs.index, fontsize=8); a.grid(alpha=.3, axis="x")
    fig.tight_layout(); fig.savefig(OUT / "ibt_variants.png", dpi=100); plt.close(fig)
    S = agg.loc["subset"].sort_values("dIS")
    fig, a = plt.subplots(figsize=(11, 9)); y = np.arange(len(S))
    a.barh(y - .2, S.dIS, .4, label="ΔIS Sharpe"); a.barh(y + .2, S.dOOS, .4, label="ΔOOS Sharpe"); a.axvline(0, c="k")
    a.set_yticks(y); a.set_yticklabels(S.index, fontsize=8); a.legend(); a.grid(alpha=.3, axis="x"); a.set_title("nus / d60+nus 抱到收盤組合：出場變體 vs 基準（平均差）")
    fig.tight_layout(); fig.savefig(OUT / "ibt_consistency.png", dpi=100)
    print("charts results/ibt_equity.png results/ibt_variants.png results/ibt_consistency.png")


if __name__ == "__main__":
    main()
