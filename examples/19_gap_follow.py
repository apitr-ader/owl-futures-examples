"""19 開盤跳空「順勢」（Gap Follow；TradingView Gap Filling Strategy 的 invert=true）

執行：
  python examples/19_gap_follow.py --five data/TXF_1c_5m_sessions.parquet --results-dir results

與 18 相同的資料、跳空定義（日盤 08:45 開盤 − 前日盤收盤 ref=day／前夜盤收盤 ref=night）、成本、樣本切分、選參方式，
只是方向反過來：向上跳空做多、向下跳空做空。順勢單「回補目標」沒有意義，所以：
- 出場：日盤收盤（13:45，時間出場）或停損。
- 停損 stop：ref＝跳空參考價（缺口被回補就認輸；原始 invert=true 的紅線就是停損）、s×ATR（s=0.25…1.5，自進場價）、none。
- 移動停損 trail（只對 s×ATR 有效）：停損價＝max(初始停損, 之前各根最高（多）− s×ATR)，只用已完成的 K（不偷看）。
- 同一根跳空越過停損價以開盤成交。
- 網格：k（最小跳空 ×ATR）× stop × d（進場延遲 1,2,3,6 根 5 分）× trail(off/on)；高原在 (k, stop, d) 三維、每個 trail 值內算，再取最高。
- 對照：原始預設反向（前一 15 分 K 高低點外跳空、第一根 15 分未回補、09:00 進場、停損＝前收、收盤出場）；
  天真「永遠順勢」（每天依跳空方向，d=1，無停損，收盤出場）。

注意：反向一個被成本拖累的策略，不會把成本的正負號翻過來——18 的逆勢版若「毛利」接近 0、淨損來自來回成本，
順勢版毛利約為其相反數（出場規則不同，並非精確相反），但每筆照樣要付 2×(手續費+滑價)+期交稅，淨利仍可能為負。
報表同時列出每筆平均成本與毛利，讓這件事看得見。

※ 純技術範例，非投資建議。
"""

import argparse
import importlib.util
import itertools
import math
from pathlib import Path

import numpy as np
import pandas as pd

_spec = importlib.util.spec_from_file_location("gf18", Path(__file__).with_name("18_gap_fill.py"))
gf = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gf)
st13 = gf.st13
POINT_VALUE, COMMISSION, SLIPPAGE, TAX_RATE = gf.POINT_VALUE, gf.COMMISSION, gf.SLIPPAGE, gf.TAX_RATE
NB, KS, DS, MODES, PERIODS_EVAL, LO, MID, EX = gf.NB, gf.KS, gf.DS, gf.MODES, gf.PERIODS_EVAL, gf.LO, gf.MID, gf.EX
STOPS = ("ref", 0.25, 0.5, 0.75, 1.0, 1.5, "none")
TRAILS = (False, True)


def run(M, ref, side_mask, d, stop, atr, trail=False):
    O, H, L, C, A, n = M["O"], M["H"], M["L"], M["C"], M["A"], M["n"]
    nd = len(n)
    side = side_mask.copy()
    pre_hi, pre_lo = H[:, :d].max(axis=1), L[:, :d].min(axis=1)
    filled = np.where(side == 1, pre_lo <= ref, np.where(side == -1, pre_hi >= ref, False))  # 缺口已回補 → 不做
    side[filled | (n <= d) | np.isnan(ref)] = 0
    ent = O[:, d]
    side[(side == 1) & (ent <= ref)] = 0
    side[(side == -1) & (ent >= ref)] = 0
    bars = np.arange(NB)[None, :]
    live = (bars >= d) & (bars < n[:, None])
    rows = np.arange(nd)
    if stop == "none":
        lvl = None
    else:
        base = ref if stop == "ref" else ent - side * stop * atr
        lvl = np.repeat(np.asarray(base, float)[:, None], NB, axis=1)
        if trail and stop != "ref":
            Hm = np.where(live, H, -np.inf); Lm = np.where(live, L, np.inf)
            rmax = np.maximum.accumulate(Hm, axis=1); rmin = np.minimum.accumulate(Lm, axis=1)
            prev_max = np.c_[np.full(nd, -np.inf), rmax[:, :-1]]; prev_min = np.c_[np.full(nd, np.inf), rmin[:, :-1]]
            tl = np.where(side[:, None] == 1, np.maximum(lvl, prev_max - stop * atr[:, None]),
                          np.minimum(lvl, prev_min + stop * atr[:, None]))
            lvl = np.where(np.isfinite(tl), tl, lvl)
    if lvl is None:
        hit = np.zeros((nd, NB), bool)
    else:
        hit = np.where(side[:, None] == 1, L <= lvl, H >= lvl) & live
    big = NB + 1
    ih = np.where(hit.any(1), hit.argmax(1), big)
    use = ih < big
    i2 = np.minimum(ih, NB - 1)
    lv = lvl[rows, i2] if lvl is not None else np.full(nd, np.nan)
    oi = O[rows, i2]
    sfill = np.where(side == 1, np.minimum(lv, oi), np.maximum(lv, oi))
    sfill = np.where(ih == d, lv, sfill)
    last = n - 1
    exit_px = np.where(use, sfill, C[rows, last]); exit_i = np.where(use, i2, last)
    raw_e = ent - A[:, d]; raw_x = exit_px - A[rows, exit_i]
    cost = 2 * (COMMISSION + SLIPPAGE * POINT_VALUE) + (raw_e + raw_x) * POINT_VALUE * TAX_RATE
    gross = np.where(side != 0, side * (exit_px - ent) * POINT_VALUE, 0.0)
    cost = np.where(side != 0, cost, 0.0)
    return gross - cost, side, use, gross, cost


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--five", required=True)
    ap.add_argument("--results-dir", default="results")
    args = ap.parse_args()
    R = Path(args.results_dir); R.mkdir(parents=True, exist_ok=True)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    days, _ = gf.build_days(args.five)
    M = {k: np.array([d[k] for d in days]) for k in ("O", "H", "L", "C", "A", "n", "ref_day", "ref_night", "ph15", "pl15")}
    M["n"] = M["n"].astype(int)
    dates = np.array([d["day"] for d in days], dtype="datetime64[ns]")
    atr = gf.daily_atr(days); ok = ~np.isnan(atr); open0 = M["O"][:, 0]

    def sides(ref, k_atr, mode):
        gap = open0 - ref
        s = np.where(gap > k_atr * atr, 1, np.where(gap < -k_atr * atr, -1, 0))  # 順勢
        s[~ok | (gap == 0) | np.isnan(ref)] = 0
        if mode == "longonly":
            s[s == -1] = 0
        elif mode == "shortonly":
            s[s == 1] = 0
        return s

    def per_metrics(pnl, side, gross, cost):
        out = {}
        for per, (a, b) in PERIODS_EVAL.items():
            m = gf.metrics(pnl, side, dates, np.datetime64(a), np.datetime64(b))
            kk = (dates >= np.datetime64(a)) & (dates < np.datetime64(b)) & (side != 0)
            m["gross"] = round(float(gross[kk].sum())); m["cost"] = round(float(cost[kk].sum()))
            m["avg_gross_pts"] = round(float(gross[kk].mean() / POINT_VALUE), 1) if kk.any() else 0.0
            m["avg_cost_pts"] = round(float(cost[kk].mean() / POINT_VALUE), 1) if kk.any() else 0.0
            for c, v in m.items():
                out[per + "_" + c] = v
        return out

    grid = []
    for ref_name, mode, kk, sp, d, tr in itertools.product(("day", "night"), MODES, KS, STOPS, DS, TRAILS):
        if tr and sp in ("ref", "none"):
            continue
        ref = M["ref_" + ref_name]
        pnl, side, hit, gross, cost = run(M, ref, sides(ref, kk, mode), d, sp, atr, tr)
        grid.append(dict(ref=ref_name, mode=mode, k=kk, stop=str(sp), d=d, trail=tr,
                         stop_hit_pct=round(100 * float(hit[side != 0].mean()), 1) if (side != 0).any() else 0.0,
                         **per_metrics(pnl, side, gross, cost)))
    G = pd.DataFrame(grid); G.to_csv(R / "gapf_grid_all.csv", index=False)

    orig_side = np.where(open0 > M["ph15"], 1, np.where(open0 < M["pl15"], -1, 0)); orig_side[~ok] = 0
    STR = [str(s) for s in STOPS]; shp = (len(KS), len(STOPS), len(DS))
    summ, legs, ovf, eqs = [], [], [], {}
    keep = ("trades", "win", "net", "mdd", "pf", "sharpe", "gross", "cost", "avg_gross_pts", "avg_cost_pts")
    for ref_name in ("day", "night"):
        ref = M["ref_" + ref_name]
        for mode in MODES:
            sub = G[(G.ref == ref_name) & (G["mode"] == mode)].reset_index(drop=True)
            sc_all = np.where(sub.IS_net > 0, sub.IS_rdd, sub.IS_net / 1e7)
            plat = np.full(len(sub), np.nan)
            for tr in TRAILS:
                idx = sub.index[sub.trail == tr]
                Z = np.full(shp, np.nan)
                ii = tuple(np.array(x) for x in (sub.k[idx].map(KS.index), sub.stop[idx].map(STR.index), sub.d[idx].map(DS.index)))
                Z[ii] = sc_all[idx]
                P = np.pad(Z, 1, constant_values=np.nan); acc = np.zeros(shp); cnt = np.zeros(shp)
                for o in np.ndindex(3, 3, 3):
                    w = P[o[0]:o[0] + shp[0], o[1]:o[1] + shp[1], o[2]:o[2] + shp[2]]
                    okw = ~np.isnan(w); acc += np.where(okw, w, 0); cnt += okw
                plat[idx] = (acc / np.maximum(cnt, 1))[ii]
            elig = sub.index[(sub.IS_trades >= 50) & (sub.IS_pf > 1)]
            fallback = len(elig) == 0
            elig = sub.index if fallback else elig
            pi = max(elig, key=lambda i: (plat[i], sub.loc[i, "IS_rdd"]))
            r = sub.loc[pi]
            sp = r.stop if r.stop in ("ref", "none") else float(r.stop)
            prm = "k={}, stop={}, d={}, trail={}{}".format(r.k, r.stop, int(r.d), "on" if r.trail else "off", " (fallback: no eligible)" if fallback else "")
            variants = [("plateau pick", sides(ref, float(r.k), mode), int(r.d), sp, bool(r.trail)),
                        ("naive always follow", sides(ref, 0.0, mode), 1, "none", False)]
            if ref_name == "day":
                os_ = orig_side.copy()
                if mode == "longonly":
                    os_[os_ == -1] = 0
                elif mode == "shortonly":
                    os_[os_ == 1] = 0
                variants.append(("original default inverted", os_, 3, "ref", False))
            for lab, sm, d, s_, tr in variants:
                pnl, side, hit, gross, cost = run(M, ref, sm, d, s_, atr, tr)
                pm = per_metrics(pnl, side, gross, cost)
                if mode == "longshort":
                    eqs[(ref_name, lab)] = (pnl, prm if lab == "plateau pick" else "")
                for per in PERIODS_EVAL:
                    summ.append(dict(ref=ref_name, mode=mode, variant=lab, params=prm if lab == "plateau pick" else "-",
                                     sample=per, **{c: pm[per + "_" + c] for c in keep}))
                if mode == "longshort" and lab == "plateau pick":
                    for leg, sv in (("long leg", 1), ("short leg", -1)):
                        m2 = side == sv
                        pl = per_metrics(np.where(m2, pnl, 0.0), np.where(m2, side, 0), np.where(m2, gross, 0.0), np.where(m2, cost, 0.0))
                        for per in PERIODS_EVAL:
                            legs.append(dict(ref=ref_name, leg=leg, params=prm, sample=per,
                                             **{c: pl[per + "_" + c] for c in ("trades", "win", "net", "mdd", "pf", "sharpe")}))
            isv, oosv = sub.IS_sharpe.to_numpy(), sub.OOS_sharpe.to_numpy()
            sr0 = st13.expected_max_sr(len(sub), float(np.var(isv, ddof=1)))
            ovf.append(dict(ref=ref_name, mode=mode, cells=len(sub), pick=prm,
                            spearman_sharpe=round(st13.spearman(isv, oosv), 3),
                            spearman_netmdd=round(st13.spearman(sc_all, np.where(sub.OOS_net > 0, sub.OOS_rdd, sub.OOS_net / 1e7)), 3),
                            pick_OOS_pctile=round(100 * float(np.mean(oosv <= r.OOS_sharpe)), 1),
                            share_IS_profitable=round(100 * float(np.mean(sub.IS_net > 0)), 1),
                            share_OOS_profitable=round(100 * float(np.mean(sub.OOS_net > 0)), 1),
                            share_IS_gross_pos=round(100 * float(np.mean(sub.IS_gross > 0)), 1),
                            sr0=round(sr0, 3), IS_sharpe_pick=r.IS_sharpe,
                            DSR_pick=round(st13.deflated_sharpe(r.IS_sharpe, r.IS_ndays, r.IS_skew, r.IS_kurt, sr0), 3)))
            if mode == "longshort":
                fig, axs = plt.subplots(1, 2, figsize=(14, 5.5))
                s2 = sub[(sub.d == r.d) & ((sub.trail == r.trail) | sub.stop.isin(["ref", "none"]))]
                for ax, per in zip(axs, ("IS", "OOS")):
                    Zh = np.full((len(KS), len(STOPS)), np.nan)
                    for _, x in s2.iterrows():
                        Zh[KS.index(x.k), STR.index(x.stop)] = x[per + "_rdd"] if x[per + "_net"] > 0 else x[per + "_net"] / 1e7
                    lim = np.nanmax(np.abs(Zh)) or 1
                    im = ax.imshow(Zh, origin="lower", cmap="RdYlGn", aspect="auto", vmin=-lim, vmax=lim)
                    ax.set_xticks(range(len(STOPS)), STR); ax.set_yticks(range(len(KS)), KS)
                    ax.set_xlabel("stop (ref = gap reference, else × daily ATR)"); ax.set_ylabel("min gap (× daily ATR)")
                    for y in range(Zh.shape[0]):
                        for xx in range(Zh.shape[1]):
                            ax.text(xx, y, "{:.2f}".format(Zh[y, xx]), ha="center", va="center", fontsize=7)
                    ax.scatter(STR.index(r.stop), KS.index(r.k), marker="o", s=420, facecolors="none", edgecolors="purple", linewidths=2)
                    ax.set_title("{} net/maxDD (loss: net/1e7), d={}, trail={}".format(per, int(r.d), "on" if r.trail else "off"))
                    fig.colorbar(im, ax=ax)
                fig.suptitle("TXF gap FOLLOW vs prior {} close (5m, long+short); circle = chosen".format(ref_name))
                fig.tight_layout(); fig.savefig(R / "gapf_heatmap_{}.png".format(ref_name), dpi=100); plt.close(fig)

    dts = pd.to_datetime(dates); k = dts >= LO
    plt.figure(figsize=(11, 5))
    for (ref_name, lab), (pnl, prm) in eqs.items():
        plt.plot(dts[k], np.cumsum(pnl[k]) / 1e3, label="{} ref={} {}".format(lab, ref_name, prm))
    plt.axvline(MID, color="k", ls=":", label="IS | OOS"); plt.axvline(EX, color="gray", ls="--", label="2025-01")
    plt.title("TXF day-session opening gap FOLLOW (5m), long+short, net of costs (k TWD)")
    plt.legend(fontsize=7); plt.grid(); plt.tight_layout(); plt.savefig(R / "gapf_equity.png", dpi=110); plt.close()

    for nm, rows_ in (("summary", summ), ("legs", legs), ("overfit", ovf)):
        pd.DataFrame(rows_).to_csv(R / "gapf_{}.csv".format(nm), index=False)
    with pd.option_context("display.width", 320, "display.max_columns", 40, "display.max_rows", 500):
        for rows_ in (summ, legs, ovf):
            print(pd.DataFrame(rows_).to_string(index=False))


if __name__ == "__main__":
    main()
