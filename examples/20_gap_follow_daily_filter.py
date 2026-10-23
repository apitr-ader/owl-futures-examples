"""20 開盤跳空順勢＋日 K 方向濾網（19 的延伸）

執行：
  python examples/20_gap_follow_daily_filter.py --five data/TXF_1c_5m_sessions.parquet \
      --daily data/TXF_1c_D_regular_2010-01-01_2026-10-22.parquet --results-dir results

- 跳空、進出場、停損／移動停損、成本、樣本切分同 19（主角 ref=night：08:45 開盤 vs 前夜盤 05:00 收盤；ref=day 作對照）。
- 濾網：只在「前一個已完成日盤日 K」（日期 < 當天，08:45 前已知，不偷看）方向一致時進場：
  向上跳空且濾網多 → 做多；向下跳空且濾網空 → 做空。
- 濾網網格刻意小（6 個）：日 K SuperTrend（SMA ATR, hl2）10/2.0、10/3.0、20/2.0、20/3.0；收盤 vs MA20、收盤 vs MA60（倒推調整價）。
- 每個濾網內 k×stop×d×trail 同 19（336 格／模式）；高原在 (k, stop, d) 內算，濾網與 trail 只是類別，取全部格中高原最高者。
  另列「濾網＋天真參數」（k=0, d=1, 無停損）——不需要調參的版本，最能看出濾網本身的價值。
- 對照：未濾網天真順勢、19 的高原選參。
- 過擬合：IS↔OOS Spearman、DSR（試驗數＝全部 6×336 格）、樣本外排除 2025–26、獲利格比例、交易數與成本。

※ 純技術範例，非投資建議。
"""

import argparse
import importlib.util
import itertools
from pathlib import Path

import numpy as np
import pandas as pd

_spec = importlib.util.spec_from_file_location("gf19", Path(__file__).with_name("19_gap_follow.py"))
g19 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(g19)
gf, st13 = g19.gf, g19.st13
POINT_VALUE, KS, DS, MODES, PERIODS_EVAL = gf.POINT_VALUE, gf.KS, gf.DS, gf.MODES, gf.PERIODS_EVAL
LO, MID, EX, STOPS, TRAILS = gf.LO, gf.MID, gf.EX, g19.STOPS, g19.TRAILS
FILTERS = ("ST10/2.0", "ST10/3.0", "ST20/2.0", "ST20/3.0", "MA20", "MA60")
# 19 的高原選參（對照用）
P19 = {("night", "longshort"): (0.0, 0.75, 2, False), ("night", "longonly"): (0.0, 0.75, 1, False),
       ("night", "shortonly"): (0.3, "none", 6, False), ("day", "longshort"): (0.75, 1.5, 6, True),
       ("day", "longonly"): (0.5, 1.0, 1, True), ("day", "shortonly"): (1.0, 1.5, 6, True)}


def daily_filters(daily_path):
    d = st13.load_daily(daily_path)
    Dd = st13.prepare(d, "daily")
    days = pd.to_datetime(Dd["days"])
    out = {}
    for f in FILTERS:
        if f.startswith("ST"):
            p, m = f[2:].split("/")
            out[f] = st13.supertrend_dir(Dd, "sma", int(p), float(m)).astype(float)
        else:
            ma = pd.Series(Dd["c"]).rolling(int(f[2:])).mean().to_numpy()
            out[f] = np.where(np.isnan(ma), 0, np.sign(Dd["c"] - ma))
    return days, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--five", required=True)
    ap.add_argument("--daily", required=True)
    ap.add_argument("--results-dir", default="results")
    args = ap.parse_args()
    R = Path(args.results_dir); R.mkdir(parents=True, exist_ok=True)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    days, _ = gf.build_days(args.five)
    M = {k: np.array([d[k] for d in days]) for k in ("O", "H", "L", "C", "A", "n", "ref_day", "ref_night")}
    M["n"] = M["n"].astype(int)
    dates = np.array([d["day"] for d in days], dtype="datetime64[ns]")
    atr = gf.daily_atr(days); ok = ~np.isnan(atr); open0 = M["O"][:, 0]

    ddays, dfilt = daily_filters(args.daily)
    # 每個交易日 T 取「日期 < T」最後一根日 K 的濾網值（不偷看）
    pos = np.searchsorted(ddays.to_numpy(), dates, side="left") - 1
    FD = {f: np.where(pos >= 0, v[np.clip(pos, 0, None)], 0) for f, v in dfilt.items()}
    FD["none"] = np.ones(len(dates))  # 佔位：未濾網

    def sides(ref, k_atr, mode, filt):
        gap = open0 - ref
        s = np.where(gap > k_atr * atr, 1, np.where(gap < -k_atr * atr, -1, 0))
        s[~ok | (gap == 0) | np.isnan(ref)] = 0
        if filt != "none":
            s[(s == 1) & (FD[filt] <= 0)] = 0
            s[(s == -1) & (FD[filt] >= 0)] = 0
        if mode == "longonly":
            s[s == -1] = 0
        elif mode == "shortonly":
            s[s == 1] = 0
        return s

    def pm(pnl, side, gross, cost):
        out = {}
        for per, (a, b) in PERIODS_EVAL.items():
            m = gf.metrics(pnl, side, dates, np.datetime64(a), np.datetime64(b))
            kk = (dates >= np.datetime64(a)) & (dates < np.datetime64(b)) & (side != 0)
            m["cost"] = round(float(cost[kk].sum()))
            m["avg_gross_pts"] = round(float(gross[kk].mean() / POINT_VALUE), 1) if kk.any() else 0.0
            for c, v in m.items():
                out[per + "_" + c] = v
        return out

    def sim(ref_name, mode, filt, k, sp, d, tr):
        ref = M["ref_" + ref_name]
        return g19.run(M, ref, sides(ref, k, mode, filt), d, sp, atr, tr)

    grid = []
    for ref_name, mode, f, kk, sp, d, tr in itertools.product(("night", "day"), MODES, FILTERS, KS, STOPS, DS, TRAILS):
        if tr and sp in ("ref", "none"):
            continue
        pnl, side, hit, gross, cost = sim(ref_name, mode, f, kk, sp, d, tr)
        grid.append(dict(ref=ref_name, mode=mode, filt=f, k=kk, stop=str(sp), d=d, trail=tr, **pm(pnl, side, gross, cost)))
    G = pd.DataFrame(grid); G.to_csv(R / "gapd_grid_all.csv", index=False)

    STR = [str(s) for s in STOPS]; shp = (len(KS), len(STOPS), len(DS))
    keep = ("trades", "win", "net", "mdd", "pf", "sharpe", "cost", "avg_gross_pts")
    summ, legs, ovf, filt_tab, eqs = [], [], [], [], {}
    for ref_name in ("night", "day"):
        for mode in MODES:
            sub = G[(G.ref == ref_name) & (G["mode"] == mode)].reset_index(drop=True)
            sc = np.where(sub.IS_net > 0, sub.IS_rdd, sub.IS_net / 1e7)
            plat = np.full(len(sub), np.nan)
            for f in FILTERS:
                for tr in TRAILS:
                    idx = sub.index[(sub.filt == f) & (sub.trail == tr)]
                    Z = np.full(shp, np.nan)
                    ii = tuple(np.array(x) for x in (sub.k[idx].map(KS.index), sub.stop[idx].map(STR.index), sub.d[idx].map(DS.index)))
                    Z[ii] = sc[idx]
                    P = np.pad(Z, 1, constant_values=np.nan); acc = np.zeros(shp); cnt = np.zeros(shp)
                    for o in np.ndindex(3, 3, 3):
                        w = P[o[0]:o[0] + shp[0], o[1]:o[1] + shp[1], o[2]:o[2] + shp[2]]
                        okw = ~np.isnan(w); acc += np.where(okw, w, 0); cnt += okw
                    plat[idx] = (acc / np.maximum(cnt, 1))[ii]
            elig = sub.index[(sub.IS_trades >= 50) & (sub.IS_pf > 1)]
            fb = len(elig) == 0
            elig = sub.index if fb else elig
            pi = max(elig, key=lambda i: (plat[i], sub.loc[i, "IS_rdd"]))
            r = sub.loc[pi]
            sp = r.stop if r.stop in ("ref", "none") else float(r.stop)
            prm = "{} k={} stop={} d={} trail={}{}".format(r.filt, r.k, r.stop, int(r.d), "on" if r.trail else "off", " (fallback)" if fb else "")
            # 濾網＋天真參數：IS 最佳濾網（依 IS 淨利/回撤）
            naive = sub[(sub.k == 0.0) & (sub.stop == "none") & (sub.d == 1)].copy()
            for _, x in naive.iterrows():
                filt_tab.append(dict(ref=ref_name, mode=mode, filt=x.filt, **{p + "_" + c: x[p + "_" + c]
                                     for p in PERIODS_EVAL for c in ("trades", "net", "pf", "sharpe")}))
            nb = naive.loc[naive.IS_rdd.idxmax()]
            p19 = P19[(ref_name, mode)]
            variants = [("plateau pick (filtered)", (r.filt, float(r.k), sp, int(r.d), bool(r.trail)), prm),
                        ("naive + best-IS filter", (nb.filt, 0.0, "none", 1, False), nb.filt + " k=0 none d=1"),
                        ("unfiltered naive", ("none", 0.0, "none", 1, False), "-"),
                        ("script 19 plateau", ("none",) + p19, "k={} stop={} d={} trail={}".format(*p19))]
            for lab, (f, k_, s_, d_, t_), pr in variants:
                pnl, side, hit, gross, cost = sim(ref_name, mode, f, k_, s_, d_, t_)
                mm = pm(pnl, side, gross, cost)
                if mode == "longshort":
                    eqs[(ref_name, lab)] = pnl
                for per in PERIODS_EVAL:
                    summ.append(dict(ref=ref_name, mode=mode, variant=lab, params=pr, sample=per, **{c: mm[per + "_" + c] for c in keep}))
                if mode == "longshort" and lab in ("plateau pick (filtered)", "naive + best-IS filter"):
                    for leg, sv in (("long", 1), ("short", -1)):
                        m2 = side == sv
                        ml = pm(np.where(m2, pnl, 0.0), np.where(m2, side, 0), np.where(m2, gross, 0.0), np.where(m2, cost, 0.0))
                        for per in PERIODS_EVAL:
                            legs.append(dict(ref=ref_name, variant=lab, leg=leg, sample=per,
                                             **{c: ml[per + "_" + c] for c in ("trades", "win", "net", "pf", "sharpe")}))
            isv, oosv = sub.IS_sharpe.to_numpy(), sub.OOS_sharpe.to_numpy()
            sr0 = st13.expected_max_sr(len(sub), float(np.var(isv, ddof=1)))
            ovf.append(dict(ref=ref_name, mode=mode, cells=len(sub), pick=prm,
                            spearman_sharpe=round(st13.spearman(isv, oosv), 3),
                            pick_OOS_pctile=round(100 * float(np.mean(oosv <= r.OOS_sharpe)), 1),
                            pct_IS_prof=round(100 * float(np.mean(sub.IS_net > 0)), 1),
                            pct_OOS_prof=round(100 * float(np.mean(sub.OOS_net > 0)), 1),
                            pct_OOSex_prof=round(100 * float(np.mean(sub.OOS_ex2526_net > 0)), 1),
                            sr0=round(sr0, 3), IS_sharpe_pick=r.IS_sharpe,
                            DSR_pick=round(st13.deflated_sharpe(r.IS_sharpe, r.IS_ndays, r.IS_skew, r.IS_kurt, sr0), 3),
                            DSR_naive_best_filter=round(st13.deflated_sharpe(nb.IS_sharpe, nb.IS_ndays, nb.IS_skew, nb.IS_kurt,
                                                        st13.expected_max_sr(len(FILTERS), float(np.var(naive.IS_sharpe, ddof=1)))), 3)))
            if mode == "longshort":
                fig, axs = plt.subplots(1, 2, figsize=(13, 5))
                naive_ls = naive.set_index("filt")
                for ax, per in zip(axs, ("IS", "OOS")):
                    Zh = np.full((len(FILTERS), len(KS)), np.nan)
                    s2 = sub[(sub.stop == r.stop) & (sub.d == r.d) & (sub.trail == r.trail)]
                    for _, x in s2.iterrows():
                        Zh[FILTERS.index(x.filt), KS.index(x.k)] = x[per + "_net"] / 1e3
                    lim = np.nanmax(np.abs(Zh)) or 1
                    im = ax.imshow(Zh, origin="lower", cmap="RdYlGn", aspect="auto", vmin=-lim, vmax=lim)
                    ax.set_xticks(range(len(KS)), KS); ax.set_yticks(range(len(FILTERS)), FILTERS)
                    ax.set_xlabel("min gap (× daily ATR)")
                    for y in range(Zh.shape[0]):
                        for xx in range(Zh.shape[1]):
                            ax.text(xx, y, "{:.0f}".format(Zh[y, xx]), ha="center", va="center", fontsize=7)
                    ax.scatter(KS.index(r.k), FILTERS.index(r.filt), marker="o", s=420, facecolors="none", edgecolors="purple", linewidths=2)
                    ax.set_title("{} net (k TWD), stop={} d={} trail={}".format(per, r.stop, int(r.d), "on" if r.trail else "off"))
                    fig.colorbar(im, ax=ax)
                fig.suptitle("TXF gap follow + daily filter, ref={} (L+S); circle = chosen".format(ref_name))
                fig.tight_layout(); fig.savefig(R / "gapd_heatmap_{}.png".format(ref_name), dpi=100); plt.close(fig)

    dts = pd.to_datetime(dates); k = dts >= LO
    plt.figure(figsize=(11, 5))
    for (ref_name, lab), pnl in eqs.items():
        plt.plot(dts[k], np.cumsum(pnl[k]) / 1e3, label="{} ref={}".format(lab, ref_name), lw=1.6 if ref_name == "night" else 0.8)
    plt.axvline(MID, color="k", ls=":", label="IS | OOS"); plt.axvline(EX, color="gray", ls="--", label="2025-01")
    plt.title("TXF gap follow ± daily direction filter (L+S), net of costs (k TWD)")
    plt.legend(fontsize=7); plt.grid(); plt.tight_layout(); plt.savefig(R / "gapd_equity.png", dpi=110); plt.close()

    for nm, rows_ in (("summary", summ), ("legs", legs), ("overfit", ovf), ("filters_naive", filt_tab)):
        pd.DataFrame(rows_).to_csv(R / "gapd_{}.csv".format(nm), index=False)
    with pd.option_context("display.width", 320, "display.max_columns", 40, "display.max_rows", 600):
        for rows_ in (summ, legs, filt_tab, ovf):
            print(pd.DataFrame(rows_).to_string(index=False))


if __name__ == "__main__":
    main()
