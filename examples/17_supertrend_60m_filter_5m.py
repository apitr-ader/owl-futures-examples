"""17 60 分 SuperTrend 當方向濾網，只做同向的 5 分 SuperTrend（event 模式）

執行：
  python examples/17_supertrend_60m_filter_5m.py \
      --minute data/TXF_1c_1m_both_2017-06-01_2026-10-22.parquet --data-dir data --results-dir results

沿用 13／15／16：SMA ATR 的 SuperTrend、倒推調整價算訊號、原始價算期交稅、每點 200、手續費單邊 50、
期交稅十萬分之二、滑價單邊 1 點、訊號於 K 棒收盤確認、下一根開盤成交、樣本內 2017-06～2022-12、樣本外 2023～。
- 5 分 K：1 分 K 依場次合成（同 13 的 resample_sessions），存 data/TXF_1c_5m_sessions.parquet。
- 60 分濾網：只用「已收完」的 60 分 K（K 棒結束時間 ≤ 5 分 K 開始時間），不偷看。
  日盤版：60 分與 5 分都只用日盤 K；日夜盤版：兩者都用日夜盤序列。
- event：空手時，5 分 ST 剛翻轉且與 60 分 ST 同向才進場；5 分反向或 60 分反向就出場（16 的單邊腿）。
  多空：多腿＋空腿（兩腿共用濾網，互斥，相加＝單一帳戶）；只做多；只做空。
- 網格：60 分 ATR 5–30（每 5）× 倍數 1–5（每 1）＝30 組；5 分 ATR 5–30 × 倍數 1.0–5.0（每 0.5）＝54 組 → 每模式 1,620 組。
- 選參：樣本內交易數 ≥ 50 且 PF > 1，四維 3×3×3×3 鄰域「淨利／最大回撤」平均最高（高原）。
- 對照：5 分 ST 單獨反手（54 組，2 維高原）、60 分 ST 單獨反手（30 組，2 維高原）、16 的日線濾網＋15 分日盤對稱 event。

※ 純技術範例，非投資建議。
"""

import argparse
import importlib.util
import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd


def _load(name, fn):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(fn))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


st13 = _load("st13", "13_supertrend_grid.py")
st15 = _load("st15", "15_supertrend_daily_filter.py")
st16 = _load("st16", "16_supertrend_asym_long_short.py")
H_PERIODS, H_MULTS = (5, 10, 15, 20, 25, 30), (1.0, 2.0, 3.0, 4.0, 5.0)
L_PERIODS, L_MULTS = st15.IN_PERIODS, st15.IN_MULTS
LO, MID, HI, EX = st16.LO, st16.MID, st16.HI, st16.EX
PERIODS_EVAL = st16.PERIODS_EVAL
MODES = ("longshort", "longonly", "shortonly")
COLS = ("trades", "win", "net", "mdd", "pf", "sharpe")

_G = {}


def h_filter(D5, D60, trend60):
    """每根 5 分 K 開盤時可用的 60 分方向：最後一根結束時間 ≤ 5 分 K 開始時間的 60 分 K。"""
    end60 = D60["end"]
    idx = np.searchsorted(end60, D5["ts"], side="right") - 1
    return np.where(idx >= 0, trend60[np.clip(idx, 0, None)], 0).astype(int)


def _job(a):
    key, hp, hm, lp, lm = a
    D = _G["D"][key]
    tr, fl = _G["tr5"][(key, lp, lm)], _G["f60"][(key, hp, hm)]
    L = st16.simulate_leg(D, tr, fl, 1)
    S = st16.simulate_leg(D, tr, fl, -1)
    LS = (L[0] + S[0], sorted(L[1] + S[1]), L[2] + S[2])
    out = dict(key=key, hp=hp, hm=hm, lp=lp, lm=lm)
    for mode, sim in (("longshort", LS), ("longonly", L), ("shortonly", S)):
        for per, (lo, hi) in PERIODS_EVAL.items():
            x = st13.metrics(D, sim[0], sim[1], lo, hi)
            for c in COLS + ("rdd", "skew", "kurt", "ndays"):
                out["{}_{}_{}".format(mode, per, c)] = x[c]
    return out


def _alone_job(a):
    key, which, p, m = a
    D = _G["D"][key] if which == "5m" else _G["D60"][key]
    pnl, trd = st13.simulate(D, "sma", p, m, "reversal")
    out = dict(key=key, which=which, p=p, m=m)
    for per, (lo, hi) in PERIODS_EVAL.items():
        x = st13.metrics(D, pnl, trd, lo, hi)
        for c in COLS + ("rdd", "skew", "kurt", "ndays"):
            out["{}_{}".format(per, c)] = x[c]
    return out


def score(net, rdd):
    return np.where(np.asarray(net) > 0, rdd, np.asarray(net) / 1e7)


def plateau_nd(idx_cols, vals, shape):
    """任意維度 3^d 鄰域平均（邊界截斷）。idx_cols: 每維的索引陣列。"""
    Z = np.full(shape, np.nan)
    Z[tuple(idx_cols)] = vals
    P = np.pad(Z, 1, constant_values=np.nan)
    acc = np.zeros_like(Z); cnt = np.zeros_like(Z)
    for s in np.ndindex(*([3] * len(shape))):
        w = P[tuple(slice(o, o + n) for o, n in zip(s, shape))]
        ok = ~np.isnan(w); acc += np.where(ok, w, 0); cnt += ok
    return (acc / cnt)[tuple(idx_cols)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--minute", required=True)
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()
    R = Path(args.results_dir); R.mkdir(parents=True, exist_ok=True)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    f5 = st13.resample_sessions(args.minute, 5)
    f5.to_parquet(Path(args.data_dir) / "TXF_1c_5m_sessions.parquet", compression="zstd", index=False)
    f60 = st13.resample_sessions(args.minute, 60)
    keep = {("sma", p) for p in set(L_PERIODS) | set(H_PERIODS)}
    D5, D60 = {}, {}
    for key, sel in (("day", lambda f: f[f["session"] == "regular"].reset_index(drop=True)), ("daynight", lambda f: f)):
        D5[key] = st13.prepare(sel(f5), "intraday")
        D5[key]["atr"] = {k: v for k, v in D5[key]["atr"].items() if k in keep}
        g = sel(f60)
        D60[key] = st13.prepare(g, "intraday")
        D60[key]["atr"] = {k: v for k, v in D60[key]["atr"].items() if k in keep}
        # 60 分 K 結束時間：開始 +60 分，但不超過場次收盤（日盤 13:45、夜盤 05:00）
        st = g["date"].dt.tz_localize(None)
        sess_end = st.dt.normalize() + np.where(g["session"] == "regular", pd.Timedelta(hours=13, minutes=45),
                                                pd.Timedelta(hours=29))
        sess_end = sess_end.where(~((g["session"] != "regular") & (st.dt.hour < 15)), st.dt.normalize() + pd.Timedelta(hours=5))
        D60[key]["end"] = np.minimum(st + pd.Timedelta(minutes=60), sess_end).to_numpy()
        print("{}: 5 分 {} 根、60 分 {} 根".format(key, len(D5[key]["o"]), len(D60[key]["o"])), flush=True)
    tr5 = {(k, p, m): st13.supertrend_dir(D5[k], "sma", p, m) for k in D5 for p in L_PERIODS for m in L_MULTS}
    f60f = {(k, p, m): h_filter(D5[k], D60[k], st13.supertrend_dir(D60[k], "sma", p, m)) for k in D5 for p in H_PERIODS for m in H_MULTS}
    _G.update(D=D5, D60=D60, tr5=tr5, f60=f60f)

    jobs = [(k, hp, hm, lp, lm) for k in D5 for hp in H_PERIODS for hm in H_MULTS for lp in L_PERIODS for lm in L_MULTS]
    ajobs = [(k, "5m", p, m) for k in D5 for p in L_PERIODS for m in L_MULTS] + \
            [(k, "60m", p, m) for k in D5 for p in H_PERIODS for m in H_MULTS]
    with ProcessPoolExecutor(args.workers, mp_context=mp.get_context("fork")) as ex:
        res = pd.DataFrame(list(ex.map(_job, jobs, chunksize=4)))
        alone = pd.DataFrame(list(ex.map(_alone_job, ajobs, chunksize=2)))
    res.to_csv(R / "h60_grid_all.csv", index=False); alone.to_csv(R / "h60_alone_all.csv", index=False)

    prev = pd.read_csv(R / "asym_summary.csv") if (R / "asym_summary.csv").exists() else None
    summ, ovf = [], []
    shape4 = (len(H_PERIODS), len(H_MULTS), len(L_PERIODS), len(L_MULTS))
    for key in D5:
        D = D5[key]
        sub = res[res.key == key].reset_index(drop=True)
        ix = [sub.hp.map(H_PERIODS.index).to_numpy(), sub.hm.map(H_MULTS.index).to_numpy(),
              sub.lp.map(L_PERIODS.index).to_numpy(), sub.lm.map(L_MULTS.index).to_numpy()]
        picks = {}
        for mode in MODES:
            c = lambda per, x: "{}_{}_{}".format(mode, per, x)
            plat = plateau_nd(ix, score(sub[c("IS", "net")], sub[c("IS", "rdd")]), shape4)
            elig = sub.index[(sub[c("IS", "trades")] >= 50) & (sub[c("IS", "pf")] > 1)]
            elig = elig if len(elig) else sub.index
            pi = max(elig, key=lambda i: (plat[i], sub.loc[i, c("IS", "rdd")]))
            r = sub.loc[pi]; picks[mode] = r
            for per in PERIODS_EVAL:
                summ.append(dict(tf=key, strategy="60m filter + 5m event", mode=mode,
                                 params="60m {}/{} | 5m {}/{}".format(r.hp, r.hm, r.lp, r.lm), sample=per,
                                 **{x: r[c(per, x)] for x in COLS}))
            isv, oosv = sub[c("IS", "sharpe")].to_numpy(), sub[c("OOS", "sharpe")].to_numpy()
            sr0 = st13.expected_max_sr(len(sub), float(np.var(isv, ddof=1)))
            ovf.append(dict(tf=key, mode=mode, cells=len(sub),
                            pick="60m {}/{} | 5m {}/{}".format(r.hp, r.hm, r.lp, r.lm),
                            spearman_sharpe=round(st13.spearman(isv, oosv), 3),
                            spearman_netmdd=round(st13.spearman(score(sub[c("IS", "net")], sub[c("IS", "rdd")]),
                                                                score(sub[c("OOS", "net")], sub[c("OOS", "rdd")])), 3),
                            pick_OOS_pctile=round(100 * float(np.mean(oosv <= r[c("OOS", "sharpe")])), 1),
                            share_IS_profitable=round(100 * float(np.mean(sub[c("IS", "net")] > 0)), 1),
                            share_OOS_profitable=round(100 * float(np.mean(sub[c("OOS", "net")] > 0)), 1),
                            sr0=round(sr0, 3), IS_sharpe_pick=r[c("IS", "sharpe")],
                            DSR_pick=round(st13.deflated_sharpe(r[c("IS", "sharpe")], r[c("IS", "ndays")], r[c("IS", "skew")],
                                                                r[c("IS", "kurt")], sr0), 3)))
        # 單獨反手
        al = {}
        for which, PP, MM in (("5m", L_PERIODS, L_MULTS), ("60m", H_PERIODS, H_MULTS)):
            s = alone[(alone.key == key) & (alone.which == which)].reset_index(drop=True)
            plat = plateau_nd([s.p.map(PP.index).to_numpy(), s.m.map(MM.index).to_numpy()],
                              score(s.IS_net, s.IS_rdd), (len(PP), len(MM)))
            elig = s.index[(s.IS_trades >= 50) & (s.IS_pf > 1)]
            elig = elig if len(elig) else s.index
            pi = max(elig, key=lambda i: (plat[i], s.loc[i, "IS_rdd"]))
            r = s.loc[pi]; al[which] = r
            for per in PERIODS_EVAL:
                summ.append(dict(tf=key, strategy="{} ST alone (reversal)".format(which), mode="longshort",
                                 params="{}/{}".format(r.p, r.m), sample=per, **{x: r["{}_{}".format(per, x)] for x in COLS}))
            isv = s.IS_sharpe.to_numpy()
            sr0 = st13.expected_max_sr(len(s), float(np.var(isv, ddof=1)))
            ovf.append(dict(tf=key, mode="{} alone".format(which), cells=len(s), pick="{}/{}".format(r.p, r.m),
                            spearman_sharpe=round(st13.spearman(isv, s.OOS_sharpe), 3),
                            spearman_netmdd=round(st13.spearman(score(s.IS_net, s.IS_rdd), score(s.OOS_net, s.OOS_rdd)), 3),
                            pick_OOS_pctile=round(100 * float(np.mean(s.OOS_sharpe <= r.OOS_sharpe)), 1),
                            share_IS_profitable=round(100 * float(np.mean(s.IS_net > 0)), 1),
                            share_OOS_profitable=round(100 * float(np.mean(s.OOS_net > 0)), 1),
                            sr0=round(sr0, 3), IS_sharpe_pick=r.IS_sharpe,
                            DSR_pick=round(st13.deflated_sharpe(r.IS_sharpe, r.IS_ndays, r.IS_skew, r.IS_kurt, sr0), 3)))
        if prev is not None:
            q = prev[(prev.tf == "15m_day") & (prev.variant == "symmetric event (15)")]
            for _, r in q.iterrows():
                summ.append(dict(tf=key, strategy="earlier best: daily filter + 15m day (event)", mode="longshort",
                                 params="D {} | 15m {}".format(*r["long"].split(" | ")), sample=r["sample"],
                                 **{x: r[x] for x in COLS}))

        # 權益圖
        tsx = pd.to_datetime(D["ts"]); k = tsx >= LO
        plt.figure(figsize=(11, 5))
        for mode in MODES:
            r = picks[mode]
            tr, fl = tr5[(key, r.lp, r.lm)], f60f[(key, r.hp, r.hm)]
            sims = [st16.simulate_leg(D, tr, fl, s)[0] for s in ((1, -1) if mode == "longshort" else (1,) if mode == "longonly" else (-1,))]
            plt.plot(tsx[k], np.cumsum(sum(sims)[k]) / 1e3, label="{}: 60m {}/{} | 5m {}/{}".format(mode, r.hp, r.hm, r.lp, r.lm))
        for which, DD in (("5m", D), ("60m", D60[key])):
            r = al[which]
            pnl, _ = st13.simulate(DD, "sma", int(r.p), float(r.m), "reversal")
            t2 = pd.to_datetime(DD["ts"]); k2 = t2 >= LO
            plt.plot(t2[k2], np.cumsum(pnl[k2]) / 1e3, label="{} ST alone {}/{}".format(which, r.p, r.m), alpha=0.7)
        plt.axvline(MID, color="k", ls=":", label="IS | OOS"); plt.axvline(EX, color="gray", ls="--", label="2025-01")
        plt.title("TXF {} 5m SuperTrend filtered by completed 60m SuperTrend (event), net of costs (k TWD)".format(key))
        plt.legend(fontsize=7); plt.grid(); plt.tight_layout()
        plt.savefig(R / "h60_equity_{}.png".format(key), dpi=110); plt.close()

        # 熱圖：多空模式，(a) 固定 60m 參數看 5m 網格；(b) 固定 5m 參數看 60m 網格
        r = picks["longshort"]
        fig, axs = plt.subplots(2, 2, figsize=(14, 10))
        for row, (fix, PP, MM, pc, mc, lab) in enumerate(((("hp", "hm"), L_PERIODS, L_MULTS, "lp", "lm", "5m"),
                                                          (("lp", "lm"), H_PERIODS, H_MULTS, "hp", "hm", "60m"))):
            s2 = sub[(sub[fix[0]] == r[fix[0]]) & (sub[fix[1]] == r[fix[1]])]
            for col, per in enumerate(("IS", "OOS")):
                ax = axs[row, col]
                Z = np.full((len(PP), len(MM)), np.nan)
                for _, x in s2.iterrows():
                    Z[PP.index(x[pc]), MM.index(x[mc])] = score(x["longshort_{}_net".format(per)], x["longshort_{}_rdd".format(per)])
                lim = np.nanmax(np.abs(Z))
                im = ax.imshow(Z, origin="lower", cmap="RdYlGn", aspect="auto", vmin=-lim, vmax=lim)
                ax.set_xticks(range(len(MM)), MM); ax.set_yticks(range(len(PP)), PP)
                ax.set_xlabel("{} ATR multiplier".format(lab)); ax.set_ylabel("{} ATR period".format(lab))
                for y in range(Z.shape[0]):
                    for xx in range(Z.shape[1]):
                        ax.text(xx, y, "{:.2f}".format(Z[y, xx]), ha="center", va="center", fontsize=7)
                ax.scatter(MM.index(r[mc]), PP.index(r[pc]), marker="o", s=420, facecolors="none", edgecolors="purple", linewidths=2)
                ax.set_title("{} grid (other fixed at {}/{}) — {} net/maxDD".format(lab, r[fix[0]], r[fix[1]], per))
                fig.colorbar(im, ax=ax)
        fig.suptitle("TXF {} 60m-filtered 5m SuperTrend, long+short event; circle = chosen".format(key))
        fig.tight_layout(); fig.savefig(R / "h60_heatmap_{}.png".format(key), dpi=100); plt.close(fig)

    pd.DataFrame(summ).to_csv(R / "h60_summary.csv", index=False)
    pd.DataFrame(ovf).to_csv(R / "h60_overfit.csv", index=False)
    with pd.option_context("display.width", 300, "display.max_columns", 40, "display.max_rows", 500, "display.max_colwidth", 50):
        print(pd.DataFrame(summ).to_string(index=False))
        print(pd.DataFrame(ovf).to_string(index=False))


if __name__ == "__main__":
    main()
