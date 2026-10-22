"""14 SuperTrend 反手策略 ＋ ATR 初始停損 ＋ 移動停利（日 K／60 分／15 分，日盤或日夜盤）

執行：
  python examples/14_supertrend_atr_trailing.py \
      --daily data/TXF_1c_D_regular_2010-01-01_2026-10-22.parquet \
      --minute data/TXF_1c_1m_both_2017-06-01_2026-10-22.parquet --results-dir results

沿用 13 的資料處理、SuperTrend 計算、成本與績效算法（同目錄 import），再加上 12 的停損／停利：
- SuperTrend 翻轉：下一根開盤反手（或停損出場後空手時，於下一次翻轉進場）。
- 初始停損：進場價 ∓ k_sl × ATR（Wilder ATR，取進場前一根已完成 K 棒的值，進場後固定）。
- 移動停利：多單＝進場後最高價 − k_tr × ATR（空單對稱），浮盈 ≥ act × ATR 才啟動；
  最高／最低價只用已完成的 K 棒更新（下一根才生效），停損價只往有利方向移動。
- 觸價以停損價成交；開盤跳空越過停損價以開盤價成交；進場當根就碰到停損一律算停損。
- 停損／停利出場後保持空手，直到下一次 SuperTrend 翻轉。
- 「none」＝不用該項（k_sl 或 k_tr），用來分離效果；兩者皆 none 即 13 的原版。

為了少做無謂的嘗試，SuperTrend 參數只用兩組：TradingView 預設 10/3.0，以及 13 在樣本內以高原法選出的那組；
停損網格：ATR 14/20 × k_sl 1.0–4.0（0.5）＋none × k_tr 1.0–5.0（0.5）＋none × act 0/1/2。
選參：樣本內、交易數門檻與 PF > 1，取 (k_sl, k_tr) 平面 3×3 鄰域「淨利／最大回撤」平均最高者。

※ 純技術範例，非投資建議。
"""

import argparse
import importlib.util
import itertools
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

_spec = importlib.util.spec_from_file_location("st13", Path(__file__).with_name("13_supertrend_grid.py"))
st13 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(st13)

POINT_VALUE, COMMISSION, SLIPPAGE, TAX_RATE = st13.POINT_VALUE, st13.COMMISSION, st13.SLIPPAGE, st13.TAX_RATE
INF = float("inf")
STOP_ATR = (14, 20)
K_SL = (1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, INF)
K_TR = (1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, INF)
ACTS = (0.0, 1.0, 2.0)
# 13 的樣本內高原選擇（SMA ATR、反手）
PLATEAU13 = {"D": (24, 5.75), "60m_daynight": (30, 1.0), "60m_day": (12, 4.25),
             "15m_daynight": (16, 3.75), "15m_day": (30, 4.5)}
MIN_TRADES = {"D": 20}


def simulate(D, st_period, st_mult, atr_n, k_sl, k_tr, act):
    trend = st13.supertrend_dir(D, "sma", st_period, st_mult).tolist()
    o, h, l, c, adj = (D[k].tolist() for k in ("o", "h", "l", "c", "adj"))
    atr = D["atr"][("rma", atr_n)].tolist()
    n = len(o)
    pnl = np.zeros(n)
    trades = []
    pos, cur = 0, 0.0
    stop = best = a0 = ep = 0.0
    active = False
    use_sl, use_tr = k_sl != INF, k_tr != INF

    def side_cost(i, px):
        return COMMISSION + (px - adj[i]) * POINT_VALUE * TAX_RATE + SLIPPAGE * POINT_VALUE

    for i in range(2, n):
        flip = trend[i - 2] != 0 and trend[i - 1] != trend[i - 2]
        bar = pos * (o[i] - c[i - 1]) * POINT_VALUE
        cur += bar
        entered = False
        if flip and pos != trend[i - 1]:
            if pos:
                k = side_cost(i, o[i]); bar -= k
                trades.append((i, cur - k)); cur = 0.0
            pos = trend[i - 1]
            k = side_cost(i, o[i]); bar -= k; cur = -k
            ep, best, active = o[i], o[i], False
            a0 = atr[i - 1]
            if a0 != a0:
                a0 = 0.0
            stop = ep - pos * k_sl * a0 if (use_sl and a0) else (-INF if pos > 0 else INF)
            entered = True
        exit_px = None
        if pos:
            if pos > 0 and l[i] <= stop:
                exit_px = min(o[i], stop) if not entered else stop
            elif pos < 0 and h[i] >= stop:
                exit_px = max(o[i], stop) if not entered else stop
        if exit_px is not None:
            r = pos * (exit_px - o[i]) * POINT_VALUE
            k = side_cost(i, exit_px)
            bar += r - k
            trades.append((i, cur + r - k)); cur = 0.0
            pos = 0
        elif pos:
            r = pos * (c[i] - o[i]) * POINT_VALUE
            bar += r; cur += r
            if use_tr and a0:
                best = max(best, h[i]) if pos > 0 else min(best, l[i])
                if not active and (best - ep) * pos >= act * a0:
                    active = True
                if active:
                    t = best - pos * k_tr * a0
                    if (pos > 0 and t > stop) or (pos < 0 and t < stop):
                        stop = t
        pnl[i] = bar
    if pos:
        trades.append((n - 1, cur))
    return pnl, trades


_DS = {}


def _init(ds):
    global _DS
    _DS = ds


def _job(a):
    key, stp, atr_n, k_sl, k_tr, act = a
    D = _DS[key]
    pnl, tr = simulate(D, stp[0], stp[1], atr_n, k_sl, k_tr, act)
    lo, mid, hi = D["split"]
    return dict(key=key, st=stp, atr_n=atr_n, k_sl=k_sl, k_tr=k_tr, act=act,
                IS=st13.metrics(D, pnl, tr, lo, mid), OOS=st13.metrics(D, pnl, tr, mid, hi))


def score(m):
    return m["rdd"] if m["net"] > 0 else m["net"] / 1e7


def fmt(v):
    return "none" if v == INF else v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--daily", required=True)
    ap.add_argument("--minute", required=True)
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    args = ap.parse_args()
    R = Path(args.results_dir); R.mkdir(parents=True, exist_ok=True)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    frames = {"D": st13.load_daily(args.daily)}
    for mins in (60, 15):
        f = st13.resample_sessions(args.minute, mins)
        frames["{}m_daynight".format(mins)] = f
        frames["{}m_day".format(mins)] = f[f["session"] == "regular"].reset_index(drop=True)
    ds = {k: st13.prepare(v, "daily" if k == "D" else "intraday") for k, v in frames.items()}

    stop_grid = []
    for a, s, t in itertools.product(STOP_ATR, K_SL, K_TR):
        for v in (ACTS if t != INF else (0.0,)):
            stop_grid.append((a, s, t, v))
    jobs = [(k, stp, *g) for k in ds for stp in ({st13.TV_DEFAULT, PLATEAU13[k]}) for g in stop_grid]
    with ProcessPoolExecutor(args.workers, initializer=_init, initargs=(ds,)) as ex:
        res = list(ex.map(_job, jobs, chunksize=8))
    pd.DataFrame([dict(key=r["key"], st="{}/{}".format(*r["st"]), atr_n=r["atr_n"], k_sl=fmt(r["k_sl"]), k_tr=fmt(r["k_tr"]),
                       act=r["act"], **{"is_" + x: y for x, y in r["IS"].items()},
                       **{"oos_" + x: y for x, y in r["OOS"].items()}) for r in res]).to_csv(R / "stx_grid_all.csv", index=False)
    print("總嘗試數（本檔）：{}；每個週期 {}（SuperTrend 參數 2 組 × 停損組合 {}）".format(
        len(res), 2 * len(stop_grid), len(stop_grid)))

    rows, ovf = [], []
    _init(ds)
    for key in ds:
        D = ds[key]
        sub_all = [r for r in res if r["key"] == key]
        mt = MIN_TRADES.get(key, 50)
        # 3×3 鄰域（k_sl × k_tr 平面，同 ATR／act／SuperTrend 參數；none 視為最外側一格）
        look = {(r["st"], r["atr_n"], r["k_sl"], r["k_tr"], r["act"]): r for r in sub_all}

        def nscore(r):
            i, j = K_SL.index(r["k_sl"]), K_TR.index(r["k_tr"])
            vals = []
            for di, dj in itertools.product((-1, 0, 1), repeat=2):
                if 0 <= i + di < len(K_SL) and 0 <= j + dj < len(K_TR):
                    kt = K_TR[j + dj]
                    q = look.get((r["st"], r["atr_n"], K_SL[i + di], kt, r["act"] if kt != INF else 0.0))
                    if q:
                        vals.append(score(q["IS"]))
            return float(np.mean(vals))

        elig = [r for r in sub_all if r["IS"]["trades"] >= mt and r["IS"]["pf"] > 1] or sub_all
        pick = max(elig, key=lambda r: (nscore(r), score(r["IS"])))
        st_p = pick["st"]
        nostop = look[(st_p, 14, INF, INF, 0.0)]
        dflt_nostop = look[(st13.TV_DEFAULT, 14, INF, INF, 0.0)]
        best_cell = max(elig, key=lambda r: score(r["IS"]))
        for lab, r in (("ST no stop", nostop), ("ST + stops (chosen)", pick), ("TV default no stop", dflt_nostop),
                       ("IS-best single cell", best_cell)):
            for per in ("IS", "OOS"):
                x = r[per]
                rows.append(dict(tf=key, variant=lab, st="{}/{}".format(*r["st"]), atr_n=r["atr_n"], k_sl=fmt(r["k_sl"]),
                                 k_tr=fmt(r["k_tr"]), act=r["act"], sample=per, trades=x["trades"], win=x["win"],
                                 net=x["net"], mdd=x["mdd"], pf=x["pf"], sharpe=x["sharpe"]))
        isv = [r["IS"]["sharpe"] for r in sub_all]
        oosv = sorted(r["OOS"]["sharpe"] for r in sub_all)
        pct = lambda r: round(100 * np.searchsorted(oosv, r["OOS"]["sharpe"], side="right") / len(oosv), 1)
        n_trials = len(sub_all) + 546  # 加上 13 的 SuperTrend 網格
        sr0 = st13.expected_max_sr(n_trials, float(np.var(isv, ddof=1)))
        p = pick["IS"]
        ovf.append(dict(tf=key, trials_this_tf=len(sub_all), trials_incl_13=n_trials,
                        spearman_IS_OOS_sharpe=round(st13.spearman(isv, [r["OOS"]["sharpe"] for r in sub_all]), 3),
                        spearman_IS_OOS_netmdd=round(st13.spearman([score(r["IS"]) for r in sub_all], [score(r["OOS"]) for r in sub_all]), 3),
                        pick_OOS_sharpe_pctile=pct(pick), ISbest_OOS_sharpe_pctile=pct(best_cell),
                        nostop_OOS_sharpe_pctile=pct(nostop),
                        share_IS_profitable=round(100 * float(np.mean([r["IS"]["net"] > 0 for r in sub_all])), 1),
                        share_OOS_profitable=round(100 * float(np.mean([r["OOS"]["net"] > 0 for r in sub_all])), 1),
                        share_OOS_beats_nostop=round(100 * float(np.mean([r["OOS"]["sharpe"] > nostop["OOS"]["sharpe"] for r in sub_all if r["st"] == st_p])), 1),
                        expected_max_IS_sharpe_null=round(sr0, 3),
                        DSR_pick=round(st13.deflated_sharpe(p["sharpe"], p["ndays"], p["skew"], p["kurt"], sr0), 3)))

        # 熱圖：選定的 ST 參數、ATR、act；k_sl × k_tr，IS vs OOS 淨利／回撤
        fig, axs = plt.subplots(1, 2, figsize=(14, 5.5))
        for ax, per in zip(axs, ("IS", "OOS")):
            Z = np.full((len(K_SL), len(K_TR)), np.nan)
            for (stp, a, s, t, v), r in look.items():
                if stp == st_p and a == pick["atr_n"] and (v == pick["act"] or t == INF):
                    Z[K_SL.index(s), K_TR.index(t)] = score(r[per])
            lim = np.nanmax(np.abs(Z))
            im = ax.imshow(Z, origin="lower", cmap="RdYlGn", aspect="auto", vmin=-lim, vmax=lim)
            ax.set_xticks(range(len(K_TR)), [fmt(x) for x in K_TR]); ax.set_yticks(range(len(K_SL)), [fmt(x) for x in K_SL])
            ax.set_xlabel("k_tr (trailing)"); ax.set_ylabel("k_sl (initial stop)")
            for y in range(Z.shape[0]):
                for x in range(Z.shape[1]):
                    ax.text(x, y, "{:.2f}".format(Z[y, x]), ha="center", va="center", fontsize=7)
            ax.scatter(K_TR.index(pick["k_tr"]), K_SL.index(pick["k_sl"]), marker="o", s=380, facecolors="none",
                       edgecolors="purple", linewidths=2, label="chosen")
            ax.scatter(len(K_TR) - 1, len(K_SL) - 1, marker="s", s=380, facecolors="none", edgecolors="blue",
                       linewidths=2, label="no stops")
            ax.set_title("{} net/maxDD (loss: net/1e7)".format(per)); fig.colorbar(im, ax=ax)
        axs[0].legend(loc="lower left", fontsize=8)
        fig.suptitle("SuperTrend {}/{} + ATR{} stops, act={}  TXF {}".format(st_p[0], st_p[1], pick["atr_n"], pick["act"], key))
        fig.tight_layout(); fig.savefig(R / "stx_heatmap_{}.png".format(key), dpi=100); plt.close(fig)

        # 權益曲線與逐年
        plt.figure(figsize=(11, 5))
        yearly = {}
        for lab, r in (("ST {}/{} + stops (chosen)".format(*st_p), pick), ("ST {}/{} no stop".format(*st_p), nostop),
                       ("TV default 10/3.0 no stop", dflt_nostop)):
            pnl, _ = simulate(D, r["st"][0], r["st"][1], r["atr_n"], r["k_sl"], r["k_tr"], r["act"])
            plt.plot(pd.to_datetime(D["ts"]), np.cumsum(pnl) / 1e3, label=lab)
            yearly[lab] = pd.Series(pnl, index=pd.to_datetime(D["ts"]).year).groupby(level=0).sum().round(0)
        plt.axvline(D["split"][1], color="k", ls=":", label="IS | OOS")
        plt.title("SuperTrend + ATR stop/trail TXF {} (mark-to-market, net of costs, k TWD)".format(key))
        plt.legend(fontsize=8); plt.grid()
        plt.savefig(R / "stx_equity_{}.png".format(key), dpi=110); plt.close()
        pd.DataFrame(yearly).to_csv(R / "stx_yearly_{}.csv".format(key))

    pd.DataFrame(rows).to_csv(R / "stx_summary.csv", index=False)
    pd.DataFrame(ovf).to_csv(R / "stx_overfit.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 40, "display.max_rows", 500):
        print(pd.DataFrame(rows).to_string(index=False))
        print(pd.DataFrame(ovf).to_string(index=False))


if __name__ == "__main__":
    main()
