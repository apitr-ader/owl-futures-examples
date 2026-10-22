"""15 日 K SuperTrend 當方向濾網，只做與日線同向的 60 分／15 分 SuperTrend 訊號

執行：
  python examples/15_supertrend_daily_filter.py \
      --daily data/TXF_1c_D_regular_2010-01-01_2026-10-22.parquet \
      --minute data/TXF_1c_1m_both_2017-06-01_2026-10-22.parquet --results-dir results

規則（沿用 13 的資料處理、SuperTrend、成本與績效）：
- 日線濾網：日 K（日盤，倒推調整價）的 SuperTrend 方向，只用「已收完」的日 K。
  期交所規則夜盤屬於次一交易日，但夜盤開始（15:00）時當天日盤已收盤，所以：
    日盤（交易日 T）用 T 之前最後一根日 K（通常是前一交易日收盤）；
    夜盤（T 日 15:00 開始）用 T 日（含）以前最後一根日 K（即當天日盤剛收的那根）。
  濾網只會在場次開始時改變，改變後於該場次第一根開盤成交（＝知道的下一根開盤）。
- 分 K 進場（訊號於 K 棒收盤確認，下一根開盤成交）：
    state：分 K 方向與濾網同向就持有該方向，不同向就空手（任何時候同向都會進場）。
    event：空手時，只在分 K「剛翻轉」且與濾網同向時進場；分 K 反向或濾網反向就出場，之後等下一次同向翻轉。
    longonly：state 規則但只做多（濾網多且分 K 多才持多單）。
  不會因為一方翻轉就直接反手到另一邊，除非兩者同向。
- 對照：分 K SuperTrend 單獨反手（不加濾網）、日 K SuperTrend 單獨反手（同期間）。
- 成本：每點 200 元、手續費單邊 50、期交稅單邊十萬分之二（原始價）、滑價單邊 1 點，1 口。
- 試驗數控制：日線濾網只用 10/3.0、10/2.0、14/3.0、20/3.0、24/5.75 五組；分 K 網格 ATR 5–30（每 5）× 倍數 1.0–5.0（每 0.5）。
- 選參：樣本內、交易數 ≥ 50 且 PF > 1，取分 K 網格 3×3 鄰域「淨利／最大回撤」平均最高者（含日線濾網的選擇）。

※ 純技術範例，非投資建議。
"""

import argparse
import importlib.util
import itertools
import math
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

_spec = importlib.util.spec_from_file_location("st13", Path(__file__).with_name("13_supertrend_grid.py"))
st13 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(st13)

POINT_VALUE, COMMISSION, SLIPPAGE, TAX_RATE = st13.POINT_VALUE, st13.COMMISSION, st13.SLIPPAGE, st13.TAX_RATE
DAILY_FILTERS = ((10, 3.0), (10, 2.0), (14, 3.0), (20, 3.0), (24, 5.75))
IN_PERIODS = (5, 10, 15, 20, 25, 30)
IN_MULTS = (1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0)
MODES = ("state", "event", "longonly")
LO, MID, HI = (pd.Timestamp(x) for x in st13.SPLITS["intraday"])


def daily_filter_for_bars(daily_df, Dd, sids, fp):
    """回傳每根分 K 可用的日線方向（+1／−1／0）。"""
    trend = st13.supertrend_dir(Dd, "sma", fp[0], fp[1])
    ddates = daily_df["date"].dt.tz_localize(None).dt.normalize().to_numpy()
    sess = np.array([s[0] for s in sids])
    sday = pd.to_datetime([s[1:] for s in sids], format="%Y%m%d").to_numpy()
    # 日盤：日期 < T 的最後一根；夜盤：日期 <= T 的最後一根
    idx = np.where(sess == "r", np.searchsorted(ddates, sday, side="left") - 1,
                   np.searchsorted(ddates, sday, side="right") - 1)
    out = np.where(idx >= 0, trend[np.clip(idx, 0, None)], 0)
    return out.astype(int)


def simulate(D, trend, filt, mode):
    """回傳 pnl（元/根）、交易清單、部位陣列、成本陣列。filt=None 表示不加濾網（反手）。"""
    o, c, adj = D["o"].tolist(), D["c"].tolist(), D["adj"].tolist()
    tr = trend.tolist()
    fl = filt.tolist() if filt is not None else None
    n = len(o)
    pnl = np.zeros(n); posa = np.zeros(n, dtype=np.int8); costs = np.zeros(n)
    trades = []
    pos, cur = 0, 0.0

    def side_cost(i, px):
        return COMMISSION + (px - adj[i]) * POINT_VALUE * TAX_RATE + SLIPPAGE * POINT_VALUE

    for i in range(2, n):
        d1, d2 = tr[i - 1], tr[i - 2]
        if fl is None:
            target = d1 if (pos or (d2 and d1 != d2)) else 0  # 13 的反手：第一次翻轉才進場
        else:
            f = fl[i]
            if mode == "state":
                target = d1 if (d1 and d1 == f) else 0
            elif mode == "longonly":
                target = 1 if (d1 == 1 and f == 1) else 0
            else:  # event
                if pos:
                    target = pos if (d1 == pos and f == pos) else 0
                    if target == 0 and d2 and d1 != d2 and d1 == f:
                        target = d1
                else:
                    target = d1 if (d2 and d1 != d2 and d1 == f) else 0
        g = pos * (o[i] - c[i - 1]) * POINT_VALUE
        bar = g; cur += g
        if target != pos:
            if pos:
                k = side_cost(i, o[i]); bar -= k; costs[i] += k
                trades.append((i, cur - k)); cur = 0.0
            if target:
                k = side_cost(i, o[i]); bar -= k; costs[i] += k; cur = -k
            pos = target
        r = pos * (c[i] - o[i]) * POINT_VALUE
        bar += r; cur += r
        pnl[i] = bar; posa[i] = pos
    if pos:
        trades.append((n - 1, cur))
    return pnl, trades, posa, costs


def metrics(D, sim, lo, hi):
    pnl, trades, posa, costs = sim
    m = st13.metrics(D, pnl, trades, lo, hi)
    ts = D["ts"]
    k = (ts >= np.datetime64(lo)) & (ts < np.datetime64(hi))
    m["tim"] = round(float((posa[k] != 0).mean() * 100), 1)
    m["costs"] = round(float(costs[k].sum()))
    return m


def score(m):
    return m["rdd"] if m["net"] > 0 else m["net"] / 1e7


_G = {}


def _init(g):
    global _G
    _G = g


def _job(a):
    key, fp, mode, p, mu = a
    D = _G["ds"][key]
    trend = st13.supertrend_dir(D, "sma", p, mu)
    filt = None if fp is None else _G["filt"][(key, fp)]
    sim = simulate(D, trend, filt, mode)
    return dict(key=key, fp=fp, mode=mode, p=p, mu=mu, IS=metrics(D, sim, LO, MID), OOS=metrics(D, sim, MID, HI))


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

    daily = st13.load_daily(args.daily)
    Dd = st13.prepare(daily, "daily")
    frames = {}
    for mins in (60, 15):
        f = st13.resample_sessions(args.minute, mins)
        frames["{}m_daynight".format(mins)] = f
        frames["{}m_day".format(mins)] = f[f["session"] == "regular"].reset_index(drop=True)
    ds = {k: st13.prepare(v, "intraday") for k, v in frames.items()}
    filt = {(k, fp): daily_filter_for_bars(daily, Dd, frames[k]["sid"].tolist(), fp) for k in ds for fp in DAILY_FILTERS}

    jobs = [(k, fp, mo, p, mu) for k in ds for fp in DAILY_FILTERS for mo in MODES for p in IN_PERIODS for mu in IN_MULTS]
    jobs += [(k, None, "nofilter", p, mu) for k in ds for p in IN_PERIODS for mu in IN_MULTS]
    with ProcessPoolExecutor(args.workers, initializer=_init, initargs=(dict(ds=ds, filt=filt),)) as ex:
        res = list(ex.map(_job, jobs, chunksize=8))
    _init(dict(ds=ds, filt=filt))
    pd.DataFrame([dict(key=r["key"], filter="none" if r["fp"] is None else "{}/{}".format(*r["fp"]), mode=r["mode"],
                       p=r["p"], mu=r["mu"], **{"is_" + a: b for a, b in r["IS"].items()},
                       **{"oos_" + a: b for a, b in r["OOS"].items()}) for r in res]).to_csv(R / "mtf_grid_all.csv", index=False)
    print("總嘗試數：{}（濾網組合 {}、不加濾網 {}）；每個週期每種模式 {}".format(
        len(res), len(jobs) - len(ds) * 54, len(ds) * 54, len(DAILY_FILTERS) * 54))

    # 日 K 單獨（反手），在分 K 的樣本內外區間計算
    daily_only = {}
    for fp in DAILY_FILTERS:
        pnl, trd = st13.simulate(Dd, "sma", fp[0], fp[1], "reversal")
        tr_ = st13.supertrend_dir(Dd, "sma", fp[0], fp[1])
        daily_only[fp] = (pnl, trd)

    rows, ovf = [], []
    for key in ds:
        D = ds[key]
        look = {(r["fp"], r["mode"], r["p"], r["mu"]): r for r in res if r["key"] == key}

        def nscore(fp, mode, p, mu):
            i, j = IN_PERIODS.index(p), IN_MULTS.index(mu)
            v = [score(look[(fp, mode, IN_PERIODS[a], IN_MULTS[b])]["IS"])
                 for a in range(max(i - 1, 0), min(i + 2, len(IN_PERIODS)))
                 for b in range(max(j - 1, 0), min(j + 2, len(IN_MULTS)))]
            return float(np.mean(v))

        nof = [r for r in res if r["key"] == key and r["mode"] == "nofilter"]
        nof_pick = max([r for r in nof if r["IS"]["trades"] >= 50 and r["IS"]["pf"] > 1] or nof,
                       key=lambda r: nscore(None, "nofilter", r["p"], r["mu"]))
        for mode in MODES:
            sub = [r for r in res if r["key"] == key and r["mode"] == mode]
            elig = [r for r in sub if r["IS"]["trades"] >= 50 and r["IS"]["pf"] > 1] or sub
            pick = max(elig, key=lambda r: (nscore(r["fp"], mode, r["p"], r["mu"]), score(r["IS"])))
            same_nof = look[(None, "nofilter", pick["p"], pick["mu"])]
            dpnl, dtr = daily_only[pick["fp"]]
            dI, dO = (st13.metrics(Dd, dpnl, dtr, a, b) for a, b in ((LO, MID), (MID, HI)))
            for lab, r in (("filtered (chosen)", pick), ("same intraday params, no filter", same_nof),
                           ("intraday no filter (own plateau)", nof_pick)):
                for per in ("IS", "OOS"):
                    x = r[per]
                    rows.append(dict(tf=key, mode=mode, variant=lab,
                                     filter="none" if r["fp"] is None else "{}/{}".format(*r["fp"]),
                                     intraday="{}/{}".format(r["p"], r["mu"]), sample=per,
                                     **{k: x[k] for k in ("trades", "win", "net", "mdd", "pf", "sharpe", "tim", "costs")}))
            for per, x in (("IS", dI), ("OOS", dO)):
                rows.append(dict(tf=key, mode=mode, variant="daily ST alone", filter="{}/{}".format(*pick["fp"]),
                                 intraday="-", sample=per, **{k: x[k] for k in ("trades", "win", "net", "mdd", "pf", "sharpe")},
                                 tim=100.0, costs=None))
            isv = [r["IS"]["sharpe"] for r in sub]
            oos_sorted = sorted(r["OOS"]["sharpe"] for r in sub)
            sr0 = st13.expected_max_sr(len(sub), float(np.var(isv, ddof=1)))
            pI = pick["IS"]
            ovf.append(dict(tf=key, mode=mode, cells=len(sub), pick="{} | {}/{}".format(pick["fp"], pick["p"], pick["mu"]),
                            spearman_sharpe=round(st13.spearman(isv, [r["OOS"]["sharpe"] for r in sub]), 3),
                            spearman_netmdd=round(st13.spearman([score(r["IS"]) for r in sub], [score(r["OOS"]) for r in sub]), 3),
                            pick_OOS_pctile=round(100 * np.searchsorted(oos_sorted, pick["OOS"]["sharpe"], side="right") / len(sub), 1),
                            share_IS_profitable=round(100 * float(np.mean([r["IS"]["net"] > 0 for r in sub])), 1),
                            share_OOS_profitable=round(100 * float(np.mean([r["OOS"]["net"] > 0 for r in sub])), 1),
                            share_OOS_profitable_nofilter=round(100 * float(np.mean([r["OOS"]["net"] > 0 for r in nof])), 1),
                            sr0=round(sr0, 3),
                            DSR_pick=round(st13.deflated_sharpe(pI["sharpe"], pI["ndays"], pI["skew"], pI["kurt"], sr0), 3)))
            # 圖：只畫 state／event 兩種模式
            if mode == "longonly":
                continue
            fig, axs = plt.subplots(1, 2, figsize=(14, 5.5))
            for ax, per in zip(axs, ("IS", "OOS")):
                Z = np.array([[score(look[(pick["fp"], mode, p, mu)][per]) for mu in IN_MULTS] for p in IN_PERIODS])
                lim = np.nanmax(np.abs(Z))
                im = ax.imshow(Z, origin="lower", cmap="RdYlGn", aspect="auto", vmin=-lim, vmax=lim)
                ax.set_xticks(range(len(IN_MULTS)), IN_MULTS); ax.set_yticks(range(len(IN_PERIODS)), IN_PERIODS)
                ax.set_xlabel("intraday ATR multiplier"); ax.set_ylabel("intraday ATR period")
                for y in range(Z.shape[0]):
                    for x in range(Z.shape[1]):
                        ax.text(x, y, "{:.2f}".format(Z[y, x]), ha="center", va="center", fontsize=7)
                ax.scatter(IN_MULTS.index(pick["mu"]), IN_PERIODS.index(pick["p"]), marker="o", s=420,
                           facecolors="none", edgecolors="purple", linewidths=2)
                ax.set_title("{} net/maxDD (loss: net/1e7); circle = chosen".format(per)); fig.colorbar(im, ax=ax)
            fig.suptitle("Intraday SuperTrend + daily filter {}/{} ({}), TXF {}".format(*pick["fp"], mode, key))
            fig.tight_layout(); fig.savefig(R / "mtf_heatmap_{}_{}.png".format(key, mode), dpi=100); plt.close(fig)

            plt.figure(figsize=(11, 5))
            yearly = {}
            tsx = pd.to_datetime(D["ts"])
            for lab, r in (("filtered {}/{} | {}/{}".format(*pick["fp"], pick["p"], pick["mu"]), pick),
                           ("no filter {}/{}".format(pick["p"], pick["mu"]), same_nof)):
                sim = simulate(D, st13.supertrend_dir(D, "sma", r["p"], r["mu"]),
                               None if r["fp"] is None else filt[(key, r["fp"])], r["mode"])
                k = tsx >= LO
                plt.plot(tsx[k], np.cumsum(sim[0][k]) / 1e3, label=lab)
                yearly[lab] = pd.Series(sim[0], index=tsx.year).groupby(level=0).sum().round(0)
            dts = pd.to_datetime(Dd["ts"])
            k = dts >= LO
            plt.plot(dts[k], np.cumsum(dpnl[k]) / 1e3, label="daily ST alone {}/{}".format(*pick["fp"]))
            yearly["daily ST alone {}/{}".format(*pick["fp"])] = pd.Series(dpnl[k], index=dts[k].year).groupby(level=0).sum().round(0)
            plt.axvline(MID, color="k", ls=":", label="IS | OOS")
            plt.title("TXF {} daily-filtered SuperTrend ({}), mark-to-market, net of costs (k TWD)".format(key, mode))
            plt.legend(fontsize=8); plt.grid()
            plt.savefig(R / "mtf_equity_{}_{}.png".format(key, mode), dpi=110); plt.close()
            pd.DataFrame(yearly).to_csv(R / "mtf_yearly_{}_{}.csv".format(key, mode))

    pd.DataFrame(rows).to_csv(R / "mtf_summary.csv", index=False)
    pd.DataFrame(ovf).to_csv(R / "mtf_overfit.csv", index=False)
    with pd.option_context("display.width", 260, "display.max_columns", 40, "display.max_rows", 500):
        print(pd.DataFrame(rows).to_string(index=False))
        print(pd.DataFrame(ovf).to_string(index=False))


if __name__ == "__main__":
    main()
