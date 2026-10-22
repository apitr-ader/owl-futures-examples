"""13 SuperTrend 策略：台指期日 K／60 分 K／15 分 K 參數網格、樣本內外、過度最佳化檢查

執行：
  python examples/13_supertrend_grid.py \
      --daily data/TXF_1c_D_regular_2010-01-01_2026-10-22.parquet \
      --minute data/TXF_1c_1m_both_2017-06-01_2026-10-22.parquet \
      --data-dir data --results-dir results

SuperTrend（經典寫法，src = hl2）：
  up = src − mult×ATR；若前一根收盤 > 前一根 up，up = max(up, 前一根 up)
  dn = src + mult×ATR；若前一根收盤 < 前一根 dn，dn = min(dn, 前一根 dn)
  趨勢 −1 且收盤 > 前一根 dn → 翻多；趨勢 +1 且收盤 < 前一根 up → 翻空
ATR 計算：sma（True Range 簡單平均，參考的 TradingView 腳本預設）或 rma（Wilder，ta.atr）。
進出場（以 K 線收盤確認訊號，「下一根開盤」成交，不偷看）：
  reversal：翻多做多、翻空做空，永遠在市場內（第一次翻轉才進場）
  longonly：翻多做多、翻空平倉（參考腳本目前版本的做法）
價格：訊號與點數損益用倒推調整價（換月價差＝新合約第一根開盤 − 舊合約最後一根收盤，近似），期交稅用原始價。
成本：每點 200 元、手續費單邊 50 元、期交稅單邊十萬分之二、滑價單邊 1 點（反手＝平倉＋開倉兩邊）。
分 K：1 分 K 依場次合成（K 線以開始時間標示）。日盤 08:45–13:45、夜盤 15:00–05:00，各自從開盤起每 N 分鐘切，
  不跨場次；13:45 收盤集合競價併入最後一根。
  day_night：日夜盤 K 線依時間串成一條序列（夜盤也交易）；
  day：只用日盤 K 線（部位會帶過夜，夜盤的漲跌反映在隔天第一根開盤的跳空）。
績效：逐根「市值」權益（含未平倉損益）；淨利、最大回撤、夏普（日損益 ×√252）以期間內權益計，
  交易筆數／勝率／PF 以「出場落在期間內」的交易計。

※ 純技術範例，非投資建議。
"""

import argparse
import math
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from statistics import NormalDist

import numpy as np
import pandas as pd

POINT_VALUE = 200
TAX_RATE = 2 / 100_000
COMMISSION = 50.0
SLIPPAGE = 1.0
PERIODS = tuple(range(5, 31))
MULTS = tuple(np.round(np.arange(1.0, 6.01, 0.25), 2).tolist())
TV_DEFAULT = (10, 3.0)
TZ = "Asia/Taipei"
SPLITS = {"daily": ("2010-01-01", "2020-01-01", "2026-12-31"),
          "intraday": ("2017-06-01", "2023-01-01", "2026-12-31")}


# ---------------- 資料 ----------------

def load_daily(path):
    d = pd.read_parquet(path)
    d["date"] = pd.to_datetime(d["date"]).dt.tz_convert(TZ) if pd.to_datetime(d["date"]).dt.tz is not None \
        else pd.to_datetime(d["date"]).dt.tz_localize(TZ)
    d["contractMonth"] = d["contractMonth"].astype(str)
    return d.sort_values("date").reset_index(drop=True)[["date", "open", "high", "low", "close", "contractMonth"]]


def resample_sessions(path, minutes):
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"]).dt.tz_convert(TZ)
    df["session"] = df["session"].astype(str)
    df["contractMonth"] = df["contractMonth"].astype(str)
    ts = df["date"]
    night = (df["session"] == "afterhours").to_numpy()
    day0 = ts.dt.normalize()
    sess_day = day0.where(~(night & (ts.dt.hour < 15).to_numpy()), day0 - pd.Timedelta(days=1))
    start = sess_day + pd.to_timedelta(np.where(night, 15 * 60, 8 * 60 + 45), unit="m")
    length = np.where(night, 14 * 60, 5 * 60)
    off = ((ts - start).dt.total_seconds() // 60).astype(int).to_numpy()
    off = np.minimum(off, length - 1)  # 13:45 收盤集合競價併入最後一根
    df["date"] = start + pd.to_timedelta((off // minutes) * minutes, unit="m")
    df["sid"] = np.where(night, "a", "r") + sess_day.dt.strftime("%Y%m%d")
    g = df.sort_values("date").groupby(["sid", "date"], sort=True)
    out = g.agg(session=("session", "first"), open=("open", "first"), high=("high", "max"),
                low=("low", "min"), close=("close", "last"), volume=("volume", "sum"),
                contractMonth=("contractMonth", "last")).reset_index().sort_values("date").reset_index(drop=True)
    for c in ("open", "high", "low", "close"):
        out[c] = out[c].astype("float64")
    return out


def back_adjust(df):
    cm = df["contractMonth"]
    roll = cm.ne(cm.shift()) & cm.shift().notna()
    gap = (df["open"] - df["close"].shift()).where(roll, 0.0)
    return gap[::-1].cumsum()[::-1].shift(-1, fill_value=0.0).to_numpy()


def prepare(df, kind):
    adj = back_adjust(df)
    o, h, l, c = (df[k].to_numpy() + adj for k in ("open", "high", "low", "close"))
    pc = np.r_[np.nan, c[:-1]]
    tr = np.nanmax(np.vstack([h - l, np.abs(h - pc), np.abs(l - pc)]), axis=0)
    trs = pd.Series(tr)
    atr = {}
    for p in PERIODS:
        atr[("sma", p)] = trs.rolling(p).mean().to_numpy()
        atr[("rma", p)] = trs.ewm(alpha=1 / p, adjust=False, min_periods=p).mean().to_numpy()
    days = df["date"].dt.tz_localize(None).dt.normalize().to_numpy()
    lo, mid, hi = (pd.Timestamp(x) for x in SPLITS[kind])
    return dict(o=o, h=h, l=l, c=c, adj=adj, src=(h + l) / 2, atr=atr, days=days,
                ts=df["date"].dt.tz_localize(None).to_numpy(), split=(lo, mid, hi))


# ---------------- 回測 ----------------

def supertrend_dir(D, method, period, mult):
    src, c, a = D["src"], D["c"], D["atr"][(method, period)]
    n = len(c)
    trend = np.zeros(n, dtype=np.int8)
    up_prev = dn_prev = None
    t = 1
    srcl, cl, al = src.tolist(), c.tolist(), a.tolist()
    for i in range(n):
        ai = al[i]
        if ai != ai:
            continue
        up = srcl[i] - mult * ai
        dn = srcl[i] + mult * ai
        if up_prev is None:
            up_prev, dn_prev = up, dn
        else:
            if cl[i - 1] > up_prev:
                up = max(up, up_prev)
            if cl[i - 1] < dn_prev:
                dn = min(dn, dn_prev)
        if t == -1 and cl[i] > dn_prev:
            t = 1
        elif t == 1 and cl[i] < up_prev:
            t = -1
        trend[i] = t
        up_prev, dn_prev = up, dn
    return trend


def simulate(D, method, period, mult, mode):
    """回傳 (逐根損益 ndarray（元，市值法）, 交易清單 [(出場索引, 該筆淨損益元)])。

    訊號在第 i−1 根收盤確認，第 i 根開盤成交。最後仍持有的部位以最後收盤計值（不扣出場成本）。
    """
    trend = supertrend_dir(D, method, period, mult).tolist()
    o, c, adj = D["o"].tolist(), D["c"].tolist(), D["adj"].tolist()
    n = len(o)
    pnl = np.zeros(n)
    trades = []
    pos, cur = 0, 0.0

    def side_cost(i, px):
        return COMMISSION + (px - adj[i]) * POINT_VALUE * TAX_RATE + SLIPPAGE * POINT_VALUE

    for i in range(2, n):
        prev, prev2 = trend[i - 1], trend[i - 2]
        target = pos
        if prev2 != 0 and prev != prev2:
            target = prev if mode == "reversal" else (1 if prev == 1 else 0)
        g = pos * (o[i] - c[i - 1]) * POINT_VALUE
        bar = g
        cur += g
        if target != pos:
            if pos:
                k = side_cost(i, o[i])
                bar -= k
                trades.append((i, cur - k))
                cur = 0.0
            if target:
                k = side_cost(i, o[i])
                bar -= k
                cur = -k
            pos = target
        r = pos * (c[i] - o[i]) * POINT_VALUE
        bar += r
        cur += r
        pnl[i] = bar
    if pos:
        trades.append((n - 1, cur))
    return pnl, trades


def metrics(D, pnl, trades, lo, hi):
    ts = D["ts"]
    m = (ts >= np.datetime64(lo)) & (ts < np.datetime64(hi))
    if not m.any():
        return {}
    p = pnl[m]
    eq = np.cumsum(p)
    mdd = float(np.min(eq - np.maximum.accumulate(np.r_[0.0, eq])[1:]))
    daily = pd.Series(p, index=D["days"][m]).groupby(level=0).sum()
    sh = float(daily.mean() / daily.std() * math.sqrt(252)) if daily.std() > 0 else 0.0
    idx0, idx1 = np.flatnonzero(m)[[0, -1]]
    nets = np.array([v for i, v in trades if idx0 <= i <= idx1])
    gw, gl = (nets[nets > 0].sum(), -nets[nets < 0].sum()) if len(nets) else (0.0, 0.0)
    net = float(eq[-1])
    return dict(trades=int(len(nets)), win=round(float((nets > 0).mean() * 100), 1) if len(nets) else 0.0,
                net=round(net), mdd=round(min(mdd, 0.0)), pf=round(float(gw / gl), 3) if gl else 0.0,
                sharpe=round(sh, 3), rdd=round(net / -mdd, 3) if mdd < 0 else 0.0,
                skew=float(daily.skew()), kurt=float(daily.kurt() + 3), ndays=int(len(daily)))


_DS = {}


def _init(ds):
    global _DS
    _DS = ds


def _job(a):
    key, method, mode, period, mult = a
    D = _DS[key]
    pnl, tr = simulate(D, method, period, mult, mode)
    lo, mid, hi = D["split"]
    return dict(key=key, method=method, mode=mode, period=period, mult=mult,
                IS=metrics(D, pnl, tr, lo, mid), OOS=metrics(D, pnl, tr, mid, hi))


# ---------------- 過度最佳化統計 ----------------

def plateau_scores(sub, field="rdd"):
    M = np.full((len(PERIODS), len(MULTS)), np.nan)
    for r in sub:
        v = r["IS"][field] if r["IS"]["net"] > 0 else r["IS"]["net"] / 1e7
        M[PERIODS.index(r["period"]), MULTS.index(r["mult"])] = v
    S = np.full_like(M, np.nan)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            S[i, j] = np.nanmean(M[max(i - 2, 0):i + 3, max(j - 2, 0):j + 3])  # 5×5 鄰域
    return M, S


def expected_max_sr(n_trials, var_sr):
    g = 0.5772156649
    nd = NormalDist()
    return math.sqrt(var_sr) * ((1 - g) * nd.inv_cdf(1 - 1 / n_trials) + g * nd.inv_cdf(1 - 1 / (n_trials * math.e)))


def deflated_sharpe(sr_ann, n_days, skew, kurt, sr0_ann):
    """Bailey & López de Prado DSR，以日夏普計算。"""
    sr, sr0 = sr_ann / math.sqrt(252), sr0_ann / math.sqrt(252)
    den = math.sqrt(max(1 - skew * sr + (kurt - 1) / 4 * sr * sr, 1e-12))
    return NormalDist().cdf((sr - sr0) * math.sqrt(n_days - 1) / den)


def spearman(a, b):
    return float(pd.Series(a).rank().corr(pd.Series(b).rank()))


# ---------------- 主程式 ----------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--daily", required=True)
    ap.add_argument("--minute", required=True)
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    args = ap.parse_args()
    R = Path(args.results_dir)
    R.mkdir(parents=True, exist_ok=True)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    frames = {"D": load_daily(args.daily)}
    for mins in (60, 15):
        f = resample_sessions(args.minute, mins)
        f.to_parquet(Path(args.data_dir) / "TXF_1c_{}m_sessions.parquet".format(mins), compression="zstd", index=False)
        frames["{}m_daynight".format(mins)] = f
        frames["{}m_day".format(mins)] = f[f["session"] == "regular"].reset_index(drop=True)
    ds = {k: prepare(v, "daily" if k == "D" else "intraday") for k, v in frames.items()}
    for k, v in frames.items():
        print("{}: {} 根，{} ~ {}".format(k, len(v), v["date"].iloc[0], v["date"].iloc[-1]))

    variants = [("sma", "reversal"), ("rma", "reversal"), ("sma", "longonly")]
    jobs = [(k, me, mo, p, m) for k in ds for me, mo in variants for p in PERIODS for m in MULTS]
    with ProcessPoolExecutor(args.workers, initializer=_init, initargs=(ds,)) as ex:
        res = list(ex.map(_job, jobs, chunksize=8))
    flat = [dict(key=r["key"], method=r["method"], mode=r["mode"], period=r["period"], mult=r["mult"],
                 **{"is_" + a: b for a, b in r["IS"].items()}, **{"oos_" + a: b for a, b in r["OOS"].items()})
            for r in res]
    pd.DataFrame(flat).to_csv(R / "st_grid_all.csv", index=False)

    rows, ovf = [], []
    _init(ds)
    min_trades = {"D": 20}
    for key in ds:
        for me, mo in variants:
            sub = [r for r in res if r["key"] == key and r["method"] == me and r["mode"] == mo]
            look = {(r["period"], r["mult"]): r for r in sub}
            M, S = plateau_scores(sub)
            mt = min_trades.get(key, 50)
            elig = [r for r in sub if r["IS"]["trades"] >= mt and r["IS"]["pf"] > 1] or sub
            best = max(elig, key=lambda r: r["IS"]["rdd"] if r["IS"]["net"] > 0 else r["IS"]["net"] / 1e7)
            plat = max(elig, key=lambda r: S[PERIODS.index(r["period"]), MULTS.index(r["mult"])])
            dflt = look[TV_DEFAULT]
            for lab, r in (("TV default", dflt), ("IS-best", best), ("plateau", plat)):
                for per in ("IS", "OOS"):
                    x = r[per]
                    rows.append(dict(tf=key, atr=me, mode=mo, pick=lab, period_n=r["period"], mult=r["mult"], sample=per,
                                     trades=x["trades"], win=x["win"], net=x["net"], mdd=x["mdd"], pf=x["pf"],
                                     sharpe=x["sharpe"], net_mdd=x["rdd"]))
            isv = [r["IS"]["sharpe"] for r in sub]
            oosv = [r["OOS"]["sharpe"] for r in sub]
            oos_sorted = sorted(oosv)
            pct = lambda r: round(100 * np.searchsorted(oos_sorted, r["OOS"]["sharpe"], side="right") / len(oosv), 1)
            sr0 = expected_max_sr(len(sub), float(np.var(isv, ddof=1)))
            b = best["IS"]
            ovf.append(dict(tf=key, atr=me, mode=mo, cells=len(sub),
                            spearman_IS_OOS_sharpe=round(spearman(isv, oosv), 3),
                            spearman_IS_OOS_netmdd=round(spearman([r["IS"]["rdd"] if r["IS"]["net"] > 0 else r["IS"]["net"] / 1e7 for r in sub],
                                                                  [r["OOS"]["rdd"] if r["OOS"]["net"] > 0 else r["OOS"]["net"] / 1e7 for r in sub]), 3),
                            best_cell=(best["period"], best["mult"]), best_OOS_sharpe_pctile=pct(best),
                            plateau_cell=(plat["period"], plat["mult"]), plateau_OOS_sharpe_pctile=pct(plat),
                            default_OOS_sharpe_pctile=pct(dflt),
                            share_IS_profitable=round(float(np.mean([r["IS"]["net"] > 0 for r in sub])) * 100, 1),
                            share_OOS_profitable=round(float(np.mean([r["OOS"]["net"] > 0 for r in sub])) * 100, 1),
                            best_sharpe_IS_to_OOS="{} → {}".format(best["IS"]["sharpe"], best["OOS"]["sharpe"]),
                            plateau_sharpe_IS_to_OOS="{} → {}".format(plat["IS"]["sharpe"], plat["OOS"]["sharpe"]),
                            default_sharpe_IS_to_OOS="{} → {}".format(dflt["IS"]["sharpe"], dflt["OOS"]["sharpe"]),
                            expected_max_IS_sharpe_under_null=round(sr0, 3),
                            DSR_best=round(deflated_sharpe(b["sharpe"], b["ndays"], b["skew"], b["kurt"], sr0), 3)))
            if (me, mo) != ("sma", "reversal"):
                continue
            # 熱圖：IS / OOS 淨利÷回撤 與 夏普
            for metric in ("rdd", "sharpe"):
                fig, axs = plt.subplots(1, 2, figsize=(15, 7))
                for ax, per in zip(axs, ("IS", "OOS")):
                    Z = np.full((len(PERIODS), len(MULTS)), np.nan)
                    for r in sub:
                        v = r[per][metric]
                        if metric == "rdd" and r[per]["net"] <= 0:
                            v = r[per]["net"] / 1e7
                        Z[PERIODS.index(r["period"]), MULTS.index(r["mult"])] = v
                    lim = np.nanmax(np.abs(Z))
                    im = ax.imshow(Z, origin="lower", cmap="RdYlGn", aspect="auto", vmin=-lim, vmax=lim)
                    ax.set_xticks(range(0, len(MULTS), 2), MULTS[::2]); ax.set_yticks(range(len(PERIODS)), PERIODS, fontsize=7)
                    ax.set_xlabel("ATR multiplier"); ax.set_ylabel("ATR period")
                    for (pp, mm), mk, col, lab in ((TV_DEFAULT, "s", "blue", "TV default 10/3.0"),
                                                   ((best["period"], best["mult"]), "*", "black", "IS-best"),
                                                   ((plat["period"], plat["mult"]), "o", "purple", "plateau")):
                        ax.scatter(MULTS.index(mm), PERIODS.index(pp), marker=mk, s=160, facecolors="none" if mk != "*" else col,
                                   edgecolors=col, linewidths=2, label=lab)
                    ax.set_title("{} {}".format(per, "net/maxDD (loss: net/1e7)" if metric == "rdd" else "Sharpe"))
                    fig.colorbar(im, ax=ax)
                axs[0].legend(loc="upper right", fontsize=8)
                fig.suptitle("SuperTrend reversal (SMA ATR) TXF {}".format(key))
                fig.tight_layout(); fig.savefig(R / "st_heatmap_{}_{}.png".format(key, metric), dpi=100); plt.close(fig)
            # 權益曲線＋逐年（plateau、IS-best、default）
            D = ds[key]
            plt.figure(figsize=(11, 5))
            yearly = {}
            for lab, r in (("TV default 10/3.0", dflt), ("IS-best {}/{}".format(best["period"], best["mult"]), best),
                           ("plateau {}/{}".format(plat["period"], plat["mult"]), plat)):
                pnl, _ = simulate(D, me, r["period"], r["mult"], mo)
                plt.plot(pd.to_datetime(D["ts"]), np.cumsum(pnl) / 1e3, label=lab)
                yearly[lab] = pd.Series(pnl, index=pd.to_datetime(D["ts"]).year).groupby(level=0).sum().round(0)
            plt.axvline(D["split"][1], color="k", ls=":", label="IS | OOS")
            plt.title("SuperTrend reversal TXF {} (mark-to-market, net of costs, k TWD)".format(key))
            plt.legend(fontsize=8); plt.grid()
            plt.savefig(R / "st_equity_{}.png".format(key), dpi=110); plt.close()
            pd.DataFrame(yearly).to_csv(R / "st_yearly_{}.csv".format(key))

    pd.DataFrame(rows).to_csv(R / "st_summary.csv", index=False)
    pd.DataFrame(ovf).to_csv(R / "st_overfit.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 40, "display.max_rows", 500):
        print(pd.DataFrame(rows).to_string(index=False))
        print(pd.DataFrame(ovf).to_string(index=False))


if __name__ == "__main__":
    main()
