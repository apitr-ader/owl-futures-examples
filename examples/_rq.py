"""60-66 共用（reddit r/quant 想法回測）：TXF 1 分 K 盤別表、成本、績效、驗證（IS/OOS、Spearman、WF、集中度）、黑金圖表。
慣例：收盤算訊號、下一根開盤成交；TXF 成本每邊 50 元 + 2e-5 期交稅 + 1 點滑價（市價），1 點 = 200 元；
MDD 為盤中最壞情況（持倉期間最大不利波動 MAE 計入）；換月：交易不跨 contractMonth（跨月者剔除）。不 commit 任何資料。
執行目錄：/workspace/fubon-threads（data/、results/ 相對路徑）。
"""
import math
from pathlib import Path
import numpy as np, pandas as pd

DATA1M = "data/TXF_1c_1m_both_2017-06-01_2026-10-22.parquet"
DATAD = "data/TXF_1c_D_regular_2010-01-01_2026-10-22.parquet"
OUT = Path("results"); PV = 200.0; CAP = 1_000_000
GOLD, BG, GRAY = "#d4af37", "#0b0b0b", "#888888"
COLORS = [GOLD, "#f5e6a8", "#c0392b", "#5dade2", "#58d68d", "#af7ac5", "#aaaaaa"]


def cost_pts(ep, xp, slip_sides=2):
    """單筆來回成本（點）：手續費 2×50/200 + 期交稅 2e-5×(進+出) + 滑價（每邊 1 點，限價邊不算）。"""
    return 0.5 + 2e-5 * (np.asarray(ep) + np.asarray(xp)) + slip_sides * 1.0


def load_1m():
    d = pd.read_parquet(DATA1M, columns=["date", "session", "contractMonth", "open", "high", "low", "close", "volume"])
    d["ts"] = d.date.dt.tz_localize(None); d = d.drop(columns="date").sort_values("ts").reset_index(drop=True)
    d["night"] = d.session.astype(str) != "regular"
    day = d.ts.dt.normalize()
    d["skey"] = np.where(~d.night, day, np.where(d.ts.dt.hour >= 15, day, day - pd.Timedelta(days=1)))
    days = pd.DatetimeIndex(sorted(d.loc[~d.night, "skey"].unique()))
    pos = np.clip(days.searchsorted(np.where(d.night, d.skey + pd.Timedelta(days=1), d.skey)), 0, len(days) - 1)
    d["tdate"] = days[pos]   # 夜盤歸屬次一交易日
    d["cm"] = d.contractMonth.astype(str)
    for k in ("open", "high", "low", "close"): d[k] = d[k].astype(float)
    return d


def sessions(d):
    g = d.groupby(["tdate", "night"])
    s = g.agg(i0=("ts", lambda x: x.index[0]), i1=("ts", lambda x: x.index[-1]), t0=("ts", "first"),
              o=("open", "first"), h=("high", "max"), l=("low", "min"), c=("close", "last"), cm0=("cm", "first"), cm1=("cm", "last")).reset_index()
    D = s[~s.night].set_index("tdate"); N = s[s.night].set_index("tdate")
    return D, N


def daily_from_trades(t, alld):
    dp = t.groupby("tdate").net.sum().reindex(alld, fill_value=0) * PV
    mae = (t.groupby("tdate").mae.min().reindex(alld).fillna(0) * PV) if "mae" in t else pd.Series(0.0, alld)
    return dp, mae


def sharpe(x):
    x = np.asarray(x, float); s = x.std()
    return x.mean() / s * math.sqrt(252) if s > 0 else np.nan


def stats(dp, mae=None, t=None):
    eq = dp.cumsum(); prev = eq.shift(fill_value=0); peak = np.maximum.accumulate(np.r_[0, eq.values])[:-1]
    intr = prev + (mae.values if mae is not None else 0) - peak
    out = dict(Sharpe=sharpe(dp), ann_ret=dp.sum() / (len(dp) / 252) / CAP, net_twd=dp.sum(),
               MDD_close=(eq - eq.cummax()).min(), MDD_intraday=min(np.min(intr), (eq - eq.cummax()).min()))
    if t is not None and len(t):
        out.update(trades=len(t), win=(t.net > 0).mean(), avg_pts=t.net.mean())
        tot = t.net.sum(); out["top10_share"] = t.net.nlargest(10).sum() / tot if tot > 0 else np.nan
    return out


def grid_eval(results, alld, is_end, min_tr=30):
    """results: {key: trades df}. 回傳 G (IS/OOS Sharpe) 與 daily pnl 矩陣。"""
    isd = alld < is_end; rows = []; DP = []
    for k, t in results.items():
        dp, _ = daily_from_trades(t, alld); DP.append(dp.values)
        ti = t[t.tdate < is_end]; to = t[t.tdate >= is_end]
        rows.append(dict(key=k, IS_trades=len(ti), OOS_trades=len(to), IS_Sharpe=sharpe(dp[isd]), OOS_Sharpe=sharpe(dp[~isd]),
                         IS_avg=ti.net.mean() if len(ti) else np.nan, OOS_avg=to.net.mean() if len(to) else np.nan))
    G = pd.DataFrame(rows); DP = np.array(DP)
    ok = G[G.IS_trades >= min_tr]
    sp = ok.IS_Sharpe.rank().corr(ok.OOS_Sharpe.rank()) if len(ok) > 3 else np.nan
    return G, DP, sp


def walk_forward(G, DP, alld, y0, train=3, min_tr=30):
    rows = []; ser = []
    yrs = alld.year
    for Y in range(y0, yrs.max() + 1):
        tr = (yrs >= Y - train) & (yrs < Y); ts = yrs == Y
        if ts.sum() == 0: continue
        X = DP[:, tr]; sh = X.mean(1) / X.std(1) * math.sqrt(252)
        ntr = (DP[:, tr] != 0).sum(1); sh[ntr < min_tr * train / 5] = np.nan
        if np.all(np.isnan(sh)): continue
        b = int(np.nanargmax(sh)); y = DP[b, ts]; ser.append(pd.Series(y, alld[ts]))
        rows.append(dict(year=Y, pick=str(G.key.iloc[b]), train_sharpe=round(sh[b], 2), test_sharpe=round(sharpe(y), 2), test_twd=round(y.sum())))
    w = pd.concat(ser) if ser else pd.Series(dtype=float)
    return pd.DataFrame(rows), w


def style():
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "Noto Sans CJK TC", "axes.unicode_minus": False, "figure.facecolor": BG, "axes.facecolor": BG,
                         "savefig.facecolor": BG, "axes.edgecolor": GOLD, "axes.labelcolor": GOLD, "xtick.color": GOLD, "ytick.color": GOLD,
                         "text.color": GOLD, "axes.titlecolor": GOLD, "grid.color": "#333333", "legend.facecolor": "#151515",
                         "legend.edgecolor": GOLD, "axes.prop_cycle": matplotlib.cycler(color=COLORS)})
    return plt


def report(name, G, sp, best_key, bt, alld, is_end, wfp, wfs, rnd=None):
    isd = alld < is_end; dp, mae = daily_from_trades(bt, alld)
    si = stats(dp[isd], mae[isd], bt[bt.tdate < is_end]); so = stats(dp[~isd], mae[~isd], bt[bt.tdate >= is_end])
    print(f"\n==== {name} | best {best_key} ====")
    print("IS ", {k: round(v, 3) for k, v in si.items()}); print("OOS", {k: round(v, 3) for k, v in so.items()})
    print(f"Spearman IS/OOS {sp:.3f} | grid N={len(G)} IS>0 {(G.IS_Sharpe>0).mean():.2f} OOS>0 {(G.OOS_Sharpe>0).mean():.2f}")
    yr = bt.assign(y=bt.tdate.dt.year).groupby("y").net.agg(["size", "sum", "mean"]).round(2); print("per year (pts):", yr.to_dict("index"))
    if len(wfp): print(wfp.to_string(index=False)); print("WF stitched Sharpe %.3f net %.0f" % (sharpe(wfs), wfs.sum()))
    if rnd is not None: print("random baseline OOS Sharpe mean %.3f 5-95%% [%.3f, %.3f]" % (np.mean(rnd), np.percentile(rnd, 5), np.percentile(rnd, 95)))
    return dict(idea=name, best=str(best_key), IS_Sharpe=si["Sharpe"], OOS_Sharpe=so["Sharpe"], spearman=sp,
                WF_Sharpe=sharpe(wfs) if len(wfs) else np.nan, OOS_MDD_intra=so["MDD_intraday"], top10=stats(dp, mae, bt).get("top10_share"),
                IS_trades=si.get("trades", 0), OOS_trades=so.get("trades", 0))


from numba import njit


@njit(cache=True)
def manage(o, h, l, c, j0, je, dr, ep, stp, tgt):
    """j0 開盤進場（ep 已含滑價）；停損（停損單，跳空以開盤，+1 點滑價）、停利（限價無滑價）、je 收盤平倉（1 點滑價）。
    同根停損停利皆碰 → 停損。回傳 xp, 出場滑價邊數, MAE(點，≤0), 出場 index。"""
    mae = 0.0
    for j in range(j0, je + 1):
        adv = (l[j] - ep) if dr > 0 else (ep - h[j])
        if dr > 0 and stp == stp and l[j] <= stp:
            px = min(o[j], stp) if j > j0 else stp
            return px - 1.0, 1, min(mae, px - 1.0 - ep), j
        if dr < 0 and stp == stp and h[j] >= stp:
            px = max(o[j], stp) if j > j0 else stp
            return px + 1.0, 1, min(mae, ep - px - 1.0), j
        mae = min(mae, adv)
        if tgt == tgt:
            if dr > 0 and h[j] >= tgt and j > j0 - 1:
                return max(o[j], tgt) if j > j0 else tgt, 0, mae, j
            if dr < 0 and l[j] <= tgt:
                return min(o[j], tgt) if j > j0 else tgt, 0, mae, j
    return c[je] - dr * 1.0, 1, mae, je


def mk_trades(rows):
    t = pd.DataFrame(rows, columns=["tdate", "dir", "ep", "xp", "mae"])
    if not len(t): return t.assign(gross=[], net=[], cost=[])
    t["gross"] = t.dir * (t.xp - t.ep); t["cost"] = 0.5 + 2e-5 * (t.ep + t.xp)   # 滑價已含在 ep/xp
    t["net"] = t.gross - t.cost; t["tdate"] = pd.to_datetime(t.tdate)
    return t


def rand_dir(bt, alld, is_end, reps=500, seed=7):
    """隨機方向基準：同進出時間、隨機多空（滑價/成本相同）。回傳 OOS Sharpe 分布。"""
    rng = np.random.default_rng(seed); oos = alld >= is_end; out = []
    di = alld.get_indexer(bt.tdate); g = np.abs(bt.gross.values); sg = np.sign(bt.gross.values)
    for _ in range(reps):
        s = rng.choice([-1, 1], len(bt)); net = s * sg * g - bt.cost.values - 2 * (s != 1) * 0  # 同成本
        dp = np.bincount(di, weights=net * PV, minlength=len(alld)); out.append(sharpe(dp[oos]))
    return np.array(out)
