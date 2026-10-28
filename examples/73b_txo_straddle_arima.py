"""73b Reddit #11：TXO 1 分鐘 ATM straddle 權利金序列 + ARIMA 預測（樣本外）。資料由 73a 建（每 2 個交易日 1 日，2017–2026）。

每日：用前 90 分鐘（08:45–10:15）以 MLE 擬合 ARIMA(p,1,q)，參數固定後對 10:15–13:44 做一步預測（apply 濾波，只用已知資料）。
比較基準：隨機漫步（不變）與「θ 漂移」（前 90 分鐘平均每分鐘變化）。指標：RMSE 比、方向命中率、Spearman(預測, 實際)。
最小 straddle 履約價切換的那一分鐘剔除（不可連續交易）。
交易檢驗：預測下一分鐘下跌 > c 點 → 賣 straddle 1 分鐘（上漲 > c → 買）；成本每腿每邊 30 元 + 0.1% + 1 tick，
  兩腿來回 ≈ 4 邊；用成交價（無買賣價差資料，偏樂觀）。IS 2017–2021 選 (order, c)，OOS 2022+，隨機同頻率基準，逐年 WF。
"""
import sys, glob, warnings, math, itertools
from concurrent.futures import ProcessPoolExecutor
import numpy as np, pandas as pd
from _rqo import *
warnings.filterwarnings("ignore")
SRC = ROOT / "data/txo_1m_straddle"; ORD = [(1, 1, 0), (0, 1, 1), (1, 1, 1), (2, 1, 0), (0, 1, 2)]; FIT = 90


def one(f):
    from statsmodels.tsa.arima.model import ARIMA
    d = pd.read_parquet(f); y = d.straddle.values; K = d.K.values
    if np.isnan(y[:FIT]).sum() > 30 or np.isnan(y[FIT:]).any(): return None
    st = np.argmax(~np.isnan(y)); yy = y[st:]; kk = K[st:]; n0 = FIT - st
    out = {"date": Path(f).stem, "y": yy[n0 - 1:], "jump": np.r_[False, kk[1:] != kk[:-1]][n0:], "drift": np.nanmean(np.diff(yy[:n0]))}
    for o in ORD:
        try:
            r = ARIMA(yy[:n0], order=o).fit(); r2 = r.apply(yy)
            fv = np.asarray(r2.fittedvalues)[n0:]          # t 時刻（用到 t-1 為止）對 y_t 的預測
            out[o] = fv
        except Exception:
            out[o] = np.full(len(yy) - n0, np.nan)
    return out


def main():
    import os; fs = sorted(glob.glob(str(SRC / "*.parquet")))[::int(os.environ.get("STEP", "1"))]
    with ProcessPoolExecutor(6) as ex: R = [r for r in ex.map(one, fs, chunksize=8) if r is not None]
    print("days", len(R))
    rows = []
    for r in R:
        y = r["y"]; prev = y[:-1]; act = y[1:] - prev; ok = ~r["jump"]
        base = dict(date=pd.Timestamp(r["date"]), prev=prev[ok], act=act[ok], drift=np.full(ok.sum(), r["drift"]))
        for o in ORD: base[o] = (r[o] - prev)[ok]
        rows.append(base)
    def cat(k, sel): return np.concatenate([x[k] for x in rows if sel(x["date"])])
    per = {"IS": lambda d: d < IS_END, "OOS": lambda d: d >= IS_END}
    M = []
    for nm, sel in per.items():
        a = cat("act", sel); rw = np.sqrt(np.mean(a ** 2)); dr = cat("drift", sel)
        M.append(dict(per=nm, model="drift θ", n=len(a), RMSE_ratio=np.sqrt(np.mean((a - dr) ** 2)) / rw, hit=np.mean(np.sign(dr[a != 0]) == np.sign(a[a != 0])), spearman=pd.Series(dr).corr(pd.Series(a), "spearman")))
        for o in ORD:
            p = cat(o, sel); m = ~np.isnan(p)
            M.append(dict(per=nm, model=str(o), n=m.sum(), RMSE_ratio=np.sqrt(np.mean((a[m] - p[m]) ** 2)) / np.sqrt(np.mean(a[m] ** 2)),
                          hit=np.mean(np.sign(p[m & (a != 0)]) == np.sign(a[m & (a != 0)])), spearman=pd.Series(p[m]).corr(pd.Series(a[m]), "spearman")))
    M = pd.DataFrame(M); print(M.round(4).to_string(index=False))
    # 交易
    cost = lambda prev: 4 * (OCOMM / MULT + 1.0) + 2 * OTAX * prev   # 兩腿 × 來回；ATM straddle 單腿 < 500 → tick 1
    T = []; daily = {}
    for o, c in itertools.product(ORD + ["drift"], (1, 2, 5, 10)):
        s = pd.Series({x["date"]: (lambda p, a, pv: np.nansum(np.where(p > c, 1, np.where(p < -c, -1, 0)) * a - np.where(np.abs(p) > c, cost(pv), 0)))(x[o], x["act"], x["prev"]) * MULT for x in rows})
        nt = pd.Series({x["date"]: int(np.nansum(np.abs(x[o]) > c)) for x in rows})
        daily[(str(o), c)] = s
        r = dict(model=str(o), c=c)
        for nm, sel in per.items():
            m = s.index.map(sel).astype(bool); ss = s[m]
            r.update({f"{nm}_trades": int(nt[m].sum()), f"{nm}_net": ss.sum(), f"{nm}_Sharpe": sharpe(ss, 126), f"{nm}_gross_per_trade": np.nan})
        T.append(r)
    T = pd.DataFrame(T); print(T.round(3).to_string(index=False))
    # 毛利（不含成本）per trade，檢查是否有任何可交易 edge
    for o in ORD + ["drift"]:
        for nm, sel in per.items():
            p = cat(o, sel); a = cat("act", sel); m = np.abs(p) > 1
            print(f"毛利/筆 {str(o):10s} {nm}: |pred|>1 筆數 {m.sum():6d} 平均毛利 {np.nanmean(np.sign(p[m]) * a[m]):+.3f} 點（成本約 {4*(OCOMM/MULT+1):.1f} 點+稅）")
    Tq = T[T.IS_trades >= 50]; best = Tq.loc[Tq.IS_Sharpe.idxmax()]; print("best IS", best.to_dict()); print("Spearman IS/OOS", round(T.IS_Sharpe.rank().corr(T.OOS_Sharpe.rank()), 3))
    # 隨機：同頻率隨機方向
    rng = np.random.default_rng(3); a = cat("act", per["OOS"]); pv = cat("prev", per["OOS"]); q = best.OOS_trades / len(a)
    rnd = [np.sum((rng.random(len(a)) < q) * (rng.choice([-1, 1], len(a)) * a - cost(pv))) * MULT for _ in range(200)]
    print(f"隨機 OOS 中位數 {np.median(rnd):.0f}，最佳 OOS {best.OOS_net:.0f}，百分位 {(np.array(rnd) < best.OOS_net).mean():.2f}")
    wf = []
    for Y in range(2018, 2027):
        sc = {k: (sharpe(w, 126) if (w != 0).sum() >= 20 else -9) for k, v in daily.items() for w in [v[(v.index.year < Y) & (v.index.year >= Y - 3)]]}
        k = max(sc, key=sc.get); v = daily[k]; wf.append(v[v.index.year == Y]); print("WF", Y, k, round(sc[k], 2), round(v[v.index.year == Y].sum()))
    wf = pd.concat(wf); wo = wf[wf.index >= IS_END]; print(f"WF 2022+ net {wo.sum():.0f} Sharpe {sharpe(wo,126):.2f}")
    M.to_csv(RES / "rqo_73_forecast.csv", index=False); T.to_csv(RES / "rqo_73_trade.csv", index=False)
    plt = plt_style(); fig, axs = plt.subplots(2, 1, figsize=(12, 9), facecolor=BG)
    ax = axs[0]; sty(ax); ex = rows[-1]; x = [r for r in R if r["date"] == str(ex["date"].date())][0]
    ax.plot(x["y"], color=GOLD, label="ATM straddle（最小 ±10 檔）"); ax.plot(x[(1, 1, 1)] if not np.isnan(x[(1, 1, 1)]).all() else x[(1, 1, 0)], color=BLUE, lw=.8, label="ARIMA 一步預測")
    ax.set_title(f"{ex['date'].date()} 1 分鐘 straddle 權利金（10:15 起）"); ax.legend(facecolor=BG, labelcolor=WH, edgecolor=GR)
    ax = axs[1]; sty(ax); bk = (best.model, best.c); s = daily[bk]
    ax.plot(s.cumsum() / 1e4, color=GOLD, lw=2, label=f"IS 最佳 {bk}"); ax.plot(wf.cumsum() / 1e4, color=BLUE, label="walk-forward"); ax.axvline(IS_END, color="white", ls="--")
    ax.set_title(f"ARIMA 預測交易 straddle（成交價、含成本）｜OOS RMSE 比 {M[(M.per=='OOS')].RMSE_ratio.min():.3f}"); ax.set_ylabel("累積淨損益（萬元）"); ax.legend(facecolor=BG, labelcolor=WH, edgecolor=GR)
    foot(fig); fig.tight_layout(rect=[0, .03, 1, 1]); fig.savefig(RES / "rqo_73_straddle_arima.png", dpi=110, facecolor=BG)


if __name__ == "__main__":
    main()
