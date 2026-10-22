"""12 台指期 15 分 K：ORB 與唐奇安（海龜式）突破，加上 ATR 停損與移動停利，並做參數網格與樣本內外驗證

執行：
  python examples/12_atr_trailing_15m.py data/TXF_1c_1m_both_2017-06-01_2026-10-22.parquet \
      --data-dir data --results-dir results

流程：
1. 把 07 下載的 1 分 K 依「場次」合成 15 分 K（K 線以開始時間標示）。日盤 08:45–13:45、夜盤 15:00–05:00
   各自切，不跨場次組 K；13:45 那根收盤集合競價併入 13:30 那根。存成 data/TXF_1c_15m.parquet。
2. 策略（一次最多 1 口）：
   - ORB：開盤區間＝場次第一根 15 分 K。之後突破上緣做多、跌破下緣做空（停損單：當根開盤就跳過區間則以開盤價成交），
     每場最多一筆，場次最後一根收盤平倉。變體：夜盤開盤區間（--orb-session afterhours）。
   - 唐奇安：突破前 N 根最高／最低進場（停損單），出場通道為前 M 根反向極值。跨場次持倉。
     使用倒推調整價（換月以「新合約第一根開盤 − 舊合約最後一根收盤」近似價差），避免換月跳空假訊號；
     期交稅以原始價格計算。
3. 停損／停利（ATR 用 Wilder 平滑，取「進場前一根」的值，進場後固定）：
   - 初始停損：進場價 ∓ k_sl × ATR。
   - 移動停利：多單＝進場後最高價 − k_tr × ATR（空單對稱），且要先有 ≥ activation × ATR 的浮盈才啟動（0＝立即）。
     最高價只用「前一根為止」更新，避免偷看同根 K 的先後。停損價只會往有利方向移動。
   - 盤中觸價以停損價成交；若開盤跳空越過停損價，以開盤價成交。同一根同時觸發進場與停損，當作停損（保守）。
4. 成本：每點 200 元、手續費每邊 50 元、期交稅每邊十萬分之二、滑價每邊 1 點。
5. 樣本內 2017-06 ~ 2022-12 選參數，樣本外 2023-01 ~ 迄今只驗證。選參以「鄰域平均」的 淨利／最大回撤 為準
   （偏好高原而非尖峰），並要求最少交易筆數與獲利因子 > 1。

※ 純技術範例，非投資建議。
"""

import argparse
import itertools
import math
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

POINT_VALUE = 200
TAX_RATE = 2 / 100_000
COMMISSION = 50.0
SLIPPAGE = 1.0
IS_END = pd.Timestamp("2023-01-01", tz="Asia/Taipei")

ATR_NS = (10, 14, 20)
K_SL = (1.0, 1.5, 2.0, 2.5, 3.0)
K_TR = (1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0)
ACTS = (0.0, 1.0, 2.0)
DC = ((20, 10), (20, 20), (40, 10), (40, 20), (55, 10), (55, 20))


# ---------------- 1. 15 分 K ----------------

def make_15m(path):
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"]).dt.tz_convert("Asia/Taipei")
    df["session"] = df["session"].astype(str)
    df["contractMonth"] = df["contractMonth"].astype(str)
    ts = df["date"]
    night = df["session"] == "afterhours"
    # 場次代號：日盤＝當天；夜盤＝開盤那天（凌晨的 K 歸前一天晚上）
    sess_day = ts.dt.normalize().where(~(night & (ts.dt.hour < 15)), ts.dt.normalize() - pd.Timedelta(days=1))
    df["sid"] = df["session"].str[0] + sess_day.dt.strftime("%Y%m%d")
    b = ts.dt.floor("15min")
    b = b.where(~((~night) & (b.dt.hour == 13) & (b.dt.minute == 45)), b - pd.Timedelta(minutes=15))
    df["bar"] = b
    g = df.sort_values("date").groupby(["sid", "bar"], sort=True)
    out = g.agg(session=("session", "first"), open=("open", "first"), high=("high", "max"),
                low=("low", "min"), close=("close", "last"), volume=("volume", "sum"),
                contractMonth=("contractMonth", "last")).reset_index()
    out = out.rename(columns={"bar": "date"}).sort_values("date").reset_index(drop=True)
    for c in ("open", "high", "low", "close"):
        out[c] = out[c].astype("float64")
    return out


def back_adjust(df):
    cm = df["contractMonth"]
    roll = cm.ne(cm.shift()) & cm.shift().notna()
    gap = (df["open"] - df["close"].shift()).where(roll, 0.0)
    adj = gap[::-1].cumsum()[::-1].shift(-1, fill_value=0.0)
    return adj.to_numpy()


def wilder_atr(h, l, c, n):
    pc = np.r_[np.nan, c[:-1]]
    tr = np.nanmax(np.vstack([h - l, np.abs(h - pc), np.abs(l - pc)]), axis=0)
    return pd.Series(tr).ewm(alpha=1 / n, adjust=False, min_periods=n).mean().to_numpy()


# ---------------- 2. 回測核心（純 Python 迴圈，最快也最不易偷看） ----------------

def run(D, strat, atr_n, k_sl, k_tr, act, entry_n=20, exit_m=10, use_stops=True):
    """回傳交易清單 [(exit_ts_index, net_twd, points)]。

    k_sl/k_tr 為 None 代表不用該停損（基準）。
    """
    o, h, l, c, adj, sid, first, last, ts = (D["o"], D["h"], D["l"], D["c"], D["adj"], D["sid"],
                                             D["first"], D["last"], D["ts"])
    atr = D["atr"][atr_n]
    n = len(o)
    trades = []
    pos = 0
    ep = stop = best = a0 = 0.0
    active = False
    if strat == "dc":
        hi_e, lo_e = D["hiN"][entry_n], D["loN"][entry_n]
        hi_x, lo_x = D["hiN"][exit_m], D["loN"][exit_m]
    rng_hi = rng_lo = None
    traded = False

    def close_pos(i, px):
        raw_ep, raw_xp = ep - adj_e, px - adj[i]
        pts = (px - ep) * pos - 2 * SLIPPAGE  # 滑價每邊 1 點，直接扣在點數
        net = pts * POINT_VALUE - (COMMISSION + raw_ep * POINT_VALUE * TAX_RATE) \
            - (COMMISSION + raw_xp * POINT_VALUE * TAX_RATE)
        trades.append((i, net, pts))

    adj_e = 0.0
    for i in range(n):
        oi, hi, li, ci = o[i], h[i], l[i], c[i]
        if strat == "orb":
            if first[i]:
                rng_hi, rng_lo, traded = hi, li, False
                continue
            if rng_hi is None:
                continue
        # ---- 有部位：先檢查出場（停損價用前一根為止的資訊） ----
        if pos:
            lvl = None
            if strat == "dc":
                lvl = lo_x[i] if pos > 0 else hi_x[i]
            if stop:  # 0＝沒有停損
                if lvl is None or (pos > 0 and stop > lvl) or (pos < 0 and stop < lvl):
                    lvl = stop
            if lvl is not None and lvl == lvl:
                if pos > 0 and li <= lvl:
                    close_pos(i, min(oi, lvl)); pos = 0
                elif pos < 0 and hi >= lvl:
                    close_pos(i, max(oi, lvl)); pos = 0
            if pos and strat == "orb" and last[i]:
                close_pos(i, ci); pos = 0
            if pos:
                # 收盤後更新最佳價與移動停利（下一根才生效）
                if use_stops and k_tr is not None:
                    best = max(best, hi) if pos > 0 else min(best, li)
                    if not active and (best - ep) * pos >= act * a0:
                        active = True
                    if active:
                        t = best - pos * k_tr * a0
                        if (pos > 0 and t > stop) or (pos < 0 and t < stop):
                            stop = t
                continue
            if strat == "orb":
                continue  # 每場最多一筆
        # ---- 空手：找進場 ----
        if strat == "orb":
            if traded or last[i]:
                continue
            up, dn = rng_hi, rng_lo
        else:
            up, dn = hi_e[i], lo_e[i]
            if up != up:
                continue
        a = atr[i - 1] if i else float("nan")
        if a != a:
            continue
        side = 0
        if hi > up:
            side, px = 1, max(oi, up) + 0.0
        elif li < dn:
            side, px = -1, min(oi, dn) + 0.0
        if not side:
            continue
        if strat == "orb":
            traded = True
        pos, ep, a0, adj_e, best, active = side, px, a, adj[i], px, False
        if use_stops and k_sl is not None:
            stop = ep - side * k_sl * a
        elif strat == "orb" and not use_stops:
            stop = dn if side > 0 else up  # 基準：原 08 的區間另一側停損
        else:
            stop = 0.0
        # 同根觸停：保守當作停損
        if stop and ((side > 0 and li <= stop) or (side < 0 and hi >= stop)):
            close_pos(i, stop); pos = 0
        elif strat == "orb" and last[i]:
            close_pos(i, ci); pos = 0
    if pos:
        close_pos(n - 1, c[n - 1])
    return trades


# ---------------- 3. 統計 ----------------

def metrics(trades, D, lo, hi):
    ts = D["ts"]
    sel = [(ts[i], net, p) for i, net, p in trades if lo <= ts[i] < hi]
    if not sel:
        return dict(trades=0, win=0, net=0, pts=0, mdd=0, pf=0, sharpe=0, rdd=0)
    nets = np.array([s[1] for s in sel])
    eq = np.cumsum(nets)
    mdd = float((eq - np.maximum.accumulate(np.r_[0, eq])[1:]).min())
    mdd = min(mdd, 0.0)
    gw, gl = nets[nets > 0].sum(), -nets[nets < 0].sum()
    days = pd.Series(nets, index=pd.DatetimeIndex([s[0] for s in sel]).normalize()).groupby(level=0).sum()
    alld = D["days"][(D["days"] >= lo) & (D["days"] < hi)]
    days = days.reindex(alld, fill_value=0.0)
    sh = float(days.mean() / days.std() * math.sqrt(252)) if days.std() > 0 else 0.0
    net = float(nets.sum())
    return dict(trades=len(nets), win=round(float((nets > 0).mean() * 100), 1), net=round(net),
                pts=round(float(sum(s[2] for s in sel)), 1), mdd=round(mdd), pf=round(float(gw / gl), 3) if gl else 99.0,
                sharpe=round(sh, 3), rdd=round(net / -mdd, 3) if mdd < 0 else 99.0)


_D = None


def _init(D):
    global _D
    _D = D


def _job(args):
    strat, p = args
    tr = run(_D, strat, p["atr"], p["k_sl"], p["k_tr"], p["act"], p.get("en", 20), p.get("ex", 10))
    return p, metrics(tr, _D, _D["start"], IS_END), metrics(tr, _D, IS_END, _D["end"])


def prep(df, strat, session=None):
    d = df if session is None else df[df["session"] == session].reset_index(drop=True)
    adj = back_adjust(d) if strat == "dc" else np.zeros(len(d))
    o, h, l, c = (d[k].to_numpy() + adj for k in ("open", "high", "low", "close"))
    sid = d["sid"].to_numpy()
    first = np.r_[True, sid[1:] != sid[:-1]]
    last = np.r_[sid[1:] != sid[:-1], True]
    D = dict(o=o.tolist(), h=h.tolist(), l=l.tolist(), c=c.tolist(), adj=adj.tolist(), sid=sid,
             first=first.tolist(), last=last.tolist(), ts=list(d["date"]),
             atr={n: wilder_atr(h, l, c, n).tolist() for n in ATR_NS})
    D["hiN"] = {k: pd.Series(h).rolling(k).max().shift().tolist() for k in (10, 20, 40, 55)}
    D["loN"] = {k: pd.Series(l).rolling(k).min().shift().tolist() for k in (10, 20, 40, 55)}
    D["days"] = pd.DatetimeIndex(d["date"].dt.normalize().unique())
    D["start"], D["end"] = d["date"].iloc[0], d["date"].iloc[-1] + pd.Timedelta(days=1)
    return D


def neighbor_score(res, keys, grids):
    """每組參數的分數＝自己與網格上相鄰（各維 ±1 格）組合的 IS 淨利／回撤 平均。"""
    idx = {tuple(r[0][k] for k in keys): r for r in res}
    out = {}
    for key, r in idx.items():
        vals = []
        pos = [grids[j].index(key[j]) for j in range(len(keys))]
        for dlt in itertools.product((-1, 0, 1), repeat=len(keys)):
            nk = []
            ok = True
            for j, dd in enumerate(dlt):
                q = pos[j] + dd
                if not 0 <= q < len(grids[j]):
                    ok = False
                    break
                nk.append(grids[j][q])
            if ok and tuple(nk) in idx:
                m = idx[tuple(nk)][1]
                vals.append(m["rdd"] if m["net"] > 0 else m["net"] / 1e6)
        out[key] = float(np.mean(vals))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    args = ap.parse_args()
    R = Path(args.results_dir)
    R.mkdir(parents=True, exist_ok=True)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    p15 = Path(args.data_dir) / "TXF_1c_15m.parquet"
    df = make_15m(args.path)
    df.to_parquet(p15, compression="zstd", index=False)
    print("15 分 K：{} 根（日盤 {}、夜盤 {}），{} ~ {}，存到 {}（{:,} bytes）".format(
        len(df), (df.session == "regular").sum(), (df.session == "afterhours").sum(),
        df.date.iloc[0], df.date.iloc[-1], p15, p15.stat().st_size))

    configs = [("orb_day", "orb", "regular", 100), ("orb_night", "orb", "afterhours", 100), ("donchian", "dc", None, 60)]
    summary = []
    for name, strat, sess, min_tr in configs:
        D = prep(df, strat, sess)
        keys = ["atr", "k_sl", "k_tr", "act"] + (["en", "ex"] if strat == "dc" else [])
        grid = [dict(atr=a, k_sl=s, k_tr=t, act=v) for a, s, t, v in itertools.product(ATR_NS, K_SL, K_TR, ACTS)]
        if strat == "dc":
            grid = [dict(g, en=e, ex=x) for g in grid for e, x in DC]
        with ProcessPoolExecutor(args.workers, initializer=_init, initargs=(D,)) as ex:
            res = list(ex.map(_job, [(strat, g) for g in grid], chunksize=4))
        pd.DataFrame([dict(**p, **{"is_" + k: v for k, v in a.items()}, **{"oos_" + k: v for k, v in b.items()})
                      for p, a, b in res]).to_csv(R / "grid_{}.csv".format(name), index=False)
        # 選參：IS 交易數與 PF 門檻 → 鄰域平均 淨利/回撤 最高
        if strat == "dc":
            # 通道組合不是有序網格，對每組通道各自算鄰域
            ns = {}
            for e, x in DC:
                sub = [r for r in res if r[0]["en"] == e and r[0]["ex"] == x]
                for k, v in neighbor_score(sub, keys[:4], [ATR_NS, K_SL, K_TR, ACTS]).items():
                    ns[k + (e, x)] = v
        else:
            ns = neighbor_score(res, keys, [ATR_NS, K_SL, K_TR, ACTS])
        ok = [r for r in res if r[1]["trades"] >= min_tr and r[1]["pf"] > 1]
        if not ok:
            ok = res
        best = max(ok, key=lambda r: (ns[tuple(r[0][k] for k in keys)], r[1]["rdd"]))
        p = best[0]
        _init(D)
        # 基準（不加 ATR 停損／移動停利）
        if strat == "dc":
            base_tr = run(D, strat, p["atr"], None, None, 0, p["en"], p["ex"], use_stops=False)
        else:
            base_tr = run(D, strat, 14, None, None, 0, use_stops=False)
        tr = run(D, strat, p["atr"], p["k_sl"], p["k_tr"], p["act"], p.get("en", 20), p.get("ex", 10))
        for label, t in (("chosen", tr), ("baseline", base_tr)):
            for per, lo, hi in (("IS", D["start"], IS_END), ("OOS", IS_END, D["end"])):
                summary.append(dict(strategy=name, variant=label, period=per,
                                    params=str(p) if label == "chosen" else "no ATR stop/trail",
                                    **metrics(t, D, lo, hi)))
        # 逐年
        yr = pd.Series([n for _, n, _ in tr], index=[D["ts"][i].year for i, _, _ in tr]).groupby(level=0).agg(["count", "sum"])
        yb = pd.Series([n for _, n, _ in base_tr], index=[D["ts"][i].year for i, _, _ in base_tr]).groupby(level=0).sum()
        yr["baseline_sum"] = yb
        yr.round(0).to_csv(R / "yearly_{}.csv".format(name))
        print("\n[{}] 選定參數 {}（鄰域分數 {:.3f}）\n逐年（元）：\n{}".format(name, p, ns[tuple(p[k] for k in keys)], yr.round(0)))
        # 權益曲線
        plt.figure(figsize=(11, 5))
        for label, t in (("ATR stop + trail " + str(p), tr), ("baseline", base_tr)):
            plt.plot([D["ts"][i] for i, _, _ in t], np.cumsum([n for _, n, _ in t]) / 1e3, label=label)
        plt.axvline(IS_END, color="k", ls=":", label="IS | OOS")
        plt.title("{} 15m TXF, net of costs (k TWD)".format(name)); plt.legend(fontsize=7); plt.grid()
        plt.savefig(R / "equity_15m_{}.png".format(name), dpi=110); plt.close()
        # 熱圖：固定選定的 ATR 期數（與通道），k_sl × k_tr，每個 activation 一張子圖，色彩＝IS 淨利/回撤
        fig, axs = plt.subplots(1, 3, figsize=(15, 4.5))
        for ax, a in zip(axs, ACTS):
            M = np.full((len(K_SL), len(K_TR)), np.nan)
            for q, m, _ in res:
                if q["atr"] == p["atr"] and q["act"] == a and q.get("en") == p.get("en") and q.get("ex") == p.get("ex"):
                    M[K_SL.index(q["k_sl"]), K_TR.index(q["k_tr"])] = m["rdd"] if m["net"] > 0 else m["net"] / 1e6
            im = ax.imshow(M, origin="lower", cmap="RdYlGn", aspect="auto")
            ax.set_xticks(range(len(K_TR)), K_TR); ax.set_yticks(range(len(K_SL)), K_SL)
            ax.set_xlabel("k_tr"); ax.set_ylabel("k_sl"); ax.set_title("act={} (ATR{})".format(a, p["atr"]))
            for y in range(M.shape[0]):
                for x in range(M.shape[1]):
                    ax.text(x, y, "{:.2f}".format(M[y, x]), ha="center", va="center", fontsize=6)
            fig.colorbar(im, ax=ax)
        fig.suptitle("{} IS net/maxDD (negative net shown as net/1e6)".format(name))
        fig.tight_layout(); fig.savefig(R / "heatmap_15m_{}.png".format(name), dpi=110); plt.close(fig)

    s = pd.DataFrame(summary)
    s.to_csv(R / "summary_15m.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 30, "display.max_colwidth", 80):
        print("\n", s.to_string(index=False))


if __name__ == "__main__":
    main()
