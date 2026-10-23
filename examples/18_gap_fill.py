"""18 開盤跳空回補（Gap Filling）：台指期日盤開盤 vs 前一日盤收盤／前一夜盤收盤

執行：
  python examples/18_gap_fill.py --five data/TXF_1c_5m_sessions.parquet --results-dir results
  （5 分 K 由 17 依場次合成；沒有的話先跑 17，或用 13 的 resample_sessions(…, 5)）

原始策略（TradingView alexgrover「Gap Filling Strategy」，2020；頁面說明，Pine 原始碼頁面需 JS 無法讀取，以下為依說明推定）：
- 新交易時段開盤價高於前一根 K 的高點＝向上跳空、低於前一根低點＝向下跳空（作者以 15 分 K 測試），
  且跳空後那根 K 沒有回補；invert=false（預設）時「逆勢」：向上跳空做空、向下跳空做多。
- 目標＝前收盤（缺口被回補即出場），無停損；Close when：新時段／新跳空／反向部位（推定預設＝新時段）。
台指期移植：
- 跳空＝日盤第一根開盤（08:45）− 參考價；參考價 ref=day：前一日盤收盤（13:45）；ref=night：前一夜盤收盤（05:00）。
- 訊號與點數用倒推調整價（日夜盤 5 分 K 序列一起調整），期交稅用原始價；成本同 12–17
  （每點 200、手續費單邊 50、期交稅十萬分之二、滑價單邊 1 點）。
- 進場：第 d 根 5 分 K 開盤（d≥1，＝看到前 d 根收盤後下一根開盤），且 0..d−1 根都沒碰到參考價（尚未回補）。
- 出場：觸及參考價（目標，限價成交；開盤已越過就以開盤價）、或停損（進場價 ± s×ATR；跳空越過以開盤價）、
  或日盤最後一根收盤（時間出場，不留倉）。同一根同時碰到目標與停損，保守假設先停損。
- ATR：日盤日 K 的 ATR(14)，只用前一日以前（不偷看）。
- 網格：最小跳空 k×ATR（k=0,0.1,0.2,0.3,0.5,0.75,1.0）× 停損 s（0.25,0.5,0.75,1,1.5,無）× 進場延遲 d（1,2,3,6 根 5 分）。
  多空／只做多（向下跳空）／只做空（向上跳空）三種，ref 兩種 → 每 ref 每模式 168 組。
- 選參：樣本內交易數 ≥ 50 且 PF > 1，(k, s, d) 三維 3×3×3 鄰域「淨利／最大回撤」平均最高（高原）。
- 對照：原始規則預設（前一 15 分 K 高低點外跳空、第一根 15 分 K 未回補、09:00 進場、無停損、收盤出場）；
  天真「永遠逆勢」（每天依跳空方向反做，d=1，無停損，目標前收，收盤出場）。
- 績效以「日」為單位（每天至多一筆，日內不留倉）：淨利、最大回撤（日權益）、PF、夏普（×√252）。

※ 純技術範例，非投資建議。
"""

import argparse
import importlib.util
import itertools
import math
from pathlib import Path

import numpy as np
import pandas as pd

_spec = importlib.util.spec_from_file_location("st13", Path(__file__).with_name("13_supertrend_grid.py"))
st13 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(st13)

POINT_VALUE, COMMISSION, SLIPPAGE, TAX_RATE = st13.POINT_VALUE, st13.COMMISSION, st13.SLIPPAGE, st13.TAX_RATE
LO, MID, HI = (pd.Timestamp(x) for x in st13.SPLITS["intraday"])
EX = pd.Timestamp("2025-01-01")
PERIODS_EVAL = {"IS": (LO, MID), "OOS": (MID, HI), "OOS_ex2526": (MID, EX)}
KS = (0.0, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0)
SS = (0.25, 0.5, 0.75, 1.0, 1.5, None)
DS = (1, 2, 3, 6)
MODES = ("longshort", "longonly", "shortonly")
NB = 60  # 日盤 08:45–13:45 共 60 根 5 分 K


def build_days(path):
    f = pd.read_parquet(path)
    f["date"] = pd.to_datetime(f["date"]).dt.tz_convert(st13.TZ) if pd.to_datetime(f["date"]).dt.tz is not None \
        else pd.to_datetime(f["date"]).dt.tz_localize(st13.TZ)
    f = f.sort_values("date").reset_index(drop=True)
    f["contractMonth"] = f["contractMonth"].astype(str); f["session"] = f["session"].astype(str)
    adj = st13.back_adjust(f)
    for c in ("open", "high", "low", "close"):
        f[c + "_a"] = f[c].astype(float) + adj
    f["adj"] = adj
    reg = (f["session"] == "regular").to_numpy()
    f["pos"] = np.arange(len(f))
    days = []
    for sid, g in f[reg].groupby("sid", sort=True):
        i0 = int(g["pos"].iloc[0])
        prev = f.iloc[:i0]
        pr = prev[prev["session"] == "regular"]
        if len(pr) < 3:
            continue
        night_ref = prev["close_a"].iloc[-1] if prev["session"].iloc[-1] != "regular" else np.nan
        last15 = pr.iloc[-3:]
        n = len(g)
        pad = lambda a: np.r_[a, np.repeat(a[-1], NB - n)] if n < NB else a[:NB]
        days.append(dict(day=pd.Timestamp(sid[1:]), n=min(n, NB), ref_day=pr["close_a"].iloc[-1], ref_night=night_ref,
                         ph15=last15["high_a"].max(), pl15=last15["low_a"].min(),
                         O=pad(g["open_a"].to_numpy()), H=pad(g["high_a"].to_numpy()), L=pad(g["low_a"].to_numpy()),
                         C=pad(g["close_a"].to_numpy()), A=pad(g["adj"].to_numpy())))
    return days, f


def daily_atr(days, period=14):
    H = np.array([d["H"][:d["n"]].max() for d in days]); L = np.array([d["L"][:d["n"]].min() for d in days])
    C = np.array([d["C"][d["n"] - 1] for d in days]); pc = np.r_[np.nan, C[:-1]]
    tr = np.nanmax(np.vstack([H - L, np.abs(H - pc), np.abs(L - pc)]), axis=0)
    return pd.Series(tr).rolling(period).mean().shift(1).to_numpy()  # 只用前一日以前


def run(M, ref, side_mask, d, stop_pts):
    """向量化：每天至多一筆。side_mask：每天 −1（做空）/+1（做多）/0（不做）。回傳每日淨損益（元）與每日方向。"""
    O, H, L, C, A, n = M["O"], M["H"], M["L"], M["C"], M["A"], M["n"]
    nd = len(n)
    side = side_mask.copy()
    # 進場前（0..d−1）已回補 → 不做
    pre_hi, pre_lo = H[:, :d].max(axis=1), L[:, :d].min(axis=1)
    filled = np.where(side == -1, pre_lo <= ref, np.where(side == 1, pre_hi >= ref, False))
    side[filled | (n <= d) | np.isnan(ref)] = 0
    ent = O[:, d]
    side[(side == -1) & (ent <= ref)] = 0
    side[(side == 1) & (ent >= ref)] = 0
    bars = np.arange(NB)[None, :]
    live = (bars >= d) & (bars < n[:, None])
    tgt_hit = np.where(side[:, None] == -1, L <= ref[:, None], H >= ref[:, None]) & live
    if stop_pts is None:
        stp_hit = np.zeros_like(tgt_hit); sp = np.full(nd, np.nan)
    else:
        sp = ent - side * stop_pts
        stp_hit = np.where(side[:, None] == -1, H >= sp[:, None], L <= sp[:, None]) & live
    big = NB + 1
    it = np.where(tgt_hit.any(1), tgt_hit.argmax(1), big)
    is_ = np.where(stp_hit.any(1), stp_hit.argmax(1), big)
    last = n - 1
    rows = np.arange(nd)
    exit_px = C[rows, last].copy(); exit_i = last.copy()
    st_first = is_ <= it
    use_s = st_first & (is_ < big); use_t = (~st_first) & (it < big)
    oi_s = O[rows, np.minimum(is_, NB - 1)]; oi_t = O[rows, np.minimum(it, NB - 1)]
    # 停損：開盤已越過停損價就用開盤
    sfill = np.where(side == -1, np.maximum(sp, oi_s), np.minimum(sp, oi_s))
    tfill = np.where(side == -1, np.minimum(ref, oi_t), np.maximum(ref, oi_t))
    # 進場那根的開盤就是進場價，沒有「越過」問題
    sfill = np.where(is_ == d, sp, sfill); tfill = np.where(it == d, ref, tfill)
    exit_px = np.where(use_s, sfill, np.where(use_t, tfill, exit_px))
    exit_i = np.where(use_s, is_, np.where(use_t, it, exit_i))
    raw_e = ent - A[:, d]; raw_x = exit_px - A[rows, np.minimum(exit_i, NB - 1)]
    cost = 2 * (COMMISSION + SLIPPAGE * POINT_VALUE) + (raw_e + raw_x) * POINT_VALUE * TAX_RATE
    pnl = np.where(side != 0, side * (exit_px - ent) * POINT_VALUE - cost, 0.0)
    return pnl, side, use_t


def metrics(pnl, side, dates, lo, hi):
    k = (dates >= lo) & (dates < hi)
    p, s = pnl[k], side[k]
    eq = np.cumsum(p)
    mdd = float(min(np.min(eq - np.maximum.accumulate(np.r_[0.0, eq])[1:]), 0.0)) if len(p) else 0.0
    tr = p[s != 0]
    gw, gl = tr[tr > 0].sum(), -tr[tr < 0].sum()
    sd = p.std(ddof=1)
    net = float(eq[-1]) if len(eq) else 0.0
    return dict(trades=int((s != 0).sum()), win=round(float((tr > 0).mean() * 100), 1) if len(tr) else 0.0,
                net=round(net), mdd=round(mdd), pf=round(float(gw / gl), 3) if gl else 0.0,
                sharpe=round(float(p.mean() / sd * math.sqrt(252)), 3) if sd > 0 else 0.0,
                rdd=round(net / -mdd, 3) if mdd < 0 else 0.0, skew=float(pd.Series(p).skew()),
                kurt=float(pd.Series(p).kurt() + 3), ndays=int(len(p)))


def score(m):
    return m["rdd"] if m["net"] > 0 else m["net"] / 1e7


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--five", required=True)
    ap.add_argument("--results-dir", default="results")
    args = ap.parse_args()
    R = Path(args.results_dir); R.mkdir(parents=True, exist_ok=True)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    days, _ = build_days(args.five)
    M = {k: np.array([d[k] for d in days]) for k in ("O", "H", "L", "C", "A", "n", "ref_day", "ref_night", "ph15", "pl15")}
    M["n"] = M["n"].astype(int)
    dates = np.array([d["day"] for d in days], dtype="datetime64[ns]")
    atr = daily_atr(days)
    ok = ~np.isnan(atr)
    open0 = M["O"][:, 0]
    print("日盤天數 {}（{} ~ {}）".format(len(days), dates[0], dates[-1]))

    # ---- 回補率統計 ----
    fill_rows = []
    for ref_name in ("day", "night"):
        ref = M["ref_" + ref_name]
        gap = open0 - ref
        hi_all = np.array([M["H"][i, :M["n"][i]].max() for i in range(len(days))])
        lo_all = np.array([M["L"][i, :M["n"][i]].min() for i in range(len(days))])
        filled = np.where(gap > 0, lo_all <= ref, hi_all >= ref)
        g_atr = np.abs(gap) / atr
        for per, (a, b) in PERIODS_EVAL.items():
            k = (dates >= np.datetime64(a)) & (dates < np.datetime64(b)) & ok & ~np.isnan(ref) & (gap != 0)
            for lo_, hi_ in ((0, 0.1), (0.1, 0.2), (0.2, 0.3), (0.3, 0.5), (0.5, 0.75), (0.75, 1.0), (1.0, 99)):
                kk = k & (g_atr >= lo_) & (g_atr < hi_)
                fill_rows.append(dict(ref=ref_name, sample=per, gap_atr="{}–{}".format(lo_, hi_ if hi_ < 99 else "∞"),
                                      days=int(kk.sum()), fill_pct=round(100 * float(filled[kk].mean()), 1) if kk.any() else np.nan,
                                      up_fill_pct=round(100 * float(filled[kk & (gap > 0)].mean()), 1) if (kk & (gap > 0)).any() else np.nan,
                                      dn_fill_pct=round(100 * float(filled[kk & (gap < 0)].mean()), 1) if (kk & (gap < 0)).any() else np.nan,
                                      med_abs_gap_pts=round(float(np.median(np.abs(gap[kk]))), 1) if kk.any() else np.nan))
            fill_rows.append(dict(ref=ref_name, sample=per, gap_atr="all", days=int(k.sum()),
                                  fill_pct=round(100 * float(filled[k].mean()), 1),
                                  up_fill_pct=round(100 * float(filled[k & (gap > 0)].mean()), 1),
                                  dn_fill_pct=round(100 * float(filled[k & (gap < 0)].mean()), 1),
                                  med_abs_gap_pts=round(float(np.median(np.abs(gap[k]))), 1)))
    fills = pd.DataFrame(fill_rows); fills.to_csv(R / "gap_fill_rates.csv", index=False)

    # ---- 網格 ----
    def sides(ref, k_atr, mode):
        gap = open0 - ref
        s = np.where(gap > k_atr * atr, -1, np.where(gap < -k_atr * atr, 1, 0))
        s[~ok | (gap == 0)] = 0
        if mode == "longonly":
            s[s == -1] = 0
        elif mode == "shortonly":
            s[s == 1] = 0
        return s

    grid = []
    for ref_name, mode, kk, ss, d in itertools.product(("day", "night"), MODES, KS, SS, DS):
        ref = M["ref_" + ref_name]
        pnl, side, ht = run(M, ref, sides(ref, kk, mode), d, None if ss is None else ss * atr)
        r = dict(ref=ref_name, mode=mode, k=kk, s="none" if ss is None else ss, d=d,
                 tgt_hit_pct=round(100 * float(ht[side != 0].mean()), 1) if (side != 0).any() else 0.0)
        for per, (a, b) in PERIODS_EVAL.items():
            for c, v in metrics(pnl, side, dates, np.datetime64(a), np.datetime64(b)).items():
                r[per + "_" + c] = v
        grid.append(r)
    G = pd.DataFrame(grid); G.to_csv(R / "gap_grid_all.csv", index=False)

    def sim_named(lab, ref_name, mode, side_override=None, kk=0.0, ss=None, d=1):
        ref = M["ref_" + ref_name]
        sm = sides(ref, kk, mode) if side_override is None else side_override
        pnl, side, ht = run(M, ref, sm, d, None if ss is None else ss * atr)
        return lab, pnl, side, ht

    # 原始規則預設：開盤在前一根 15 分 K 高低點之外、第一根 15 分（3 根 5 分）未回補、09:00 進場、無停損
    orig_side = np.where(open0 > M["ph15"], -1, np.where(open0 < M["pl15"], 1, 0))
    orig_side[~ok] = 0

    summ, ovf, legs = [], [], []
    shp = (len(KS), len(SS), len(DS))
    best_eq = {}
    for ref_name in ("day", "night"):
        for mode in MODES:
            sub = G[(G.ref == ref_name) & (G["mode"] == mode)].reset_index(drop=True)
            Z = np.full(shp, np.nan)
            ii = [sub.k.map(KS.index), sub.s.map(lambda x: SS.index(None if x == "none" else x)), sub.d.map(DS.index)]
            sc = np.where(sub.IS_net > 0, sub.IS_rdd, sub.IS_net / 1e7)
            Z[tuple(np.array(i) for i in ii)] = sc
            P = np.pad(Z, 1, constant_values=np.nan); acc = np.zeros(shp); cnt = np.zeros(shp)
            for o in np.ndindex(3, 3, 3):
                w = P[o[0]:o[0] + shp[0], o[1]:o[1] + shp[1], o[2]:o[2] + shp[2]]
                okw = ~np.isnan(w); acc += np.where(okw, w, 0); cnt += okw
            plat = (acc / cnt)[tuple(np.array(i) for i in ii)]
            elig = sub.index[(sub.IS_trades >= 50) & (sub.IS_pf > 1)]
            elig = elig if len(elig) else sub.index
            pi = max(elig, key=lambda i: (plat[i], sub.loc[i, "IS_rdd"]))
            r = sub.loc[pi]
            ss = None if r.s == "none" else float(r.s)
            prm = "k={} ATR, stop={}, d={}".format(r.k, r.s, int(r.d))
            variants = [sim_named("plateau pick", ref_name, mode, kk=float(r.k), ss=ss, d=int(r.d)),
                        sim_named("naive always fade", ref_name, mode, kk=0.0, ss=None, d=1)]
            if ref_name == "day":
                os_ = orig_side.copy()
                if mode == "longonly":
                    os_[os_ == -1] = 0
                elif mode == "shortonly":
                    os_[os_ == 1] = 0
                variants.append(sim_named("original rule default", "day", mode, side_override=os_, d=3))
            for lab, pnl, side, ht in variants:
                if lab == "plateau pick" and mode == "longshort":
                    best_eq[ref_name] = (pnl, side, prm)
                for per, (a, b) in PERIODS_EVAL.items():
                    x = metrics(pnl, side, dates, np.datetime64(a), np.datetime64(b))
                    kk_ = (dates >= np.datetime64(a)) & (dates < np.datetime64(b)) & (side != 0)
                    row = dict(ref=ref_name, mode=mode, variant=lab, params=prm if lab == "plateau pick" else "-",
                               sample=per, **{c: x[c] for c in ("trades", "win", "net", "mdd", "pf", "sharpe")},
                               tgt_hit_pct=round(100 * float(ht[kk_].mean()), 1) if kk_.any() else 0.0)
                    summ.append(row)
                    if mode == "longshort" and lab == "plateau pick":
                        for leg, sv in (("long leg", 1), ("short leg", -1)):
                            p2 = np.where(side == sv, pnl, 0.0); s2 = np.where(side == sv, side, 0)
                            y = metrics(p2, s2, dates, np.datetime64(a), np.datetime64(b))
                            legs.append(dict(ref=ref_name, leg=leg, params=prm, sample=per,
                                             **{c: y[c] for c in ("trades", "win", "net", "mdd", "pf", "sharpe")}))
            isv, oosv = sub.IS_sharpe.to_numpy(), sub.OOS_sharpe.to_numpy()
            sr0 = st13.expected_max_sr(len(sub), float(np.var(isv, ddof=1)))
            ovf.append(dict(ref=ref_name, mode=mode, cells=len(sub), pick=prm,
                            spearman_sharpe=round(st13.spearman(isv, oosv), 3),
                            spearman_netmdd=round(st13.spearman(sc, np.where(sub.OOS_net > 0, sub.OOS_rdd, sub.OOS_net / 1e7)), 3),
                            pick_OOS_pctile=round(100 * float(np.mean(oosv <= r.OOS_sharpe)), 1),
                            share_IS_profitable=round(100 * float(np.mean(sub.IS_net > 0)), 1),
                            share_OOS_profitable=round(100 * float(np.mean(sub.OOS_net > 0)), 1),
                            sr0=round(sr0, 3), IS_sharpe_pick=r.IS_sharpe,
                            DSR_pick=round(st13.deflated_sharpe(r.IS_sharpe, r.IS_ndays, r.IS_skew, r.IS_kurt, sr0), 3)))
            # 熱圖（多空）：k × stop，d 固定在選定值
            if mode == "longshort":
                fig, axs = plt.subplots(1, 2, figsize=(14, 5.5))
                for ax, per in zip(axs, ("IS", "OOS")):
                    s2 = sub[sub.d == r.d]
                    Zh = np.full((len(KS), len(SS)), np.nan)
                    for _, x in s2.iterrows():
                        Zh[KS.index(x.k), SS.index(None if x.s == "none" else x.s)] = \
                            x[per + "_rdd"] if x[per + "_net"] > 0 else x[per + "_net"] / 1e7
                    lim = np.nanmax(np.abs(Zh)) or 1
                    im = ax.imshow(Zh, origin="lower", cmap="RdYlGn", aspect="auto", vmin=-lim, vmax=lim)
                    ax.set_xticks(range(len(SS)), [str(x) if x else "none" for x in SS]); ax.set_yticks(range(len(KS)), KS)
                    ax.set_xlabel("stop (× daily ATR)"); ax.set_ylabel("min gap (× daily ATR)")
                    for y in range(Zh.shape[0]):
                        for xx in range(Zh.shape[1]):
                            ax.text(xx, y, "{:.2f}".format(Zh[y, xx]), ha="center", va="center", fontsize=7)
                    ax.scatter(SS.index(ss), KS.index(r.k), marker="o", s=420, facecolors="none", edgecolors="purple", linewidths=2)
                    ax.set_title("{} net/maxDD (loss: net/1e7), entry delay d={}".format(per, int(r.d))); fig.colorbar(im, ax=ax)
                fig.suptitle("TXF gap fade vs prior {} close (5m, long+short); circle = chosen".format(ref_name))
                fig.tight_layout(); fig.savefig(R / "gap_heatmap_{}.png".format(ref_name), dpi=100); plt.close(fig)

    # 權益圖
    dts = pd.to_datetime(dates); k = dts >= LO
    plt.figure(figsize=(11, 5))
    for ref_name, (pnl, side, prm) in best_eq.items():
        plt.plot(dts[k], np.cumsum(pnl[k]) / 1e3, label="plateau, ref={} ({})".format(ref_name, prm))
    for lab, ref_name, sm, d in (("naive always fade, ref=day", "day", None, 1), ("naive always fade, ref=night", "night", None, 1),
                                 ("original rule default (ref=day)", "day", orig_side, 3)):
        _, pnl, side, _ = sim_named(lab, ref_name, "longshort", side_override=sm, d=d)
        plt.plot(dts[k], np.cumsum(pnl[k]) / 1e3, label=lab, alpha=0.8)
    plt.axvline(MID, color="k", ls=":", label="IS | OOS"); plt.axvline(EX, color="gray", ls="--", label="2025-01")
    plt.title("TXF day-session opening gap fade (5m bars), net of costs (k TWD)")
    plt.legend(fontsize=7); plt.grid(); plt.tight_layout()
    plt.savefig(R / "gap_equity.png", dpi=110); plt.close()

    pd.DataFrame(summ).to_csv(R / "gap_summary.csv", index=False)
    pd.DataFrame(legs).to_csv(R / "gap_legs.csv", index=False)
    pd.DataFrame(ovf).to_csv(R / "gap_overfit.csv", index=False)
    with pd.option_context("display.width", 300, "display.max_columns", 40, "display.max_rows", 500):
        print(fills.to_string(index=False))
        print(pd.DataFrame(summ).to_string(index=False))
        print(pd.DataFrame(legs).to_string(index=False))
        print(pd.DataFrame(ovf).to_string(index=False))


if __name__ == "__main__":
    main()
