"""16 日 K SuperTrend 濾網（event 模式）＋多空「不同參數」的 60 分／15 分 SuperTrend

執行：
  python examples/16_supertrend_asym_long_short.py \
      --daily data/TXF_1c_D_regular_2010-01-01_2026-10-22.parquet \
      --minute data/TXF_1c_1m_both_2017-06-01_2026-10-22.parquet --results-dir results

沿用 13／15 的資料處理、SuperTrend（SMA ATR）、成本（每點 200、手續費單邊 50、期交稅十萬分之二（原始價）、
滑價單邊 1 點）、下一根開盤成交、樣本內 2017-06～2022-12、樣本外 2023～。
規則（15 的 event 模式，拆成多、空兩條腿）：
- 多方腿：空手時，分 K ST_long(pL, mL)「剛翻多」且日線濾網 fL 為多才進場；ST_long 翻空或 fL 轉空就出場。
- 空方腿：同上，用 ST_short(pS, mS) 與日線濾網 fS。
- fL = fS 時兩條腿互斥（濾網同時只有一個方向），合併損益＝兩腿相加，與單一帳戶完全相同；
  pL=pS、mL=mS 時就等於 15 的對稱 event 模式（程式會自我核對）。
- fL ≠ fS（日線也不對稱）時兩腿可能同時持有（一多一空），以「兩個子帳戶」相加計（各自付成本），並回報重疊時間比例。
試驗：分 K 多、空各 54 組（ATR 5–30 每 5 × 倍數 1.0–5.0 每 0.5）＝2,916 組；
  A「分 K 不對稱」：fL=fS，5 組濾網 → 14,580 組；B「分 K＋日線都不對稱」：25 組濾網 → 72,900 組。
選參：樣本內交易數 ≥ 50 且 PF > 1，在 (pL, mL, pS, mS) 四維 3×3×3×3 鄰域取「淨利／最大回撤」平均最高（高原），不取單點最佳。
最大回撤以逐根市值權益計（同 13）。

※ 純技術範例，非投資建議。
"""

import argparse
import importlib.util
import math
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
POINT_VALUE, COMMISSION, SLIPPAGE, TAX_RATE = st13.POINT_VALUE, st13.COMMISSION, st13.SLIPPAGE, st13.TAX_RATE
DAILY_FILTERS = st15.DAILY_FILTERS
IN_PERIODS, IN_MULTS = st15.IN_PERIODS, st15.IN_MULTS
CELLS = [(p, m) for p in IN_PERIODS for m in IN_MULTS]
LO, MID, HI = st15.LO, st15.MID, st15.HI
EX = pd.Timestamp("2025-01-01")  # 樣本外排除 2025–2026
PERIODS_EVAL = {"IS": (LO, MID), "OOS": (MID, HI), "OOS_ex2526": (MID, EX)}


def simulate_leg(D, trend, filt, side):
    """單邊 event：回傳 pnl（元/根）、交易 [(出場索引, 淨損益)]、部位陣列。"""
    o, c, adj = D["o"].tolist(), D["c"].tolist(), D["adj"].tolist()
    tr, fl = trend.tolist(), filt.tolist()
    n = len(o)
    pnl = np.zeros(n); posa = np.zeros(n, dtype=np.int8)
    trades = []
    pos, cur = 0, 0.0

    def side_cost(i, px):
        return COMMISSION + (px - adj[i]) * POINT_VALUE * TAX_RATE + SLIPPAGE * POINT_VALUE

    for i in range(2, n):
        d1, d2, f = tr[i - 1], tr[i - 2], fl[i]
        if pos:
            target = side if (d1 == side and f == side) else 0
        else:
            target = side if (d2 and d1 != d2 and d1 == side and f == side) else 0
        g = pos * (o[i] - c[i - 1]) * POINT_VALUE
        bar = g; cur += g
        if target != pos:
            if pos:
                k = side_cost(i, o[i]); bar -= k
                trades.append((i, cur - k)); cur = 0.0
            if target:
                k = side_cost(i, o[i]); bar -= k; cur = -k
            pos = target
        r = pos * (c[i] - o[i]) * POINT_VALUE
        bar += r; cur += r
        pnl[i] = bar; posa[i] = pos
    if pos:
        trades.append((n - 1, cur))
    return pnl, trades, posa


def leg_summary(D, sim):
    """每個評估區間：該腿逐根損益（區間內）、交易數、毛利、毛損。"""
    pnl, trades, posa = sim
    ts = D["ts"]
    out = {}
    ti = np.array([i for i, _ in trades], dtype=int)
    tv = np.array([v for _, v in trades], dtype=float)
    for per, (a, b) in PERIODS_EVAL.items():
        m = (ts >= np.datetime64(a)) & (ts < np.datetime64(b))
        i0, i1 = np.flatnonzero(m)[[0, -1]]
        k = (ti >= i0) & (ti <= i1)
        v = tv[k]
        out[per] = dict(n=int(k.sum()), gw=float(v[v > 0].sum()), gl=float(-v[v < 0].sum()), win=int((v > 0).sum()))
    return out


_G = {}


def _init(g):
    global _G
    _G = g


def _leg_job(a):
    key, fp, side, p, mu = a
    D = _G["ds"][key]
    sim = simulate_leg(D, st13.supertrend_dir(D, "sma", p, mu), _G["filt"][(key, fp)], side)
    return a, sim[0].astype(np.float32), leg_summary(D, sim), sim[2]


def _combo_job(a):
    """一個 (key, fL, fS, 多方參數) 對全部 54 組空方參數，向量化算各區間指標。"""
    key, fL, fS, li = a
    g = _G["g"][key]
    out = []
    for per, m in g["masks"].items():
        L = g["pnl"][(fL, 1)][li][m].astype(np.float64)
        S = g["pnl"][(fS, -1)][:, m].astype(np.float64)
        eq = np.cumsum(S, axis=1) + np.cumsum(L)[None, :]
        peak = np.maximum.accumulate(np.concatenate([np.zeros((len(S), 1)), eq], axis=1), axis=1)[:, 1:]
        mdd = np.minimum((eq - peak).min(axis=1), 0.0)
        net = eq[:, -1]
        dl = np.add.reduceat(L, g["dstart"][per]); dsd = np.add.reduceat(S, g["dstart"][per], axis=1)
        dd = dsd + dl[None, :]
        sd = dd.std(axis=1, ddof=1)
        sh = np.where(sd > 0, dd.mean(axis=1) / np.where(sd > 0, sd, 1) * math.sqrt(252), 0.0)
        tl, ts_ = g["tr"][(fL, 1)][li][per], [g["tr"][(fS, -1)][j][per] for j in range(len(CELLS))]
        n = np.array([tl["n"] + t["n"] for t in ts_])
        gw = np.array([tl["gw"] + t["gw"] for t in ts_]); gl = np.array([tl["gl"] + t["gl"] for t in ts_])
        pf = np.where(gl > 0, gw / np.where(gl > 0, gl, 1), 0.0)
        rdd = np.where(mdd < 0, net / np.where(mdd < 0, -mdd, 1), 0.0)
        out.append((per, net, mdd, sh, n, pf, rdd))
    rows = []
    for j in range(len(CELLS)):
        r = dict(key=key, fL=fL, fS=fS, pL=CELLS[li][0], mL=CELLS[li][1], pS=CELLS[j][0], mS=CELLS[j][1])
        for per, net, mdd, sh, n, pf, rdd in out:
            r.update({per + "_net": net[j], per + "_mdd": mdd[j], per + "_sharpe": sh[j], per + "_trades": n[j],
                      per + "_pf": pf[j], per + "_rdd": rdd[j]})
        rows.append(r)
    return rows


def score(net, rdd):
    return np.where(net > 0, rdd, net / 1e7)


def plateau(df):
    """四維 3×3×3×3 鄰域平均（邊界截斷），df 為單一 (key, fL, fS) 的 2916 組。"""
    nP, nM = len(IN_PERIODS), len(IN_MULTS)
    Z = np.full((nP, nM, nP, nM), np.nan)
    ip = {p: i for i, p in enumerate(IN_PERIODS)}; im = {m: i for i, m in enumerate(IN_MULTS)}
    a = df["pL"].map(ip).to_numpy(); b = df["mL"].map(im).to_numpy()
    c = df["pS"].map(ip).to_numpy(); d = df["mS"].map(im).to_numpy()
    Z[a, b, c, d] = score(df["IS_net"].to_numpy(), df["IS_rdd"].to_numpy())
    P = np.pad(Z, 1, constant_values=np.nan)
    acc = np.zeros_like(Z); cnt = np.zeros_like(Z)
    for s in np.ndindex(3, 3, 3, 3):
        w = P[s[0]:s[0] + nP, s[1]:s[1] + nM, s[2]:s[2] + nP, s[3]:s[3] + nM]
        ok = ~np.isnan(w); acc += np.where(ok, w, 0); cnt += ok
    return (acc / cnt)[a, b, c, d], Z


def full_metrics(D, pnl, trades, posa, lo, hi):
    m = st13.metrics(D, pnl, trades, lo, hi)
    k = (D["ts"] >= np.datetime64(lo)) & (D["ts"] < np.datetime64(hi))
    m["tim"] = round(float((posa[k] != 0).mean() * 100), 1)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--daily", required=True)
    ap.add_argument("--minute", required=True)
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--combo-workers", type=int, default=3)
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
    filt = {(k, fp): st15.daily_filter_for_bars(daily, Dd, frames[k]["sid"].tolist(), fp) for k in ds for fp in DAILY_FILTERS}

    legjobs = [(k, fp, s, p, mu) for k in ds for fp in DAILY_FILTERS for s in (1, -1) for p, mu in CELLS]
    with ProcessPoolExecutor(args.workers, initializer=_init, initargs=(dict(ds=ds, filt=filt),)) as ex:
        legs = {a: (pn, sm) for a, pn, sm, pa in ex.map(_leg_job, legjobs, chunksize=4)}

    allrows = []
    for k, D in ds.items():  # 逐週期建立共享資料，避免記憶體爆掉
        masks, dstart = {}, {}
        for per, (a, b) in PERIODS_EVAL.items():
            m = (D["ts"] >= np.datetime64(a)) & (D["ts"] < np.datetime64(b))
            masks[per] = m
            dd = D["days"][m]
            dstart[per] = np.r_[0, np.flatnonzero(dd[1:] != dd[:-1]) + 1]
        gk = dict(masks=masks, dstart=dstart,
                  pnl={(fp, s): np.stack([legs[(k, fp, s) + c][0] for c in CELLS]) for fp in DAILY_FILTERS for s in (1, -1)},
                  tr={(fp, s): [legs[(k, fp, s) + c][1] for c in CELLS] for fp in DAILY_FILTERS for s in (1, -1)})
        cjobs = [(k, fL, fS, li) for fL in DAILY_FILTERS for fS in DAILY_FILTERS for li in range(len(CELLS))]
        with ProcessPoolExecutor(args.combo_workers, initializer=_init, initargs=(dict(g={k: gk}),)) as ex:
            allrows += [r for rows in ex.map(_combo_job, cjobs, chunksize=4) for r in rows]
        del gk
        print("{}: {} 組完成".format(k, len(cjobs) * len(CELLS)), flush=True)
    grid = pd.DataFrame(allrows)
    grid["fL"] = grid["fL"].map(lambda x: "{}/{}".format(*x)); grid["fS"] = grid["fS"].map(lambda x: "{}/{}".format(*x))
    grid["plateau"] = np.nan
    for (k, a, b), sub in grid.groupby(["key", "fL", "fS"]):
        grid.loc[sub.index, "plateau"] = plateau(sub)[0]
    grid.to_csv(R / "asym_grid_all.csv", index=False)
    fstr = lambda fp: "{}/{}".format(*fp)

    def rebuild(key, fL, fS, pL, mL, pS, mS):
        D = ds[key]
        fpL = DAILY_FILTERS[[fstr(x) for x in DAILY_FILTERS].index(fL)]
        fpS = DAILY_FILTERS[[fstr(x) for x in DAILY_FILTERS].index(fS)]
        sl = simulate_leg(D, st13.supertrend_dir(D, "sma", pL, mL), filt[(key, fpL)], 1)
        ss = simulate_leg(D, st13.supertrend_dir(D, "sma", pS, mS), filt[(key, fpS)], -1)
        tot = (sl[0] + ss[0], sorted(sl[1] + ss[1]), (sl[2] + ss[2]))
        return tot, sl, ss

    summ, ovf, legrows = [], [], []
    COLS = ("trades", "win", "net", "mdd", "pf", "sharpe", "tim")
    for key in ds:
        D = ds[key]
        g15 = pd.read_csv(R / "mtf_overfit.csv") if (R / "mtf_overfit.csv").exists() else None
        sym_row = g15[(g15.tf == key) & (g15["mode"] == "event")].iloc[0]["pick"]  # "(20, 3.0) | 30/4.5"
        fpart, ipart = sym_row.split(" | ")
        sfp = tuple(float(x) for x in fpart.strip("()").split(","))
        sfp = (int(sfp[0]), sfp[1]); sp, smu = ipart.split("/"); sp, smu = int(sp), float(smu)
        cands = {}
        for lab, sub in (("asym intraday (fL=fS)", grid[(grid.key == key) & (grid.fL == grid.fS)]),
                         ("asym intraday+daily", grid[grid.key == key])):
            elig = sub[(sub.IS_trades >= 50) & (sub.IS_pf > 1)]
            elig = elig if len(elig) else sub
            pick = elig.sort_values(["plateau", "IS_rdd"], ascending=False).iloc[0]
            best = elig.assign(s=score(elig.IS_net, elig.IS_rdd)).sort_values("s", ascending=False).iloc[0]
            cands[lab] = (pick, sub, best)
        cands["symmetric event (15)"] = (dict(fL=fstr(sfp), fS=fstr(sfp), pL=sp, mL=smu, pS=sp, mS=smu), None, None)
        eq = {}
        for lab, (pick, sub, best) in cands.items():
            prm = [pick[x] for x in ("fL", "fS", "pL", "mL", "pS", "mS")]
            prm[2], prm[4] = int(prm[2]), int(prm[4]); prm[3], prm[5] = float(prm[3]), float(prm[5])
            tot, sl, ss = rebuild(key, *prm)
            eq[lab] = (tot[0], prm)
            overlap = float(((sl[2] != 0) & (ss[2] != 0))[D["ts"] >= np.datetime64(MID)].mean() * 100)
            for leg, sim in (("combined", tot), ("long leg", sl), ("short leg", ss)):
                for per, (a, b) in PERIODS_EVAL.items():
                    x = full_metrics(D, sim[0], sim[1], sim[2], a, b)
                    (summ if leg == "combined" else legrows).append(dict(
                        tf=key, variant=lab, leg=leg, long="{} | {}/{}".format(prm[0], prm[2], prm[3]),
                        short="{} | {}/{}".format(prm[1], prm[4], prm[5]), sample=per,
                        **{c: x[c] for c in COLS}, overlap_oos_pct=round(overlap, 2)))
            if sub is not None:
                isv, oosv = sub.IS_sharpe.to_numpy(), sub.OOS_sharpe.to_numpy()
                sr0 = st13.expected_max_sr(len(sub), float(np.var(isv, ddof=1)))
                x = full_metrics(D, tot[0], tot[1], tot[2], LO, MID)
                ovf.append(dict(tf=key, variant=lab, cells=len(sub),
                                pick="L {} {}/{} ; S {} {}/{}".format(*[pick[c] for c in ("fL", "pL", "mL", "fS", "pS", "mS")]),
                                IS_best="L {} {}/{} ; S {} {}/{}".format(*[best[c] for c in ("fL", "pL", "mL", "fS", "pS", "mS")]),
                                IS_best_OOS_net=round(best.OOS_net), IS_best_OOS_sharpe=round(best.OOS_sharpe, 3),
                                spearman_sharpe=round(st13.spearman(isv, oosv), 3),
                                spearman_netmdd=round(st13.spearman(score(sub.IS_net, sub.IS_rdd), score(sub.OOS_net, sub.OOS_rdd)), 3),
                                pick_OOS_pctile=round(100 * float(np.mean(oosv <= pick["OOS_sharpe"])), 1),
                                share_OOS_profitable=round(100 * float(np.mean(sub.OOS_net > 0)), 1),
                                sr0=round(sr0, 3), IS_sharpe_pick=x["sharpe"],
                                DSR_pick=round(st13.deflated_sharpe(x["sharpe"], x["ndays"], x["skew"], x["kurt"], sr0), 3)))
        # 自我核對：對稱參數 → 應與 15 的 event 結果一致
        # 日 K 單獨
        dpnl, dtr = st13.simulate(Dd, "sma", sfp[0], sfp[1], "reversal")
        for per, (a, b) in PERIODS_EVAL.items():
            x = st13.metrics(Dd, dpnl, dtr, a, b)
            summ.append(dict(tf=key, variant="daily ST alone", leg="combined", long=fstr(sfp), short=fstr(sfp), sample=per,
                             **{c: x[c] for c in COLS if c != "tim"}, tim=100.0, overlap_oos_pct=0.0))

        # 圖：權益
        tsx = pd.to_datetime(D["ts"]); k = tsx >= LO
        plt.figure(figsize=(11, 5))
        for lab, (pn, prm) in eq.items():
            plt.plot(tsx[k], np.cumsum(pn[k]) / 1e3, label="{}: L {} {}/{}, S {} {}/{}".format(lab, prm[0], prm[2], prm[3], prm[1], prm[4], prm[5]))
        dts = pd.to_datetime(Dd["ts"]); kd = dts >= LO
        plt.plot(dts[kd], np.cumsum(dpnl[kd]) / 1e3, label="daily ST alone " + fstr(sfp))
        plt.axvline(MID, color="k", ls=":", label="IS | OOS"); plt.axvline(EX, color="gray", ls="--", label="2025-01")
        plt.title("TXF {} asymmetric long/short SuperTrend + daily filter (event), net of costs (k TWD)".format(key))
        plt.legend(fontsize=7); plt.grid(); plt.tight_layout()
        plt.savefig(R / "asym_equity_{}.png".format(key), dpi=110); plt.close()

        # 熱圖：選定 (fL=fS) 下，固定空方看多方參數、固定多方看空方參數
        pick = cands["asym intraday (fL=fS)"][0]
        sub = grid[(grid.key == key) & (grid.fL == pick.fL) & (grid.fS == pick.fS)]
        fig, axs = plt.subplots(2, 2, figsize=(14, 10))
        for row, (side, fixed) in enumerate((("L", ("pS", "mS")), ("S", ("pL", "mL")))):
            s2 = sub[(sub[fixed[0]] == pick[fixed[0]]) & (sub[fixed[1]] == pick[fixed[1]])]
            pc, mc = ("pL", "mL") if side == "L" else ("pS", "mS")
            for col, per in enumerate(("IS", "OOS")):
                ax = axs[row, col]
                Z = np.full((len(IN_PERIODS), len(IN_MULTS)), np.nan)
                for _, r in s2.iterrows():
                    Z[IN_PERIODS.index(r[pc]), IN_MULTS.index(r[mc])] = score(r[per + "_net"], r[per + "_rdd"])
                lim = np.nanmax(np.abs(Z))
                im = ax.imshow(Z, origin="lower", cmap="RdYlGn", aspect="auto", vmin=-lim, vmax=lim)
                ax.set_xticks(range(len(IN_MULTS)), IN_MULTS); ax.set_yticks(range(len(IN_PERIODS)), IN_PERIODS)
                ax.set_xlabel("{} ATR multiplier".format("long" if side == "L" else "short"))
                ax.set_ylabel("{} ATR period".format("long" if side == "L" else "short"))
                for y in range(Z.shape[0]):
                    for x in range(Z.shape[1]):
                        ax.text(x, y, "{:.2f}".format(Z[y, x]), ha="center", va="center", fontsize=7)
                ax.scatter(IN_MULTS.index(pick[mc]), IN_PERIODS.index(pick[pc]), marker="o", s=420,
                           facecolors="none", edgecolors="purple", linewidths=2)
                ax.set_title("{} params, other side fixed at {}/{} — {} net/maxDD".format(
                    "Long" if side == "L" else "Short", pick[fixed[0]], pick[fixed[1]], per))
                fig.colorbar(im, ax=ax)
        fig.suptitle("TXF {} asymmetric SuperTrend, daily filter {} (event); circle = chosen".format(key, pick.fL))
        fig.tight_layout(); fig.savefig(R / "asym_heatmap_{}.png".format(key), dpi=100); plt.close(fig)

    pd.DataFrame(summ).to_csv(R / "asym_summary.csv", index=False)
    pd.DataFrame(legrows).to_csv(R / "asym_legs.csv", index=False)
    pd.DataFrame(ovf).to_csv(R / "asym_overfit.csv", index=False)
    with pd.option_context("display.width", 300, "display.max_columns", 40, "display.max_rows", 500, "display.max_colwidth", 60):
        print(pd.DataFrame(summ).to_string(index=False))
        print(pd.DataFrame(legrows).to_string(index=False))
        print(pd.DataFrame(ovf).to_string(index=False))


if __name__ == "__main__":
    main()
