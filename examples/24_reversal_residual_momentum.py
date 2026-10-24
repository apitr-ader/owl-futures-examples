"""24 個股期貨：短期反轉 與 殘差動能（Residual Momentum）

沿用 21/22（不修改）：同資料、可選池規則、成本模型（每口每邊 50 元＋稅 2/100000＋1 跳動點、持有換月另扣來回）、
月/週/季末收盤訊號 → 次一交易日開盤成交、只用當時已知資料、5 檔、單檔上限 40%、22 的分數口數模擬邏輯。
IS 2014-2019、OOS 2020-2026/10。

1) 短期反轉（rev）：做多回看 lb（5/10/15/20 日、skip 0/1）報酬最差的 5 檔（無絕對濾網）。
   換倉 週/月；降週轉：無 / 排名緩衝（已持有且排名在前 2N 內續抱）/ 最短持有 20 交易日；
   流動性：(價>=0,量>=50) (50,200) (100,200) (50,500) (100,500)；大盤濾網 關/開；波動目標 無/15%；EW/IV/MV/ERC。
2) 殘差動能（rres）：每檔日報酬對 TXF 回歸（窗 126/252 日，只用到訊號日），排序用回看 63/126/252（skip 0/21）的
   累計殘差（可選除以殘差標準差），做多前 5；絕對濾網：原始動能 > 0 或 累計殘差 > 0；換倉 月/季；其餘同上。

對沖版：做多 5 檔，同時放空 Σ w·β 名目的 TXF（β 用換倉日前 120 日對 TXF 回歸），計 TXF 成本
（每邊 50 元＋1 點滑價＋稅，近似每月換月一次來回）。換倉當日沿用舊避險比例（新避險自隔日起）以免用到開盤前資訊。
用來看純選股 alpha：報告避險後年化報酬、Sharpe、t 值，以及未避險日報酬對 TXF 的回歸 alpha/beta。
挑參：IS plateau（自己＋每個參數只差一格的鄰居平均 IS Sharpe），資格 IS 平均空槽 < 30%（避免早年幾乎全現金的假贏家）。DSR 用各策略的試驗數。
注意：products.csv 只有現存契約 → 倖存者偏差（偏樂觀，反轉策略尤其吃到「跌深後沒下市」的偏差）。
"""
import argparse
import importlib.util
import itertools
import math
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

_spec = importlib.util.spec_from_file_location("dmo22", Path(__file__).with_name("22_dual_momentum_opt.py"))
o22 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(o22)
dm, D = o22.dm, o22.D
IS0, IS1, EX1, END = o22.IS0, o22.IS1, o22.EX1, o22.END
N = 5
LIQ = [(0, 50), (50, 200), (100, 200), (50, 500), (100, 500)]   # (最低價, 最低量)

G_REV = dict(lb=[5, 10, 15, 20], skip=[0, 1], reb=["W", "M"], tv=["none", "buf", "hold"], liq=list(range(len(LIQ))),
             mkt=[0, 1], vt=[0.0, 0.15], scheme=["EW", "IV", "MV", "ERC"])
G_RES = dict(lb=[63, 126, 252], skip=[0, 21], win=[126, 252], scaled=[0, 1], absf=["raw", "res"], reb=["M", "Q"],
             mkt=[0, 1], vt=[0.0, 0.15], scheme=["EW", "IV", "MV", "ERC"])

_S = {}


def schedule(kind):
    if kind in _S:
        return _S[kind]
    days = D["days"]
    s = pd.Series(np.arange(len(days)), days)
    if kind == "W":
        iso = days.isocalendar()
        last = s.groupby([iso.year.values, iso.week.values]).last()
    else:
        last = s.groupby([days.year, days.month]).last()
        if kind == "Q":
            last = last[[m in (3, 6, 9, 12) for _, m in last.index]]
    v = sorted(int(i) for i in last.values if pd.Timestamp("2013-12-20") <= days[i] < END)
    _S[kind] = v
    return v


def score_rev(P, i, U):
    I = D["I"]
    m = I[i - P["skip"], U] / I[i - P["lb"], U] - 1
    return -m, np.ones(len(U), bool)          # 越跌分數越高；無絕對濾網


def score_res(P, i, U):
    W, L, sk = P["win"], P["lb"], P["skip"]
    R = D["ret"][i - W + 1:i + 1, U]
    m = D["txv"][i - W + 1:i + 1]
    mc = m - m.mean()
    b = (mc @ (R - R.mean(0))) / (mc @ mc)
    a = R.mean(0) - b * m.mean()
    e = R - a - np.outer(m, b)
    seg = e[W - L:W - sk] if sk else e[W - L:]
    cum = seg.sum(0)
    sc = cum / (e.std(0) * math.sqrt(L) + 1e-12) if P["scaled"] else cum
    if P["absf"] == "raw":
        ok = D["I"][i - sk, U] / D["I"][i - L, U] - 1 > 0
    else:
        ok = cum > 0
    return sc, ok


def run(P, kind, lo=None, hi=None, path=False):
    """kind: rev / res。回傳 dict(r=日報酬, h=避險比例(日), turn, cost(占權益比例總和), cash, yrs)"""
    days, mult, A, O, RO, CR = D["days"], D["mult"], D["A"], D["O"], D["RO"], D["CR"]
    sched = schedule(P["reb"])
    if lo is not None:
        sched = [i for i in sched if lo - pd.Timedelta(days=10) <= days[i] < hi]
    nd = len(days)
    minpx, minvol = LIQ[P["liq"]] if kind == "rev" else (0, 50)
    tv = P.get("tv", "none")
    eq, q, k_prev = 1.0, np.zeros(len(D["cols"])), None
    vals, hser = np.full(nd, np.nan), np.zeros(nd)
    turn = cfrac = 0.0
    held, since = [], {}
    cash_slots = 0
    h_old = 0.0
    for n, i in enumerate(sched):
        k0 = i + 1
        if k0 >= nd:
            break
        k1 = sched[n + 1] + 1 if n + 1 < len(sched) else (nd if hi is None else min(nd, int(np.searchsorted(days, hi))))
        if k_prev is not None:
            h = np.nonzero(q)[0]
            if len(h):
                rc = q[h] @ (CR[k0, h] - CR[k_prev, h])
                eq += (q[h] * mult[h]) @ (O[k0, h] - O[k_prev, h]) - rc
                cfrac += rc / eq
        sel, w = [], np.zeros(0)
        U = o22.univ(i, minvol, minpx)
        need = max(P["lb"] + 1, P.get("win", 0))
        if len(U) >= dm.MIN_UNIVERSE and not (P["mkt"] and not D["txf_mom"][i] > 0) and i - need >= 0:
            sc, ok = (score_rev if kind == "rev" else score_res)(P, i, U)
            order = np.argsort(-sc)
            rk = U[order]
            okd = dict(zip(U, ok))
            pos = {s: r for r, s in enumerate(rk)}
            keep = []
            if tv == "buf":
                keep = [s for s in held if s in pos and pos[s] < 2 * N and okd[s]]
            elif tv == "hold":
                keep = [s for s in held if s in pos and okd[s] and i - since[s] < 20]
            keep = keep[:N]
            sel = keep + [s for s in rk if s not in keep and okd[s]][:N - len(keep)]
            cash_slots += N - len(sel)
            if sel:
                sel = sorted(int(s) for s in sel)
                w = o22.wts(i, P["scheme"], tuple(sel), len(sel) / N, P["vt"])
        else:
            cash_slots += N
        since = {s: (since[s] if s in since and s in held else i) for s in sel}
        held = list(sel)
        # 避險比例
        h_new = 0.0
        if sel:
            Rb = D["ret"][i - 119:i + 1, sel]; mb = D["txv"][i - 119:i + 1]; mc = mb - mb.mean()
            beta = (mc @ (Rb - Rb.mean(0))) / (mc @ mc)
            h_new = float(w @ beta)
        hser[k0] = h_old
        hser[k0 + 1:k1] = h_new
        h_old = h_new
        newq = np.zeros_like(q)
        if sel:
            newq[sel] = w * eq / (RO[k0, sel] * mult[sel])
        dq = np.abs(newq - q)
        ch = np.nonzero(dq)[0]
        if len(ch):
            px = RO[k0, ch]
            c = dq[ch] @ (dm.COMMISSION + mult[ch] * (dm.TAX * px + D["tickRO"][k0, ch]))
            turn += (dq[ch] * mult[ch] * px).sum() / eq
            eq -= c; cfrac += c / eq
        q = newq
        hh = np.nonzero(q)[0]
        if len(hh):
            vals[k0:k1] = eq + (A[k0:k1, hh] - O[k0, hh]) @ (q[hh] * mult[hh]) - (CR[k0 + 1:k1 + 1, hh] - CR[k0, hh]) @ q[hh]
        else:
            vals[k0:k1] = eq
        k_prev = k0
    e = pd.Series(vals, days).dropna()
    r = e.pct_change().fillna(0.0)
    hs = pd.Series(hser, days).reindex(r.index)
    return dict(r=r, h=hs, turn=turn, cost=cfrac, cash=cash_slots / max(len(sched) * N, 1), yrs=len(r) / 252)


def hedged(o):
    """r - h·TXF − TXF 交易成本（|Δh| 與每月換月來回）。"""
    r, h = o["r"], o["h"]
    tx = D["txr_s"].reindex(r.index).fillna(0.0)
    px = D["txpx"].reindex(r.index).ffill()
    cside = (dm.COMMISSION + 200 * 1.0) / (200 * px) + dm.TAX
    dh = h.diff().abs().fillna(h.abs())
    mon = pd.Series(r.index.to_period("M"), r.index)
    roll = (mon != mon.shift(-1)).astype(float)
    return r - h * tx - dh * cside - roll * 2 * h.abs() * cside


def stats(r):
    if len(r) < 30 or r.std() == 0:
        return dict(CAGR=0.0, vol=0.0, sharpe=0.0, mdd=0.0, n=len(r), skew=0.0, kurt=3.0)
    e = (1 + r).cumprod(); yrs = len(r) / 252
    return dict(CAGR=e.iloc[-1] ** (1 / yrs) - 1, vol=r.std() * math.sqrt(252), sharpe=r.mean() / r.std() * math.sqrt(252),
                mdd=(e / e.cummax() - 1).min(), n=len(r), skew=float(r.skew()), kurt=float(r.kurt() + 3))


def win(r, lo, hi):
    return r[(r.index >= lo) & (r.index < hi)]


def _work(args):
    P, kind = args
    o = run(P, kind)
    row = dict(P)
    for nm, lo, hi in (("IS", IS0, IS1), ("OOS", IS1, END), ("EX", IS1, EX1)):
        row[nm + "_sharpe"] = stats(win(o["r"], lo, hi))["sharpe"]
    row["IS_cash"] = o["cash"]
    return row


def plateau(res, G):
    keys = list(G)
    idx = {k: {x: j for j, x in enumerate(G[k])} for k in keys}
    code = pd.DataFrame({k: res[k].map(idx[k]) for k in keys})
    look = pd.Series(res.IS_sharpe.values, pd.MultiIndex.from_frame(code))
    tot, cnt = res.IS_sharpe.values.copy(), np.ones(len(res))
    for k in keys:
        if k == "scheme":
            continue
        for d in (-1, 1):
            nb = code.copy(); nb[k] += d
            v = look.reindex(pd.MultiIndex.from_frame(nb)).values; ok = ~np.isnan(v)
            tot[ok] += v[ok]; cnt[ok] += 1
    return tot / cnt


def study(kind, G, prefix, out, procs, bm):
    C = [(dict(zip(G, v)), kind) for v in itertools.product(*G.values())]
    C = [c for c in C if not (kind == "rev" and c[0]["skip"] >= c[0]["lb"]) and not (kind == "res" and c[0]["lb"] > c[0]["win"])]
    with Pool(procs) as pool:
        rows = pool.map(_work, C, chunksize=16)
    res = pd.DataFrame(rows)
    res["plateau"] = plateau(res, G)
    res.to_csv(out / f"{prefix}_grid.csv", index=False)
    ntr = len(res)
    sr0 = dm.expected_max_sr(ntr, float(res.IS_sharpe.var()))
    rho = res[["IS_sharpe", "OOS_sharpe"]].rank().corr().iloc[0, 1]
    elig = res[res.IS_cash < 0.3]
    picks = {"chosen": elig.sort_values(["plateau", "IS_sharpe"], ascending=False).iloc[0]}
    for sch in G["scheme"]:
        picks[sch] = elig[elig.scheme == sch].sort_values(["plateau", "IS_sharpe"], ascending=False).iloc[0]
    tab, curves = [], {}
    for nm, p in picks.items():
        P = {k: (p[k].item() if hasattr(p[k], "item") else p[k]) for k in G}
        full = run(P, kind)
        hr = hedged(full)
        if nm == "chosen":
            curves["chosen"], curves["chosen hedged"] = full["r"], hr
        for per, lo, hi in (("IS", IS0, IS1), ("OOS", IS1, END), ("OOS_ex2526", IS1, EX1)):
            o = run(P, kind, lo, hi)
            s = stats(win(full["r"], lo, hi)); sh = stats(win(hr, lo, hi))
            rr, tx = win(full["r"], lo, hi), D["txr_s"].reindex(win(full["r"], lo, hi).index).fillna(0)
            X = np.c_[np.ones(len(tx)), tx.values]
            coef, *_ = np.linalg.lstsq(X, rr.values, rcond=None)
            resid = rr.values - X @ coef
            se = math.sqrt(resid.var(ddof=2) / len(rr) * (np.linalg.inv(X.T @ X)[0, 0] * len(rr)))
            hm = win(hr, lo, hi)
            tab.append(dict(row=nm, cfg=" ".join(f"{k}={P[k]}" for k in G), period=per,
                            CAGR=s["CAGR"], vol=s["vol"], sharpe=s["sharpe"], mdd=s["mdd"],
                            turnover_yr=o["turn"] / o["yrs"], cost_pct_yr=o["cost"] / o["yrs"] * 100,
                            DSR=dm.dsr(s["sharpe"], s["n"], s["skew"], s["kurt"], sr0) if per == "OOS" else np.nan,
                            alpha_ann=coef[0] * 252, alpha_t=coef[0] / se, beta=coef[1],
                            hedged_CAGR=sh["CAGR"], hedged_sharpe=sh["sharpe"], hedged_mdd=sh["mdd"],
                            hedged_t=hm.mean() / hm.std() * math.sqrt(len(hm)) if hm.std() > 0 else 0.0,
                            avg_hedge=float(win(full["h"], lo, hi).mean())))
    for n, r in bm.items():
        for per, lo, hi in (("IS", IS0, IS1), ("OOS", IS1, END), ("OOS_ex2526", IS1, EX1)):
            s = stats(win(r, lo, hi))
            tab.append(dict(row=n, period=per, CAGR=s["CAGR"], vol=s["vol"], sharpe=s["sharpe"], mdd=s["mdd"]))
    tab = pd.DataFrame(tab)
    tab.to_csv(out / f"{prefix}_summary.csv", index=False)
    sens = []
    for k in G:
        g = res.groupby(k)[["IS_sharpe", "OOS_sharpe", "EX_sharpe"]].median()
        sens += [dict(param=k, value=str(v), **r.to_dict()) for v, r in g.iterrows()]
    pd.DataFrame(sens).to_csv(out / f"{prefix}_sensitivity.csv", index=False)
    print(f"\n==== {prefix}: 試驗數 {ntr}  SR0 {sr0:.3f}  Spearman(IS,OOS) {rho:.3f}")
    print("IS Sharpe", res.IS_sharpe.describe().round(3).to_dict())
    print("OOS Sharpe", res.OOS_sharpe.describe().round(3).to_dict())
    print(pd.DataFrame(sens).round(3).to_string())
    print(tab.round(3).to_string())
    charts(out, prefix, curves, bm, res, G)
    return ntr, sr0, rho


def charts(out, prefix, curves, bm, res, G):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.figure(figsize=(11, 5.5))
    for n, r in list(curves.items()) + list(bm.items()):
        r = win(r, IS0, END)
        plt.plot(r.index, (1 + r).cumprod(), label=n, lw=2 if n.startswith("chosen") else 1.2,
                 ls="-" if n.startswith("chosen") else "--")
    plt.axvline(IS1, color="k", ls="--", lw=.8); plt.yscale("log"); plt.grid(alpha=.3); plt.legend(fontsize=8)
    plt.title(f"{prefix}: IS-plateau pick (5 holdings, net) and TXF-hedged version vs benchmarks")
    plt.tight_layout(); plt.savefig(out / f"{prefix}_equity.png", dpi=110); plt.close()
    keys = [k for k in G if len(G[k]) > 1]
    fig, axs = plt.subplots(2, math.ceil(len(keys) / 2), figsize=(15, 7)); axs = axs.ravel()
    for ax, k in zip(axs, keys):
        g = res.groupby(k)[["IS_sharpe", "OOS_sharpe"]].median()
        x = np.arange(len(g)); ax.bar(x - .2, g.IS_sharpe, .4, label="IS"); ax.bar(x + .2, g.OOS_sharpe, .4, label="OOS")
        ax.set_xticks(x, [str(v) if k != "liq" else "px{}/v{}".format(*LIQ[v]) for v in g.index], fontsize=7)
        ax.set_title(k); ax.axhline(0, color="k", lw=.5); ax.grid(alpha=.3)
    axs[0].legend()
    for ax in axs[len(keys):]:
        ax.axis("off")
    fig.suptitle(f"{prefix}: median Sharpe by parameter value"); fig.tight_layout()
    fig.savefig(out / f"{prefix}_sensitivity.png", dpi=110); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/stock_futures")
    ap.add_argument("--txf", default="data/TXF_1c_D_regular_2010-01-01_2026-10-22.parquet")
    ap.add_argument("--out", default="results")
    ap.add_argument("--procs", type=int, default=6)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    o22.prepare(a.data, a.txf)
    D["txv"] = D["txr"].values
    D["txr_s"] = D["txr"]
    t = pd.read_parquet(a.txf)
    D["txpx"] = pd.Series(t.close.values.astype(float), t["date"].dt.tz_localize(None).dt.normalize()).reindex(D["days"]).ffill()
    for k in ("W", "M", "Q"):
        schedule(k)
    bm = {"TXF B&H": D["txr"]}
    mes = schedule("M")
    ew = pd.Series(0.0, D["days"])
    for n, i in enumerate(mes[:-1]):
        U = o22.univ(i, 50, 0); k0, k1 = i + 1, mes[n + 1] + 1
        if len(U) >= dm.MIN_UNIVERSE:
            ew.iloc[k0:k1] = D["ret"][k0:k1, U].mean(1)
    bm["EW universe (gross)"] = ew
    pd.set_option("display.width", 260)
    study("rev", G_REV, "rev", out, a.procs, bm)
    study("res", G_RES, "rres", out, a.procs, bm)


if __name__ == "__main__":
    main()
