"""22 個股期貨 Dual Momentum 參數最佳化（防過度擬合版）

延伸 21（21 不動）：同樣的資料、可選池規則、成本模型、次一交易日開盤成交、只用當時已知資料。
這支用向量化的「分數口數」模擬器（等同 21 的 frac 模式：口數 = 權重×權益/(開盤價×乘數)，持有期間口數固定），
跑大網格，再用 IS plateau 挑參數 → OOS、DSR、排除 2025-26、walk-forward（滾動 5 年訓練 / 1 年測試）檢查。

網格
  lb 回看 21/42/63/126/189/252 日、skip 0/5/21（skip < lb）、N_TOP 3/5/8/10、換倉 每月/每 2 月/每季、
  緩衝（已持有且排名在前 2N 內就續抱）開/關、最小 60 日量中位數 50/200/500 口、最低價（真實收盤）0/50/100 元、
  絕對動能門檻 0/rf、大盤濾網（TXF 12 個月報酬 > 0）開/關、波動目標 無/10%/15%/20%、
  權重 EW / IV / MV（Ledoit-Wolf 收縮、回看 120 日、只做多）/ ERC，單檔上限 40%。

選參
  1. 資格：IS Sharpe > 0、IS 平均現金 < 80%。
  2. plateau 分數 = 自己與所有「只差一個有序參數一格」的鄰居 IS Sharpe 平均；取最高。
  3. N_TOP=5 另外單獨挑一組（原始需求是 5 檔）。
  4. DSR：以全網格試驗數與 IS Sharpe 變異算預期最大 Sharpe。
  5. walk-forward：每年 Y（2019..2026）用 Y-5..Y-1 訓練（同樣 plateau 規則），Y 年測試，串接。

注意：products.csv 只有目前仍掛牌的個股期貨，已下市契約不在資料中 → 倖存者偏差，回測偏樂觀。
基準：TXF 買進持有；可選池等權（每月重新等權、不計成本）。
"""
import argparse
import importlib.util
import itertools
import math
import os
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

_spec = importlib.util.spec_from_file_location("dm21", Path(__file__).with_name("21_dual_momentum_stock_futures.py"))
dm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(dm)

IS0, IS1, EX1, END = dm.IS0, dm.IS1, pd.Timestamp("2025-01-01"), dm.OOS1
YEARS = list(range(2014, 2027))
G = dict(lb=[21, 42, 63, 126, 189, 252], skip=[0, 5, 21], ntop=[3, 5, 8, 10], freq=[1, 2, 3], buf=[0, 1],
         minvol=[50, 200, 500], minpx=[0, 50, 100], thr=["0", "rf"], mkt=[0, 1], vt=[0.0, 0.10, 0.15, 0.20],
         scheme=["EW", "IV", "MV", "ERC"])
ORD = ["lb", "skip", "ntop", "freq", "minvol", "minpx", "vt"]   # 有序參數（plateau 鄰居）

D = {}   # fork 後共用的唯讀資料


def tick_arr(p):
    return np.select([p < 10, p < 50, p < 100, p < 500, p < 1000], [0.01, 0.05, 0.1, 0.5, 1.0], 5.0)


def prepare(data, txf):
    M = dm.load(Path(data), False)
    days = M["ret"].index
    cols = list(M["A"].columns)
    mult = M["mult"].values
    A, O, R, RO = (M[k].values for k in ("Af", "Of", "Rf", "ROf"))
    roll = M["ROLL"].values
    rc = np.where(roll, 2 * (dm.COMMISSION + mult * (dm.TAX * np.nan_to_num(R) + tick_arr(np.nan_to_num(R)))), 0.0)
    CR = np.vstack([np.zeros(len(cols)), np.cumsum(rc, 0)])     # CR[k+1] = 第 0..k 日累計換月成本（每口）
    ret = M["ret"].values
    I = np.cumprod(1 + ret, 0)
    txr = dm.txf_returns(txf, days)
    txf_mom = ((1 + txr).rolling(252).apply(np.prod, raw=True) - 1).values
    mes = [days.get_loc(t) for t in dm.month_ends(days) if pd.Timestamp("2013-12-01") <= t < END]
    # 可選池基本條件（不含量、價門檻）
    base = (M["nhist"].values >= dm.MIN_HIST) & M["traded"].values & (np.nan_to_num(M["GAP"].values) <= dm.GAP_OUTLIER)
    medvol = np.nan_to_num(M["medvol"].values)
    und, mini = M["und"].values, M["mini"].values.astype(bool)
    order = np.lexsort((mini, und))            # 同標的標準優先
    D.update(days=days, cols=cols, mult=mult, A=A, O=O, R=R, RO=RO, CR=CR, ret=ret, I=I, txr=txr,
             txf_mom=txf_mom, mes=mes, base=base, medvol=medvol, und=und, order=order,
             tickRO=tick_arr(np.nan_to_num(RO)), M=M)
    return D


_U, _W = {}, {}


def mv_fast(X, total, cap=dm.MAX_W):
    """只做多、上限 cap 的最小變異（Ledoit-Wolf）：active-set，固定上限/下限後對其餘解等式限制最小變異。"""
    S = dm.ledoit_wolf(X)
    n = len(S)
    T = min(total, n * cap)
    fixed = np.full(n, np.nan)
    for _ in range(3 * n):
        free = np.isnan(fixed)
        rem = T - np.nansum(fixed)
        Sf = S[np.ix_(free, free)]
        # 固定部分對自由部分的梯度修正
        b = -S[np.ix_(free, ~free)] @ np.nan_to_num(fixed[~free]) if (~free).any() else np.zeros(free.sum())
        Si = np.linalg.pinv(Sf)
        one = np.ones(free.sum())
        lam = (rem - one @ Si @ b) / (one @ Si @ one)
        wf = Si @ (b + lam * one)
        w = fixed.copy(); w[free] = wf
        if (wf >= -1e-12).all() and (wf <= cap + 1e-12).all():
            return np.clip(w, 0, cap)
        j = np.argmin(wf) if wf.min() < -1e-12 else np.argmax(wf)
        idx = np.nonzero(free)[0][j]
        fixed[idx] = 0.0 if wf[j] < 0 else cap
    return np.full(n, T / n)


def univ(i, minvol, minpx):
    k = (i, minvol, minpx)
    if k not in _U:
        ok = D["base"][i] & (D["medvol"][i] >= minvol) & (np.nan_to_num(D["R"][i]) >= minpx)
        seen, keep = set(), []
        for j in D["order"]:
            if ok[j] and D["und"][j] not in seen:
                seen.add(D["und"][j]); keep.append(j)
        _U[k] = np.array(sorted(keep), dtype=int)
    return _U[k]


def wts(i, scheme, sel, total, vt):
    k = (i, scheme, sel, total, vt)
    if k not in _W:
        sel_a = list(sel)
        lbk = 120 if scheme == "MV" else 60
        X = D["ret"][max(0, i - lbk + 1):i + 1, sel_a]
        w = mv_fast(X, total) if scheme == "MV" else dm.weights(scheme, X, total)
        if vt:
            S = dm.ledoit_wolf(D["ret"][i - 59:i + 1, sel_a])
            pv = math.sqrt(max(w @ S @ w, 1e-12) * 252)
            w = w * min(vt / pv, 2.0)
        _W[k] = w
    return _W[k]


def run(P, keep_path=False):
    days, mult, A, O, RO, CR, I = D["days"], D["mult"], D["A"], D["O"], D["RO"], D["CR"], D["I"]
    mes = D["mes"][::P["freq"]]
    nd = len(days)
    N = P["ntop"]
    thr = (1 + dm.RF) ** (P["lb"] / 252) - 1 if P["thr"] == "rf" else 0.0
    eq = 1.0
    q = np.zeros(len(D["cols"]))      # 每單位初始資本的口數（線性，可縮放）
    k_prev = None
    vals = np.full(nd, np.nan)
    turn = cost = cfrac = 0.0
    held = []
    cash_slots = 0
    for n, i in enumerate(mes):
        k0 = i + 1
        if k0 >= nd:
            break
        k1 = mes[n + 1] + 1 if n + 1 < len(mes) else nd
        # 舊部位結算到今日開盤
        if k_prev is not None:
            h = np.nonzero(q)[0]
            if len(h):
                rcst = q[h] @ (CR[k0, h] - CR[k_prev, h])
                eq += (q[h] * mult[h]) @ (O[k0, h] - O[k_prev, h]) - rcst
                cost += rcst; cfrac += rcst / eq
        # 訊號
        sel, w = [], np.zeros(0)
        U = univ(i, P["minvol"], P["minpx"])
        if len(U) >= dm.MIN_UNIVERSE and not (P["mkt"] and not D["txf_mom"][i] > 0) and i - P["lb"] >= 0:
            m = I[i - P["skip"], U] / I[i - P["lb"], U] - 1
            rk = U[np.argsort(-m)]
            mm = dict(zip(U, m))
            if P["buf"]:
                pos = {s: r for r, s in enumerate(rk)}
                keep = [s for s in held if s in pos and pos[s] < 2 * N and mm[s] > thr][:N]
                sel = keep + [s for s in rk if s not in keep and mm[s] > thr][:N - len(keep)]
            else:
                sel = [s for s in rk[:N] if mm[s] > thr]
            cash_slots += N - len(sel)
            if sel:
                sel = sorted(int(s) for s in sel)
                w = wts(i, P["scheme"], tuple(sel), len(sel) / N, P["vt"])
        else:
            cash_slots += N
        held = sel
        newq = np.zeros_like(q)
        if sel:
            newq[sel] = w * eq / (RO[k0, sel] * mult[sel])
        dq = np.abs(newq - q)
        ch = np.nonzero(dq)[0]
        if len(ch):
            px = RO[k0, ch]
            c = dq[ch] @ (dm.COMMISSION + mult[ch] * (dm.TAX * px + D["tickRO"][k0, ch]))
            turn += (dq[ch] * mult[ch] * px).sum() / eq
            eq -= c; cost += c; cfrac += c / eq
        q = newq
        h = np.nonzero(q)[0]
        if len(h):
            path = (A[k0:k1, h] - O[k0, h]) @ (q[h] * mult[h]) - (CR[k0 + 1:k1 + 1, h] - CR[k0, h]) @ q[h]
            vals[k0:k1] = eq + path
        else:
            vals[k0:k1] = eq
        k_prev = k0
    e = pd.Series(vals, days).dropna()
    r = e.pct_change().fillna(0.0)
    out = dict(turn=turn, cost=cfrac, cash=cash_slots / max(len(mes) * N, 1), r=r if keep_path else None)
    yrs = r.index.year
    for y in YEARS:
        ry = r.values[yrs == y]
        out[f"s{y}"], out[f"q{y}"], out[f"n{y}"], out[f"g{y}"] = ry.sum(), (ry ** 2).sum(), len(ry), np.log1p(ry).sum()
    # 年化 turn/cost：全期間
    out["yrs"] = len(r) / 252
    return out


def sharpe_from(row, ys):
    s = sum(row[f"s{y}"] for y in ys); q2 = sum(row[f"q{y}"] for y in ys); n = sum(row[f"n{y}"] for y in ys)
    if n < 20:
        return np.nan
    mu = s / n; var = q2 / n - mu * mu
    return mu / math.sqrt(var) * math.sqrt(252) if var > 0 else 0.0


def sharpe_cols(df, ys):
    s = sum(df[f"s{y}"] for y in ys); q2 = sum(df[f"q{y}"] for y in ys); n = sum(df[f"n{y}"] for y in ys)
    mu = s / n; sd = np.sqrt((q2 / n - mu * mu).clip(lower=1e-18))
    return mu / sd * math.sqrt(252)


def configs():
    for v in itertools.product(*G.values()):
        P = dict(zip(G, v))
        if P["skip"] >= P["lb"]:
            continue
        yield P


def _work(P):
    o = run(P)
    o.pop("r")
    return {**P, **o}


def stats(r, lo, hi):
    r = r[(r.index >= lo) & (r.index < hi)]
    e = (1 + r).cumprod()
    yrs = len(r) / 252
    return dict(CAGR=e.iloc[-1] ** (1 / yrs) - 1, vol=r.std() * math.sqrt(252),
                sharpe=r.mean() / r.std() * math.sqrt(252), mdd=(e / e.cummax() - 1).min(),
                n=len(r), skew=float(r.skew()), kurt=float(r.kurt() + 3))


def plateau(res, score):
    """自己＋只差一格有序參數的鄰居 score 平均。"""
    keys = list(G)
    idx = {c: {v: k for k, v in enumerate(G[c])} for c in keys}
    code = res[keys].copy()
    for c in keys:
        code[c] = res[c].map(idx[c])
    lookup = pd.Series(score.values, index=pd.MultiIndex.from_frame(code))
    tot, cnt = score.values.copy(), np.ones(len(res))
    for c in ORD:
        for d in (-1, 1):
            nb = code.copy(); nb[c] = nb[c] + d
            v = lookup.reindex(pd.MultiIndex.from_frame(nb)).values
            ok = ~np.isnan(v)
            tot[ok] += v[ok]; cnt[ok] += 1
    return pd.Series(tot / cnt, res.index)


def pick(res, ys, extra=None):
    sc = sharpe_cols(res, ys)
    cash_ok = res["cash"] < 0.8
    pl = plateau(res, sc)
    m = (sc > 0) & cash_ok
    if extra is not None:
        m &= extra
    cand = res[m].assign(_pl=pl[m], _sc=sc[m])
    return cand.sort_values(["_pl", "_sc"], ascending=False).iloc[0], sc, pl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/stock_futures")
    ap.add_argument("--txf", default="data/TXF_1c_D_regular_2010-01-01_2026-10-22.parquet")
    ap.add_argument("--out", default="results")
    ap.add_argument("--procs", type=int, default=6)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    prepare(a.data, a.txf)
    C = list(configs())
    gp = out / "dmo_grid.parquet"
    if gp.exists():
        res = pd.read_parquet(gp)
    else:
        t0 = time.time()
        rows = []
        with Pool(a.procs) as pool:
            for n, o in enumerate(pool.imap_unordered(_work, C, chunksize=64)):
                rows.append(o)
                if n % 5000 == 0:
                    print(f"{n}/{len(C)} {time.time() - t0:.0f}s", flush=True)
        res = pd.DataFrame(rows)
        res.to_parquet(gp)
    ntr = len(res)
    IS_Y, OOS_Y, EX_Y = list(range(2014, 2020)), list(range(2020, 2027)), list(range(2020, 2025))
    res["IS_sharpe"], res["OOS_sharpe"], res["EX_sharpe"] = (sharpe_cols(res, y) for y in (IS_Y, OOS_Y, EX_Y))
    var_sr = float(res.IS_sharpe.var())
    sr0 = dm.expected_max_sr(ntr, var_sr)
    rho = res[["IS_sharpe", "OOS_sharpe"]].rank().corr().iloc[0, 1]

    best, sc, pl = pick(res, IS_Y)
    best5, _, _ = pick(res, IS_Y, res.ntop == 5)
    res["plateau_IS"] = pl
    res.drop(columns=[c for c in res.columns if c[:1] in "sqng" and c[1:].isdigit()]).to_csv(out / "dmo_grid.csv", index=False)

    keys = list(G)
    def P_of(row):
        return {k: (row[k].item() if hasattr(row[k], "item") else row[k]) for k in keys}

    # walk-forward
    wf_r, wf_rows = [], []
    for Y in range(2019, 2027):
        tr = list(range(Y - 5, Y))
        tr = [y for y in tr if y >= 2014]
        b, _, _ = pick(res, tr)
        b5, _, _ = pick(res, tr, res.ntop == 5)
        for tag, bb in (("all", b), ("n5", b5)):
            r = run(P_of(bb), keep_path=True)["r"]
            ry = r[r.index.year == Y]
            if tag == "all":
                wf_r.append(ry)
            else:
                wf_rows.append(("n5", Y, ry))
            wf_rows.append((tag + "_cfg", Y, P_of(bb)))
    wf = pd.concat(wf_r)
    wf5 = pd.concat([x[2] for x in wf_rows if x[0] == "n5"])
    pd.DataFrame([dict(test_year=Y, which=t[:-4], **c) for t, Y, c in wf_rows if t.endswith("_cfg")]).to_csv(out / "dmo_walkforward_picks.csv", index=False)

    # 基準
    days = D["days"]
    bm = {"TXF B&H": D["txr"]}
    ew = pd.Series(0.0, days)
    mes = D["mes"]
    for n, i in enumerate(mes[:-1]):
        U = univ(i, 50, 0)
        k0, k1 = i + 1, mes[n + 1] + 1
        if len(U) >= dm.MIN_UNIVERSE:
            ew.iloc[k0:k1] = D["ret"][k0:k1, U].mean(1)
    bm["EW universe (gross)"] = ew

    summ = []
    paths = {}
    for name, row in (("best_any_N", best), ("best_N5", best5)):
        o = run(P_of(row), keep_path=True)
        paths[name] = o["r"]
        for per, lo, hi in (("IS", IS0, IS1), ("OOS", IS1, END), ("OOS_ex2526", IS1, EX1)):
            s = stats(o["r"], lo, hi)
            summ.append(dict(strategy=name, period=per, **{k: s[k] for k in ("CAGR", "vol", "sharpe", "mdd")},
                             turnover_yr=o["turn"] / o["yrs"], cost_pct_yr=o["cost"] / o["yrs"] * 100,
                             cash_pct=o["cash"] * 100,
                             DSR=dm.dsr(s["sharpe"], s["n"], s["skew"], s["kurt"], sr0) if per == "OOS" else np.nan))
    for name, r in (("walkforward_any_N", wf), ("walkforward_N5", wf5)):
        for per, lo, hi in (("WF 2019-2026", pd.Timestamp("2019-01-01"), END), ("WF 2020-2026", IS1, END), ("WF ex2526", IS1, EX1)):
            s = stats(r, lo, hi)
            summ.append(dict(strategy=name, period=per, **{k: s[k] for k in ("CAGR", "vol", "sharpe", "mdd")},
                             DSR=dm.dsr(s["sharpe"], s["n"], s["skew"], s["kurt"], sr0) if per == "WF 2020-2026" else np.nan))
    for name, r in bm.items():
        for per, lo, hi in (("IS", IS0, IS1), ("OOS", IS1, END), ("OOS_ex2526", IS1, EX1), ("WF 2019-2026", pd.Timestamp("2019-01-01"), END)):
            s = stats(r, lo, hi)
            summ.append(dict(strategy=name, period=per, **{k: s[k] for k in ("CAGR", "vol", "sharpe", "mdd")}))
    summ = pd.DataFrame(summ)
    summ.to_csv(out / "dmo_summary.csv", index=False)

    # 敏感度：每個參數值的 IS / OOS Sharpe 中位數
    sens = []
    for k in keys:
        g = res.groupby(k)[["IS_sharpe", "OOS_sharpe", "EX_sharpe"]].median()
        for v, r in g.iterrows():
            sens.append(dict(param=k, value=str(v), **r.to_dict()))
    sens = pd.DataFrame(sens); sens.to_csv(out / "dmo_sensitivity.csv", index=False)

    pd.set_option("display.width", 250)
    print(f"試驗數 {ntr}，IS Sharpe 變異 {var_sr:.4f}，SR0(年化) {sr0:.3f}，Spearman(IS,OOS) {rho:.3f}")
    print("IS Sharpe 分布", res.IS_sharpe.describe().round(3).to_dict())
    print("OOS Sharpe 分布", res.OOS_sharpe.describe().round(3).to_dict())
    print("IS 前 1% 的 OOS Sharpe 中位數", round(res[res.IS_sharpe >= res.IS_sharpe.quantile(.99)].OOS_sharpe.median(), 3))
    for nm, b in (("best any N", best), ("best N=5", best5)):
        print(nm, P_of(b), "IS", round(b.IS_sharpe, 3), "plateau", round(b._pl, 3), "OOS", round(b.OOS_sharpe, 3), "EX", round(b.EX_sharpe, 3))
    print(summ.round(3).to_string())
    print(pd.read_csv(out / "dmo_walkforward_picks.csv").to_string())
    print(sens.round(3).to_string())
    charts(out, res, paths, wf, wf5, bm, best, best5)


def charts(out, res, paths, wf, wf5, bm, best, best5):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    lo = IS0
    plt.figure(figsize=(11, 5.5))
    for n, r in list(paths.items()) + list(bm.items()):
        r = r[(r.index >= lo) & (r.index < END)]
        plt.plot(r.index, (1 + r).cumprod(), label=n, lw=2 if n.startswith("best") else 1)
    for n, r in (("walk-forward (any N)", wf), ("walk-forward (N=5)", wf5)):
        plt.plot(r.index, (1 + r).cumprod(), ls="--", label=n)
    plt.axvline(IS1, color="k", ls="--", lw=.8); plt.yscale("log"); plt.grid(alpha=.3); plt.legend(fontsize=8)
    plt.title("Optimized dual momentum (TW stock futures, net of costs) vs benchmarks; walk-forward chained from 2019")
    plt.tight_layout(); plt.savefig(out / "dmo_equity.png", dpi=110); plt.close()

    plt.figure(figsize=(11, 3.8))
    for n, r in (("best_N5", paths["best_N5"]), ("TXF B&H", bm["TXF B&H"])):
        r = r[(r.index >= lo) & (r.index < END)]; e = (1 + r).cumprod()
        plt.fill_between(e.index, (e / e.cummax() - 1) * 100, 0, alpha=.4, label=n)
    plt.axvline(IS1, color="k", ls="--", lw=.8); plt.legend(); plt.grid(alpha=.3); plt.title("Drawdown %")
    plt.tight_layout(); plt.savefig(out / "dmo_drawdown.png", dpi=110); plt.close()

    # heatmap：lb × ntop（其他參數取最佳 N5 設定以外的中位數）
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, col in zip(axs, ("IS_sharpe", "OOS_sharpe")):
        hm = res.pivot_table(index="lb", columns="ntop", values=col, aggfunc="median")
        im = ax.imshow(hm.values, cmap="RdYlGn", vmin=-0.6, vmax=0.6, aspect="auto")
        ax.set_xticks(range(hm.shape[1]), hm.columns); ax.set_yticks(range(hm.shape[0]), hm.index)
        for (y, x), v in np.ndenumerate(hm.values):
            ax.text(x, y, f"{v:.2f}", ha="center", va="center", fontsize=8)
        ax.set_xlabel("N_TOP"); ax.set_ylabel("lookback (days)"); ax.set_title(f"median {col} over all other params")
    fig.colorbar(im, ax=axs); plt.savefig(out / "dmo_heatmap_lb_ntop.png", dpi=110); plt.close(fig)

    # 敏感度：最佳 N5 附近一次只動一個參數
    keys = list(G)
    fig, axs = plt.subplots(3, 4, figsize=(15, 9)); axs = axs.ravel()
    for ax, k in zip(axs, keys):
        m = np.ones(len(res), bool)
        for o in keys:
            if o != k:
                m &= (res[o] == best5[o]).values
        sub = res[m].sort_values(k)
        xs = [str(v) for v in sub[k]]
        ax.plot(xs, sub.IS_sharpe, "o-", label="IS"); ax.plot(xs, sub.OOS_sharpe, "s-", label="OOS")
        ax.axhline(0, color="k", lw=.5); ax.set_title(k); ax.grid(alpha=.3)
    axs[0].legend(); axs[-1].axis("off")
    fig.suptitle("One-at-a-time sensitivity around chosen N=5 config (Sharpe)"); fig.tight_layout()
    fig.savefig(out / "dmo_sensitivity.png", dpi=110); plt.close(fig)

    plt.figure(figsize=(6, 6))
    plt.scatter(res.IS_sharpe, res.OOS_sharpe, s=1, alpha=.15)
    plt.scatter([best.IS_sharpe, best5.IS_sharpe], [best.OOS_sharpe, best5.OOS_sharpe], c="r", s=40)
    plt.axhline(0, color="k", lw=.5); plt.axvline(0, color="k", lw=.5)
    plt.xlabel("IS Sharpe 2014-2019"); plt.ylabel("OOS Sharpe 2020-2026"); plt.title("All configs: IS vs OOS")
    plt.tight_layout(); plt.savefig(out / "dmo_is_vs_oos.png", dpi=110); plt.close()


if __name__ == "__main__":
    main()
