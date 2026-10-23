"""21 個股期貨 Dual Momentum（Antonacci 風格）5 檔組合＋考慮共變異數的權重

執行：
  python examples/21_dual_momentum_stock_futures.py --data <stock_futures 資料夾> --txf <TXF 日 K parquet> --out <結果資料夾>

資料：tools/fetch_stock_futures.py 下載的 daily/<SYM>_adj.parquet（訊號與損益）與 _raw（真實價格算稅、名目、跳動點），
products.csv（契約乘數、小型、標的）。

規則
- 月底收盤算訊號，次一交易日開盤成交；持有到下次換倉。
- 每次換倉的可選池（只用當時以前的資料，避免倖存者偏差與垃圾標的）：
  已有 >=252 個交易日、近 60 日成交量中位數 >= MIN_VOL 口、當日有成交；排除代號結尾 1 的調整型契約、
  歷史上換月跳空 > 15% 的契約；同一標的只取一檔（標準優先，否則小型）；ETF 期貨預設排除（--etf 另測）。
  可選池少於 10 檔就全部空手。
- 相對動能：回看 L 個月（63/126/252 日）報酬，可略過最近 1 個月；取前 5。
- 絕對動能：自身回看報酬 > 門檻（0 或無風險 1.5%/年）才持有，沒過的名額放現金（報酬 0）。
- 大盤濾網（可選）：TXF 12 個月報酬 <= 0 時全部空手。
- 權重：EW 等權、IV 反波動、MV 最小變異（Ledoit-Wolf 收縮共變異數，回看 60/120/252 日，只做多，單檔上限 40%）、
  ERC 風險平價。每檔名額 1/5，沒過濾網的名額留現金：風險資產總權重 = 通過檔數/5，再依方法分配（上限 40%）。
- 可選波動目標 15%/年：用 60 日共變異數估組合波動，名目放大縮小（上限 2 倍）。
- 整數口數：名目資本 10M TWD（逐月複利），口數 = round(權重×權益/(價格×乘數))；另報分數口（理想）版本。

成本（每口每邊）：手續費 50 元、期交稅 2/100000×真實價格×乘數、滑價 1 跳動點×乘數
（股票期貨跳動點：<10 0.01、<50 0.05、<100 0.1、<500 0.5、<1000 1、其餘 5）。
換月：調整價已吸收價差，持有部位在換月日另扣一次來回（兩邊）成本。

樣本：可選池 2013 年以前太少，IS 改為 2014-01..2019-12，OOS 2020-01..2026-10。
參數挑選：IS 上依 (回看, 略過) 鄰域平均分數（plateau）選，資格：IS 換倉次數 >= 24 且 Sharpe > 0。
"""
import argparse
import itertools
import math
from pathlib import Path
from statistics import NormalDist

import numpy as np
import pandas as pd

MIN_VOL = 50
MIN_HIST = 252
N_TOP = 5
MAX_W = 0.40
MIN_UNIVERSE = 10
GAP_OUTLIER = 0.15
COMMISSION = 50.0
TAX = 2e-5
RF = 0.015
CAPITAL = 10_000_000.0
IS0, IS1, OOS1 = pd.Timestamp("2014-01-01"), pd.Timestamp("2020-01-01"), pd.Timestamp("2026-11-01")


def tick(p):
    return 0.01 if p < 10 else 0.05 if p < 50 else 0.1 if p < 100 else 0.5 if p < 500 else 1.0 if p < 1000 else 5.0


# ---------------- 資料 ----------------

def load(data, include_etf):
    prod = pd.read_csv(data / "products.csv", dtype={"underlying_code": str})
    prod = prod[~prod.symbol.str.endswith("1")]
    if not include_etf:
        prod = prod[prod.underlying_type == "S"]
    A, R, O, RO, V, ROLL = {}, {}, {}, {}, {}, {}
    gapmax = {}
    for s in prod.symbol:
        pa, pr = data / "daily" / f"{s}_adj.parquet", data / "daily" / f"{s}_raw.parquet"
        if not pa.exists():
            continue
        a, r = pd.read_parquet(pa), pd.read_parquet(pr)
        if len(a) < MIN_HIST:
            continue
        idx = a["time"].dt.tz_localize(None).dt.normalize()
        A[s], O[s] = pd.Series(a.close.values, idx), pd.Series(a.open.values, idx)
        R[s], RO[s] = pd.Series(r.close.values, idx), pd.Series(r.open.values, idx)
        V[s], ROLL[s] = pd.Series(r.volume.values, idx), pd.Series(r.is_roll.values, idx)
        g = (r.open - r.close.shift()).abs() / r.close.shift()
        gapmax[s] = pd.Series(np.where(r.is_roll, g, 0.0), idx).cummax()  # 截至當日最大換月跳空
    df = lambda d: pd.DataFrame(d).sort_index()
    M = dict(A=df(A), O=df(O), R=df(R), RO=df(RO), V=df(V), ROLL=df(ROLL).fillna(False).astype(bool), GAP=df(gapmax))
    days = M["A"].index
    M = {k: v.reindex(days) for k, v in M.items()}
    M["traded"] = M["R"].notna()
    M["GAP"] = M["GAP"].ffill()
    # 每日報酬：調整價差 / 前一日真實收盤（加法調整下最合理的報酬定義）
    Af = M["A"].ffill()
    M["ret"] = (Af.diff() / M["R"].ffill().shift()).fillna(0.0)
    M["Af"], M["Rf"] = Af, M["R"].ffill()
    M["Of"] = M["O"].fillna(Af)                     # 沒成交的日子用前收當開盤
    M["ROf"] = M["RO"].fillna(M["Rf"])
    pm = prod.set_index("symbol").reindex(M["A"].columns)
    M["mult"] = pm.contract_size.astype(float)
    M["und"], M["mini"] = pm.underlying_code, pm["mini"].astype(bool)
    M["nhist"] = M["traded"].cumsum()
    M["medvol"] = M["V"].fillna(0).rolling(60, min_periods=20).median()
    return M


def txf_returns(path, days):
    d = pd.read_parquet(path)
    idx = d["date"].dt.tz_localize(None).dt.normalize()
    c, o, cm = d.close.values.astype(float), d.open.values.astype(float), d.contractMonth.astype(str)
    roll = (cm.ne(cm.shift()) & cm.shift().notna()).values
    pc = np.r_[np.nan, c[:-1]]
    r = np.where(roll, (c - o) / pc, (c - pc) / pc)       # 換月日只算新合約開到收（同 back_adjust）
    return pd.Series(r, idx).reindex(days).fillna(0.0)


def month_ends(days):
    s = pd.Series(days, days)
    return list(s.groupby([days.year, days.month]).last())


# ---------------- 權重 ----------------

def ledoit_wolf(X):
    X = X - X.mean(0)
    n, p = X.shape
    S = X.T @ X / n
    mu = np.trace(S) / p
    F = mu * np.eye(p)
    d2 = ((S - F) ** 2).sum()
    b2 = (((X ** 2).sum(1) ** 2).sum() / n - (S ** 2).sum()) / n
    sh = min(b2, d2) / d2 if d2 > 0 else 1.0
    return sh * F + (1 - sh) * S


def proj_capped_simplex(v, total, cap):
    lo, hi = v.min() - total, v.max()
    for _ in range(60):
        m = (lo + hi) / 2
        if np.clip(v - m, 0, cap).sum() > total:
            lo = m
        else:
            hi = m
    return np.clip(v - (lo + hi) / 2, 0, cap)


def min_var(S, total, cap):
    n = len(S)
    if n * cap < total - 1e-9:
        return np.full(n, cap)
    w = np.full(n, total / n)
    step = 1.0 / (np.linalg.eigvalsh(S).max() * 2 + 1e-12)
    for _ in range(250):
        w = proj_capped_simplex(w - step * (S @ w), total, cap)
    return w


def erc(S, total, cap):
    n = len(S)
    w = np.full(n, 1.0 / n)
    for _ in range(300):
        rc = w * (S @ w)
        w = w * (rc.mean() / np.maximum(rc, 1e-18)) ** 0.5
        w /= w.sum()
    w = w * total
    for _ in range(20):  # 上限：多的平均分給其他
        over = w > cap
        if not over.any():
            break
        ex = (w[over] - cap).sum(); w[over] = cap
        free = ~over & (w < cap)
        if not free.any():
            break
        w[free] += ex * w[free] / w[free].sum()
    return w


def weights(scheme, X, total):
    n = X.shape[1]
    if n == 0:
        return np.zeros(0)
    cap = MAX_W
    if scheme == "EW":
        return np.full(n, total / n) if total / n <= cap else np.full(n, cap)
    vol = X.std(0) + 1e-12
    if scheme == "IV":
        w = (1 / vol) / (1 / vol).sum() * total
        return proj_capped_simplex(w, min(total, n * cap), cap) if (w > cap).any() else w
    S = ledoit_wolf(X)
    if scheme.startswith("MV"):
        return min_var(S, min(total, n * cap), cap)
    if scheme == "ERC":
        return erc(S, min(total, n * cap), cap)
    raise ValueError(scheme)


# ---------------- 回測 ----------------

_UCACHE, _MCACHE, _WCACHE = {}, {}, {}


def universe(M, t):
    """t（月底）可選的代號，只用 t 以前資料。"""
    if t not in _UCACHE:
        _UCACHE[t] = _universe(M, t)
    return _UCACHE[t]


def _universe(M, t):
    ok = (M["nhist"].loc[t] >= MIN_HIST) & (M["medvol"].loc[t] >= MIN_VOL) & M["traded"].loc[t] \
        & (M["GAP"].loc[t].fillna(0) <= GAP_OUTLIER)
    c = ok[ok].index
    df = pd.DataFrame({"s": c, "und": M["und"][c].values, "mini": M["mini"][c].values})
    df = df.sort_values(["und", "mini"]).drop_duplicates("und")   # 標準優先
    return list(df.s)


def momentum(M, t, L, skip):
    k = (t, L, skip)
    if k not in _MCACHE:
        _MCACHE[k] = _momentum(M, t, L, skip)
    return _MCACHE[k]


def _momentum(M, t, L, skip):
    i = M["ret"].index.get_loc(t)
    a, b = i - L, i - skip
    if a < 0:
        return None
    r = M["ret"].iloc[a + 1:b + 1]
    return (1 + r).prod() - 1


def simulate(M, P, txf_ret, txf_mom, integer=False, start=IS0, end=OOS1):
    """P：lb, skip, thr, mkt, scheme, cov_lb, vt。回傳 daily equity 等。"""
    days = M["ret"].index
    mes = [t for t in month_ends(days) if start - pd.Timedelta(days=40) <= t < end]
    ret, Af, Of, Rf, ROf = M["ret"], M["Af"], M["Of"], M["Rf"], M["ROf"]
    mult, ROLL = M["mult"], M["ROLL"]
    pos = pd.Series(dtype=float)       # 口數（可為分數）
    eq = CAPITAL
    out_eq, out_d = [], []
    turn = cost_tot = 0.0
    hold_log = []
    cash_slots = 0
    nreb = 0
    di = {d: k for k, d in enumerate(days)}
    Afv, Ofv, Rfv, ROfv, ROLLv = Af.values, Of.values, Rf.values, ROf.values, ROLL.values
    col = {s: k for k, s in enumerate(Af.columns)}
    mv = mult.values
    for mi, t in enumerate(mes):
        k0 = di[t] + 1
        if k0 >= len(days):
            break
        k1 = di[mes[mi + 1]] + 1 if mi + 1 < len(mes) else min(len(days), di[days[days < end][-1]] + 1)
        if k1 > len(days):
            k1 = len(days)
        # ---- 訊號 ----
        target = {}
        U = universe(M, t)
        if len(U) >= MIN_UNIVERSE and not (P["mkt"] and txf_mom.loc[t] <= 0):
            mom = momentum(M, t, P["lb"], P["skip"])
            if mom is not None:
                m = mom[U].dropna().sort_values(ascending=False)
                top = list(m.index[:N_TOP])
                thr = {"rf": (1 + RF) ** (P["lb"] / 252) - 1, "0": 0.0, "none": -np.inf}[P["thr"]]
                pas = [s for s in top if m[s] > thr]
                cash_slots += N_TOP - len(pas)
                if pas:
                    lbk = P["cov_lb"] if P["scheme"].startswith("MV") else 60
                    i = di[t]
                    X = ret[pas].iloc[max(0, i - lbk + 1):i + 1].values
                    wk = (t, P["scheme"], tuple(pas))
                    if wk not in _WCACHE:
                        _WCACHE[wk] = weights(P["scheme"], X, len(pas) / N_TOP)
                    w = _WCACHE[wk]
                    if P["vt"]:
                        S = ledoit_wolf(ret[pas].iloc[i - 59:i + 1].values)
                        pv = math.sqrt(max(w @ S @ w, 1e-12) * 252)
                        w = w * min(P["vt"] / pv, 2.0)
                    target = dict(zip(pas, w))
            else:
                cash_slots += N_TOP
        else:
            cash_slots += N_TOP
        nreb += 1
        hold_log.append((t, target))
        # ---- 次日開盤換倉 ----
        newpos = {}
        for s, w in target.items():
            c = col[s]
            n = w * eq / (ROfv[k0, c] * mv[c])
            newpos[s] = round(n) if integer else n
        names = set(pos.index) | set(newpos)
        if k0 > 0:  # 舊部位：前收 → 今開
            g = sum(pos.get(s, 0) * mv[col[s]] * (Ofv[k0, col[s]] - Afv[k0 - 1, col[s]]) for s in pos.index)
            eq += g
        c_reb = 0.0
        for s in names:
            dq = abs(newpos.get(s, 0) - pos.get(s, 0))
            if dq:
                c = col[s]; px = ROfv[k0, c]
                c_reb += dq * (COMMISSION + TAX * px * mv[c] + float(tick(px)) * mv[c])
                turn += dq * px * mv[c] / eq
        eq -= c_reb; cost_tot += c_reb
        pos = pd.Series(newpos, dtype=float)
        # ---- 持有期逐日 ----
        for k in range(k0, k1):
            if k == k0:
                g = sum(q * mv[col[s]] * (Afv[k, col[s]] - Ofv[k, col[s]]) for s, q in pos.items())
            else:
                g = sum(q * mv[col[s]] * (Afv[k, col[s]] - Afv[k - 1, col[s]]) for s, q in pos.items())
            rc = sum(abs(q) * 2 * (COMMISSION + TAX * Rfv[k, col[s]] * mv[col[s]] + float(tick(Rfv[k, col[s]])) * mv[col[s]])
                     for s, q in pos.items() if ROLLv[k, col[s]])
            eq += g - rc; cost_tot += rc
            if k + 1 < k1 or mi + 1 == len(mes):
                out_eq.append(eq); out_d.append(days[k])
            elif k + 1 == k1:
                out_eq.append(eq); out_d.append(days[k])
    e = pd.Series(out_eq, pd.DatetimeIndex(out_d))
    e = e[~e.index.duplicated(keep="last")]
    return dict(eq=e, turn=turn, cost=cost_tot, cash=cash_slots / max(nreb * N_TOP, 1), holds=hold_log)


def stats(eq, lo, hi, turn_cost=None):
    e = eq[(eq.index >= lo) & (eq.index < hi)]
    if len(e) < 30:
        return dict(CAGR=np.nan, vol=np.nan, sharpe=np.nan, mdd=np.nan, days=0, skew=0, kurt=3)
    r = e.pct_change().dropna()
    yrs = len(r) / 252
    cagr = (e.iloc[-1] / e.iloc[0]) ** (1 / yrs) - 1
    dd = (e / e.cummax() - 1).min()
    sh = r.mean() / r.std() * math.sqrt(252) if r.std() > 0 else 0.0
    return dict(CAGR=cagr, vol=r.std() * math.sqrt(252), sharpe=sh, mdd=dd, days=len(r),
                skew=float(r.skew()), kurt=float(r.kurt() + 3))


def expected_max_sr(n, var):
    g = 0.5772156649; nd = NormalDist()
    return math.sqrt(var) * ((1 - g) * nd.inv_cdf(1 - 1 / n) + g * nd.inv_cdf(1 - 1 / (n * math.e)))


def dsr(sr, n, skew, kurt, sr0):
    s, s0 = sr / math.sqrt(252), sr0 / math.sqrt(252)
    den = math.sqrt(max(1 - skew * s + (kurt - 1) / 4 * s * s, 1e-12))
    return NormalDist().cdf((s - s0) * math.sqrt(n - 1) / den)


# ---------------- 主程式 ----------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/stock_futures")
    ap.add_argument("--txf", default="data/TXF_1c_D_regular_2010-01-01_2026-10-22.parquet")
    ap.add_argument("--out", default="results")
    ap.add_argument("--etf", action="store_true", help="可選池也納入 ETF 期貨")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    tag = "_etf" if a.etf else ""
    M = load(Path(a.data), a.etf)
    days = M["ret"].index
    txr = txf_returns(a.txf, days)
    txf_mom = (1 + txr).rolling(252).apply(np.prod, raw=True) - 1
    mes = month_ends(days)
    usz = pd.Series({t: len(universe(M, t)) for t in mes if t >= pd.Timestamp("2012-01-01")})
    print("可選池檔數（每年 12 月底）：", usz[usz.index.month == 12].to_dict())

    G = dict(lb=[63, 126, 252], skip=[0, 21], thr=["0", "rf"], mkt=[False, True],
             scheme=["EW", "IV", "MV60", "MV120", "MV252", "ERC"], vt=[0.0, 0.15])
    rows, eqs = [], {}
    for lb, skip, thr, mkt, scheme, vt in itertools.product(*G.values()):
        P = dict(lb=lb, skip=skip, thr=thr, mkt=mkt, scheme=scheme, vt=vt,
                 cov_lb=int(scheme[2:]) if scheme.startswith("MV") else 60)
        r = simulate(M, P, txr, txf_mom)
        s_is, s_oos = stats(r["eq"], IS0, IS1), stats(r["eq"], IS1, OOS1)
        s_ex = stats(r["eq"], IS1, pd.Timestamp("2025-01-01"))
        key = (lb, skip, thr, mkt, scheme, vt)
        eqs[key] = r
        rows.append(dict(lb=lb, skip=skip, thr=thr, mkt=mkt, scheme=scheme, vt=vt,
                         IS_sharpe=s_is["sharpe"], IS_cagr=s_is["CAGR"], IS_mdd=s_is["mdd"],
                         OOS_sharpe=s_oos["sharpe"], OOS_cagr=s_oos["CAGR"], OOS_mdd=s_oos["mdd"],
                         EX_sharpe=s_ex["sharpe"], cash=r["cash"]))
        print(key, "IS {:.2f} OOS {:.2f}".format(s_is["sharpe"], s_oos["sharpe"]), flush=True)
    res = pd.DataFrame(rows)
    res.to_csv(out / f"dm_grid{tag}.csv", index=False)

    # plateau：同 (thr, mkt, scheme, vt) 下，(lb, skip) 鄰域（lb ±1 格、skip 兩值）IS Sharpe 平均
    LBs = G["lb"]
    def plat(r):
        sub = res[(res.thr == r.thr) & (res.mkt == r.mkt) & (res.scheme == r.scheme) & (res.vt == r.vt)]
        i = LBs.index(r.lb)
        nb = sub[sub.lb.isin(LBs[max(0, i - 1):i + 2])]
        return nb.IS_sharpe.mean()
    res["plateau"] = res.apply(plat, axis=1)
    var_sr = float(res.IS_sharpe.var())
    nd_oos = stats(next(iter(eqs.values()))["eq"], IS1, OOS1)["days"]
    picks = []
    for scheme in G["scheme"]:
        sub = res[(res.scheme == scheme) & (res.IS_sharpe > 0)]
        sub = sub if len(sub) else res[res.scheme == scheme]
        p = sub.sort_values(["plateau", "IS_sharpe"], ascending=False).iloc[0]
        picks.append(p)
    picks = pd.DataFrame(picks)

    # 選定參數：各權重法的 pick 再跑整數口數版本
    summ = []
    sr0 = expected_max_sr(len(res), var_sr)
    rc = res[["IS_sharpe", "OOS_sharpe"]].rank().corr().iloc[0, 1]
    for _, p in picks.iterrows():
        key = (p.lb, p.skip, p.thr, p.mkt, p.scheme, p.vt)
        P = dict(lb=p.lb, skip=p.skip, thr=p.thr, mkt=p.mkt, scheme=p.scheme, vt=p.vt,
                 cov_lb=int(p.scheme[2:]) if p.scheme.startswith("MV") else 60)
        for mode in ("frac", "int"):
            r = eqs[key] if mode == "frac" else simulate(M, P, txr, txf_mom, integer=True)
            for per, lo, hi in (("IS", IS0, IS1), ("OOS", IS1, OOS1), ("OOS_ex2526", IS1, pd.Timestamp("2025-01-01"))):
                rr = simulate(M, P, txr, txf_mom, integer=(mode == "int"), start=lo, end=hi)
                s = stats(rr["eq"], lo, hi)
                yrs = max(s["days"], 1) / 252
                summ.append(dict(scheme=p.scheme, params="lb{} skip{} thr{} mkt{} vt{}".format(p.lb, p.skip, p.thr, int(p.mkt), p.vt),
                                 mode=mode, period=per, CAGR=s["CAGR"], vol=s["vol"], sharpe=s["sharpe"], mdd=s["mdd"],
                                 turnover_yr=rr["turn"] / yrs, cost_pct_yr=rr["cost"] / CAPITAL / yrs * 100,
                                 cash_pct=rr["cash"] * 100,
                                 DSR=dsr(s["sharpe"], s["days"], s["skew"], s["kurt"], sr0) if per == "OOS" else np.nan))
            if mode == "int":
                eqs[("int",) + key] = r
    summ = pd.DataFrame(summ)

    # 基準
    def bench_eq(rets, lo, hi):
        r = rets[(rets.index >= lo) & (rets.index < hi)]
        return CAPITAL * (1 + r).cumprod()
    bm = {}
    bm["TXF B&H"] = txr
    # 等權可選池（月換倉，含成本近似：以 EW 全池模擬）
    ew_rows = []
    for mi, t in enumerate(mes[:-1]):
        U = universe(M, t)
        k0, k1 = days.get_loc(t) + 1, days.get_loc(mes[mi + 1]) + 1
        r = M["ret"][U].iloc[k0:k1].mean(axis=1) if len(U) >= MIN_UNIVERSE else pd.Series(0.0, days[k0:k1])
        ew_rows.append(r)
    bm["EW universe"] = pd.concat(ew_rows)
    best_ew = picks[picks.scheme == "EW"].iloc[0]
    Pn = dict(lb=best_ew.lb, skip=best_ew.skip, thr="0", mkt=False, scheme="EW", vt=0.0, cov_lb=60)
    bm_rows = []
    for per, lo, hi in (("IS", IS0, IS1), ("OOS", IS1, OOS1), ("OOS_ex2526", IS1, pd.Timestamp("2025-01-01"))):
        for name, rr in bm.items():
            s = stats(bench_eq(rr, lo, hi), lo, hi)
            bm_rows.append(dict(bench=name, period=per, **{k: s[k] for k in ("CAGR", "vol", "sharpe", "mdd")}))
        # top5 不加絕對動能：把 thr 設成 -inf 的效果 → 用 mom > -1
        rr = simulate_noabs(M, Pn, txr, txf_mom, lo, hi)
        s = stats(rr["eq"], lo, hi)
        bm_rows.append(dict(bench="Top5 rel-mom EW (no abs)", period=per, **{k: s[k] for k in ("CAGR", "vol", "sharpe", "mdd")}))
    bmdf = pd.DataFrame(bm_rows)
    picks.to_csv(out / f"dm_picks{tag}.csv", index=False)
    summ.to_csv(out / f"dm_summary{tag}.csv", index=False)
    bmdf.to_csv(out / f"dm_bench{tag}.csv", index=False)
    pd.set_option("display.width", 220)
    print("\n可選池大小", usz.describe().round(1).to_dict())
    print("\nSpearman(IS Sharpe, OOS Sharpe) 全網格 =", round(rc, 3), " 試驗數", len(res), " SR0(年化) =", round(sr0, 3))
    print(picks[["scheme", "lb", "skip", "thr", "mkt", "vt", "plateau", "IS_sharpe", "OOS_sharpe", "EX_sharpe", "cash"]].round(3).to_string())
    print(summ.round(3).to_string())
    print(bmdf.round(3).to_string())

    # 持有名單統計（選定 EW 以外依 OOS 最佳 IS pick 的 scheme：取 ERC 或 plateau 最高者）
    lead = picks.sort_values("plateau", ascending=False).iloc[0]
    lk = (lead.lb, lead.skip, lead.thr, lead.mkt, lead.scheme, lead.vt)
    holds = eqs[lk]["holds"]
    cnt = pd.Series([s for t, h in holds for s in h]).value_counts()
    names = pd.read_csv(Path(a.data) / "products.csv").set_index("symbol")["name"]
    print("\n主選（plateau 最高）", lk, "最常持有：", [(s, names.get(s, ""), int(n)) for s, n in cnt.head(10).items()])
    charts(out, tag, eqs, picks, lk, bm, M, holds, names)


def simulate_noabs(M, P, txr, txf_mom, lo, hi):
    """基準：前 5 相對動能一定持有（沒有絕對動能濾網）。"""
    P = dict(P); P["thr"] = "none"
    return simulate(M, P, txr, txf_mom, start=lo, end=hi)


def charts(out, tag, eqs, picks, lk, bm, M, holds, names):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    lo = IS0
    # 1 權益 vs 基準
    plt.figure(figsize=(11, 5.5))
    e = eqs[lk]["eq"]; e = e[e.index >= lo]
    plt.plot(e.index, e / e.iloc[0], label="DM pick {} (frac)".format(lk[4]), lw=2)
    ei = eqs[("int",) + lk]["eq"]; ei = ei[ei.index >= lo]
    plt.plot(ei.index, ei / ei.iloc[0], label="DM pick integer contracts 10M", lw=1)
    for n, r in bm.items():
        r = r[(r.index >= lo) & (r.index < OOS1)]
        plt.plot(r.index, (1 + r).cumprod(), label=n, lw=1)
    plt.axvline(IS1, color="k", ls="--", lw=0.8); plt.yscale("log"); plt.grid(alpha=.3); plt.legend(fontsize=8)
    plt.title("Dual momentum on TW stock futures vs benchmarks (log, net of costs; IS | OOS)")
    plt.tight_layout(); plt.savefig(out / f"dm_equity{tag}.png", dpi=110); plt.close()
    # 2 回撤
    plt.figure(figsize=(11, 3.8))
    for n, s in (("DM pick", e), ("TXF B&H", None)):
        if s is None:
            r = bm["TXF B&H"]; r = r[(r.index >= lo) & (r.index < OOS1)]; s = (1 + r).cumprod()
        plt.fill_between(s.index, (s / s.cummax() - 1) * 100, 0, alpha=.4, label=n)
    plt.axvline(IS1, color="k", ls="--", lw=0.8); plt.legend(); plt.grid(alpha=.3); plt.ylabel("%")
    plt.title("Drawdown"); plt.tight_layout(); plt.savefig(out / f"dm_drawdown{tag}.png", dpi=110); plt.close()
    # 3 權重法比較
    plt.figure(figsize=(11, 5.5))
    for _, p in picks.iterrows():
        k = (p.lb, p.skip, p.thr, p.mkt, p.scheme, p.vt)
        s = eqs[k]["eq"]; s = s[s.index >= lo]
        plt.plot(s.index, s / s.iloc[0], label="{} lb{} skip{} {} mkt{} vt{}".format(p.scheme, p.lb, p.skip, p.thr, int(p.mkt), p.vt))
    plt.axvline(IS1, color="k", ls="--", lw=0.8); plt.yscale("log"); plt.grid(alpha=.3); plt.legend(fontsize=8)
    plt.title("Weight schemes (IS-plateau pick per scheme), fractional contracts, net")
    plt.tight_layout(); plt.savefig(out / f"dm_schemes{tag}.png", dpi=110); plt.close()
    # 4 持有
    H = pd.DataFrame({t: h for t, h in holds}).T.fillna(0.0)
    H = H[H.index >= lo]
    top = H.sum().sort_values(ascending=False).index[:12]
    Hp = pd.concat([H[top], H.drop(columns=top).sum(axis=1).rename("others")], axis=1)
    Hp["cash"] = (1 - H.sum(axis=1)).clip(lower=0)
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.stackplot(Hp.index, Hp.T.values, labels=["{} {}".format(c, names.get(c, "")) if c in names else c for c in Hp.columns], step="post")
    ax.legend(fontsize=7, ncol=2, loc="upper left", bbox_to_anchor=(1, 1))
    ax.set_title("Holdings (weights at each rebalance), pick {}".format(lk[4])); ax.grid(alpha=.3)
    plt.rcParams["font.sans-serif"] = ["Noto Sans CJK TC", "DejaVu Sans"]
    fig.tight_layout(); fig.savefig(out / f"dm_holdings{tag}.png", dpi=110); plt.close(fig)


if __name__ == "__main__":
    main()
