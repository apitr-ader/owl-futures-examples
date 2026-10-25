"""共用現股月/週換股回測引擎（42/43/44 共用），含成交模式選項。

fill="next_open"（預設）：第 t 日收盤算訊號，所有進出場（換股、停損、停利）於 t+1 開盤成交；
  當日報酬 = 前收→開盤（舊部位）× 成本 × 開盤→收盤（新部位）。開盤價用調整開盤（與調整收盤同因子）。
  漲跌停：未還原開盤 >= 漲停價（參考價×(1+限幅) 向下取 tick）不能買、<= 跌停價不能賣 → 該檔委託保留，
  下一個交易日開盤重試（直到成交或被新的換股訊號取代）；停牌（無成交）同樣隔日重試；已下市以最後價格出清。
  限幅：2015-06-01 前 7%，之後 10%。參考價 = 前一日調整收盤 × 當日還原因子（除權息日即除權息參考價）。
fill="close"：舊做法，t 日收盤成交、無漲跌停限制（僅供比較）。
價格：調整收盤向前填補（停牌期間報酬 0，復牌日計入停牌期間累積報酬；舊 42 版本此段報酬遺失）。
成本：手續費 comm/邊、證交稅 tax（賣）、滑價 1 tick/邊（以成交時未還原價格計）。
"""
import argparse, math
import numpy as np, pandas as pd

FILL_MODES = ("next_open", "close")


def fill_arg(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("--fill", choices=FILL_MODES, default="next_open")
    return ap.parse_known_args(argv)[0].fill


def tick(p):
    return np.select([p < 10, p < 50, p < 100, p < 500, p < 1000], [0.01, 0.05, 0.1, 0.5, 1.0], 5.0)


def lw_cov(X):
    X = X - X.mean(0); n, p = X.shape; S = X.T @ X / n
    mu = np.trace(S) / p; F = mu * np.eye(p)
    d2 = ((S - F) ** 2).sum(); b2 = min(sum(((np.outer(x, x) - S) ** 2).sum() for x in X) / n ** 2, d2)
    a = b2 / d2 if d2 > 0 else 1.0
    return a * F + (1 - a) * S


def prepare(sd, start="2008-01-01"):
    adj = sd.panel("close", adjusted=True, start=start)
    g = lambda f, a: sd.panel(f, adjusted=a, start=start).reindex_like(adj).values
    P, tov, AO, PO = g("close", False), g("turnover", False), g("open", True), g("open", False)
    A = adj.values; traded = ~np.isnan(P)
    Af = pd.DataFrame(A).ffill().values; Pf = pd.DataFrame(P).ffill().values
    R = np.vstack([np.full(A.shape[1], np.nan), Af[1:] / Af[:-1] - 1])      # close-to-close, gaps carried
    R0 = np.where(traded, np.nan_to_num(R), 0.0)
    Rv = np.where(traded, R, np.nan)                                        # for vol estimates
    days = adj.index
    idx = np.arange(len(days))[:, None]
    last = np.where(traded, idx, -1).max(0)
    # limit prices
    ref = np.vstack([np.full(A.shape[1], np.nan), Af[:-1]]) * (P / A)       # raw reference price
    lim = np.where(days < pd.Timestamp("2015-06-01"), 0.07, 0.10)[:, None]
    with np.errstate(invalid="ignore", divide="ignore"):
        up = ref * (1 + lim); dn = ref * (1 - lim)
        up = np.floor(up / tick(up) + 1e-9) * tick(up); dn = np.ceil(dn / tick(dn) - 1e-9) * tick(dn)
        lu = traded & (PO >= up - 1e-9); ld = traded & (PO <= dn + 1e-9)
    t20 = pd.DataFrame(np.where(traded, tov, 0.0)).rolling(20).mean().values
    return dict(days=days, cols=adj.columns, A=Af, AO=np.where(traded, AO, np.nan), P=Pf, PO=PO, R=Rv, R0=R0, traded=traded,
                nhist=np.cumsum(traded, 0), t20=t20, last=last, lu=lu, ld=ld)


def run(D, sig, legs=(21, 63, 126), freq="M", N=10, liq=5e7, vt=None, stop=None, trail=None, act=0.10, rawrank=False, rev=False,
        absm=True, rf=0.015, comm=0.001425 * 0.5, tax=0.003, fill="next_open", start=pd.Timestamp("2010-01-01")):
    days, A, AO, P, PO, R, R0, traded = D["days"], D["A"], D["AO"], D["P"], D["PO"], D["R"], D["R0"], D["traded"]
    nhist, t20, last, lu, ld = D["nhist"], D["t20"], D["last"], D["lu"], D["ld"]
    nd, nS = A.shape
    if freq == "M":
        reb = np.r_[days[1:].month != days[:-1].month, True]
    else:
        wk = days.to_period("W-SUN"); reb = np.r_[wk[1:] != wk[:-1], True]
    thr = (1 + rf) ** (np.mean(legs) / 252) - 1
    no = fill == "next_open"
    w = np.zeros(nS); pend = np.full(nS, np.nan); E = np.full(nS, np.nan); G = np.zeros(nS)
    out = np.zeros(nd); expo = np.zeros(nd); turn = cost = 0.0
    ev = dict(pos=0, stop=0, trail=0, blocked_up=0, blocked_dn=0, blocked_susp=0, orders=0)

    def tc(d, px):
        px = np.where(np.isfinite(px) & (px > 0), px, 1.0); s = tick(px) / px
        return (np.clip(d, 0, None) * (comm + s)).sum() + (np.clip(-d, 0, None) * (comm + tax + s)).sum()

    def drift(w, r):
        pr = w @ r; g = 1 + pr
        return (w * (1 + r) / g if g > 0 else w * 0), pr

    t0 = max(130, int(np.searchsorted(days, start - pd.Timedelta(days=40))))
    for t in range(1, nd):
        c = 0.0
        if no:
            ro = np.where(traded[t], AO[t] / A[t - 1] - 1, 0.0); ro = np.nan_to_num(ro)
            w, p1 = drift(w, ro)
            ip = np.where(~np.isnan(pend))[0]
            if len(ip):
                d = pend[ip] - w[ip]; buy = d > 0
                dl = t > last[ip]
                okx = (traded[t, ip] & ~(buy & lu[t, ip]) & ~(~buy & ld[t, ip])) | (dl & ~buy)
                ev["blocked_up"] += int((traded[t, ip] & buy & lu[t, ip]).sum())
                ev["blocked_dn"] += int((traded[t, ip] & ~buy & ld[t, ip]).sum())
                ev["blocked_susp"] += int((~traded[t, ip] & ~(dl & ~buy)).sum())
                ii = ip[okx]
                if len(ii):
                    px = np.where(traded[t, ii], PO[t, ii], P[t, ii])
                    dd = pend[ii] - w[ii]; c = tc(dd, px); turn += np.abs(dd).sum() / 2
                    pos = pend[ii] > 0          # entry price reset at every (re)balance fill, as in 43
                    E[ii[pos]] = AO[t, ii[pos]]; G[ii[pos]] = 0.0; E[ii[~pos]] = np.nan
                    w[ii] = pend[ii]; pend[ii] = np.nan
            rc = np.where(traded[t], A[t] / AO[t] - 1, 0.0); rc = np.nan_to_num(rc)
            w, p2 = drift(w, rc)
            out[t] = (1 + p1) * (1 - c) * (1 + p2) - 1
        else:
            w, out[t] = drift(w, R0[t])
        cost += c
        # exits (signal at close)
        sells = np.zeros(nS, bool)
        if (stop or trail) and not reb[t]:
            h = np.where((w > 0) & traded[t] & np.isnan(pend) & np.isfinite(E))[0]
            if len(h):
                g = A[t, h] / E[h] - 1; G[h] = np.maximum(G[h], g)
                hs = (g <= -stop) if stop else np.zeros(len(h), bool)
                ht = ((G[h] >= act) & (g <= (1 - trail) * G[h]) & ~hs) if trail else np.zeros(len(h), bool)
                sells[h[hs | ht]] = True; ev["stop"] += int(hs.sum()); ev["trail"] += int(ht.sum())
        tgt = None
        if reb[t] and t >= t0:
            tgt = np.zeros(nS)
            if sig[t]:
                ok = np.where(traded[t] & (nhist[t] >= 127) & (t20[t] >= liq))[0]
                raw_m = np.zeros(len(ok)); vs = np.zeros(len(ok))
                for L in legs:
                    rL = A[t, ok] / A[t - L, ok] - 1
                    v = np.nanstd(R[t - L + 1:t + 1, ok], 0) * math.sqrt(L)
                    raw_m += rL; vs += rL / np.maximum(v, 1e-6)
                raw_m /= len(legs); vs /= len(legs)
                fin = np.isfinite(vs) & np.isfinite(raw_m)
                score = raw_m if rawrank else vs
                if rev:
                    cand = ok[fin][np.argsort(score[fin])][:N]
                else:
                    good = fin & ((raw_m > thr) if absm else True)
                    cand = ok[good][np.argsort(-score[good])][:N]
                if len(cand):
                    tgt[cand] = 1.0 / N
                    if vt:
                        C = lw_cov(np.nan_to_num(R[t - 59:t + 1, cand]))
                        pv = math.sqrt(max(tgt[cand] @ C @ tgt[cand], 1e-12) * 252)
                        tgt[cand] *= min(vt / pv, 1.0)
            ev["pos"] += int(((tgt > 0) & (w <= 1e-12)).sum())
        if no:
            if tgt is not None:
                ch = (np.abs(tgt - w) > 1e-12) | ~np.isnan(pend)
                pend = np.where(ch, tgt, np.nan); ev["orders"] += int(ch.sum())
            pend[sells] = 0.0; ev["orders"] += int(sells.sum())
        else:
            if tgt is None and sells.any():
                tgt = w.copy(); tgt[sells] = 0.0
            if tgt is not None:
                d = tgt - w; c = tc(d, P[t]); out[t] -= c; cost += c; turn += np.abs(d).sum() / 2
                if reb[t]:
                    E[tgt > 0] = A[t, tgt > 0]; G[tgt > 0] = 0.0
                E[tgt <= 0] = np.nan
                w = tgt
        expo[t] = w.sum()
    keep = days >= start
    return pd.Series(out, days)[keep], pd.Series(expo, days)[keep], turn, cost, ev
