"""68 深入：EMA 趨勢（只做多/空手）+ 波動目標（65 候選 EMA20/100 LF tv10%）。日K 2010-，TXF 倒推報酬（同 37/65 換月處理）。
A. 參數穩健性：EMA 對 ×  tv 8–15% × 波動估計（EWMA hl10/hl20、SD20、SD60）× 再平衡（每日／偏離>10%／>20%／每週一）。連續口數，成本率同 65。
B. 整數口數：資金 200 萬／500 萬／1000 萬，商品 TX(200/點)、MTX(50/點)、TMF(10/點)、TX+MTX 混合；
   手續費 TX/MTX 50 元/邊、TMF 20 元/邊（假設，依券商）、稅 2e-5、滑價 1 點/邊；換月兩腿成本；保證金檢查（約名目 7.5%，2026-10 TX 701,000 / 指數≈46,600）。
C. 對照：B&H 同波動（固定槓桿調到同已實現波動）、B&H + 同波動目標（無趨勢濾網）。
D. 年度表；E. 與 67 夜盤溢酬（N1500→X0500，2017-06 起）組合，各自調到 10% 波動後等風險合成。
IS 2011-2019、OOS 2020-2026/10；WF：每年以前 3 年 Sharpe 選參數。
"""
import sys, itertools, importlib
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent)); import _rq as R
M65 = importlib.import_module("65_txf_tsmom_voltarget"); M67 = importlib.import_module("67_dd_overnight_premium")


def vol(r, k):
    if k == "ewm10": v = np.sqrt((r ** 2).ewm(halflife=10, min_periods=10).mean())
    elif k == "ewm20": v = np.sqrt((r ** 2).ewm(halflife=20, min_periods=20).mean())
    elif k == "sd20": v = r.rolling(20).std()
    else: v = r.rolling(60).std()
    return v * np.sqrt(252)


def target(X, cum, f, s, tv, vk, cap=2.0):
    trend = (cum.ewm(span=f).mean() > cum.ewm(span=s).mean()).astype(float) if f else pd.Series(1.0, X.index)
    return (trend * (tv / vol(X.r.fillna(0), vk)).clip(upper=cap)).fillna(0)


def sim_cont(X, tgt, reb):
    tg = tgt.shift(1).fillna(0).values; n = len(tg); w = np.zeros(n); cur = 0.0; wd = X.index.weekday.values
    for t in range(n):
        want = tg[t]
        if reb == "daily": cur = want
        elif reb == "weekly": cur = want if (wd[t] == 0 or (t and wd[t] < wd[t - 1])) or (want == 0) != (cur == 0) else cur
        else:
            th = float(reb[3:]) / 100
            if (want == 0) != (cur == 0) or abs(want - cur) > th * max(abs(cur), 1e-9): cur = want
        w[t] = cur
    wp = np.r_[0, w[:-1]]
    ret = wp * X.gap.values + w * X.intr.values - np.abs(w - wp) * X.cr.values - np.abs(wp) * X.roll.values * 2 * X.cr.values
    worst = wp * X.gap.values + np.minimum(w * X.lo.values, w * X.hi.values)
    return pd.Series(ret, X.index), pd.Series(np.minimum(worst, 0), X.index), np.abs(w - wp).sum() / (n / 252), pd.Series(w, X.index)


def st(r, wst=None):
    eq = (1 + r).cumprod(); pk = eq.cummax(); mdd = (eq / pk - 1).min()
    if wst is not None: mdd = min(mdd, ((eq.shift(1).fillna(1) * (1 + wst)) / pk.shift(1).fillna(1) - 1).min())
    return dict(Sharpe=r.mean() / r.std() * np.sqrt(252), CAGR=eq.iloc[-1] ** (252 / len(r)) - 1, vol=r.std() * np.sqrt(252), MDD=mdd)


SPEC = {"TX": (200, 50), "MTX": (50, 50), "TMF": (10, 20)}


def sim_int(X, tgt, E0, units, thr=0.2):
    """整數口數。units: 'TX'|'MTX'|'TMF'|'TX+MTX'. 開盤依前收訊號調整（偏離 > thr 或方向切換才交易）。"""
    o, c = X.o.values, X.c.values; tg = tgt.shift(1).fillna(0).values; roll = X.roll.values; lo = X.lo.values
    parts = units.split("+"); q = {u: 0 for u in parts}; E = E0; eqs = np.zeros(len(o)); cost_tot = 0; ntr = 0; worst = np.zeros(len(o)); marg_breach = 0
    for t in range(len(o)):
        pc = c[t - 1] if t else o[t]; e0 = E
        pnl = sum(q[u] * SPEC[u][0] * (0 if roll[t] else (o[t] - pc)) for u in parts)
        if roll[t]:
            for u in parts:
                if q[u]: cc = abs(q[u]) * 2 * (SPEC[u][1] + SPEC[u][0] * (2e-5 * o[t] + 1)); pnl -= cc; cost_tot += cc; ntr += 2 * abs(q[u])
        E += pnl
        notional = tg[t] * E; cur = sum(q[u] * SPEC[u][0] for u in parts) * o[t]
        if (notional == 0) != (cur == 0) or abs(notional - cur) > thr * max(abs(cur), 1):
            newq = {}; rem = notional
            for u in parts:
                m = SPEC[u][0] * o[t]; newq[u] = int(rem // m) if u != parts[-1] else int(round(rem / m)); rem -= newq[u] * m
            for u in parts:
                dq = newq[u] - q[u]
                if dq: cc = abs(dq) * (SPEC[u][1] + SPEC[u][0] * (2e-5 * o[t] + 1)); E -= cc; cost_tot += cc; ntr += abs(dq)
            q = newq
        mval = sum(abs(q[u]) * SPEC[u][0] for u in parts) * o[t] * 0.075
        if mval > E: marg_breach += 1
        expo = sum(q[u] * SPEC[u][0] for u in parts)
        worst[t] = min(0, expo * pc * lo[t]) if t else 0
        E += expo * (c[t] - o[t]); eqs[t] = E
    eq = pd.Series(eqs, X.index); r = eq.pct_change().fillna(0)
    return eq, r, cost_tot, ntr, marg_breach, pd.Series(worst, X.index)


def main():
    X = M65.load(); cum = (1 + X.r.fillna(0)).cumprod(); X = X[X.index >= "2010-01-01"]; cum = cum.reindex(X.index)
    w11 = X.index >= "2011-01-01"; IS = X.index[w11] < "2020-01-01"
    # A. 穩健性網格
    rows = []; ser = {}
    for (f, s), tv, vk, reb in itertools.product([(10, 50), (20, 100), (20, 150), (30, 120), (50, 200)], (0.08, 0.10, 0.12, 0.15), ("ewm10", "ewm20", "sd20", "sd60"), ("daily", "thr10", "thr20", "weekly")):
        ret, wst, turn, _ = sim_cont(X, target(X, cum, f, s, tv, vk), reb); ret, wst = ret[w11], wst[w11]
        a, b = st(ret[IS], wst[IS]), st(ret[~IS], wst[~IS]); key = (f, s, tv, vk, reb); ser[key] = ret
        rows.append(dict(ema=f"{f}/{s}", tv=tv, vol=vk, reb=reb, IS=a["Sharpe"], OOS=b["Sharpe"], IS_MDD=a["MDD"], OOS_MDD=b["MDD"], OOS_CAGR=b["CAGR"], turn=turn))
    G = pd.DataFrame(rows); G.to_csv(R.OUT / "dd68_grid.csv", index=False)
    print("grid N", len(G), "IS>0 %.2f OOS>0 %.2f | IS Sharpe median %.2f [%.2f,%.2f] OOS median %.2f [%.2f,%.2f]" % ((G.IS > 0).mean(), (G.OOS > 0).mean(), G.IS.median(), G.IS.quantile(.05), G.IS.quantile(.95), G.OOS.median(), G.OOS.quantile(.05), G.OOS.quantile(.95)))
    print("Spearman IS/OOS %.3f" % G.IS.rank().corr(G.OOS.rank()))
    for k in ("ema", "tv", "vol", "reb"): print(G.groupby(k)[["IS", "OOS", "OOS_MDD", "OOS_CAGR", "turn"]].mean().round(3).to_string(), "\n")
    print("IS top 5:\n", G.sort_values("IS").tail(5).round(3).to_string())
    base_key = (20, 100, 0.10, "ewm20", "daily"); print("65 原規格:", G[(G.ema == "20/100") & (G.tv == .1) & (G.vol == "ewm20") & (G.reb == "daily")].round(3).to_string())
    wf = []; picks = []
    for Y in range(2014, 2027):
        sc = {k: v[(v.index.year >= Y - 3) & (v.index.year < Y)] for k, v in ser.items()}; sc = {k: v.mean() / v.std() for k, v in sc.items()}
        b = max(sc, key=sc.get); wf.append(ser[b][ser[b].index.year == Y]); picks.append((Y, b))
    wf = pd.concat(wf); print("WF picks", picks[-7:]); print("WF Sharpe 2014+ %.2f 2020+ %.2f" % (st(wf)["Sharpe"], st(wf[wf.index >= "2020"])["Sharpe"]))
    # C. 對照
    S0, W0, T0, w0 = sim_cont(X, target(X, cum, 20, 100, .10, "ewm20"), "daily"); S0, W0 = S0[w11], W0[w11]
    Svt, Wvt, Tvt, _ = sim_cont(X, target(X, cum, 0, 0, .10, "ewm20"), "daily"); Svt, Wvt = Svt[w11], Wvt[w11]
    bh, bw, _, _ = sim_cont(X, pd.Series(1.0, X.index), "daily"); bh, bw = bh[w11], bw[w11]
    lev = S0.std() / bh.std(); bhs = bh * lev
    print(f"\n== 對照（2011-）== B&H 同波動槓桿 {lev:.2f}")
    comp = {}
    for k, (r_, w_) in {"EMA20/100+tv10": (S0, W0), "B&H+tv10(無趨勢)": (Svt, Wvt), "B&H 同波動固定槓桿": (bhs, bw * lev), "B&H 1x": (bh, bw)}.items():
        a, b = st(r_[IS], w_[IS]), st(r_[~IS], w_[~IS]); comp[k] = r_
        print(f"{k:20s} IS Sharpe {a['Sharpe']:.2f} CAGR {a['CAGR']:.1%} MDD {a['MDD']:.1%} | OOS Sharpe {b['Sharpe']:.2f} CAGR {b['CAGR']:.1%} vol {b['vol']:.1%} MDD {b['MDD']:.1%}")
    print("time in market %.0f%%, avg leverage when in %.2f, turnover %.1fx/yr" % ((w0[w11] > 0).mean() * 100, w0[w11][w0[w11] > 0].mean(), T0))
    # D. 年度
    Y = pd.DataFrame({k: v.groupby(v.index.year).apply(lambda x: (1 + x).prod() - 1) for k, v in comp.items()})
    Y["EMA_MDD"] = S0.groupby(S0.index.year).apply(lambda x: st(x)["MDD"]); print("\n== 年度報酬 =="); print((Y * 100).round(1).to_string())
    # B. 整數口數
    print("\n== 整數口數（EMA20/100 tv10 ewm20，偏離>20% 調整）==")
    ir = []; tg = target(X, cum, 20, 100, .10, "ewm20")
    Xi = X[X.index >= "2011-01-01"]; tgi = tg.reindex(Xi.index)
    for E0, u in itertools.product((2e6, 5e6, 1e7), ("TX", "MTX", "TMF", "TX+MTX", "MTX+TMF")):
        eq, r_, cst, ntr, mb, _ = sim_int(Xi, tgi, E0, u); ii = r_.index < "2020-01-01"
        a, b = st(r_[ii]), st(r_[~ii]); idle = (r_[r_.index >= "2024-01-01"] == 0).mean()
        ir.append(dict(capital=E0, units=u, IS=a["Sharpe"], OOS=b["Sharpe"], OOS_CAGR=b["CAGR"], MDD=st(r_)["MDD"], cost_pct_yr=cst / E0 / (len(r_) / 252) * 100, trades_yr=ntr / (len(r_) / 252), idle_2024plus=idle, margin_breach_days=mb, final=eq.iloc[-1]))
    I = pd.DataFrame(ir); print(I.round(3).to_string()); I.to_csv(R.OUT / "dd68_integer.csv", index=False)
    px = X.o.iloc[-1]; tgt_now = tg.iloc[-1]
    print(f"目前（{X.index[-1].date()}）指數 {px:.0f}，目標曝險 {tgt_now:.2f}x；一口名目 TX {px*200:,.0f} / MTX {px*50:,.0f} / TMF {px*10:,.0f}")
    for E0 in (2e6, 5e6, 1e7): print(f"  資金 {E0/1e4:.0f}萬 → 目標名目 {E0*tgt_now:,.0f} = TX {E0*tgt_now/(px*200):.2f} / MTX {E0*tgt_now/(px*50):.2f} / TMF {E0*tgt_now/(px*10):.1f} 口")
    # E. 組合
    d, S, o, h, l, c = M67.build(); nt = M67.trade(S, o, h, l, c, "N1500", "X0500"); nr = (nt.net / nt.ep).reindex(S.index).fillna(0)
    nr.index = pd.DatetimeIndex(nr.index); ema = S0.copy(); J = pd.concat([ema, nr], axis=1, keys=["ema", "night"]).dropna(); J = J[J.index >= "2017-07-01"]
    sc = 0.10 / (J.std() * np.sqrt(252)); Js = J * sc; Js["combo"] = Js.mean(axis=1); Js["combo"] *= 0.10 / (Js.combo.std() * np.sqrt(252))
    print(f"\n== 與夜盤溢酬組合（2017-07~，各調 10% 波動，等風險）== 相關係數 {J.corr().iloc[0,1]:.2f}；夜盤槓桿 {sc.night:.2f}x（名目/權益）")
    ii = Js.index < "2022-01-01"
    for k in Js: a, b = st(Js[k][ii]), st(Js[k][~ii]); print(f"{k:6s} 2017-21 Sharpe {a['Sharpe']:.2f} MDD {a['MDD']:.1%} | 2022+ Sharpe {b['Sharpe']:.2f} CAGR {b['CAGR']:.1%} MDD {b['MDD']:.1%}")
    print("combo yearly %:", (Js.combo.groupby(Js.index.year).apply(lambda x: (1 + x).prod() - 1) * 100).round(1).to_dict())
    plt = R.style(); fig, ax = plt.subplots(2, 2, figsize=(16, 10))
    for k, v in comp.items(): ax[0, 0].plot((1 + v).cumprod(), label=k)
    ax[0, 0].set_yscale("log"); ax[0, 0].axvline(pd.Timestamp("2020-01-01"), c=R.GRAY, ls="--"); ax[0, 0].legend(fontsize=8); ax[0, 0].set_title("EMA20/100+tv10 vs B&H（淨值 log）")
    ax[0, 1].scatter(G.IS, G.OOS, s=8, c=R.GOLD); ax[0, 1].set_xlabel("IS Sharpe"); ax[0, 1].set_ylabel("OOS Sharpe"); ax[0, 1].set_title(f"320 組參數 IS vs OOS")
    for k in ("ema", "night", "combo"): ax[1, 0].plot((1 + Js[k]).cumprod(), label=k)
    ax[1, 0].legend(fontsize=8); ax[1, 0].set_title("組合（各 10% 波動）")
    for k, v in comp.items():
        e = (1 + v).cumprod(); ax[1, 1].plot(e / e.cummax() - 1, label=k, lw=.8)
    ax[1, 1].legend(fontsize=8); ax[1, 1].set_title("回撤")
    for a in ax.flat: a.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(R.OUT / "dd_68_ema_voltarget.png", dpi=100); print("chart results/dd_68_ema_voltarget.png")


if __name__ == "__main__":
    main()
