"""30 診斷：為何主要策略沒選到強勢股（不修改 21-29）

策略（到期換倉框架，同 26-28）：27 pick（vol-scaled 複合）、28 s126、26 lb20 每到期、26 DM lb252 每季。
每年強勢股 = 各標的（同標的取標準契約，無則小型）調整連續契約的年度總報酬前 10，另加 2330/3661/2454/2382/3017/2308/6669。
每個到期期間記錄：是否在可選池（不在的原因）、分數排名、是否過絕對濾網、大盤濾網是否強制空手、是否持有、權重；
彙總空手比例、強勢股狀態分佈、捕捉報酬、實際持有標的、各假說的量化。輸出 results/why_*.csv/png。
"""
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

_s = importlib.util.spec_from_file_location("mv28", Path(__file__).with_name("28_mom_over_vol.py"))
m28 = importlib.util.module_from_spec(_s); _s.loader.exec_module(m28)
m26, m27, o22, dm, D = m28.m26, m28.m27, m28.o22, m28.dm, m28.D
OUT = Path("results")
N = 5
FOCUS = ["2330", "3661", "2454", "2382", "3017", "2308", "6669"]
STRATS = {
    "27 vol-scaled composite": ("CMP", dict(skip=0, thr="rf", vs=1, cad=1, buf=0, mkt=1, vt=0.15, scheme="EW", lb=126)),
    "28 s126": ("MOV", dict(sc="s126", thr="rf", cad=1, buf=0, mkt=1, vt=0.15, scheme="EW", lb=126, win=252)),
    "26 lb20 monthly": ("DMS", dict(lb=20, skip=0, cad=1, buf=1, mkt=1, vt=0.15, scheme="EW")),
    "26 DM lb252 quarterly": ("DM", dict(lb=252, skip=21, cad=3, buf=0, mkt=0, vt=0.15, scheme="EW")),
}
ALT = {  # 假說測試用的排名分數（同一可選池）
    "raw composite": ("CMP", dict(skip=0, thr="rf", vs=0, lb=126)),
    "vol-scaled composite": ("CMP", dict(skip=0, thr="rf", vs=1, lb=126)),
    "composite/vol126": ("MOV", dict(sc="s126", thr="rf", lb=126)),
    "63+126 only (no 1m leg)": None,
    "252d return": ("DM", dict(lb=252, skip=0)),
}


def setup():
    o22.prepare("data/stock_futures", "data/TXF_1c_D_regular_2010-01-01_2026-10-22.parquet")
    D["txv"] = D["txr"].values; D["txr_s"] = D["txr"]; D["exp"] = m26.expiry_calendar()
    tI = (1 + D["txr"]).cumprod()
    D["txf_cmp"] = np.mean([(tI / tI.shift(L) - 1).values for L in m27.LEGS], 0)
    D["txf_252"] = D["txf_mom"].copy()


def trace(fam, P):
    """複製 26.simulate 的選股邏輯，逐到期期間記錄。"""
    days, X, nd = D["days"], D["exp"], len(D["days"])
    X = [x for x in X if days[x] >= pd.Timestamp("2013-11-01")]
    m28.market(fam)
    held, w_held, recs = [], None, []
    for n, x in enumerate(X):
        k0 = x + 1
        if k0 >= nd:
            break
        k1 = X[n + 1] + 1 if n + 1 < len(X) else nd
        rec = dict(n=n, x=x, k0=k0, k1=k1, date=days[x], rebal=(n % P["cad"] == 0))
        if rec["rebal"]:
            U = o22.univ(x, 50, 0)
            need = max(P["lb"] + 1, P.get("win", 0))
            mcash = bool(P["mkt"] and not D["txf_mom"][x] > 0)
            sel, rank, okd = [], {}, {}
            if len(U) >= dm.MIN_UNIVERSE and x - need >= 0:
                sc, ok = m26.score(fam, P, x, U)
                rk = U[np.argsort(-sc)]
                rank = {int(s): r + 1 for r, s in enumerate(rk)}
                okd = {int(s): bool(o) for s, o in zip(U, ok)}
                if not mcash:
                    pos = {s: r - 1 for s, r in rank.items()}
                    keep = [s for s in held if s in pos and pos[s] < 2 * N and okd[s]][:N] if P.get("buf") else []
                    sel = keep + [int(s) for s in rk if int(s) not in keep and okd[int(s)]][:N - len(keep)]
            sel = sorted(sel)
            w = o22.wts(x, P["scheme"], tuple(sel), len(sel) / N, P["vt"]) if sel else np.zeros(0)
            held, w_held = sel, w
            rec.update(U=set(int(u) for u in U), rank=rank, ok=okd, mcash=mcash)
        else:
            rec.update(U=recs[-1]["U"], rank=recs[-1]["rank"], ok=recs[-1]["ok"], mcash=recs[-1]["mcash"])
        rec.update(sel=list(held), w=dict(zip(held, np.asarray(w_held, float))) if held else {})
        recs.append(rec)
    return recs


def main():
    setup()
    days, cols, I = D["days"], D["cols"], D["I"]
    M = D["M"]
    und = pd.Series(D["und"], index=range(len(cols)))
    mini = D["M"]["mini"].values.astype(bool)
    prod = pd.read_csv("data/stock_futures/products.csv", dtype={"underlying_code": str})
    # 每個標的的代表契約：標準優先，且有資料
    rep = {}
    for j in np.lexsort((mini, D["und"])):
        rep.setdefault(D["und"][j], j)
    sym_of = {j: cols[j] for j in range(len(cols))}
    ye = pd.Series(range(len(days)), days).groupby(days.year).last()
    yrs = list(range(2014, 2027))
    # 年度報酬（調整連續契約，以 I 指數；當年無交易/未上市則 NaN）
    traded = M["traded"].values
    rows = []
    for u, j in rep.items():
        for y in yrs:
            a = ye.get(y - 1); b = ye.get(y)
            if a is None or b is None:
                continue
            t = traded[a + 1:b + 1, j]
            if t.sum() < 20:
                continue
            first = a + 1 + np.argmax(t)
            rows.append(dict(und=u, sym=cols[j], j=j, year=y, ret=I[b, j] / I[max(a, first - 1), j] - 1, tdays=int(t.sum())))
    yr = pd.DataFrame(rows)
    top = yr.sort_values("ret", ascending=False).groupby("year").head(10).assign(top10=True)
    foc = yr[yr.und.isin(FOCUS)]
    W = pd.concat([top, foc]).drop_duplicates(["und", "year"]).fillna({"top10": False})
    missing_focus = [u for u in FOCUS if u not in set(yr.und)]
    nm = prod.drop_duplicates("underlying_code").set_index("underlying_code").name.str.replace("期貨", "")

    # 未載入的商品（歷史 < 252 根 或 代號結尾 1 / ETF）
    loaded = set(cols)

    def reason(j, x, U, rank, okd, mcash, sel):
        u = D["und"][j]
        sib = [s for s in U if D["und"][s] == u]
        if not sib:
            if not traded[x, j]:
                return "not trading yet / no trade", None
            if M["nhist"].values[x, j] < dm.MIN_HIST:
                return "history<252d", None
            if D["medvol"][x, j] < 50:
                return "median vol<50", None
            if np.nan_to_num(M["GAP"].values[x, j]) > dm.GAP_OUTLIER:
                return "roll-gap>15% (permanent)", None
            return "other", None
        s = sib[0]
        if s in sel:
            return "held", s
        if mcash:
            return "market filter cash", s
        if not okd.get(s, True):
            return "failed abs filter", s
        return "ranked below top5", s

    allrec, summ, held_rows, cap_rows = [], [], [], []
    for name, (fam, P) in STRATS.items():
        R = trace(fam, P)
        Rd = pd.DataFrame([dict(date=r["date"], mcash=r["mcash"], nsel=len(r["sel"]), wsum=sum(r["w"].values())) for r in R])
        for per, lo, hi in (("IS", "2014-01-01", "2020-01-01"), ("OOS", "2020-01-01", "2026-11-01")):
            q = Rd[(Rd.date >= lo) & (Rd.date < hi)]
            summ.append(dict(strategy=name, period=per, periods=len(q), mkt_cash_share=q.mcash.mean(), empty_slot_share=1 - q.nsel.mean() / N,
                             avg_gross_weight=q.wsum.mean()))
        for r in R:
            per_ret = I[r["k1"] - 1] / I[r["x"]] - 1
            y = r["date"].year if r["date"].month < 12 else r["date"].year + 1  # 期間主要落在下一月
            y = days[min(r["k1"] - 1, len(days) - 1)].year
            for wv in W[W.year == y].itertuples():
                st, s = reason(wv.j, r["x"], r["U"], r["rank"], r["ok"], r["mcash"], r["sel"])
                allrec.append(dict(strategy=name, year=y, und=wv.und, name=nm.get(wv.und, ""), top10=wv.top10, date=r["date"].date(), status=st,
                                   rank=r["rank"].get(s) if s is not None else np.nan, n_univ=len(r["U"]), weight=r["w"].get(s, 0.0) if s is not None else 0.0,
                                   period_ret=per_ret[wv.j], contrib=(r["w"].get(s, 0.0) if s is not None else 0.0) * per_ret[s if s is not None else wv.j]))
            for s, wv in r["w"].items():
                held_rows.append(dict(strategy=name, year=days[min(r["k1"] - 1, len(days) - 1)].year, und=D["und"][s], sym=cols[s], weight=wv,
                                      period_ret=per_ret[s], contrib=wv * per_ret[s]))
    A = pd.DataFrame(allrec); A.to_csv(OUT / "why_winner_months.csv", index=False)
    S = pd.DataFrame(summ)
    H = pd.DataFrame(held_rows)
    # 強勢股狀態分佈
    st = A[A.top10].groupby(["strategy", A.year >= 2020]).status.value_counts(normalize=True).unstack().fillna(0)
    st.index = st.index.set_levels(["IS", "OOS"], level=1)
    st.to_csv(OUT / "why_status_share.csv")
    # 捕捉：持有期間報酬 vs 全年
    def cap(g):
        held = g[g.status == "held"]
        return pd.Series(dict(year_ret=np.prod(1 + g.period_ret) - 1, held_periods=len(held), periods=len(g),
                              ret_while_held=np.prod(1 + held.period_ret) - 1 if len(held) else 0.0,
                              weighted_contrib=g.contrib.sum(), avg_w_when_held=held.weight.mean() if len(held) else np.nan,
                              best_rank=g["rank"].min(), median_rank=g["rank"].median(), n_univ=g.n_univ.median(),
                              main_status=g.status.value_counts().index[0]))
    C = A.groupby(["strategy", "year", "und", "name", "top10"]).apply(cap).reset_index()
    C.to_csv(OUT / "why_winners_capture.csv", index=False)
    # 實際持有
    HH = H.groupby(["strategy", "year", "und"]).agg(periods=("weight", "size"), avg_w=("weight", "mean"), ret_held=("period_ret", lambda r: np.prod(1 + r) - 1),
                                                    contrib=("contrib", "sum")).reset_index()
    HH["name"] = HH.und.map(nm)
    HH = HH.merge(yr[["und", "year", "ret"]].rename(columns={"ret": "year_ret"}), on=["und", "year"], how="left")
    HH.sort_values(["strategy", "year", "periods"], ascending=[True, True, False]).to_csv(OUT / "why_held.csv", index=False)

    # 假說：同一可選池下，強勢股在不同分數的排名百分位（每月，強勢股「之前」與「當中」）
    hyp = []
    alt_scores = {}
    for x in sorted(set(r for r in D["exp"] if days[r] >= pd.Timestamp("2014-01-01"))):
        U = o22.univ(x, 50, 0)
        if len(U) < 10 or x < 260:
            continue
        sc = {}
        for k, v in ALT.items():
            if v is None:
                s = np.mean([I[x, U] / I[x - L, U] - 1 for L in (63, 126)], 0)
            else:
                s = m26.score(v[0], v[1], x, U)[0]
            sc[k] = pd.Series(s, index=U).rank(ascending=False)
        alt_scores[x] = (U, sc)
    for wv in W[W.top10].itertuples():
        for x, (U, sc) in alt_scores.items():
            if days[x].year != wv.year:
                continue
            s = [u for u in U if D["und"][u] == wv.und]
            if not s:
                continue
            row = dict(year=wv.year, und=wv.und, date=days[x].date(), n_univ=len(U),
                       vol63=D["ret"][x - 62:x + 1, s[0]].std() * np.sqrt(252))
            row.update({k: v[s[0]] for k, v in sc.items()})
            hyp.append(row)
    Hy = pd.DataFrame(hyp); Hy.to_csv(OUT / "why_rank_by_score.csv", index=False)
    # 可選池其他股票的 63 日波動中位數（比較強勢股波動）
    uv = np.median([np.median(D["ret"][x - 62:x + 1, U].std(0) * np.sqrt(252)) for x, (U, _) in alt_scores.items()])
    # 進場延遲：強勢股年度第一次進入 27 pick 前 5 名的月份 vs 年內最大漲幅起點
    # 輪出太早：持有後下一期被賣出，之後同年剩餘報酬
    rot = []
    for name in STRATS:
        a = A[(A.strategy == name) & A.top10].sort_values("date")
        for (y, u), g in a.groupby(["year", "und"]):
            h = (g.status == "held").values
            pr = g.period_ret.values
            for k in range(1, len(g)):
                if h[k - 1] and not h[k]:
                    rot.append(dict(strategy=name, year=y, und=u, exit=g.date.values[k], ret_after_exit_same_year=np.prod(1 + pr[k:]) - 1,
                                    status_after=g.status.values[k]))
    Ro = pd.DataFrame(rot); Ro.to_csv(OUT / "why_rotation.csv", index=False)

    pd.set_option("display.width", 260); pd.set_option("display.max_rows", 400)
    print("missing focus names (no stock-future data loaded):", missing_focus)
    print("\n== cash / slots ==\n", S.round(3).to_string())
    print("\n== top-10 winners: status share of periods ==\n", (st * 100).round(1).to_string())
    k = C[C.top10 & (C.year >= 2020)]
    print("\n== OOS top-10 winners capture (per strategy) ==")
    print(k.groupby("strategy").agg(winners=("und", "size"), held_any=("held_periods", lambda s: (s > 0).mean()),
                                    held_period_share=("held_periods", "sum"), periods=("periods", "sum"),
                                    med_year_ret=("year_ret", "median"), med_ret_while_held=("ret_while_held", "median"),
                                    sum_contrib=("weighted_contrib", "sum"), avg_w=("avg_w_when_held", "mean")).round(3).to_string())
    print("\n== hypotheses: median rank of top-10 winners (OOS months), universe median size", int(Hy.n_univ.median()), "==")
    o = Hy[Hy.year >= 2020]
    print(o[list(ALT)].median().round(1).to_string(), "\n top5 share:", (o[list(ALT)] <= 5).mean().round(3).to_dict())
    print(f" winners' 63d vol median {o.vol63.median():.2f} vs universe median {uv:.2f}")
    print("\n== rotation exits (top-10 winners, OOS) ==\n", Ro[Ro.year >= 2020].groupby("strategy").agg(exits=("und", "size"),
          med_ret_after=("ret_after_exit_same_year", "median"), status_after=("status_after", lambda s: s.value_counts().to_dict())).to_string())
    print("\n== per-year table (27 pick & 26 lb20) ==")
    for name in ("27 vol-scaled composite", "26 lb20 monthly"):
        t = C[(C.strategy == name) & (C.year >= 2020) & (C.top10 | C.und.isin(FOCUS))].sort_values(["year", "year_ret"], ascending=[True, False])
        t = t.groupby("year").head(6)
        print(name); print(t[["year", "und", "name", "year_ret", "main_status", "held_periods", "periods", "best_rank", "median_rank", "ret_while_held", "weighted_contrib"]].round(3).to_string(index=False))
    for name in STRATS:
        h = HH[(HH.strategy == name) & (HH.year >= 2020)].groupby(["und", "name"]).agg(periods=("periods", "sum"), contrib=("contrib", "sum")).sort_values("periods", ascending=False).head(8)
        print(f"\nmost held OOS {name}:\n", h.round(3).to_string())
    # focus names
    print("\n== focus names (27 pick), status by year ==")
    f = A[(A.strategy == "27 vol-scaled composite") & A.und.isin(FOCUS)]
    print(f.groupby(["und", "year"]).status.agg(lambda s: s.value_counts().to_dict()).unstack(0).to_string())

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for name in ("27 vol-scaled composite", "26 lb20 monthly"):
        a = A[(A.strategy == name) & A.top10 & (A.year >= 2020)].copy()
        a["key"] = a.year.astype(str) + " " + a.und + " " + a.name
        a["v"] = np.where(a.status == "held", 0, a["rank"])
        piv = a.pivot_table(index="key", columns="date", values="v", aggfunc="first")
        stat = a.pivot_table(index="key", columns="date", values="status", aggfunc="first")
        fig, ax = plt.subplots(figsize=(18, 14))
        im = ax.imshow(np.clip(piv.values, 0, 60), aspect="auto", cmap="viridis_r", vmin=0, vmax=60)
        for (i, j), s in np.ndenumerate(stat.values):
            if s == "held":
                ax.text(j, i, "H", ha="center", va="center", fontsize=5, color="red")
            elif isinstance(s, str) and s not in ("ranked below top5", "failed abs filter", "market filter cash"):
                ax.text(j, i, "x", ha="center", va="center", fontsize=5, color="k")
            elif s == "market filter cash":
                ax.text(j, i, "c", ha="center", va="center", fontsize=5, color="w")
        ax.set_yticks(range(len(piv))); ax.set_yticklabels(piv.index, fontsize=6)
        ax.set_xticks(range(0, len(piv.columns), 3)); ax.set_xticklabels([str(d)[:7] for d in piv.columns[::3]], rotation=90, fontsize=6)
        plt.colorbar(im, ax=ax, label="rank by score (H=held, x=not in universe, c=market-filter cash)")
        ax.set_title(f"{name}: OOS yearly top-10 winners, rank at each expiry")
        plt.tight_layout(); plt.savefig(OUT / f"why_rank_heatmap_{name.split()[0]}.png", dpi=110); plt.close()
    fig, ax = plt.subplots(figsize=(10, 4.5))
    (st.xs("OOS", level=1) * 100).plot.barh(stacked=True, ax=ax)
    ax.set_xlabel("% of winner-periods (OOS yearly top-10)"); ax.legend(fontsize=7, loc="lower right"); plt.tight_layout()
    plt.savefig(OUT / "why_status_share.png", dpi=110); plt.close()
    S.to_csv(OUT / "why_cash.csv", index=False)


if __name__ == "__main__":
    main()
