"""共用工具 for 70+ (reddit quant ideas, TXO 部分)：資料載入、成本、統計、黑金圖表。資料不在 repo。"""
import sys, math
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.optdaily import load_txo_daily, expiries

ROOT = Path("/workspace/fubon-threads")
FUT = ROOT / "data/TXF_1c_D_regular_2010-01-01_2026-10-22.parquet"
RES = ROOT / "results"
OCOMM, OTAX, MULT = 30.0, 0.001, 50.0          # TXO: 30 元/邊、0.1% 權利金稅、50 元/點
FCOMM, FTAX, FSLIP, FMULT = 50.0, 2e-5, 1.0, 200.0
IS_END = pd.Timestamp("2022-01-01")
GOLD, BG, GR, WH, BLUE = "#C9A227", "#111111", "#8a8a8a", "#e6e6e6", "#5fb0ff"


def tick(p):
    return 0.1 if p < 10 else 0.5 if p < 50 else 1 if p < 500 else 5 if p < 1000 else 10


def opt_cost_pts(prem):
    """單邊成本（點）：手續費 + 稅 + 1 tick 滑價。"""
    return OCOMM / MULT + OTAX * prem + tick(max(prem, 0.1))


def load_fut(start="2010-01-01"):
    d = pd.read_parquet(FUT)
    d.index = d["date"].dt.tz_localize(None).dt.normalize()
    cm = d.contractMonth.astype(str)
    d = d[["open", "high", "low", "close"]].astype(float).assign(roll=(cm.ne(cm.shift()) & cm.shift().notna()).values)
    return d[d.index >= pd.Timestamp(start)]


def fut_ret(F):
    """換月不跨合約的日報酬（換月日用 open→close）。"""
    r = np.log(F.close / F.close.shift())
    r[F.roll] = np.log(F.close / F.open)[F.roll]
    return r


def load_opt(start, end, cp=None):
    o = load_txo_daily(start, end, cp=cp, columns=["date", "expiry", "strike", "cp", "day_open", "day_high", "day_low", "day_close", "day_volume"])
    o["date"] = pd.to_datetime(o.date).dt.normalize()
    return o[o.expiry.str.match(r"^\d{6}([WF]\d)?$")]


def sharpe(x, ann=252):
    x = np.asarray(x, float); s = x.std()
    return float(x.mean() / s * math.sqrt(ann)) if s > 0 else 0.0


def mdd(eq):
    eq = np.asarray(eq, float); return float((eq - np.maximum.accumulate(np.r_[0, eq])[1:]).min())


def plt_style():
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = "Noto Sans CJK TC"; plt.rcParams["axes.unicode_minus"] = False
    return plt


def sty(ax):
    ax.set_facecolor(BG); ax.tick_params(colors=WH); ax.grid(alpha=.15, color=GR)
    for s in ax.spines.values(): s.set_color(GR)
    ax.xaxis.label.set_color(WH); ax.yaxis.label.set_color(WH); ax.title.set_color(GOLD)


def foot(fig):
    fig.text(0.98, 0.01, "夜盤貓頭鷹｜含成本與滑價｜回測不代表未來績效", color=GOLD, ha="right", fontsize=9)


def garch11(r, params=None):
    """GARCH(1,1) 條件變異（MLE 只用傳入的 fit 樣本）。r: np array（日報酬，去均值）。"""
    from scipy.optimize import minimize
    r = np.asarray(r, float)
    def filt(p, x):
        w, a, b = p; h = np.empty(len(x)); h[0] = x.var()
        for t in range(1, len(x)): h[t] = w + a * x[t - 1] ** 2 + b * h[t - 1]
        return h
    if params is None:
        v = r.var()
        def nll(p):
            w, a, b = p
            if w <= 0 or a < 0 or b < 0 or a + b >= .999: return 1e10
            h = filt(p, r); return 0.5 * np.sum(np.log(h) + r ** 2 / h)
        params = minimize(nll, [v * .05, .08, .9], method="Nelder-Mead", options=dict(maxiter=4000)).x
    return params, filt(params, r)


def load_atm_iv(ok_only=True):
    """TXO 每日 ATM IV（由 examples/78_txo_atm_iv.py 產生）。index=date；atm_iv_* 為小數（0.2=20%）。"""
    d = pd.read_parquet(ROOT / "data/txo_atm_iv_daily.parquet")
    if ok_only: d = d[d.ok]
    return d.set_index("date")
