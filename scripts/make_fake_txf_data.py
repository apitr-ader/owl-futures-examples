"""產生「假的」台指期資料，格式同 07（1 分 K）／09（日 K），只用來測試 10、11 能不能跑。

不是真實行情！數字是隨機漫步＋換月跳空，請勿拿來做任何判斷。

  python scripts/make_fake_txf_data.py            # 產生 data/FAKE_TXF_D.parquet 與 data/FAKE_TXF_1m.parquet
"""
import numpy as np
import pandas as pd
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "data"
rng = np.random.default_rng(42)


def settle_month(d):
    """第三個星期三後換月：回傳當天所屬近月 yyyyMM。"""
    first = d.replace(day=1)
    third_wed = first + pd.Timedelta(days=(2 - first.dayofweek) % 7 + 14)
    m = d if d <= third_wed else d + pd.offsets.MonthBegin(1)
    return m.strftime("%Y%m")


def daily(start="2010-01-04", end="2026-10-20"):
    days = pd.bdate_range(start, end)
    ret = rng.standard_t(4, len(days)) * 0.011 + 0.0002
    # 加一點趨勢段，讓海龜有東西可抓
    ret += np.repeat(rng.normal(0, 0.002, len(days) // 60 + 1), 60)[: len(days)]
    close = 7800 * np.exp(np.cumsum(ret))
    cm = [settle_month(d) for d in days]
    roll = np.r_[False, np.array(cm[1:]) != np.array(cm[:-1])]
    close = close + np.cumsum(np.where(roll, rng.normal(-30, 20, len(days)), 0))  # 換月價差
    open_ = np.r_[close[0], close[:-1]] * np.exp(rng.normal(0, 0.004, len(days)))
    hi = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.006, len(days))))
    lo = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.006, len(days))))
    df = pd.DataFrame({"date": days.tz_localize("Asia/Taipei"), "session": "regular", "contractMonth": cm,
                       "open": open_.round(), "high": hi.round(), "low": lo.round(), "close": close.round(),
                       "volume": rng.integers(50_000, 200_000, len(days))})
    df["change"] = df["close"].diff().fillna(0)
    return df


def minute(start="2025-01-02", end="2026-10-20"):
    rows = []
    px = 22000.0
    for d in pd.bdate_range(start, end):
        cm = settle_month(d)
        for sess, t0, n in (("afterhours", d - pd.Timedelta(days=1) + pd.Timedelta(hours=15), 840),
                            ("regular", d + pd.Timedelta(hours=8, minutes=45), 300)):
            if d.dayofweek == 0 and sess == "afterhours":
                t0 = d - pd.Timedelta(days=3) + pd.Timedelta(hours=15)  # 週五夜盤歸屬週一
            steps = rng.normal(0.02, 6, n)
            c = px + np.cumsum(steps)
            o = np.r_[px, c[:-1]]
            h = np.maximum(o, c) + np.abs(rng.normal(0, 3, n))
            l = np.minimum(o, c) - np.abs(rng.normal(0, 3, n))
            ts = pd.date_range(t0, periods=n, freq="min", tz="Asia/Taipei")
            rows.append(pd.DataFrame({"date": ts, "session": sess, "contractMonth": cm,
                                      "open": o.round(), "high": h.round(), "low": l.round(),
                                      "close": c.round(), "volume": rng.integers(10, 800, n)}))
            px = c[-1]
    return pd.concat(rows, ignore_index=True)


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    d = daily()
    d.to_parquet(OUT / "FAKE_TXF_D.parquet", index=False)
    d.assign(date=d["date"].dt.strftime("%Y-%m-%d")).to_csv(OUT / "FAKE_TXF_D.csv", index=False)
    m = minute()
    m.to_parquet(OUT / "FAKE_TXF_1m.parquet", index=False)
    print("日 K {} 根、1 分 K {} 根 → {}".format(len(d), len(m), OUT))
