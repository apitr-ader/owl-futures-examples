"""回測共用：把 07（分 K）或 09（日 K）下載的 Parquet／CSV 讀成 OHLCV DataFrame。

回傳：index 為台北時間（Asia/Taipei）的 DatetimeIndex，欄位 Open/High/Low/Close/Volume
（backtesting.py 要求的大寫欄名，Backtrader 的 PandasData 也吃），另保留 contractMonth（若有）。

換月（連續合約 1!）注意事項：
- 官方連續合約「不做換月價差調整」，換月那根 K 線會跳空（近遠月價差，常見幾十點）。
- roll_adjust="none"（預設）：照原始價格，跳空會被當成真的漲跌，可能誤觸突破或停損。
- roll_adjust="back"：倒推調整（back-adjust）。因為 API 只給 1! 的價格，拿不到換月當下
  新舊合約的同時價格，這裡用「新合約第一根開盤 − 舊合約最後一根收盤」近似價差，
  會把那一晚真實的隔夜跳空一起抹掉，只是近似。調整後早年價格會偏離真實價位，
  點數損益不受影響，但以價格計算的期交稅（10、11 用成交價算）會有些微誤差。
  要精準請另外下載各月份合約（contractMonth=yyyyMM）自行接合。
"""

from pathlib import Path

import pandas as pd

TZ = "Asia/Taipei"


def _read(path):
    path = Path(path)
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path, dtype={"contractMonth": "string"})


def _trading_day(df):
    """把分 K 歸到交易日：夜盤（15:00 後與隔天凌晨）歸屬次一交易日（期交所規則）。"""
    ts = df.index
    cal = ts.normalize()
    is_night_eve = ts.hour >= 15
    # 先以「日盤出現過的日期」當交易日曆；沒有日盤資料時退回平日
    if "session" in df and (df["session"].astype(str) == "regular").any():
        days = pd.DatetimeIndex(sorted(cal[df["session"].astype(str) == "regular"].unique()))
    else:
        days = pd.DatetimeIndex(sorted(cal[cal.dayofweek < 5].unique()))
    target = cal.where(~is_night_eve, cal + pd.Timedelta(days=1))
    pos = days.searchsorted(target)  # 第一個 >= target 的交易日
    pos = pos.clip(max=len(days) - 1)
    return days[pos]


def load_ohlcv(path, session="both", resample=None, roll_adjust="none"):
    """讀檔並回傳 OHLCV。

    session：both / regular / afterhours（只對有 session 欄的檔案有效）
    resample：None（原樣）、"D"（分 K 依交易日合成日 K）、"60min" 等 pandas 週期字串
    roll_adjust：none / back
    """
    df = _read(path)
    df["date"] = pd.to_datetime(df["date"], utc=False)
    if df["date"].dt.tz is None:
        df["date"] = df["date"].dt.tz_localize(TZ)  # 日 K 是 yyyy-MM-dd，當作台北 00:00
    else:
        df["date"] = df["date"].dt.tz_convert(TZ)
    df = df.set_index("date").sort_index()
    df = df[~df.index.duplicated(keep="last")]
    if session != "both" and "session" in df:
        df = df[df["session"].astype(str) == session]
    df = df.rename(columns=str.capitalize).rename(columns={"Contractmonth": "contractMonth"})
    has_cm = "contractMonth" in df
    if has_cm:
        df["contractMonth"] = df["contractMonth"].astype("string")

    if resample:
        agg = {"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}
        if has_cm:
            agg["contractMonth"] = "last"
        if resample.upper() == "D":
            df = df.assign(_td=_trading_day(df)).groupby("_td").agg(agg)
            df.index.name = "date"
        else:
            # 60 分 K：以台北時間整點切；夜盤跨午夜沒關係，空區間會被丟掉
            df = df.resample(resample, label="left", closed="left").agg(agg).dropna(subset=["Open"])

    if has_cm and roll_adjust == "back":
        roll = df["contractMonth"].ne(df["contractMonth"].shift()) & df["contractMonth"].shift().notna()
        gap = (df["Open"] - df["Close"].shift()).where(roll, 0.0)
        # 每個換月點之前的價格都加上之後所有的價差
        adj = gap[::-1].cumsum()[::-1].shift(-1, fill_value=0.0)
        for c in ("Open", "High", "Low", "Close"):
            df[c] = df[c].astype("float64") + adj
    elif roll_adjust not in ("none", "back"):
        raise ValueError("roll_adjust 只接受 none / back")

    out = df[["Open", "High", "Low", "Close", "Volume"]].astype("float64")
    if has_cm:
        out["contractMonth"] = df["contractMonth"]
    return out.dropna(subset=["Open", "High", "Low", "Close"])


def roll_dates(df):
    if "contractMonth" not in df:
        return []
    cm = df["contractMonth"]
    return list(df.index[cm.ne(cm.shift()) & cm.shift().notna()])


# ---- 10、11 共用：Turtle 指標與台指期成本 ----

POINT_VALUE = 200          # 大台 TXF 每點 200 元
TAX_RATE = 2 / 100_000     # 期交稅：契約金額 × 十萬分之二，買賣各收
DEFAULT_COMMISSION = 50    # 每口單邊手續費（元），各券商不同，請用 --commission 改
DEFAULT_SLIPPAGE = 1       # 每口單邊滑價（點）


def add_turtle_indicators(df, entry=20, exit_=10, atr_n=20):
    """先在 pandas 算好通道與 N，兩個框架吃同一份指標，結果才好比對。

    通道都用「前一根為止」的高低（shift(1)），避免偷看當根。
    N 用 Wilder 平滑的 ATR（同原版 Turtle）。
    """
    d = df.copy()
    prev_close = d["Close"].shift()
    tr = pd.concat([d["High"] - d["Low"], (d["High"] - prev_close).abs(),
                    (d["Low"] - prev_close).abs()], axis=1).max(axis=1)
    d["N"] = tr.ewm(alpha=1 / atr_n, adjust=False, min_periods=atr_n).mean()
    d["hi_entry"] = d["High"].rolling(entry).max().shift()
    d["lo_entry"] = d["Low"].rolling(entry).min().shift()
    d["hi_exit"] = d["High"].rolling(exit_).max().shift()
    d["lo_exit"] = d["Low"].rolling(exit_).min().shift()
    return d.dropna(subset=["N", "hi_entry", "lo_entry", "hi_exit", "lo_exit"])


def cost_per_side(price_points, contracts, commission=DEFAULT_COMMISSION, slippage=DEFAULT_SLIPPAGE):
    """單邊成本（元）＝ 手續費 ＋ 期交稅 ＋ 滑價。price_points 為點數。"""
    q = abs(contracts)
    return q * (commission + price_points * POINT_VALUE * TAX_RATE + slippage * POINT_VALUE)
