"""示範用日K小樣本（sample_data/，最近約 20 個交易日）。沒有 API 帳號時可用來跑範例。

    from tools.sample_data import load_daily, use_sample
    df = load_daily("TXF")              # date, symbol, open, high, low, close, volume
use_sample() 在未設定 FUBON_ID 環境變數時回傳 True（只檢查是否存在，不讀取內容）。
"""
import os
from pathlib import Path

import pandas as pd

DIR = Path(__file__).resolve().parent.parent / "sample_data"


def use_sample():
    return not os.environ.get("FUBON_ID")


def load_daily(symbol="TXF"):
    f = DIR / ("TXF_D_sample.parquet" if symbol == "TXF" else "stock_futures_D_sample.parquet")
    d = pd.read_parquet(f)
    return d[d.symbol == symbol].reset_index(drop=True)
