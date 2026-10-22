"""09 下載台指期 2010 年起的日 K，存成 Parquet／CSV（回測系列 3：給 10、11 的 Turtle 回測用）

執行：
  python examples/09_download_daily_history.py                       # TXF 近月連續 1!，2010-01-01 至今，日盤
  python examples/09_download_daily_history.py --session afterhours  # 夜盤日 K
  python examples/09_download_daily_history.py --format csv

為什麼另外寫（不併進 07）：07 只收分 K（--timeframe 1/5/10/15/30/60，最早 2017-06），
日 K 最早可到 2010 年，欄位也不同（日 K 有 change，沒有 average／transaction）。

重點（官方文件，SDK v2.4.0 起）：
- timeframe="D" 為日 K；日／週／月 K 可回溯至 2010 年。
- 單次查詢區間須「小於 1 年」（恰好 1 年回 400），這裡每段 360 天。
- data.date 日 K 為交易日 yyyy-MM-dd（不含時區），這裡存成台北時間 00:00。
- 查無資料回 404（不是空陣列）；超過 60 次／分回 429。
- 連續合約 1! 不做換月價差調整；data.contractMonth 標示實際合約，輸出檔會保留。
- session=afterhours 的日 K 歸屬「次一交易日」（期交所規則）。

官方文件：
https://www.fbs.com.tw/TradeAPI/docs/market-data-future/http-api/historical/candles.txt
https://www.fbs.com.tw/TradeAPI/docs/market-data-future/rate-limit.txt
"""

import argparse
import csv
import datetime as dt
import sys
import time

from _common import REPO_ROOT, import_sdk, login

FubonSDK = import_sdk()
from fubon_neo.fugle_marketdata.rest.base_rest import FugleAPIError  # noqa: E402

FIELDS = "open,high,low,close,volume,change"  # change 僅日／週／月 K 有
COLUMNS = ["date", "session", "contractMonth"] + FIELDS.split(",")
MIN_START = dt.date(2010, 1, 1)
SLEEP_SEC = 1.1


def has_pyarrow():
    try:
        import pandas  # noqa: F401
        import pyarrow  # noqa: F401
        return True
    except ImportError:
        return False


def parse_args():
    p = argparse.ArgumentParser(description="分段下載期貨歷史日 K 並存成 Parquet／CSV")
    p.add_argument("--product", default="TXF")
    p.add_argument("--contract-month", default="1!", help="1! / 2! / 3! 或 yyyyMM（預設 1!）")
    p.add_argument("--session", default="regular", choices=("regular", "afterhours"))
    p.add_argument("--start", default=MIN_START.isoformat(), help="yyyy-MM-dd（最早 2010-01-01）")
    p.add_argument("--end", default=dt.date.today().isoformat())
    p.add_argument("--format", default=None, choices=("csv", "parquet"))
    p.add_argument("--out", default=None)
    args = p.parse_args()
    if args.format is None:
        args.format = "parquet" if has_pyarrow() else "csv"
    elif args.format == "parquet" and not has_pyarrow():
        sys.exit("--format parquet 需要 pandas 與 pyarrow：pip install pandas pyarrow")
    return args


def chunks(start, end, days=360):
    cur = start
    while cur <= end:
        stop = min(cur + dt.timedelta(days=days - 1), end)
        yield cur, stop
        cur = stop + dt.timedelta(days=1)


def fetch(restfutopt, query):
    for _ in range(3):
        try:
            result = restfutopt.historical.candles(**query)
        except FugleAPIError as e:
            if e.status_code == 404:
                return []
            if e.status_code == 429:
                print("  觸發速率限制，等 60 秒…")
                time.sleep(60)
                continue
            raise
        # 已實測（SDK 2.4.0）：Python SDK 回傳 dict（product、contractMonth、exchange、session、timeframe、sort、data[]）
        return result.get("data", []) if isinstance(result, dict) else []
    sys.exit("連續 3 次 429，先停下來，晚點再跑。")


def main():
    args = parse_args()
    start = max(dt.date.fromisoformat(args.start), MIN_START)
    end = dt.date.fromisoformat(args.end)
    if start > end:
        sys.exit("--start 不可晚於 --end")

    sdk = FubonSDK()
    login(sdk)
    sdk.init_realtime()
    restfutopt = sdk.marketdata.rest_client.futopt

    rows = []
    try:
        for frm, to in chunks(start, end):
            data = fetch(restfutopt, {
                "product": args.product,
                "contractMonth": args.contract_month,
                "timeframe": "D",
                "session": args.session,
                "from": frm.isoformat(),
                "to": to.isoformat(),
                "fields": FIELDS,
                "sort": "asc",
            })
            for r in data:
                r["session"] = args.session
            rows.extend(data)
            print("{} ~ {}：{} 根".format(frm, to, len(data)))
            time.sleep(SLEEP_SEC)
    finally:
        sdk.logout()

    if not rows:
        print("整段都沒有資料，請檢查商品代號與日期。")
        return
    uniq = {r["date"]: r for r in rows}
    rows = [uniq[k] for k in sorted(uniq)]

    (REPO_ROOT / "data").mkdir(exist_ok=True)
    out = args.out or str(REPO_ROOT / "data" / "{}_{}_D_{}_{}_{}.{}".format(
        args.product, args.contract_month.replace("!", "c"), args.session, start, end, args.format))
    if args.format == "parquet":
        import pandas as pd
        df = pd.DataFrame([{c: r.get(c) for c in COLUMNS} for r in rows], columns=COLUMNS)
        df["date"] = pd.to_datetime(df["date"]).dt.tz_localize("Asia/Taipei")
        df["session"] = df["session"].astype("category")
        df["contractMonth"] = df["contractMonth"].astype("string").astype("category")
        for c in ("open", "high", "low", "close", "change"):
            df[c] = pd.to_numeric(df[c]).astype("float64")
        df["volume"] = pd.to_numeric(df["volume"]).astype("Int64")
        df.to_parquet(out, engine="pyarrow", compression="zstd", index=False)
    else:
        with open(out, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=COLUMNS, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)

    rolls = sum(1 for a, b in zip(rows, rows[1:]) if a.get("contractMonth") != b.get("contractMonth"))
    print("共 {} 根日 K，已存到 {}；換月 {} 次（未做價差調整）".format(len(rows), out, rolls))


if __name__ == "__main__":
    main()
