"""07 把台指期多年份分 K 一次抓下來存成 Parquet／CSV（對應回測系列第 1 篇）

執行：
  python examples/07_download_minute_history.py                          # TXF 近月 1 分 K，2017-06 至今，日盤＋夜盤
  python examples/07_download_minute_history.py --start 2024-01-01 --session regular
  python examples/07_download_minute_history.py --product MXF --timeframe 5
  python examples/07_download_minute_history.py --format csv             # 不想裝 pandas／pyarrow 就存 CSV

輸出格式（--format）：有裝 pyarrow 預設存 Parquet（zstd 壓縮，檔案小很多、讀取也快很多），
沒裝就自動改存 CSV 並提示。Parquet 欄位型別：date 為 datetime64（+08:00 台北時間）、
價格 float32（台指期價格是整數點，float32 可精確表示）、average 保留 float64、
volume／transaction 為整數、session／contractMonth 為 category。

為什麼要另外寫一支（03 只能抓一段）：官方規定單次查詢區間須「小於 1 年」，
歷史行情又有 60 次／分 的速率限制，所以要自己切段、慢慢抓、最後合併。

重點（官方文件，SDK v2.4.0 起）：
- 分 K（日盤與夜盤）自 2017-06 起提供；夜盤自 15:00 起，歸屬「次一交易日」。
- from／to 以交易日計；區間內查無資料回 404（不是空陣列），這裡當成「這段沒資料」略過。
- 超過速率限制回 429，這裡等 60 秒再重試。
- 連續合約 1! 不做換月價差調整，換月那根會跳空；data.contractMonth 標示實際合約，輸出檔會保留。
- 分 K 只算單式委託成交，價差委託的兩腿與鉅額交易不計入，所以分 K 加總量可能小於日 K 量，屬正常。

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

MINUTE_FRAMES = ("1", "5", "10", "15", "30", "60")
FIELDS = "open,high,low,close,volume,average,transaction"  # average／transaction 僅分 K 有
MIN_START = dt.date(2017, 6, 1)
SLEEP_SEC = 1.1  # 60 次／分 → 每次間隔 > 1 秒，保守一點

# TODO: 未驗證，請對照官方文件
# 官方文件沒寫單次回傳有沒有筆數上限或分頁。1 分 K 一整年約七、八萬根，
# 這裡保守用「每段一個月」查詢；若你實測一次抓更長沒問題，可把 --chunk-days 調大（須 < 365）。


def has_pyarrow():
    try:
        import pyarrow  # noqa: F401
        import pandas  # noqa: F401
        return True
    except ImportError:
        return False


def parse_args():
    p = argparse.ArgumentParser(description="分段下載期貨歷史分 K 並合併成 Parquet／CSV")
    p.add_argument("--product", default="TXF")
    p.add_argument("--contract-month", default="1!", help="1! / 2! / 3! 或 yyyyMM（預設 1!）")
    p.add_argument("--timeframe", default="1", choices=MINUTE_FRAMES)
    p.add_argument("--session", default="both", choices=("regular", "afterhours", "both"))
    p.add_argument("--start", default=MIN_START.isoformat(), help="yyyy-MM-dd（最早 2017-06-01）")
    p.add_argument("--end", default=dt.date.today().isoformat(), help="yyyy-MM-dd（預設今天）")
    p.add_argument("--chunk-days", type=int, default=31, help="每段查詢天數（須 < 365）")
    p.add_argument("--format", default=None, choices=("csv", "parquet"),
                   help="輸出格式（預設：有裝 pyarrow 用 parquet，否則 csv）")
    p.add_argument("--out", default=None)
    args = p.parse_args()
    if args.format is None:
        if has_pyarrow():
            args.format = "parquet"
        else:
            args.format = "csv"
            print("提示：沒偵測到 pandas／pyarrow，改存 CSV。"
                  "pip install pandas pyarrow 後預設會存 Parquet（檔案小很多、讀取更快）。")
    elif args.format == "parquet" and not has_pyarrow():
        sys.exit("--format parquet 需要 pandas 與 pyarrow：pip install pandas pyarrow")
    return args


def chunks(start, end, days):
    cur = start
    while cur <= end:
        stop = min(cur + dt.timedelta(days=days - 1), end)
        yield cur, stop
        cur = stop + dt.timedelta(days=1)


def fetch(restfutopt, query):
    """回傳 data 清單；404 視為這段沒資料；429 等一分鐘重試（最多 3 次）。"""
    for attempt in range(3):
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
        # TODO: 未驗證，請對照官方文件（同 03：假設 Python SDK 回傳 dict）
        return result.get("data", []) if isinstance(result, dict) else []
    sys.exit("連續 3 次 429，先停下來，晚點再跑。")


COLUMNS = ["date", "session", "contractMonth"] + FIELDS.split(",")


def write_csv(rows, out):
    with open(out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_parquet(rows, out):
    """pandas.to_parquet，zstd 壓縮（pyarrow 不支援時退回 snappy）。"""
    import pandas as pd

    df = pd.DataFrame([{c: r.get(c) for c in COLUMNS} for r in rows], columns=COLUMNS)
    # date 是 ISO 8601 含 +08:00，轉成帶時區的 datetime64，統一成台北時間
    df["date"] = pd.to_datetime(df["date"], utc=True).dt.tz_convert("Asia/Taipei")
    df["session"] = df["session"].astype("category")
    df["contractMonth"] = df["contractMonth"].astype("string").astype("category")
    for c in ("open", "high", "low", "close"):
        df[c] = pd.to_numeric(df[c]).astype("float32")  # 整數點位，float32 精確到 1600 萬
    df["average"] = pd.to_numeric(df["average"]).astype("float64")  # 均價有小數，保留 float64
    for c in ("volume", "transaction"):
        v = pd.to_numeric(df[c])
        df[c] = v.astype("int64") if v.notna().all() else v.astype("Int64")
    try:
        df.to_parquet(out, engine="pyarrow", compression="zstd", index=False)
    except Exception as e:  # 少數 pyarrow 編譯版本沒有 zstd
        print("zstd 壓縮不可用（{}），改用 snappy".format(e))
        df.to_parquet(out, engine="pyarrow", compression="snappy", index=False)


def main():
    args = parse_args()
    start = max(dt.date.fromisoformat(args.start), MIN_START)
    end = dt.date.fromisoformat(args.end)
    if start > end:
        sys.exit("--start 不可晚於 --end")
    if not 0 < args.chunk_days < 365:
        sys.exit("--chunk-days 需介於 1 與 364 之間（官方：單次查詢區間須小於 1 年）")
    sessions = ("regular", "afterhours") if args.session == "both" else (args.session,)

    sdk = FubonSDK()
    login(sdk)
    sdk.init_realtime()
    restfutopt = sdk.marketdata.rest_client.futopt

    rows = []
    try:
        for session in sessions:
            for frm, to in chunks(start, end, args.chunk_days):
                data = fetch(restfutopt, {
                    "product": args.product,
                    "contractMonth": args.contract_month,
                    "timeframe": args.timeframe,
                    "session": session,
                    "from": frm.isoformat(),
                    "to": to.isoformat(),
                    "fields": FIELDS,
                    "sort": "asc",
                })
                for r in data:
                    r["session"] = session
                rows.extend(data)
                print("{} {} ~ {}：{} 根".format(session, frm, to, len(data)))
                time.sleep(SLEEP_SEC)
    finally:
        sdk.logout()

    if not rows:
        print("整段都沒有資料，請檢查商品代號與日期。")
        return

    # 分 K 的 date 是 ISO 8601 含時區（同為 +08:00），字串排序即時間排序；用 (date, session) 去重
    uniq = {(r["date"], r["session"]): r for r in rows}
    rows = [uniq[k] for k in sorted(uniq)]

    out = args.out or str(REPO_ROOT / "data" / "{}_{}_{}m_{}_{}_{}.{}".format(
        args.product, args.contract_month.replace("!", "c"), args.timeframe,
        args.session, start, end, args.format))
    (REPO_ROOT / "data").mkdir(exist_ok=True)
    if args.format == "parquet":
        write_parquet(rows, out)
    else:
        write_csv(rows, out)

    rolls = sorted({r["date"][:10] for prev, r in zip(rows, rows[1:])
                    if r.get("contractMonth") != prev.get("contractMonth")})
    print("共 {} 根 K 線，已存到 {}".format(len(rows), out))
    print("換月次數：{}（連續合約未做價差調整，回測前要處理跳空）".format(len(rolls)))


if __name__ == "__main__":
    main()
