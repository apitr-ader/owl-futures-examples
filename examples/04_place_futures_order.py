"""04 期貨限價下單（對應 Threads P9）—— 預設只「模擬」，不會送出真實委託

!!! 這支程式可以送出真實期貨委託，可能造成實際損失 !!!
期貨有測試帳號（官網文件寫只有證券，但實測可用），建議先用測試帳號；
所以本範例預設 DRY_RUN（只印出委託內容、不登入、不送單）。

要真的送單，必須「同時」滿足：
  1. 環境變數 FUBON_LIVE_ORDER=I_UNDERSTAND_THE_RISK（一字不差）
  2. 命令列加上 --live
  3. 執行時在終端機手動輸入商品代號確認
另外本範例硬性限制最多 1 口（MAX_LOT），且只允許限價單。

執行（模擬）：
  python examples/04_place_futures_order.py --symbol TXFK6 --side buy --price 20000
執行（真實送單，請先確認價格遠離市價、只用 1 口測流程）：
  FUBON_LIVE_ORDER=I_UNDERSTAND_THE_RISK python examples/04_place_futures_order.py \
      --symbol TXFK6 --side buy --price 20000 --live

symbol 用具體契約代碼（例：TXFK6 = 2026 年 11 月台指期；月份碼 A~L = 1~12 月）。
夜盤請加 --night（market_type 改用 FutOptMarketType.FutureNight）。

官方文件：
https://www.fbs.com.tw/TradeAPI/docs/trading-future/library/python/trade/PlaceOrder.txt
https://www.fbs.com.tw/TradeAPI/docs/trading-future/library/python/EnumMatrix.txt
https://www.fbs.com.tw/TradeAPI/docs/trading-future/guide/advance/asyn_order.txt
https://www.fbs.com.tw/TradeAPI/docs/trading-future/library/python/trade/GetOrderResults.txt
"""

import argparse
import os
import re
import sys

from _common import import_sdk, login, mask, pick_futopt_account

FubonSDK = import_sdk()
from fubon_neo.sdk import FutOptOrder  # noqa: E402
from fubon_neo.constant import (  # noqa: E402  官方範例逐一 import，不用 import *
    BSAction,
    FutOptMarketType,
    FutOptOrderType,
    FutOptPriceType,
    TimeInForce,
)

# ===== 安全設定 =====
DRY_RUN = True  # 預設模擬；只有通過下方所有檢查才會變成 False
LIVE_FLAG_ENV = "FUBON_LIVE_ORDER"
LIVE_FLAG_VALUE = "I_UNDERSTAND_THE_RISK"
MAX_LOT = 1
# ====================


def parse_args():
    p = argparse.ArgumentParser(description="期貨限價下單範例（預設模擬）")
    p.add_argument("--symbol", required=True, help="具體契約代碼，例如 TXFK6（不要用 TXF1!）")
    p.add_argument("--side", required=True, choices=("buy", "sell"))
    p.add_argument("--price", required=True, help="限價（字串或數字）")
    p.add_argument("--lot", type=int, default=1)
    p.add_argument("--night", action="store_true", help="夜盤（FutOptMarketType.FutureNight）")
    p.add_argument("--blocking", action="store_true",
                   help="改用阻塞下單（預設 unblock=True 非阻塞）")
    p.add_argument("--live", action="store_true", help="真的送單（還需環境變數與手動確認）")
    return p.parse_args()


def decide_dry_run(args):
    if not args.live:
        return True
    if os.environ.get(LIVE_FLAG_ENV, "") != LIVE_FLAG_VALUE:
        print("有 --live 但環境變數 {}={} 未設定，維持模擬。".format(LIVE_FLAG_ENV, LIVE_FLAG_VALUE))
        return True
    typed = input("真的要送出真實委託嗎？請輸入商品代號 {} 確認：".format(args.symbol)).strip()
    if typed != args.symbol:
        print("輸入不符，取消送單。")
        return True
    return False


def main():
    args = parse_args()

    if "!" in args.symbol:
        sys.exit("下單請用具體契約代碼（例 TXFK6）。連續月別名 TXF1! 是行情用的。")
    if not re.fullmatch(r"\d+(\.\d+)?", str(args.price)):
        sys.exit("--price 必須是數字（只允許限價單）")
    if not 0 < args.lot <= MAX_LOT:
        sys.exit("本範例最多只允許 {} 口".format(MAX_LOT))

    order = FutOptOrder(
        buy_sell=BSAction.Buy if args.side == "buy" else BSAction.Sell,
        symbol=args.symbol,
        price=str(args.price),
        lot=args.lot,
        market_type=FutOptMarketType.FutureNight if args.night else FutOptMarketType.Future,
        price_type=FutOptPriceType.Limit,
        time_in_force=TimeInForce.ROD,
        order_type=FutOptOrderType.Auto,
        user_def="owlpy",  # 選填；證券 user_def 規則為英數最多 10 字元，這裡沿用同規則
    )
    unblock = not args.blocking
    print("委託內容：{} {} {} 口 @ {}（{}，ROD，Auto，unblock={}）".format(
        args.side, args.symbol, args.lot, args.price,
        "夜盤" if args.night else "日盤", unblock))

    dry_run = DRY_RUN
    if dry_run and not decide_dry_run(args):
        dry_run = False  # 只有 --live＋環境變數＋手動確認三關都過才會走到這裡
    if dry_run:
        print("[DRY_RUN] 沒有登入、沒有送單。確認無誤後請看檔頭說明開啟真實送單。")
        return

    sdk = FubonSDK()
    accounts = login(sdk)
    try:
        acc = pick_futopt_account(accounts)
        print("使用期貨帳號：{}".format(mask(acc.account)))
        # 第三個參數 unblock：True = 非阻塞（送出後立即回覆）
        result = sdk.futopt.place_order(acc, order, unblock)
        if result.is_success:
            print("送單成功：{}".format(result.data))
        else:
            print("送單失敗：{}".format(result.message))

        # 盤後檢討用：查今日委託
        orders = sdk.futopt.get_order_results(acc)
        if orders.is_success:
            print("今日委託共 {} 筆".format(len(orders.data)))
    finally:
        sdk.logout()


if __name__ == "__main__":
    main()
