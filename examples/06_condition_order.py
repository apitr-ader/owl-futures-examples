"""06 期貨單一條件單＋停損停利（對應 Threads P7、P12）—— 預設只「模擬」

情境：台指期成交價跌破（小於）觸發價時，以限價新倉買進 1 口，
成交後自動掛停利／停損（平倉）。全部參數由命令列給，程式不預設任何價位。

!!! 條件單觸發後會送出真實委託，可能造成實際損失 !!!
前置：需先登入富邦證券簽署中心，選「期貨」簽署期貨條件單同意書。
限制（官方文件）：分時分量、移動鎖利、時間觸發條件不支援期貨夜盤；
停損停利僅為「觸發送單」，不保證成交，跳空時照樣會滑價。

送出真實條件單需同時滿足（同 04）：
  1. 環境變數 FUBON_LIVE_ORDER=I_UNDERSTAND_THE_RISK
  2. 命令列 --live
  3. 終端機手動輸入商品代號確認

執行（模擬）：
  python examples/06_condition_order.py --symbol TXFK6 --trigger 20000 --price 20000 \
      --take-profit 20100 --stop-loss 19950

官方文件：
https://www.fbs.com.tw/TradeAPI/docs/smart-condition-future/prepare.txt
https://www.fbs.com.tw/TradeAPI/docs/smart-condition-future/library/python/SingleConditionTPSL.txt
https://www.fbs.com.tw/TradeAPI/docs/smart-condition-future/library/python/SingleCondition.txt
https://www.fbs.com.tw/TradeAPI/docs/smart-condition-future/library/python/EnumMatrix.txt
https://www.fbs.com.tw/TradeAPI/docs/download/download-sdk.txt （>=2.2.0 夜盤限制說明）
"""

import argparse
import datetime as dt
import os
import re
import sys

from _common import import_sdk, login, mask, pick_futopt_account

FubonSDK = import_sdk()
from fubon_neo.sdk import Condition, FutOptConditionOrder  # noqa: E402
from fubon_neo.constant import (  # noqa: E402
    BSAction,
    FutOptConditionMarketType,
    FutOptConditionOrderType,
    FutOptConditionPriceType,
    FutOptTPSLOrder,
    FutOptTPSLWrapper,
    Operator,
    StopSign,
    TimeInForce,
    TradingType,
    TriggerContent,
)

DRY_RUN = True
LIVE_FLAG_ENV = "FUBON_LIVE_ORDER"
LIVE_FLAG_VALUE = "I_UNDERSTAND_THE_RISK"
MAX_LOT = 1


def num(value):
    if not re.fullmatch(r"\d+(\.\d+)?", str(value)):
        raise argparse.ArgumentTypeError("必須是數字：{}".format(value))
    return str(value)


def parse_args():
    p = argparse.ArgumentParser(description="期貨單一條件單＋停損停利（預設模擬）")
    p.add_argument("--symbol", required=True, help="具體契約代碼，例如 TXFK6")
    p.add_argument("--trigger", required=True, type=num, help="觸發價：成交價 < 此值時觸發")
    p.add_argument("--price", required=True, type=num, help="觸發後的限價買進價")
    p.add_argument("--take-profit", required=True, type=num, help="停利觸發價（同時當限價）")
    p.add_argument("--stop-loss", required=True, type=num, help="停損觸發價（同時當限價）")
    p.add_argument("--days", type=int, default=0, help="監控天數，0 = 只有今天")
    p.add_argument("--live", action="store_true")
    return p.parse_args()


def decide_dry_run(args):
    if not args.live:
        return True
    if os.environ.get(LIVE_FLAG_ENV, "") != LIVE_FLAG_VALUE:
        print("有 --live 但環境變數 {}={} 未設定，維持模擬。".format(LIVE_FLAG_ENV, LIVE_FLAG_VALUE))
        return True
    typed = input("真的要送出真實條件單嗎？請輸入商品代號 {} 確認：".format(args.symbol)).strip()
    return typed != args.symbol


def main():
    args = parse_args()
    if "!" in args.symbol:
        sys.exit("條件單請用具體契約代碼（例 TXFK6）。")
    if not float(args.stop_loss) < float(args.price) < float(args.take_profit):
        sys.exit("多單應滿足：停損 < 進場價 < 停利，請檢查參數。")

    start = dt.date.today()
    end = start + dt.timedelta(days=max(args.days, 0))
    start_date, end_date = start.strftime("%Y%m%d"), end.strftime("%Y%m%d")  # 官方範例格式 yyyyMMdd

    # TODO: 未驗證，請對照官方文件
    # 官方期貨條件單 Python 範例的 symbol 都是選擇權（TXO20000E4），
    # 沒有直接示範台指期代碼（TXFK6）；理論上同為期權商品代碼，請先小量確認。
    condition = Condition(
        market_type=TradingType.Reference,  # 自動參考委託物件
        symbol=args.symbol,
        trigger=TriggerContent.MatchedPrice,  # 成交價
        trigger_value=args.trigger,
        comparison=Operator.LessThan,
    )

    # TODO: 未驗證，請對照官方文件
    # 參數對照表的 FutOptConditionOrder 欄位寫 quantity，但官方請求範例用 lot；這裡照範例用 lot。
    # 參數對照表的 FutOptConditionMarketType 常數表只列 Future / Option（欄位說明另提到 FutureNight），
    # 本範例只示範日盤 Future。
    order = FutOptConditionOrder(
        buy_sell=BSAction.Buy,
        symbol=args.symbol,
        price=args.price,
        lot=MAX_LOT,
        market_type=FutOptConditionMarketType.Future,
        price_type=FutOptConditionPriceType.Limit,
        time_in_force=TimeInForce.ROD,
        order_type=FutOptConditionOrderType.New,
    )

    tp = FutOptTPSLOrder(
        time_in_force=TimeInForce.ROD,
        price_type=FutOptConditionPriceType.Limit,
        order_type=FutOptConditionOrderType.Close,
        target_price=args.take_profit,
        price=args.take_profit,
    )
    sl = FutOptTPSLOrder(
        time_in_force=TimeInForce.ROD,
        price_type=FutOptConditionPriceType.Limit,
        order_type=FutOptConditionOrderType.Close,
        target_price=args.stop_loss,
        price=args.stop_loss,
    )
    tpsl = FutOptTPSLWrapper(
        stop_sign=StopSign.Full,
        tp=tp,
        sl=sl,
        end_date=end_date,
    )

    print("條件單：{}~{} 成交價 < {} 時，限價 {} 買進 {} {} 口；停利 {}／停損 {}".format(
        start_date, end_date, args.trigger, args.price, args.symbol, MAX_LOT,
        args.take_profit, args.stop_loss))

    dry_run = DRY_RUN
    if dry_run and not decide_dry_run(args):
        dry_run = False
    if dry_run:
        print("[DRY_RUN] 沒有登入、沒有送出條件單。")
        return

    sdk = FubonSDK()
    accounts = login(sdk)
    try:
        acc = pick_futopt_account(accounts)
        print("使用期貨帳號：{}".format(mask(acc.account)))
        res = sdk.futopt.single_condition(acc, start_date, end_date, StopSign.Full,
                                          condition, order, tpsl)
        if res.is_success:
            print("條件單已建立，guid={}".format(res.data.guid))
        else:
            print("建立失敗：{}".format(res.message))
    finally:
        sdk.logout()


if __name__ == "__main__":
    main()
