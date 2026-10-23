# 夜盤貓頭鷹的台指期 × 富邦新一代 API 範例 🦉

這是 Threads 帳號 **夜盤貓頭鷹（@apitr_ader）** 的貼文配套程式碼：
用 [富邦新一代 API（Fubon Neo API）](https://www.fbs.com.tw/TradeAPI/docs/welcome/) 的 Python SDK，
示範台指期（TXF）的登入、即時行情、歷史 K 線、下單、斷線重連與條件單。

每支範例都盡量短、只做一件事，並依官方文件撰寫；文件沒寫清楚、我無法驗證的地方，
程式裡會標註 `# TODO: 未驗證，請對照官方文件`。

## ⚠️ 免責聲明（請先看完）

- **個人學習範例**：這是我個人的學習紀錄與範例程式，不保證正確、完整或適用於你的情況。
- **非投資建議**：本 repo 不提供任何投資建議、不報明牌，程式裡的價格都是你自己輸入的參數。
- **期貨交易有風險**：期貨具高槓桿，可能在短時間內產生超過保證金的損失。
- **程式交易有系統與網路風險**：程式錯誤、斷線、延遲、主機當機、API 異動都可能造成非預期的委託或損失。真實下單前請自行充分測試，並準備其他下單平台作為備援。
- **作者非富邦員工，無業配**：我只是一般使用者，與富邦證券／富邦期貨沒有任何僱傭、合作或贊助關係。「富邦」「Fubon Neo API」等名稱與商標屬其權利人所有。
- 一切以 [官方文件](https://www.fbs.com.tw/TradeAPI/docs/welcome/) 為準；使用本程式碼的任何後果由使用者自行負責（另見 [LICENSE](LICENSE)）。

## 環境需求

| 項目 | 需求（依官方文件） |
| --- | --- |
| Python | **3.8–3.13**（v2.0.1 起不支援 3.7，**目前不支援 3.14**） |
| 作業系統 | Windows（x86-64）／macOS（Arm64、x86-64）／Linux（Arm64、x86-64） |
| 富邦新一代 API SDK | Python `.whl`；建議用最新版（撰寫時為 v2.4.0）。歷史 K 線需 ≥ v2.4.0、連續月別名需 ≥ v2.2.9、API Key 登入需 ≥ v2.2.7 |
| 帳戶 | 富邦期貨帳戶、已簽署 API 風險使用聲明書、已申請憑證（見 [事前準備](https://www.fbs.com.tw/TradeAPI/docs/trading-future/prepare.txt)） |
| 條件單（06） | 另需於富邦證券簽署中心選「期貨」簽署期貨條件單同意書（見 [條件單事前準備](https://www.fbs.com.tw/TradeAPI/docs/smart-condition-future/prepare.txt)） |

> 期貨也有測試帳號（官網文件寫只有證券，但實測可用），建議先用測試帳號跑。即便如此，下單相關範例預設都是 DRY_RUN（只印、不送）。

## 安裝

### 1. 取得官方 SDK（本 repo 不附 SDK 檔）

到官方 [SDK 下載頁](https://www.fbs.com.tw/TradeAPI/docs/download/download-sdk.txt) 下載你平台對應的 `.whl`，然後：

```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install /path/to/fubon_neo-<version>-<platform>.whl
pip install -r requirements.txt   # python-dotenv；pandas／pyarrow 為選用（只用 CSV 可不裝）
```

安裝細節與相容性：[安裝與版本相容性](https://www.fbs.com.tw/TradeAPI/docs/install-compatibility.txt)

### 2. 準備憑證

- Windows：可用 [憑證e總管](https://www.fbs.com.tw/Certificate/Management/) 申請。
- macOS／Linux：可用官方 [金鑰管理與憑證匯出](https://www.fbs.com.tw/TradeAPI/docs/key.txt) 頁面匯出網頁憑證（SDK ≥ v2.2.8）。
- 憑證檔請放在 repo 外，或放在本 repo 的 `certs/`（已被 `.gitignore` 排除）。

### 3. 設定帳密（環境變數或 .env）

```bash
cp .env.example .env
# 編輯 .env，填入你自己的值
```

| 變數 | 說明 |
| --- | --- |
| `FUBON_LOGIN_MODE` | `password`（帳密＋憑證，預設）或 `apikey` |
| `FUBON_ID` | 身分證字號 |
| `FUBON_PASSWORD` | 登入密碼（password 模式） |
| `FUBON_API_KEY` | API Key（apikey 模式，[申請說明](https://www.fbs.com.tw/TradeAPI/docs/trading/api-key-apply.txt)） |
| `FUBON_CERT_PATH` | 憑證檔路徑（相對路徑以 repo 根目錄為準） |
| `FUBON_CERT_PASSWORD` | 憑證密碼；password 模式使用「預設密碼憑證」時可留空 |
| `FUBON_ACCOUNT` | 選填，多帳號歸戶時指定期貨帳號 |
| `FUBON_WS_SYMBOL` / `FUBON_WS_SYMBOLS` | 選填，行情訂閱商品，預設 `TXF1!` / `TXF1!,MXF1!` |
| `FUBON_AFTER_HOURS` | 選填，`1` = 訂閱夜盤行情 |
| `FUBON_REALTIME_MODE` | 選填，`speed`（預設）或 `normal` |
| `FUBON_LIVE_ORDER` | **真實送單開關**，平常請留空（見下方安全說明） |

也可以不用 `.env`，直接 `export FUBON_ID=...`。程式裡不寫任何帳密。

> 小建議：研究行情用「只開行情權限」的 API Key；真正下單的 Key 綁固定 IP。官方提醒 API Key 權限變更時，受影響的 Key 連線會被強制登出，盤中別亂改。

## 範例列表（對應 Threads 貼文）

在 repo 根目錄執行，例如 `python examples/01_login.py`。

| 檔案 | 內容 | 對應貼文 |
| --- | --- | --- |
| [`examples/01_login.py`](examples/01_login.py) | 帳密＋憑證或 API Key 登入，列出帳號（遮罩）後登出 | P5 第一支程式、P10 API Key |
| [`examples/02_subscribe_txf_websocket.py`](examples/02_subscribe_txf_websocket.py) | WebSocket 訂閱 `TXF1!`（連續月別名）的 `trades` 成交與 `books` 五檔，可切夜盤 | P5、P6 連續月別名 |
| [`examples/03_historical_candles.py`](examples/03_historical_candles.py) | 查期貨歷史 K 線（日／分 K、日盤／夜盤、連續合約 `1!`），存 CSV 並標出換月跳空 | P8 歷史 K 線 |
| [`examples/04_place_futures_order.py`](examples/04_place_futures_order.py) | 期貨限價單（非阻塞 `unblock=True`），**預設 DRY_RUN** | P9 下單 |
| [`examples/05_reconnect_websocket.py`](examples/05_reconnect_websocket.py) | 行情斷線重連：背景執行緒、2/4/8/16/30 秒退避、最多 5 次、整份重新訂閱 | P11 限制、P13 斷線重連 |
| [`examples/06_condition_order.py`](examples/06_condition_order.py) | 期貨單一條件單＋停損停利，**預設 DRY_RUN** | P7 條件單、P12 流程 |
| [`examples/07_download_minute_history.py`](examples/07_download_minute_history.py) | 分段下載 2017-06 起的台指期 1 分 K（日盤＋夜盤），處理 429／404，合併去重存 Parquet 或 CSV（`--format`） | 回測系列 1 |
| [`examples/08_backtest_orb.py`](examples/08_backtest_orb.py) | 讀 07 的 CSV 或 Parquet，跑開盤區間突破當沖回測，含交易稅、手續費、滑價，逐年統計（讀 CSV 純標準函式庫） | 回測系列 2 |
| [`examples/09_download_daily_history.py`](examples/09_download_daily_history.py) | 分段下載 2010 年起的台指期日 K（`timeframe="D"`，每段 < 1 年），存 Parquet 或 CSV | 回測系列 3 |
| [`examples/10_backtesting_py_turtle.py`](examples/10_backtesting_py_turtle.py) | 用 [backtesting.py](https://kernc.github.io/backtesting.py/) 跑海龜（唐奇安 20/10、2N 停損、1 口），含成本，存 HTML 圖 | 回測系列 3 |
| [`examples/11_backtrader_turtle.py`](examples/11_backtrader_turtle.py) | 同一策略用 [Backtrader](https://www.backtrader.com/) 跑（mult=200 期貨模式），對照 10，說明成交模型差異 | 回測系列 3 |
| [`examples/12_atr_trailing_15m.py`](examples/12_atr_trailing_15m.py) | 15 分 K 的 ORB／唐奇安加 ATR 停損與移動停利，參數網格＋樣本內外＋鄰域熱圖 | 回測系列 5 |
| [`examples/13_supertrend_grid.py`](examples/13_supertrend_grid.py) | SuperTrend（日 K／60 分／15 分，日盤或日夜盤）參數網格，樣本內外與過度最佳化檢查（排名相關、DSR） | 回測系列 6 |
| [`examples/14_supertrend_atr_trailing.py`](examples/14_supertrend_atr_trailing.py) | SuperTrend 反手加 ATR 初始停損與移動停利（停損後空手等下次翻轉），停損網格＋樣本內外＋熱圖 | 回測系列 6 |
| [`examples/15_supertrend_daily_filter.py`](examples/15_supertrend_daily_filter.py) | 日 K SuperTrend 當方向濾網（只用已收完的日 K），只做同向的 60／15 分 SuperTrend；state／event／只做多三種，樣本內外與過度最佳化統計 | 回測系列 7 |
| [`examples/16_supertrend_asym_long_short.py`](examples/16_supertrend_asym_long_short.py) | 15 的 event 模式拆成多、空兩條腿，各用不同的分 K SuperTrend 參數（可選日線濾網也不對稱）；四維鄰域高原選參、樣本內外、排除 2025–26、多空分腿與 DSR | 回測系列 8 |
| [`examples/17_supertrend_60m_filter_5m.py`](examples/17_supertrend_60m_filter_5m.py) | 60 分 SuperTrend（只用已收完的 60 分 K）當方向濾網，只做同向的 5 分 SuperTrend（event）；日盤／日夜盤、多空／只做多／只做空，四維鄰域選參、樣本內外、排除 2025–26、DSR | 回測系列 9 |
| `examples/18_gap_fill.py` | 回測系列 10：日盤開盤跳空回補（逆勢補缺口，參考前日盤／前夜盤收盤；最小跳空×停損×進場延遲網格、原始規則與永遠逆勢對照） |
| `examples/19_gap_follow.py` | 回測系列 11：日盤開盤跳空「順勢」（18 的 invert；收盤出場＋前收/ATR/移動停損網格、原始預設反向與永遠順勢對照、毛利 vs 成本拆解） |
| `examples/20_gap_follow_daily_filter.py` | 回測系列 12：跳空順勢＋日 K 方向濾網（SuperTrend 10/20×2/3、MA20/MA60，前夜盤／前日盤參考價，與未濾網及 19 選參對照） |
| `examples/_bt_data.py` | 回測共用：Parquet／CSV → OHLCV（台北時間索引）、分 K 合成日／60 分 K、換月倒推調整、海龜指標與成本 | — |
| `scripts/make_fake_txf_data.py` | 產生**假**台指期資料（隨機漫步）只為測試 10、11 能跑，不是行情 | — |
| `examples/_common.py` | 共用：讀環境變數、登入、挑期貨帳號、遮罩帳號 | — |

### 📦 分 K 檔案格式：Parquet 或 CSV（07、08）

07 加了 `--format {csv,parquet}`：有裝 `pandas`、`pyarrow` 時預設存 Parquet（zstd 壓縮，不支援時退回 snappy），沒裝就自動存 CSV 並提示。08 看副檔名自動判斷，`.csv` 照舊只用標準函式庫，`.parquet` 才載入 pandas。

為什麼建議 Parquet：9 年 1 分 K 有上百萬根，CSV 每個數字都存成文字、時間字串一再重複；Parquet 是欄式儲存、有壓縮、有型別（時間 datetime64、價格 float32、session 用 category），**檔案小很多，讀取也快很多**，載入後不用再自己 parse 字串。同一份資料用兩種格式跑 08，回測結果完全相同。

```bash
python examples/07_download_minute_history.py                  # 預設 Parquet（有 pyarrow 時）
python examples/07_download_minute_history.py --format csv     # 只要 CSV
python examples/08_backtest_orb.py data/TXF_1c_1m_both_2017-06-01_2026-10-20.parquet
```

### 🧪 回測管線：富邦新一代 API 下載 → Parquet → backtesting.py／Backtrader（09、10、11）

```text
富邦新一代 API（historical.candles）
   ├─ 07：1 分 K（2017-06 起，日盤＋夜盤）
   └─ 09：日 K（2010 起）
        ↓  data/*.parquet（或 CSV）
   _bt_data.load_ohlcv()  → Open/High/Low/Close/Volume，台北時間索引
        ├─ 10：backtesting.py（價格 × 200 換成元，1 單位＝1 口，輸出 HTML 圖）
        └─ 11：Backtrader（CommInfo mult=200、保證金制）
```

```bash
pip install -r requirements.txt            # 含 pandas、pyarrow、backtesting、backtrader
python examples/09_download_daily_history.py
python examples/10_backtesting_py_turtle.py data/TXF_1c_D_regular_2010-01-01_<今天>.parquet
python examples/11_backtrader_turtle.py     data/TXF_1c_D_regular_2010-01-01_<今天>.parquet
python examples/13_supertrend_grid.py --daily data/TXF_1c_D_regular_2010-01-01_<今天>.parquet --minute data/TXF_1c_1m_both_2017-06-01_<今天>.parquet
# 用 07 的 1 分 K：加 --resample D 或 --resample 60min（夜盤依期交所規則歸屬次一交易日）
# 沒帳號想先試跑：python scripts/make_fake_txf_data.py，再把 data/FAKE_TXF_D.parquet 丟給 10、11
```

- **策略**：收盤突破前 20 根高（低）→ 下一根開盤做多（空）；收盤破前 10 根反向極值，或收盤超出「進場價 ∓ 2N」→ 下一根開盤出場。N＝20 根 Wilder ATR，固定 1 口、不加碼。
- **成本**（每口單邊）：手續費 `--commission`（預設 50 元，依券商）＋ 期交稅（價格 × 200 × 十萬分之二）＋ 滑價 `--slippage`（預設 1 點＝200 元）。保證金用 `--margin-ratio`（預設 0.1，實際以期交所公告為準）。
- **換月**：連續合約 `1!` 不做價差調整，換月會跳空，可能誤觸突破或停損。`--roll-adjust back` 做倒推調整（用新合約開盤 − 舊合約收盤近似價差，屬近似）；要精準請下載各月份合約自行接合。
- **兩個框架結果**：指標由同一份 pandas 計算、都用「下一根開盤」市價成交，用假資料測試時逐筆進出場價與損益相同；差別只在最後一筆未平倉：10 以 `finalize_trades=True` 在最後收盤結算並計入，11 列為未平倉。改用盤中停損單時兩者成交規則不同，詳見 11 檔頭。
- Backtrader 最後一版是 2023 年，Python 3.13 實測可跑；若 import 失敗，改用 Python 3.11 venv 或 `pip install backtrader2`。

### 🔒 下單範例的安全機制（04、06）

預設 `DRY_RUN = True`：只印出委託內容，**不登入、不送單**。要送出真實委託，必須三個條件同時成立：

1. 環境變數 `FUBON_LIVE_ORDER=I_UNDERSTAND_THE_RISK`
2. 命令列加 `--live`
3. 執行時在終端機手動輸入商品代號確認

此外硬性限制最多 1 口、只允許限價單、不接受連續月別名（下單請用具體契約代碼，例如 `TXFK6`）。
第一次實單測流程，請用 1 口、遠離市價的限價單，並在其他平台確認委託狀態。

## 幾個踩坑提醒（出自官方文件）

- **連續月別名**：WebSocket 只在訂閱當下解析一次，跨結算日不會自動轉倉，長時間執行要在換月後 unsubscribe 再 subscribe；Web API 每次呼叫都重新解析。
- **斷線**：SDK 不會自動重連、也不保留訂閱；約 60 秒沒收到伺服器訊息會判定逾時並觸發 disconnect。
- **速率限制**（期貨行情）：日內行情 300 次／分、快照 300 次／分、歷史 60 次／分、WebSocket 300 訂閱／7 連線，超過回 429。
- **歷史 K 線**：單次查詢區間須小於 1 年；連續合約不做換月價差調整。
- **條件單**：分時分量、移動鎖利、時間觸發不支援期貨夜盤；停損停利只保證「觸發送單」，不保證成交。

## 官方資源

- 官方文件：<https://www.fbs.com.tw/TradeAPI/docs/welcome/>
  - 期貨交易：<https://www.fbs.com.tw/TradeAPI/docs/trading-future/introduction.txt>
  - 期貨行情 WebSocket：<https://www.fbs.com.tw/TradeAPI/docs/market-data-future/websocket-api/getting-started.txt>
  - 期貨行情 Web API：<https://www.fbs.com.tw/TradeAPI/docs/market-data-future/http-api/getting-started.txt>
  - 期貨條件單：<https://www.fbs.com.tw/TradeAPI/docs/smart-condition-future/introduction.txt>
  - SDK 下載：<https://www.fbs.com.tw/TradeAPI/docs/download/download-sdk.txt>
- 官方 Discord 社群（程式交易討論）：<https://discord.com/invite/M8Gv9yKfza>

有問題歡迎到 Threads 留言，或在本 repo 開 issue；開戶、手續費等請洽官方或營業員。

## 授權

[MIT](LICENSE) © 夜盤貓頭鷹
