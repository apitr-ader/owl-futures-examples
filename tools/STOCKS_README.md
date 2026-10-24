# Taiwan common stocks (TWSE 上市 + TPEx 上櫃), daily, with corporate actions

Built by `tools/fetch_stocks.py`, checked by `tools/check_stocks.py`, loaded with `tools/stockdata.py`.
Period 2008-01-02 .. latest trading day. Common stocks only: 4-digit codes, excluding ETFs (0xxx) and TDRs (91xx).

## Update (incremental)
    cd /workspace/fubon-threads/tools
    nohup /workspace/.venv-pq/bin/python fetch_stocks.py all >> run/stocks.out 2>&1 &
- daily: only dates not yet in `cache/<twse|tpex>/YYYYMMDD.json.gz` (empty file = non-trading day)
- actions: re-pulls only the last 2 years (`--full` for everything)
- crfix: FinMind TPEx capital reductions only for stocks not yet cached in `_finmind_cr.json`
- build: rebuilds raw/adj/corporate_actions/adj_factors/universe from the caches, no network
Stages can be run on their own: `meta | actions | daily | crfix | build`.

## Files
| file | content |
|---|---|
| `raw/<code>.parquet` | unadjusted market,name,open,high,low,close,volume(shares),turnover(TWD),trades,shares(issued, TPEx only) |
| `adj/<code>.parquet` | total-return adjusted open/high/low/close, volume, turnover, `factor` (cumulative), `ret` (daily total return) |
| `corporate_actions.parquet/.csv` | one row per event: code, market, name, ex_date, trade_date, type, cash_div, stock_div (shares/share, TPEx), rights_value, div_plus_rights, cash_issue_shares/price, ref_price, prev_close, prev_close_data, adj_factor, applied, mismatch, inferred, reason, source |
| `adj_factors.parquet` | per stock/date event factors (`event_factor`, `cum_factor_before`) |
| `universe.csv` | code, market, name, industry, list_date, first_date, last_date, n_days, status (listed/delisted), delist_date, delist_src |
| `_actions_official.csv`, `_meta_current.csv`, `_delist_lists.csv`, `_finmind_cr.json`, `_tpex_cr_candidates.csv`, `_state.json` | intermediate / state |

`type`: cash_div, stock_div_or_rights (TWSE 權: stock dividend or rights issue, not split by the source), cash_div+stock_div_or_rights,
TPEx combinations of cash_div / stock_div / rights_issue, cap_reduction (減資), par_change (面額變更/split).

## Adjustment
For event at ex/resumption date e: `adj_factor = ref_price / prev_close` (除權息參考價 / 除權息前收盤價; 減資 恢復買賣參考價 / 停止買賣前收盤價; par change after/before).
`factor(t) = prod_{e>t} adj_factor_e`, `adj = raw * factor` → latest adjusted price = real price, and the return across e is `close_e / ref_e`
(cash dividends reinvested at the reference price; rights valued at their theoretical value).

## Sources & caveats
TWSE MI_INDEX / TWT49U / TWTAUU / suspendListing; TPEx afterTrading/otc, exDailyQ; isin.twse.com.tw; FinMind (par value change, delisting list, TPEx capital reductions).
Delisted stocks are included because the prices come from whole-market daily snapshots. TPEx capital reductions that are not in FinMind are
inferred from the change in issued shares (`inferred=1`, cash refunds not captured). TWSE capital reductions start in 2011 (TWTAUU floor).
Industry is only known for currently listed names.

## Use
    import sys; sys.path.insert(0, "/workspace/fubon-threads/tools"); import stockdata as sd
    px = sd.panel("close", adjusted=True, start="2010-01-01")      # date x code total-return prices
    u = sd.universe(as_of="2016-01-04")                             # point-in-time universe
    ca = sd.load_corporate_actions(codes=["2330"])
