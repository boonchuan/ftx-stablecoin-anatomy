# Data

This directory holds the on-chain transaction data and Binance aggregate trades used in the paper. Files are gitignored due to size; regenerate using `scripts/01_download_ftx_flows.py` and `scripts/02_download_binance.py`.

## Schema: `data/onchain/ftx_flows.parquet`

Raw ERC-20 stablecoin (USDT, USDC) transfers at four publicly tagged FTX-affiliated wallets between 2022-10-15 and 2022-11-25.

| Column | Type | Description |
|---|---|---|
| datetime | timestamp[UTC] | Block timestamp of the transfer |
| wallet | string | Wallet label (`ftx_hot`, `ftx_cold`, `ftx_other`, `alameda`) |
| direction | string | `in` (deposit to wallet) or `out` (withdrawal from wallet) |
| amount | float | Transfer value in USD (USDT and USDC both treated 1:1 with USD) |
| token | string | `USDT` or `USDC` |
| tx_hash | string | Ethereum transaction hash |

## Schema: `data/hourly_merged_v2.parquet`

Hourly aggregation produced by `scripts/03_merge_and_figures.py`. Used by stats and regression scripts.

| Column | Type | Description |
|---|---|---|
| index | timestamp[UTC] | Hour bin |
| net_inflow_usd | float | Signed flow at FTX hot wallet (negative = net outflow) |
| abs_flow_usd | float | Absolute flow magnitude at FTX hot wallet |
| onchain_tx_count | int | Number of on-chain transactions in the hour |
| volume | float | Binance BTC/USDT spot volume |
| rv | float | Realised volatility from Binance trades |
| hl_range | float | Hourly high-low range |

## Address reference

See Appendix B of the paper for full hex addresses of the four FTX wallets and the two stablecoin contracts.
