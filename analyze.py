import os
import zipfile
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime

# ----- Load on-chain flows -----
flows = pd.read_parquet("data/onchain/ftx_flows.parquet")
flows = flows[flows["wallet"] == "ftx_hot"].copy()  # focus on the active wallet
flows["datetime"] = pd.to_datetime(flows["datetime"])
flows["signed_amount"] = flows.apply(
    lambda r: -r["amount"] if r["direction"] == "out" else r["amount"], axis=1
)
# Hourly net flow into ftx_hot (negative = net outflow = users withdrawing)
flows_hourly = (
    flows.set_index("datetime")
    .resample("1h")["signed_amount"]
    .sum()
    .rename("net_inflow_usd")
)
print("On-chain hourly bins:", len(flows_hourly))
print(flows_hourly.describe())

# ----- Load Binance aggTrades and compute trade-flow asymmetry -----
agg_dir = Path("data/binance_aggtrades")
zips = sorted(agg_dir.glob("*.zip"))
print(f"\nLoading {len(zips)} aggTrades zips...")

all_trades = []
for z in zips:
    with zipfile.ZipFile(z) as zf:
        name = zf.namelist()[0]
        # aggTrades schema: agg_trade_id, price, qty, first_trade_id, last_trade_id, timestamp_ms, is_buyer_maker, is_best_match
        df = pd.read_csv(
            zf.open(name),
            header=None,
            names=["agg_id", "price", "qty", "first_id", "last_id", "ts_ms", "is_buyer_maker", "is_best_match"],
        )
        all_trades.append(df)
trades = pd.concat(all_trades, ignore_index=True)
print(f"Total trades: {len(trades):,}")

trades["datetime"] = pd.to_datetime(trades["ts_ms"], unit="ms")
trades["notional"] = trades["price"] * trades["qty"]
# is_buyer_maker = True means the taker was a SELLER (sold into a resting bid)
# is_buyer_maker = False means the taker was a BUYER (lifted a resting offer)
trades["taker_side"] = trades["is_buyer_maker"].apply(lambda x: "sell" if x else "buy")

# Hourly bins
trades_hourly = (
    trades.set_index("datetime")
    .groupby([pd.Grouper(freq="1h"), "taker_side"])["notional"]
    .sum()
    .unstack(fill_value=0)
)
trades_hourly["total"] = trades_hourly["buy"] + trades_hourly["sell"]
trades_hourly["net_taker_buy"] = trades_hourly["buy"] - trades_hourly["sell"]
trades_hourly["asymmetry"] = trades_hourly["net_taker_buy"] / trades_hourly["total"]
print(f"\nTrade hourly bins: {len(trades_hourly)}")

# Hourly mid price (use last trade price in each hour)
trades_hourly["price"] = trades.set_index("datetime")["price"].resample("1h").last()

# ----- Merge -----
df = pd.concat([flows_hourly, trades_hourly], axis=1).dropna(subset=["price"])
df["net_inflow_usd"] = df["net_inflow_usd"].fillna(0)
df["net_outflow_usd"] = -df["net_inflow_usd"]  # positive = users withdrawing
df["cum_outflow"] = df["net_outflow_usd"].cumsum()

# ----- Plot -----
fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)

# Panel 1: BTC price
axes[0].plot(df.index, df["price"], color="black", linewidth=1)
axes[0].set_ylabel("BTC/USDT Price")
axes[0].set_title("FTX Collapse: On-Chain Flows vs Binance Trade Flow")
for date, label in [
    ("2022-11-02", "CoinDesk"),
    ("2022-11-06", "CZ tweet"),
    ("2022-11-08", "Binance LOI"),
    ("2022-11-09", "LOI off"),
    ("2022-11-10", "Halt withdr."),
    ("2022-11-11", "Bankruptcy"),
]:
    for ax in axes:
        ax.axvline(pd.Timestamp(date), color="red", linestyle="--", alpha=0.4, linewidth=0.8)
    axes[0].text(pd.Timestamp(date), df["price"].max() * 0.98, label, rotation=90, fontsize=8, va="top")

# Panel 2: hourly net outflow from ftx_hot (positive = withdrawals)
axes[1].bar(df.index, df["net_outflow_usd"] / 1e6, width=0.04, color="steelblue")
axes[1].set_ylabel("Net Outflow ($M/hr)\nfrom ftx_hot")
axes[1].axhline(0, color="black", linewidth=0.5)

# Panel 3: cumulative outflow
axes[2].plot(df.index, df["cum_outflow"] / 1e6, color="darkblue")
axes[2].set_ylabel("Cum. Net Outflow ($M)")
axes[2].axhline(0, color="black", linewidth=0.5)

# Panel 4: trade asymmetry (positive = taker buying, negative = taker selling)
axes[3].bar(df.index, df["asymmetry"], width=0.04,
            color=df["asymmetry"].apply(lambda x: "green" if x > 0 else "red"))
axes[3].set_ylabel("Taker Asymmetry\n(buy - sell) / total")
axes[3].set_xlabel("Date (UTC)")
axes[3].axhline(0, color="black", linewidth=0.5)

plt.tight_layout()
plt.savefig("prototype_chart.png", dpi=120)
print("\nSaved chart: prototype_chart.png")

# ----- Quick lead-lag check -----
print("\n=== Lead-lag correlation (Pearson) ===")
print("Hypothesis: rising outflows -> taker sell pressure (negative asymmetry)")
print("Expect NEGATIVE correlations if hypothesis holds, stronger at small positive lags.")
for lag in [-6, -3, -1, 0, 1, 3, 6, 12, 24]:
    if lag >= 0:
        corr = df["net_outflow_usd"].corr(df["asymmetry"].shift(-lag))
    else:
        corr = df["net_outflow_usd"].shift(-lag).corr(df["asymmetry"])
    print(f"  outflow leads asymmetry by {lag:+d}h: corr = {corr:+.3f}")

plt.show()