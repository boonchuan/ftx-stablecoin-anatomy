import zipfile
import polars as pl
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ----- Load on-chain flows (small, use pandas) -----
flows = pd.read_parquet("data/onchain/ftx_flows.parquet")
flows = flows[flows["wallet"] == "ftx_hot"].copy()
flows["datetime"] = pd.to_datetime(flows["datetime"])
flows["signed"] = np.where(flows["direction"] == "out", -flows["amount"], flows["amount"])
flows_hourly = (
    flows.set_index("datetime")
    .resample("1h")["signed"].sum()
    .rename("net_inflow_usd")
)
print("On-chain hourly bins:", len(flows_hourly))

# ----- Stream Binance aggTrades, building hourly aggregates per day -----
agg_dir = Path("data/binance_aggtrades")
zips = sorted(agg_dir.glob("*.zip"))
print(f"Streaming {len(zips)} days...")

hourly_chunks = []
for i, z in enumerate(zips, 1):
    with zipfile.ZipFile(z) as zf:
        name = zf.namelist()[0]
        # Read directly into Polars - much faster than pandas
        df = pl.read_csv(
            zf.read(name),
            has_header=False,
            new_columns=["agg_id","price","qty","first_id","last_id","ts_ms","is_buyer_maker","is_best_match"],
            schema_overrides={"price": pl.Float64, "qty": pl.Float64, "ts_ms": pl.Int64},
        )
    
    # Add datetime, notional, signed_qty
    df = df.with_columns([
        pl.from_epoch("ts_ms", time_unit="ms").alias("datetime"),
        (pl.col("price") * pl.col("qty")).alias("notional"),
        pl.when(pl.col("is_buyer_maker")).then(-pl.col("qty")).otherwise(pl.col("qty")).alias("signed_qty"),
    ])
    
    # Build 1-min bars
    minute = df.group_by_dynamic("datetime", every="1m").agg([
        pl.col("price").first().alias("open_"),
        pl.col("price").max().alias("high"),
        pl.col("price").min().alias("low"),
        pl.col("price").last().alias("close"),
        pl.col("notional").sum().alias("vol"),
        pl.col("price").count().alias("n_trades"),
        pl.col("signed_qty").sum().alias("signed_qty"),
    ])
    
    # Compute returns
    minute = minute.with_columns(
        (pl.col("close").log() - pl.col("close").log().shift(1)).alias("ret")
    )
    
    # Aggregate to hourly stress metrics
    hourly = minute.group_by_dynamic("datetime", every="1h").agg([
        pl.col("ret").std().alias("rv"),
        ((pl.col("high").max() - pl.col("low").min()) / pl.col("close").mean()).alias("hl_range"),
        pl.col("signed_qty").std().alias("flow_var"),
        pl.col("vol").sum().alias("volume"),
        pl.col("n_trades").sum().alias("n_trades"),
        pl.col("close").mean().alias("mean_price"),
    ])
    
    hourly_chunks.append(hourly)
    
    if i % 5 == 0 or i == len(zips):
        print(f"  Processed {i}/{len(zips)} days")

# Combine and convert to pandas for the merge / plot
hourly_pl = pl.concat(hourly_chunks).sort("datetime")
hourly = hourly_pl.to_pandas().set_index("datetime")
print(f"Total hourly bins: {len(hourly)}")

# ----- Merge with on-chain flows -----
df = pd.concat([flows_hourly, hourly], axis=1)
df["net_inflow_usd"] = df["net_inflow_usd"].fillna(0)
df["outflow_usd"] = -df["net_inflow_usd"]
df["abs_flow_usd"] = df["net_inflow_usd"].abs()
df = df.dropna(subset=["rv", "mean_price"])
print(f"Merged bins (post-dropna): {len(df)}")

# ----- Lead-lag correlations -----
print("\n" + "="*70)
print("LEAD-LAG CORRELATIONS (positive lag = outflow then stress)")
print("="*70)

stress_metrics = ["rv", "hl_range", "flow_var", "volume", "n_trades"]
lags = [-6, -3, -1, 0, 1, 3, 6, 12, 24]

for flow_col in ["outflow_usd", "abs_flow_usd"]:
    print(f"\n--- Flow signal: {flow_col} ---")
    print(f"{'metric':<12} " + " ".join(f"{l:+d}h".rjust(7) for l in lags))
    for metric in stress_metrics:
        row = []
        for lag in lags:
            if lag >= 0:
                c = df[flow_col].corr(df[metric].shift(-lag))
            else:
                c = df[flow_col].shift(-lag).corr(df[metric])
            row.append(f"{c:+.3f}".rjust(7))
        print(f"{metric:<12} " + " ".join(row))

# ----- Event-window analysis -----
print("\n" + "="*70)
print("EVENT WINDOW: response to top-10 outflow hours")
print("="*70)
top_outflows = df["outflow_usd"].nlargest(10)
print("\nTop 10 outflow hours:")
for ts, val in top_outflows.items():
    print(f"  {ts}: ${val/1e6:.1f}M")

print(f"\n{'h_after':<8} {'rv':>10} {'hl_range':>10} {'volume_M':>10}")
for h in range(0, 7):
    rvs, hls, vols = [], [], []
    for ts in top_outflows.index:
        future = df.loc[df.index > ts].head(7)
        if len(future) > h:
            rvs.append(future["rv"].iloc[h])
            hls.append(future["hl_range"].iloc[h])
            vols.append(future["volume"].iloc[h])
    if rvs:
        print(f"{h:>4}h    {np.nanmean(rvs):>10.5f} {np.nanmean(hls):>10.5f} {np.nanmean(vols)/1e6:>10.2f}")

print(f"\nBaseline median across all hours:")
print(f"         {df['rv'].median():>10.5f} {df['hl_range'].median():>10.5f} {df['volume'].median()/1e6:>10.2f}")

# ----- Plot -----
fig, axes = plt.subplots(5, 1, figsize=(14, 13), sharex=True)

axes[0].plot(df.index, df["mean_price"], color="black", linewidth=1)
axes[0].set_ylabel("BTC/USDT")
axes[0].set_title("FTX Collapse: On-Chain Outflows vs Binance Liquidity Stress")

axes[1].bar(df.index, df["outflow_usd"] / 1e6, width=0.04, color="steelblue")
axes[1].set_ylabel("Net Outflow ($M/h)")
axes[1].axhline(0, color="black", linewidth=0.5)

axes[2].plot(df.index, df["rv"], color="purple", linewidth=0.8)
axes[2].set_ylabel("Realized Vol (1h)")

axes[3].plot(df.index, df["hl_range"], color="darkorange", linewidth=0.8)
axes[3].set_ylabel("High-Low Range")

axes[4].plot(df.index, df["volume"] / 1e6, color="darkgreen", linewidth=0.8)
axes[4].set_ylabel("Volume ($M/h)")
axes[4].set_xlabel("Date (UTC)")

events = [
    ("2022-11-02", "CoinDesk"),
    ("2022-11-06", "CZ tweet"),
    ("2022-11-08", "Binance LOI"),
    ("2022-11-09", "LOI off"),
    ("2022-11-10", "Halt"),
    ("2022-11-11", "Bankruptcy"),
]
for date, label in events:
    for ax in axes:
        ax.axvline(pd.Timestamp(date), color="red", linestyle="--", alpha=0.4, linewidth=0.8)
    axes[0].text(pd.Timestamp(date), df["mean_price"].max() * 0.99, label,
                 rotation=90, fontsize=8, va="top")

plt.tight_layout()
plt.savefig("stress_chart.png", dpi=120)
print("\nSaved: stress_chart.png")

# Save the merged hourly dataset for later use
df.to_parquet("data/hourly_merged.parquet")
print("Saved: data/hourly_merged.parquet")