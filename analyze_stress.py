import zipfile
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ----- Load on-chain flows -----
flows = pd.read_parquet("data/onchain/ftx_flows.parquet")
flows = flows[flows["wallet"] == "ftx_hot"].copy()
flows["datetime"] = pd.to_datetime(flows["datetime"])
flows["signed"] = flows.apply(
    lambda r: -r["amount"] if r["direction"] == "out" else r["amount"], axis=1
)
flows_hourly = (
    flows.set_index("datetime")
    .resample("1h")["signed"].sum()
    .rename("net_inflow_usd")
)
outflow_usd = (-flows_hourly).rename("outflow_usd")
abs_flow = flows_hourly.abs().rename("abs_flow_usd")
print("On-chain hourly bins:", len(flows_hourly))

# ----- Load Binance aggTrades -----
agg_dir = Path("data/binance_aggtrades")
zips = sorted(agg_dir.glob("*.zip"))
print(f"Loading {len(zips)} aggTrades zips...")

frames = []
for z in zips:
    with zipfile.ZipFile(z) as zf:
        name = zf.namelist()[0]
        df = pd.read_csv(
            zf.open(name),
            header=None,
            names=["agg_id", "price", "qty", "first_id", "last_id", "ts_ms", "is_buyer_maker", "is_best_match"],
        )
        frames.append(df)
trades = pd.concat(frames, ignore_index=True)
print(f"Total trades: {len(trades):,}")

trades["datetime"] = pd.to_datetime(trades["ts_ms"], unit="ms")
trades["notional"] = trades["price"] * trades["qty"]
trades["signed_qty"] = trades.apply(
    lambda r: -r["qty"] if r["is_buyer_maker"] else r["qty"], axis=1
)
trades = trades.set_index("datetime").sort_index()

# ----- Build hourly stress metrics -----
print("Computing stress metrics...")

# 1-min bars first, then aggregate to hourly stress measures
minute = trades.resample("1min").agg(
    open_=("price", "first"),
    high=("price", "max"),
    low=("price", "min"),
    close=("price", "last"),
    vol=("notional", "sum"),
    n_trades=("price", "count"),
    signed_qty=("signed_qty", "sum"),
).dropna()

minute["ret"] = np.log(minute["close"]).diff()

# Hourly stress metrics
def hourly_metrics(g):
    if len(g) < 2:
        return pd.Series({k: np.nan for k in ["rv","hl_range","flow_var","tail_count","volume","n_trades","mean_price"]})
    rv = g["ret"].std() * np.sqrt(60)  # annualize within-hour roughly
    price_ref = g["close"].mean()
    hl = (g["high"].max() - g["low"].min()) / price_ref
    flow_var = g["signed_qty"].std()
    sigma = g["ret"].std()
    tail = (g["ret"].abs() > 3 * sigma).sum() if sigma > 0 else 0
    return pd.Series({
        "rv": rv,
        "hl_range": hl,
        "flow_var": flow_var,
        "tail_count": tail,
        "volume": g["vol"].sum(),
        "n_trades": g["n_trades"].sum(),
        "mean_price": price_ref,
    })

hourly = minute.groupby(pd.Grouper(freq="1h")).apply(hourly_metrics).dropna()
print(f"Hourly stress bins: {len(hourly)}")

# ----- Merge -----
df = pd.concat([outflow_usd, abs_flow, hourly], axis=1).dropna(subset=["rv"])
df["outflow_usd"] = df["outflow_usd"].fillna(0)
df["abs_flow_usd"] = df["abs_flow_usd"].fillna(0)

# ----- Lead-lag table for each stress metric -----
print("\n" + "="*70)
print("LEAD-LAG CORRELATIONS")
print("Outflow leads stress by k hours (positive k = outflow then stress)")
print("="*70)

stress_metrics = ["rv", "hl_range", "flow_var", "tail_count", "volume", "n_trades"]
lags = [-6, -3, -1, 0, 1, 3, 6, 12, 24]

# Two flow signals: net outflow (signed) and abs flow magnitude
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

# ----- Plot stress vs outflow -----
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

# ----- Event window: response to top-N outflow spikes -----
print("\n" + "="*70)
print("EVENT WINDOW ANALYSIS")
print("="*70)
top_outflows = df["outflow_usd"].nlargest(10)
print(f"\nTop 10 outflow hours:")
print(top_outflows.apply(lambda x: f"${x/1e6:.1f}M"))

print("\nResponse over next 6 hours (mean across top-10 events):")
print(f"{'h_after':<8} {'rv':>10} {'hl_range':>10} {'volume_M':>10}")
for h in range(0, 7):
    rv_resp = []
    hl_resp = []
    vol_resp = []
    for ts in top_outflows.index:
        future = df.loc[df.index > ts].head(7)
        if len(future) > h:
            rv_resp.append(future["rv"].iloc[h])
            hl_resp.append(future["hl_range"].iloc[h])
            vol_resp.append(future["volume"].iloc[h])
    if rv_resp:
        print(f"{h:>4}h    {np.mean(rv_resp):>10.4f} {np.mean(hl_resp):>10.4f} {np.mean(vol_resp)/1e6:>10.2f}")

# Compare to baseline (all hours not in event windows)
baseline_rv = df["rv"].median()
baseline_hl = df["hl_range"].median()
baseline_vol = df["volume"].median() / 1e6
print(f"\nBaseline (median across all hours):")
print(f"         {baseline_rv:>10.4f} {baseline_hl:>10.4f} {baseline_vol:>10.2f}")

print("\nDone.")