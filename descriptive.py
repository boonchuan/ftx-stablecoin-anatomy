import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ----- Load on-chain flows -----
flows = pd.read_parquet("data/onchain/ftx_flows.parquet")
flows = flows[flows["wallet"] == "ftx_hot"].copy()
flows["datetime"] = pd.to_datetime(flows["datetime"])
flows["signed"] = np.where(flows["direction"]=="out", -flows["amount"], flows["amount"])

# Hourly aggregation
hourly = flows.set_index("datetime").resample("1h").agg(
    net_signed=("signed", "sum"),
    n_tx=("signed", "count"),
    abs_total=("amount", "sum"),
)
hourly["net_outflow"] = -hourly["net_signed"]  # positive = users withdrawing
hourly["cum_net_outflow"] = hourly["net_outflow"].cumsum()
print(f"Total hours: {len(hourly)}")

# Restrict to study window
hourly = hourly.loc["2022-10-15":"2022-11-25"]

# ===========================================================================
# Section 4.1: The slow-burn outflow pattern
# ===========================================================================
print("\n" + "="*70)
print("4.1 SLOW-BURN OUTFLOW PATTERN")
print("="*70)

pre_news_end = pd.Timestamp("2022-11-01 23:59:59")
news_to_halt = pd.Timestamp("2022-11-09 23:59:59")  # Nov 2 to Nov 9 inclusive
halt_period = pd.Timestamp("2022-11-25 00:00:00")

pre = hourly.loc[:pre_news_end]
during = hourly.loc["2022-11-02":news_to_halt]
post = hourly.loc["2022-11-10":]

print(f"Pre-news window (Oct 15 - Nov 1):")
print(f"  Hours: {len(pre)}")
print(f"  Total net outflow: ${pre['net_outflow'].sum()/1e6:.1f}M")
print(f"  Average hourly net outflow: ${pre['net_outflow'].mean()/1e6:.2f}M")

print(f"\nNews window (Nov 2 - Nov 9):")
print(f"  Hours: {len(during)}")
print(f"  Total net outflow: ${during['net_outflow'].sum()/1e6:.1f}M")
print(f"  Average hourly net outflow: ${during['net_outflow'].mean()/1e6:.2f}M")

print(f"\nPost-halt (Nov 10 onwards):")
print(f"  Hours: {len(post)}")
print(f"  Total net outflow: ${post['net_outflow'].sum()/1e6:.1f}M")
print(f"  Average hourly net outflow: ${post['net_outflow'].mean()/1e6:.2f}M")

# Date when cumulative outflow first turns positive (visible run begins)
run_start = hourly[hourly["cum_net_outflow"] > 0].index[0]
print(f"\nDate cumulative net outflow first turns positive: {run_start}")

# ===========================================================================
# Section 4.2: Activity concentration
# ===========================================================================
print("\n" + "="*70)
print("4.2 ACTIVITY CONCENTRATION")
print("="*70)

abs_flow = hourly["abs_total"].dropna().sort_values(ascending=False)
total = abs_flow.sum()

for pct in [1, 5, 10, 25, 50]:
    top_n = max(1, int(len(abs_flow) * pct / 100))
    top_share = abs_flow.head(top_n).sum() / total * 100
    print(f"  Top {pct}% of hours ({top_n} hours): {top_share:.1f}% of total flow")

# Gini coefficient
sorted_vals = np.sort(abs_flow.values)
n = len(sorted_vals)
cumvals = np.cumsum(sorted_vals)
gini = (2 * np.sum((np.arange(1, n+1)) * sorted_vals) / (n * np.sum(sorted_vals))) - (n + 1) / n
print(f"\n  Gini coefficient of hourly flow: {gini:.3f}")

# ===========================================================================
# Section 4.3: Pre-news vs post-news regime shift
# ===========================================================================
print("\n" + "="*70)
print("4.3 PRE-NEWS vs POST-NEWS REGIME SHIFT (in flow characteristics)")
print("="*70)

post_news = hourly.loc["2022-11-02":]

print(f"\n{'metric':<25} {'pre-news':>15} {'post-news':>15} {'ratio':>10}")
metrics = [
    ("Mean hourly flow ($M)", pre["abs_total"].mean()/1e6, post_news["abs_total"].mean()/1e6),
    ("Median hourly flow ($M)", pre["abs_total"].median()/1e6, post_news["abs_total"].median()/1e6),
    ("Std hourly flow ($M)", pre["abs_total"].std()/1e6, post_news["abs_total"].std()/1e6),
    ("Skewness", pre["abs_total"].skew(), post_news["abs_total"].skew()),
    ("Kurtosis", pre["abs_total"].kurt(), post_news["abs_total"].kurt()),
    ("Mean transactions/hour", pre["n_tx"].mean(), post_news["n_tx"].mean()),
    ("Max hourly flow ($M)", pre["abs_total"].max()/1e6, post_news["abs_total"].max()/1e6),
]
for name, p, q in metrics:
    ratio = q / p if p else float('nan')
    print(f"{name:<25} {p:>15.2f} {q:>15.2f} {ratio:>10.2f}x")

# ===========================================================================
# Section 4.4: The withdrawal halt visible on-chain
# ===========================================================================
print("\n" + "="*70)
print("4.4 WITHDRAWAL HALT VISIBLE ON-CHAIN")
print("="*70)

halt_window = hourly.loc["2022-11-07":"2022-11-12"]
print(f"\nHourly net outflow Nov 7 - Nov 12:")
print(halt_window[["net_outflow", "n_tx"]].assign(
    net_outflow_M = lambda x: x["net_outflow"]/1e6
)[["net_outflow_M", "n_tx"]].to_string())

# Identify hours with essentially zero activity
quiet = halt_window[halt_window["n_tx"] == 0]
print(f"\nHours with zero transactions in halt window: {len(quiet)}")
if len(quiet) > 0:
    print(f"  First quiet hour: {quiet.index.min()}")
    print(f"  Last quiet hour: {quiet.index.max()}")
    print(f"  Total quiet duration: {len(quiet)} hours")

# ===========================================================================
# Make the headline figure
# ===========================================================================
fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

# Panel A: Cumulative net outflow
axes[0].plot(hourly.index, hourly["cum_net_outflow"]/1e6, color="darkblue", linewidth=2)
axes[0].fill_between(hourly.index, 0, hourly["cum_net_outflow"]/1e6,
                     where=hourly["cum_net_outflow"]>0, alpha=0.2, color="blue")
axes[0].axhline(0, color="black", linewidth=0.5)
axes[0].set_ylabel("Cumulative Net Outflow ($M)")
axes[0].set_title("FTX Hot Wallet: Stablecoin Outflows During the November 2022 Collapse")

# Panel B: Hourly net outflow bars
axes[1].bar(hourly.index, hourly["net_outflow"]/1e6, width=0.04, color="steelblue")
axes[1].set_ylabel("Hourly Net Outflow ($M)")
axes[1].axhline(0, color="black", linewidth=0.5)

# Panel C: Transaction count
axes[2].plot(hourly.index, hourly["n_tx"], color="darkgreen", linewidth=1)
axes[2].set_ylabel("Transactions per Hour")
axes[2].set_xlabel("Date (UTC)")

events = [
    ("2022-11-02", "CoinDesk article"),
    ("2022-11-06", "CZ tweet"),
    ("2022-11-08", "Binance LOI"),
    ("2022-11-09", "LOI withdrawn"),
    ("2022-11-10", "Withdrawals halted"),
    ("2022-11-11", "Bankruptcy"),
]
for date, label in events:
    for ax in axes:
        ax.axvline(pd.Timestamp(date), color="red", linestyle="--", alpha=0.4, linewidth=0.8)
    axes[0].text(pd.Timestamp(date), hourly["cum_net_outflow"].max()/1e6 * 0.98,
                 label, rotation=90, fontsize=8, va="top")

plt.tight_layout()
plt.savefig("paper_figure_1.png", dpi=150)
print("\nSaved: paper_figure_1.png")

# Histogram of hourly flows (Section 4.2 figure)
fig2, ax = plt.subplots(figsize=(10, 6))
ax.hist(np.log10(abs_flow[abs_flow > 0].values), bins=50, color="darkblue", edgecolor="white")
ax.set_xlabel("log10(Hourly Absolute Flow, USD)")
ax.set_ylabel("Frequency")
ax.set_title("Distribution of Hourly Absolute Stablecoin Flows from FTX Hot Wallet")
plt.tight_layout()
plt.savefig("paper_figure_2.png", dpi=150)
print("Saved: paper_figure_2.png")

# Save the hourly aggregated data for the paper
hourly.to_csv("paper_data.csv")
print("Saved: paper_data.csv")
print("\nDone.")