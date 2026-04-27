import pandas as pd, numpy as np
df = pd.read_parquet("data/hourly_merged_v2.parquet")
if df.index.tz is None: df.index = df.index.tz_localize("UTC")
df["net_outflow_usd"] = -df["net_inflow_usd"]

T_PRE_S, T_PRE_E   = pd.Timestamp("2022-10-15", tz="UTC"), pd.Timestamp("2022-11-01 23:59", tz="UTC")
T_NEWS_S, T_NEWS_E = pd.Timestamp("2022-11-02", tz="UTC"), pd.Timestamp("2022-11-09 23:59", tz="UTC")
T_POST_S, T_POST_E = pd.Timestamp("2022-11-10", tz="UTC"), pd.Timestamp("2022-11-25 23:59", tz="UTC")

def stats(df_slice, label):
    n_hours = len(df_slice)
    total_outflow = df_slice["net_outflow_usd"].sum() / 1e6
    hourly_mean = df_slice["net_outflow_usd"].mean() / 1e6
    abs_mean = df_slice["abs_flow_usd"].mean() / 1e6
    abs_median = df_slice["abs_flow_usd"].median() / 1e6
    abs_std = df_slice["abs_flow_usd"].std() / 1e6
    abs_max = df_slice["abs_flow_usd"].max() / 1e6
    abs_skew = df_slice["abs_flow_usd"].skew()
    abs_kurt = df_slice["abs_flow_usd"].kurt()
    tx_mean = df_slice["onchain_tx_count"].mean()
    print(f"\n=== {label} ===")
    print(f"  Hours:                {n_hours}")
    print(f"  Total net outflow:    ${total_outflow:.1f}M")
    print(f"  Hourly mean outflow:  ${hourly_mean:.2f}M")
    print(f"  Abs flow mean:        ${abs_mean:.2f}M")
    print(f"  Abs flow median:      ${abs_median:.2f}M")
    print(f"  Abs flow std:         ${abs_std:.2f}M")
    print(f"  Abs flow max:         ${abs_max:.2f}M")
    print(f"  Abs flow skew:        {abs_skew:.2f}")
    print(f"  Abs flow kurt:        {abs_kurt:.2f}")
    print(f"  Mean tx/hour:         {tx_mean:.1f}")

stats(df.loc[T_PRE_S:T_PRE_E],   "PRE-NEWS  Oct 15 - Nov 1")
stats(df.loc[T_NEWS_S:T_NEWS_E], "NEWS      Nov 2 - Nov 9")
stats(df.loc[T_POST_S:T_POST_E], "POST-HALT Nov 10+")

# Concentration / Gini using onchain abs flow
# (same dataset the paper currently reports on)
abs_flows = df.loc[T_PRE_S:T_POST_E, "abs_flow_usd"].sort_values(ascending=False).values
total = abs_flows.sum()
n = len(abs_flows)
print(f"\n=== CONCENTRATION (Oct 15 - Nov 25, n={n} hours) ===")
for pct in [1, 5, 10, 25, 50]:
    k = max(1, int(round(n * pct/100)))
    share = abs_flows[:k].sum() / total * 100
    print(f"  Top {pct:>2}%: {k:>3} hours, {share:.1f}% of flow")

# Gini
sorted_flows = np.sort(df.loc[T_PRE_S:T_POST_E, "abs_flow_usd"].values)
n = len(sorted_flows)
cum = np.cumsum(sorted_flows)
gini = (n + 1 - 2 * cum.sum() / cum[-1]) / n if cum[-1] > 0 else 0
print(f"  Gini coefficient:        {gini:.3f}")

# Halt timeline
print(f"\n=== HALT WINDOW (Nov 8 17:00 -> Nov 13 00:00) ===")
halt_view = df.loc[pd.Timestamp("2022-11-08 17:00", tz="UTC"):
                   pd.Timestamp("2022-11-13 00:00", tz="UTC"), "onchain_tx_count"]
zero_hours = halt_view[halt_view == 0]
print(f"  Total hours in window:   {len(halt_view)}")
print(f"  Zero-tx hours:           {len(zero_hours)}")
if len(zero_hours):
    print(f"  First zero-tx hour:      {zero_hours.index.min()}")
    print(f"  Last zero-tx hour:       {zero_hours.index.max()}")
