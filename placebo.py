import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the merged dataset we saved earlier
df = pd.read_parquet("data/hourly_merged.parquet")
print(f"Loaded {len(df)} hourly bins")

# The result we want to test
TARGET_LAGS = [0, 1, 3, 6, 12, 24]
N_PERMUTATIONS = 1000
RNG_SEED = 42

print("\n" + "="*70)
print("OBSERVED CORRELATIONS (abs_flow_usd vs volume)")
print("="*70)
observed = {}
for lag in TARGET_LAGS:
    c = df["abs_flow_usd"].corr(df["volume"].shift(-lag))
    observed[lag] = c
    print(f"  lag +{lag:>2}h:  corr = {c:+.4f}")

# ---- Placebo Test 1: Random circular shifts ----
# Rotate the abs_flow series by a random offset and recompute correlations
# This preserves the autocorrelation structure of both series
print("\n" + "="*70)
print(f"PLACEBO TEST 1: Random circular shifts ({N_PERMUTATIONS} permutations)")
print("Preserves autocorrelation in both series; breaks alignment.")
print("="*70)

rng = np.random.default_rng(RNG_SEED)
n = len(df)
flow = df["abs_flow_usd"].values
vol = df["volume"].values

placebo_corrs = {lag: [] for lag in TARGET_LAGS}
for _ in range(N_PERMUTATIONS):
    shift = rng.integers(24, n - 24)  # avoid trivial near-zero shifts
    rolled = np.roll(flow, shift)
    flow_series = pd.Series(rolled, index=df.index)
    vol_series = pd.Series(vol, index=df.index)
    for lag in TARGET_LAGS:
        c = flow_series.corr(vol_series.shift(-lag))
        placebo_corrs[lag].append(c)

print(f"\n{'lag':<6} {'observed':>10} {'placebo_mean':>14} {'placebo_p5':>12} {'placebo_p95':>12} {'p-value':>10}")
for lag in TARGET_LAGS:
    obs = observed[lag]
    placebo = np.array(placebo_corrs[lag])
    p_value = (np.abs(placebo) >= abs(obs)).mean()
    p5, p95 = np.percentile(placebo, [5, 95])
    star = " ***" if p_value < 0.01 else " **" if p_value < 0.05 else " *" if p_value < 0.10 else ""
    print(f"+{lag:<5}h {obs:>+10.4f} {placebo.mean():>+14.4f} {p5:>+12.4f} {p95:>+12.4f} {p_value:>10.4f}{star}")

# ---- Placebo Test 2: Pre-stress window only ----
# Use only the period BEFORE FTX stress was visible (Oct 15 - Oct 31)
# If correlation exists in normal times too, it's not FTX-specific
print("\n" + "="*70)
print("PLACEBO TEST 2: Pre-stress window (Oct 15 - Oct 31)")
print("If our result is FTX-specific, correlation should be weaker here.")
print("="*70)

pre = df.loc["2022-10-15":"2022-10-31"]
print(f"Pre-stress sample size: {len(pre)} hours")
print(f"\n{'lag':<6} {'observed':>10} {'pre-stress':>12}")
for lag in TARGET_LAGS:
    full = observed[lag]
    pre_corr = pre["abs_flow_usd"].corr(pre["volume"].shift(-lag))
    print(f"+{lag:<5}h {full:>+10.4f} {pre_corr:>+12.4f}")

# ---- Placebo Test 3: Stress window only ----
# Use only the FTX collapse window (Nov 1 - Nov 14)
# This is where the result SHOULD be strongest
print("\n" + "="*70)
print("PLACEBO TEST 3: FTX stress window (Nov 1 - Nov 14)")
print("If our result is FTX-specific, correlation should be STRONGER here.")
print("="*70)

stress = df.loc["2022-11-01":"2022-11-14"]
print(f"Stress sample size: {len(stress)} hours")
print(f"\n{'lag':<6} {'full':>10} {'stress':>10}")
for lag in TARGET_LAGS:
    full = observed[lag]
    s = stress["abs_flow_usd"].corr(stress["volume"].shift(-lag))
    print(f"+{lag:<5}h {full:>+10.4f} {s:>+10.4f}")

# ---- Robustness: drop top-N outflow events ----
print("\n" + "="*70)
print("ROBUSTNESS: drop largest outflow hours")
print("Tests whether result is driven by a few extreme events.")
print("="*70)

print(f"\n{'lag':<6} {'all':>10} {'drop_top5':>12} {'drop_top10':>12} {'drop_top20':>12}")
for lag in TARGET_LAGS:
    full = observed[lag]
    
    top5_idx = df["abs_flow_usd"].nlargest(5).index
    no5 = df.drop(top5_idx)
    c5 = no5["abs_flow_usd"].corr(no5["volume"].shift(-lag))
    
    top10_idx = df["abs_flow_usd"].nlargest(10).index
    no10 = df.drop(top10_idx)
    c10 = no10["abs_flow_usd"].corr(no10["volume"].shift(-lag))
    
    top20_idx = df["abs_flow_usd"].nlargest(20).index
    no20 = df.drop(top20_idx)
    c20 = no20["abs_flow_usd"].corr(no20["volume"].shift(-lag))
    
    print(f"+{lag:<5}h {full:>+10.4f} {c5:>+12.4f} {c10:>+12.4f} {c20:>+12.4f}")

# ---- Visualize the +1h lag distribution under permutation ----
fig, ax = plt.subplots(figsize=(10, 6))
placebo_1h = np.array(placebo_corrs[1])
ax.hist(placebo_1h, bins=40, color="lightgray", edgecolor="black", alpha=0.7, label="Placebo distribution")
ax.axvline(observed[1], color="red", linewidth=2, label=f"Observed: {observed[1]:+.3f}")
ax.axvline(np.percentile(placebo_1h, 5), color="blue", linestyle="--", alpha=0.5, label="5th/95th percentile")
ax.axvline(np.percentile(placebo_1h, 95), color="blue", linestyle="--", alpha=0.5)
ax.set_xlabel("Correlation (abs_flow vs volume, +1h lag)")
ax.set_ylabel("Frequency")
ax.set_title(f"Placebo Test: Random Shifts ({N_PERMUTATIONS} permutations)\np-value = {(np.abs(placebo_1h) >= abs(observed[1])).mean():.4f}")
ax.legend()
plt.tight_layout()
plt.savefig("placebo_distribution.png", dpi=120)
print("\nSaved: placebo_distribution.png")
print("\nDone.")