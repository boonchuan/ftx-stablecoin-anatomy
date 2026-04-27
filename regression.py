import pandas as pd
import numpy as np
import statsmodels.api as sm

# Load merged hourly data we saved earlier
df = pd.read_parquet("data/hourly_merged.parquet")
print(f"Loaded {len(df)} hourly observations")

# Define the windows
df["log_volume"] = np.log(df["volume"].replace(0, np.nan))
df["log_volume_lag1"] = df["log_volume"].shift(1)
df["log_abs_flow"] = np.log(df["abs_flow_usd"] + 1)  # +1 to handle zeros
df["log_abs_flow_lag1"] = df["log_abs_flow"].shift(1)
df["log_abs_flow_lag3"] = df["log_abs_flow"].shift(3)
df["log_abs_flow_lag6"] = df["log_abs_flow"].shift(6)
df["log_abs_flow_lag12"] = df["log_abs_flow"].shift(12)
df["hour_of_day"] = df.index.hour

stress_window = (df.index >= "2022-11-01") & (df.index <= "2022-11-14")
df["is_stress"] = stress_window.astype(int)

print("\n" + "="*70)
print("REGRESSION: log(volume_t) on log(|flow|) at multiple lags")
print("Standard errors: Newey-West HAC (lag=24)")
print("="*70)

specs = [
    ("Spec 1: contemporaneous only",
     ["log_abs_flow"]),
    ("Spec 2: 1h lag only",
     ["log_abs_flow_lag1"]),
    ("Spec 3: 1h lag + AR(1) volume",
     ["log_abs_flow_lag1", "log_volume_lag1"]),
    ("Spec 4: multiple lags",
     ["log_abs_flow_lag1", "log_abs_flow_lag3", "log_abs_flow_lag6", "log_abs_flow_lag12"]),
    ("Spec 5: 1h lag + AR(1) + hour FE",
     ["log_abs_flow_lag1", "log_volume_lag1"]),  # add hour dummies below
    ("Spec 6: 1h lag interacted with stress window",
     ["log_abs_flow_lag1", "is_stress", "log_volume_lag1"]),
]

results = {}
for name, features in specs:
    work = df.dropna(subset=["log_volume"] + features).copy()
    
    if "hour FE" in name:
        hour_dummies = pd.get_dummies(work["hour_of_day"], prefix="h", drop_first=True).astype(float)
        X = pd.concat([work[features], hour_dummies], axis=1)
    elif "interacted" in name:
        work["flow_x_stress"] = work["log_abs_flow_lag1"] * work["is_stress"]
        X = work[features + ["flow_x_stress"]]
    else:
        X = work[features]
    
    X = sm.add_constant(X)
    y = work["log_volume"]
    
    model = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": 24})
    results[name] = model
    
    print(f"\n--- {name} ---")
    print(f"N = {int(model.nobs)},  R² = {model.rsquared:.4f},  Adj R² = {model.rsquared_adj:.4f}")
    print(f"{'variable':<28} {'coef':>10} {'se':>10} {'t':>8} {'p':>8}")
    for var in model.params.index:
        if var.startswith("h_"):  # skip hour dummies in display
            continue
        coef = model.params[var]
        se = model.bse[var]
        t = model.tvalues[var]
        p = model.pvalues[var]
        sig = "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else ""
        print(f"{var:<28} {coef:>+10.4f} {se:>10.4f} {t:>+8.2f} {p:>8.4f} {sig}")

# ---- Stress window subsample regression ----
print("\n" + "="*70)
print("SUBSAMPLE: stress window only (Nov 1 - Nov 14)")
print("="*70)

stress_df = df.loc["2022-11-01":"2022-11-14"].dropna(subset=["log_volume", "log_abs_flow_lag1", "log_volume_lag1"])
X = sm.add_constant(stress_df[["log_abs_flow_lag1", "log_volume_lag1"]])
y = stress_df["log_volume"]
model_stress = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": 24})

print(f"N = {int(model_stress.nobs)},  R² = {model_stress.rsquared:.4f}")
print(f"{'variable':<28} {'coef':>10} {'se':>10} {'t':>8} {'p':>8}")
for var in model_stress.params.index:
    coef = model_stress.params[var]
    se = model_stress.bse[var]
    t = model_stress.tvalues[var]
    p = model_stress.pvalues[var]
    sig = "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else ""
    print(f"{var:<28} {coef:>+10.4f} {se:>10.4f} {t:>+8.2f} {p:>8.4f} {sig}")

# ---- Pre-stress window subsample for comparison ----
print("\n" + "="*70)
print("SUBSAMPLE: pre-stress window only (Oct 15 - Oct 31)")
print("="*70)

pre_df = df.loc["2022-10-15":"2022-10-31"].dropna(subset=["log_volume", "log_abs_flow_lag1", "log_volume_lag1"])
X = sm.add_constant(pre_df[["log_abs_flow_lag1", "log_volume_lag1"]])
y = pre_df["log_volume"]
model_pre = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": 24})

print(f"N = {int(model_pre.nobs)},  R² = {model_pre.rsquared:.4f}")
print(f"{'variable':<28} {'coef':>10} {'se':>10} {'t':>8} {'p':>8}")
for var in model_pre.params.index:
    coef = model_pre.params[var]
    se = model_pre.bse[var]
    t = model_pre.tvalues[var]
    p = model_pre.pvalues[var]
    sig = "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else ""
    print(f"{var:<28} {coef:>+10.4f} {se:>10.4f} {t:>+8.2f} {p:>8.4f} {sig}")

print("\nDone.")