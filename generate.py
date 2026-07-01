import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

# Set seeds for reproducibility
np.random.seed(42)
random.seed(42)

N = 10_000

# --- Base Distributions ---
regions = ["NA", "EMEA", "APAC", "LATAM"]
region_weights = [0.4, 0.3, 0.2, 0.1]

segments = ["SMB", "Enterprise", "Startup"]
segment_weights = [0.5, 0.3, 0.2]

products = ["Cloud_SaaS", "On_Prem_Hardware", "Consulting", "Support_ Retainer"]

# Generate base arrays
dates = pd.date_range(end=datetime(2023, 10, 31), periods=N, freq="h")
region_arr = np.random.choice(regions, N, p=region_weights)
segment_arr = np.random.choice(segments, N, p=segment_weights)
product_arr = np.random.choice(products, N)

# Base logic for realistic numbers
base_revenue = np.where(segment_arr == "Enterprise", 50000, 
               np.where(segment_arr == "SMB", 5000, 2000))
base_units = np.where(segment_arr == "Enterprise", 50, 
             np.where(segment_arr == "SMB", 10, 5))
base_marketing = np.where(segment_arr == "Enterprise", 5000, 1000)

# Add regional multiplier
region_mult = {"NA": 1.2, "EMEA": 1.0, "APAC": 0.8, "LATAM": 0.6}
rev_mult = np.vectorize(region_mult.get)(region_arr)

# Generate actuals with noise
actual_revenue = base_revenue * rev_mult + np.random.normal(0, base_revenue * 0.2, N)
actual_units = base_units + np.random.normal(0, base_units * 0.3, N)
actual_marketing = base_marketing * rev_mult + np.random.normal(0, base_marketing * 0.5, N)

# --- Q3 Artificial Dip (For temporal analysis & web verification hooks) ---
q3_mask = (dates.month >= 7) & (dates.month <= 9)
actual_revenue[q3_mask] = actual_revenue[q3_mask] * 0.65  # 35% drop in Q3

df = pd.DataFrame({
    "transaction_id": [f"TXN-{10000+i}" for i in range(N)],
    "date": dates,
    "region": region_arr,
    "customer_segment": segment_arr,
    "product_category": product_arr,
    "marketing_spend": np.round(actual_marketing, 2),
    "units_sold": np.round(actual_units).astype(int),
    "unit_price": np.round(actual_revenue / actual_units, 2),
    "total_revenue": np.round(actual_revenue, 2),
    "customer_tenure_days": np.random.randint(30, 2000, N),
    "churn_risk_score": np.random.uniform(0, 1, N)
})

# =====================================================================
# PHASE 2: INJECT ANOMALIES (The "Messy" Part)
# =====================================================================

# 1. Messy Dates: Mix formats for 5% of rows
messy_date_idx = random.sample(range(N), int(N * 0.05))
for i in messy_date_idx:
    d = df.loc[i, 'date']
    if random.random() > 0.5:
        df.loc[i, 'date'] = d.strftime('%m/%d/%Y')  # US format
    else:
        df.loc[i, 'date'] = d.strftime('%d-%b-%y')   # UK abbreviated

# 2. Messy Strings: Inconsistent casing in segments
case_idx = random.sample(range(N), int(N * 0.1))
df.loc[case_idx, 'customer_segment'] = df.loc[case_idx, 'customer_segment'].str.lower()

# 3. Messy Currency: Add '$' and ',' to 3% of unit_prices
currency_idx = random.sample(range(N), int(N * 0.03))
df['unit_price'] = df['unit_price'].astype(object)
df.loc[currency_idx, 'unit_price'] = "$" + df.loc[currency_idx, 'unit_price'].astype(str).str.replace('.', ',', regex=False)

# 4. Missing Values (MCAR & MNAR)
# MCAR (Completely random)
df.loc[random.sample(range(N), int(N * 0.04)), 'marketing_spend'] = np.nan
df.loc[random.sample(range(N), int(N * 0.02)), 'units_sold'] = np.nan

# MNAR (Missing Not At Random - Support contracts have no marketing spend)
support_mask = df['product_category'] == "Support_ Retainer"
support_idx = df[support_mask].index.tolist()
df.loc[random.sample(support_idx, int(len(support_idx) * 0.6)), 'marketing_spend'] = np.nan

# 5. Exact Duplicates
dupes = df.loc[random.sample(range(N), 50)]
df = pd.concat([df, dupes], ignore_index=True)

# 6. Outliers (Errors vs. Valid Extremes)
# Error: 10x revenue mistake
error_idx = random.sample(range(N), 15)
df.loc[error_idx, 'total_revenue'] = df.loc[error_idx, 'total_revenue'] * 10

# Valid Extreme: Enterprise mega-deals
mega_idx = random.sample(range(N), 10)
df.loc[mega_idx, 'total_revenue'] = 500000.0
df.loc[mega_idx, 'units_sold'] = 500

# Error: Negative units sold
neg_idx = random.sample(range(N), 5)
df.loc[neg_idx, 'units_sold'] = -df.loc[neg_idx, 'units_sold']

# 7. Churn Score Bounds Violations
df.loc[random.sample(range(N), 20), 'churn_risk_score'] = 1.5
df.loc[random.sample(range(N), 20), 'churn_risk_score'] = -0.2

# 8. Spurious Correlation Injection
# Make marketing spend perfectly correlated to a random subset's revenue
spoof_idx = random.sample(range(N), 500)
df.loc[spoof_idx, 'total_revenue'] = df.loc[spoof_idx, 'marketing_spend'] * 5.5

# =====================================================================
# PHASE 3: SAVE
# =====================================================================

# Shuffle to make it not perfectly ordered
df = df.sample(frac=1).reset_index(drop=True)

out_path = "Ai-Analysis-Engine/src/test_data/complex_10k.csv"
df.to_csv(out_path, index=False)

print(f"Generated {len(df)} rows.")
print(f"Saved to {out_path}")
print("\n--- Data Summary ---")
print(f"Missing Values:\n{df.isnull().sum()}\n")
print(f"Dtypes:\n{df.dtypes}")