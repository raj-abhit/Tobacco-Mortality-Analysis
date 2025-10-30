import pandas as pd
import numpy as np

# Load merged file
df = pd.read_csv("merged.csv")

def clean_numeric(val):
    if pd.isna(val):
        return np.nan
    s = str(val).replace(",", "").strip()
    if "/" in s:  # handle things like 2004/05
        parts = [p for p in s.split("/") if p.isdigit()]
        if parts:
            nums = [int(p) for p in parts]
            return sum(nums) / len(nums)
    try:
        return float(s)
    except:
        return np.nan

# Clean Year column if present
if "Year" in df.columns:
    df["Year"] = df["Year"].apply(clean_numeric)

# Convert all possible numeric columns
for col in df.columns:
    try:
        df[col] = df[col].apply(clean_numeric)
    except:
        pass

# Save cleaned version
df.to_csv("merged_cleaned.csv", index=False)
print("✅ Cleaned data saved as 'merged_cleaned.csv'")
