
import pandas as pd

INPUT_PATH    = "results/dataset_summary.csv"
FILTERED_PATH = "results/filtered_datasets.csv"
EXCLUDED_PATH = "results/excluded_datasets.csv"

# Thresholds
MIN_SAMPLES   = 3_000
MIN_FEATURES  = 4
MAX_DIM_RATIO = 0.10 

# ── Loading the datasetSummary
df = pd.read_csv(INPUT_PATH)
df["exclusion_reason"] = ""

# Filter 1: Minimum samples
mask = df["n_samples"] < MIN_SAMPLES
df.loc[mask, "exclusion_reason"] += "too_few_samples; "

# Filter 2: Minimum features
mask = df["n_features"] < MIN_FEATURES
df.loc[mask, "exclusion_reason"] += "too_few_features; "

# Filter 3: d/n ratio
df["dim_ratio"] = df["n_features"] / df["n_samples"]
mask = df["dim_ratio"] >= MAX_DIM_RATIO
df.loc[mask, "exclusion_reason"] += "high_dimensional; "

# Filter 4: Missing data
mask = df["has_missing"] == True
df.loc[mask, "exclusion_reason"] += "has_missing_data; "

# Split into kept / excluded
excluded = df[df["exclusion_reason"] != ""].copy()
filtered = df[df["exclusion_reason"] == ""].drop(columns=["exclusion_reason"])

# Saving the filtered and excluded files separately 
filtered.to_csv(FILTERED_PATH, index=False)
excluded.to_csv(EXCLUDED_PATH, index=False)
