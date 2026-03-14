# ==============================================================================
# This is how we are cleaning the data 
# ==============================================================================

# ── Loading libraries ─────────────────────────────────────────────────────────
import pandas as pd
import os

# ── Load data ─────────────────────────────────────────────────────────────────
RAW_DATA_PATH = 'data/raw/chicago_crimes_2001_2025_raw.csv'

if not os.path.exists(RAW_DATA_PATH):
    print("Raw data not found. Please run download_data.py first.")
    exit(1)

print("Loading raw data...")
df = pd.read_csv(RAW_DATA_PATH, low_memory=False)

# ── Basic info ────────────────────────────────────────────────────────────────
print(df.describe())
print(df.shape)
print(df.dtypes)
print(df.columns.tolist())

# ── Filter to 2001-2025 ───────────────────────────────────────────────────────
df = df[df['year'].between(2001, 2025)]
print(df['year'].value_counts().sort_index())
print("\nTotal records (2001-2025):", len(df))

# ── Missing value summary ─────────────────────────────────────────────────────
missing = df.isnull().sum()
missing_pct = (missing / len(df) * 100).round(2)

missing_summary = pd.DataFrame({
    "missing_count": missing,
    "missing_pct": missing_pct
}).sort_values("missing_pct", ascending=False)

missing_summary

# ── Data type checks ──────────────────────────────────────────────────────────
# fbi_code and iucr are strings, which is correct - they are category codes
print(df.dtypes)
df.info()

# Verify arrest and domestic are boolean
print("arrest unique values:", df["arrest"].unique())
print("domestic unique values:", df["domestic"].unique())
#ok good

# ── Type conversions ──────────────────────────────────────────────────────────

# date-time conversions

date_series = []
for date in df['date']:
    date_series.append(date[0:10])

time_series = []
for time in df['date']:
    time_series.append(time[11:19])

df['date'] = date_series
df['time'] = time_series

df['date'] = pd.to_datetime(df['date'], format = "%Y-%m-%d")
df['time'] = pd.to_datetime(df['time'], format = "%H:%M:%S").dt.time

df[['date', 'time']].tail(10)

# district, ward, community_area should be integer
# Int64 (capital I) is used because these columns have missing values
# regular int64 cannot store NaN, Int64 can
df["district"]       = pd.to_numeric(df["district"], errors="coerce").astype("Int64")
df["ward"]           = pd.to_numeric(df["ward"], errors="coerce").astype("Int64")
df["community_area"] = pd.to_numeric(df["community_area"], errors="coerce").astype("Int64")

# Recode categorical text columns to save memory
df["primary_type"]        = df["primary_type"].astype("category")
df["description"]         = df["description"].astype("category")
df["location_description"] = df["location_description"].astype("category")

# Verify missing values did not increase after type conversions
missing = df.isnull().sum()
missing_pct = (missing / len(df) * 100).round(2)
missing_summary = pd.DataFrame({
    "missing_count": missing,
    "missing_pct": missing_pct
}).sort_values("missing_pct", ascending=False)
missing_summary

# ── Remove coordinates outside Chicago ───────────────────────────────────────

# we first remove the location column: is redundant ('y-coordinate' and
# 'x-coordinate' already have the same information).

df = df.drop(columns=['location'])

# ── Remove coordinates outside Chicago ───────────────────────────────────────
# Valid Chicago latitude range: 41.6 - 42.1
# Valid Chicago longitude range: -87.9 - -87.5
print(f'Observations with invalid latitude coordinates: ',len(df[~df['latitude'].between(41.6, 42.1)]))
print(f'Obsevations with null values in latitude coordinates: ',len(df[df['latitude'].isnull()]))
print(f'Thus, observations with non-null values yet out-of-range latitude coordinates:',
      len(df[~df['latitude'].between(41.6, 42.1)]) - len(df[df['latitude'].isnull()]))

print(f'Observations with invalid longitude coordinates: ',len(df[~df['longitude'].between(-87.9, -87.5)]))
print(f'Obsevations with null values in longitude coordinates: ',len(df[df['longitude'].isnull()]))
print(f'Thus, observations with non-null values yet out-of-range longitude coordinates:',
      len(df[~df['longitude'].between(-87.9, -87.5)]) - len(df[df['longitude'].isnull()]))

# It would be safe to delete observations with null-values in either latitude or longitude.
# Deleting 149 observations without valid latitude coordinates might be safe, but doing so
# for 27759 observations without valid longitud coordinates might not. Further exploration
# might be required before making a decision.

# ── Missing data analysis ─────────────────────────────────────────────────────

# Total rows with at least one missing value
print("Rows with at least one missing value:", df.isnull().any(axis=1).sum())
print("That is", round(df.isnull().any(axis=1).sum() / len(df) * 100, 2), "% of the data")

# Which columns are driving the missing values
print("\nMissing values per column for incomplete rows:")
print(df[df.isnull().any(axis=1)][df.columns].isnull().sum())

# ── Is missing data concentrated in certain years? ────────────────────────────
df["has_missing"] = df.isnull().any(axis=1)

missing_by_year = df.groupby("year")["has_missing"].agg(["sum", "count"])
missing_by_year["pct"] = (missing_by_year["sum"] / missing_by_year["count"] * 100).round(2)
missing_by_year.columns = ["missing_count", "total_rows", "missing_pct"]
print(missing_by_year)
# Note: 2001 has high missing data - worth considering in analysis

# Which specific columns are missing by year
df.groupby("year")[["latitude", "longitude", "ward", "community_area"]].apply(lambda x: x.isnull().sum())

#double checking for things that may be considered full but just empty text, seems fine
# ── Check for empty strings masquerading as non-null ─────────────────────────
for col in df.select_dtypes(include="object").columns:
    empty_count = (df[col].astype(str).str.strip() == "").sum()
    nan_string  = (df[col].astype(str) == "nan").sum()
    if empty_count > 0 or nan_string > 0:
        print(col, "- empty strings:", empty_count, "| 'nan' strings:", nan_string)

# ── Save cleaned data ─────────────────────────────────────────────────────────
df = df.drop(columns=["has_missing"])

CLEAN_DATA_PATH = 'data/cleaned/chicago_crimes_cleaned.csv'
os.makedirs('data/cleaned', exist_ok=True)
df.to_csv(CLEAN_DATA_PATH, index=False)
print(f"Cleaned data saved to {CLEAN_DATA_PATH}")
