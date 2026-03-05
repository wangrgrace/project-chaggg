#Comment: this is how we originally downloaded the data from the chicago data portal

import requests
import pandas as pd
import time
import os
import json

# Download Data
# the results will be stored in the folder "data"
# the folder structure should be:
# project-chaggg/
# ├── data/
# │   └── chicago_crime_data/
# │       ├── batches/
# │       └── chicago_crimes_2001_2025.csv

# ── Config ────────────────────────────────────────────────────────────────────
APP_TOKEN    = "YOUR APP TOKEN HERE"  # Get your token from Chicago Data portal
BASE_URL     = "https://data.cityofchicago.org/resource/ijzp-q8t2.json"
BATCH_SIZE   = 50000
BASE_DIR = os.getcwd()
OUTPUT_DIR   = os.path.join(BASE_DIR, "data", "chicago_crime_data")
FINAL_OUTPUT = os.path.join(OUTPUT_DIR, "chicago_crimes_2001_2025.csv")
PROGRESS_FILE = os.path.join(OUTPUT_DIR, "progress.json")

BATCH_DIR = os.path.join(OUTPUT_DIR, "batches")
os.makedirs(BATCH_DIR, exist_ok=True)

EXPECTED_COLUMNS = [
    'id', 'case_number', 'date', 'block', 'iucr', 'primary_type',
    'description', 'location_description', 'arrest', 'domestic', 'beat',
    'district', 'ward', 'community_area', 'fbi_code', 'year', 'updated_on',
    'x_coordinate', 'y_coordinate', 'latitude', 'longitude', 'location'
]

# ── Progress helpers ──────────────────────────────────────────────────────────
def load_progress():
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE) as f:
            return json.load(f)
    return {"offset": 0, "batch_num": 1, "total_records": 0}

def save_progress(offset, batch_num, total):
    with open(PROGRESS_FILE, "w") as f:
        json.dump({"offset": offset, "batch_num": batch_num, "total_records": total}, f)

# ── Fetch ─────────────────────────────────────────────────────────────────────
def fetch_batch(offset, retries=3):
    params = {
        "$limit":      BATCH_SIZE,
        "$offset":     offset,
        "$order":      "date ASC",
        "$$app_token": APP_TOKEN
    }
    for attempt in range(1, retries + 1):
        try:
            r = requests.get(BASE_URL, params=params, timeout=60)
            r.raise_for_status()
            return r.json()
        except requests.RequestException as e:
            print(f"  [Attempt {attempt}/{retries}] Failed: {e}")
            if attempt < retries:
                time.sleep(5 * attempt)
            else:
                raise

# ── Align columns ─────────────────────────────────────────────────────────────
def align_columns(df):
    for col in EXPECTED_COLUMNS:
        if col not in df.columns:
            df[col] = None
    return df[EXPECTED_COLUMNS]

# ── Download ──────────────────────────────────────────────────────────────────
def download_all():
    if os.path.exists(FINAL_OUTPUT):
        print(f"File already exists: {FINAL_OUTPUT}\nDownload skipped.")
        return
    progress  = load_progress()
    offset    = progress["offset"]
    batch_num = progress["batch_num"]
    total     = progress["total_records"]

    print(f"{'Resuming' if offset > 0 else 'Starting'} download "
          f"(batch={batch_num}, offset={offset}, saved={total})\n")

    while True:
        batch_file = os.path.join(BATCH_DIR, f"batch_{batch_num:04d}.csv")

        if os.path.exists(batch_file):
            print(f"Batch {batch_num} already exists, skipping...")
            offset    += BATCH_SIZE
            batch_num += 1
            continue

        print(f"Fetching batch {batch_num} (offset={offset})...")
        batch = fetch_batch(offset)

        if not batch:
            print("No more data. Download complete.")
            break

        df = align_columns(pd.DataFrame(batch))
        df.to_csv(batch_file, index=False)

        total     += len(batch)
        offset    += BATCH_SIZE
        batch_num += 1
        save_progress(offset, batch_num, total)
        print(f"  -> {len(batch)} records. Total: {total}")
        time.sleep(0.5)

    if os.path.exists(PROGRESS_FILE):
        os.remove(PROGRESS_FILE)

# ── Stream-merge batches ──────────────────────────────────────────────────────
def combine_batches():
    if os.path.exists(FINAL_OUTPUT):
        print(f"Final file already exists: {FINAL_OUTPUT}, skipping merge.")
        return

    batch_files = sorted(
        os.path.join(BATCH_DIR, f)
        for f in os.listdir(BATCH_DIR) if f.endswith(".csv")
    )
    print(f"\nMerging {len(batch_files)} batches into final CSV...")

    with open(FINAL_OUTPUT, "w", encoding="utf-8") as out:
        for i, path in enumerate(batch_files):
            df = pd.read_csv(path, low_memory=False)
            df = align_columns(df)                      # re-align just in case
            df.to_csv(out, index=False, header=(i == 0))
            print(f"  Merged {path} ({len(df)} rows)")

    print(f"\nDone! Saved to: {FINAL_OUTPUT}")

# ── Utility helpers ───────────────────────────────────────────────────────────
def convert_to_parquet(parquet_path=None):
    """Generate a parquet copy of the final CSV.
    Parquet is smaller and faster to load than CSV.
    If parquet already exists the function does nothing.
    """
    parquet_path = parquet_path or os.path.join(OUTPUT_DIR, "chicago_crimes_cleaned.parquet")
    if os.path.exists(parquet_path):
        print(f"Parquet already exists: {parquet_path}")
        return
    if not os.path.exists(FINAL_OUTPUT):
        raise FileNotFoundError("CSV not found. Run download_all() and combine_batches() first.")
    df = pd.read_csv(FINAL_OUTPUT, low_memory=False)
    df.to_parquet(parquet_path, index=False)
    print(f"Saved parquet: {parquet_path}")

def load_data(parquet_path=None, csv_path=None):
    """Load dataset, preferring parquet if available, otherwise falls back to CSV.

    Args:
        parquet_path: optional override for parquet location.
        csv_path: optional override for csv location.

    Returns:
        pandas.DataFrame containing the crime data.

    Raises:
        FileNotFoundError: if neither file is present.
    """
    parquet_path = parquet_path or os.path.join(OUTPUT_DIR, "chicago_crimes_cleaned.parquet")
    csv_path = csv_path or FINAL_OUTPUT

    if os.path.exists(parquet_path):
        print("Loading from parquet...")
        return pd.read_parquet(parquet_path)
    if os.path.exists(csv_path):
        print("Loading from CSV...")
        return pd.read_csv(csv_path, low_memory=False)
    raise FileNotFoundError(
        f"No data found. Run download_all() then combine_batches(), or place a file at {parquet_path} or {csv_path}.")

# ── Execution ──────────────────────────────────────────────────────────────
if __name__ == "__main__":

  # ── Run download ──────────────────────────────────────────────────────────────
  # Set DOWNLOAD_DATA = True if you need to download the data for the first time
  # Set CONVERT_PARQUET = True if you already have the CSV but want to convert to parquet
  # Set both to False if you already have the parquet and just want to load and explore
  DOWNLOAD_DATA    = True
  CONVERT_PARQUET  = False

  if DOWNLOAD_DATA:
      download_all()
      combine_batches()

  if DOWNLOAD_DATA or CONVERT_PARQUET:
      convert_to_parquet()

  # ── Load data (uses parquet if available, otherwise CSV) ──────────────────────
  # If neither file exists, a FileNotFoundError will tell you to set DOWNLOAD_DATA = True
  df = load_data()
