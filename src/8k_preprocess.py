import os
import requests
import zipfile
import orjson
import polars as pl
from tqdm import tqdm

def download_8k(
    url: str = "https://www.sec.gov/Archives/edgar/daily-index/bulkdata/submissions.zip",
    zip_path: str = "../data/raw/submissions.zip",
    parquet_path: str = "../data/processed/submissions_8k.parquet",
    force_download: bool = False
) -> pl.DataFrame:
    """
    Download, parse, and extract SEC 8-K and 8-K/A filings from the official submissions.zip.

    Args:
        url (str): URL to the SEC submissions ZIP file.
        zip_path (str): Local path to store the downloaded ZIP.
        parquet_path (str): Output path for the processed Parquet file.
        force_download (bool): If True, re-download the ZIP even if it exists.

    Returns:
        pl.DataFrame: Polars DataFrame with essential columns from 8-K and 8-K/A filings.
    """

    # --- Create necessary directories ---
    os.makedirs(os.path.dirname(zip_path), exist_ok=True)
    os.makedirs(os.path.dirname(parquet_path), exist_ok=True)

    # --- Step 0: Skip if parquet already exists ---
    if os.path.exists(parquet_path):
        print(f"✅ Parquet file already exists at {parquet_path}. Reading...")
        return pl.read_parquet(parquet_path)

    # --- Step 1: Download ZIP if missing or forced ---
    if not os.path.exists(zip_path) or force_download:
        print("⬇️  Downloading submissions.zip from SEC...")
        headers = {"User-Agent": "DataScienceStudent/EPFL (matthias@example.com)"}
        with requests.get(url, headers=headers, stream=True) as r:
            r.raise_for_status()
            with open(zip_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        print("✅ Download complete.")
    else:
        print("📦 Zip file already exists locally, skipping download.")

    # --- Step 2: Parse JSONs inside the ZIP ---
    rows = []

    def to_str_list(x):
        """Normalize to list[str]."""
        if x is None:
            return []
        if isinstance(x, list):
            return [str(v) for v in x]
        if x == "":
            return []
        return [str(x)]

    def to_int_or_none(x):
        """Convert safely to int or None."""
        try:
            return int(x)
        except Exception:
            return None

    with zipfile.ZipFile(zip_path, "r") as zf:
        json_files = [name for name in zf.namelist() if name.endswith(".json")]
        for name in tqdm(json_files, desc="Parsing SEC JSONs"):
            with zf.open(name) as f:
                data = orjson.loads(f.read())

            cik_raw = data.get("cik")
            cik_int = str(int(cik_raw)) if cik_raw else None
            company = data.get("name")

            recent = data.get("filings", {}).get("recent", {}) or {}
            forms = recent.get("form", []) or []
            filing_dt = recent.get("filingDate", []) or []
            accessions = recent.get("accessionNumber", []) or []
            accept_ts = recent.get("acceptanceDateTime", []) or []

            # --- Loop over filings ---
            for i, (form, acc, acc_time) in enumerate(zip(forms, accessions, accept_ts)):
                # keep all filings, filter later (avoid missing 8-K/A variants)
                rows.append({
                    "cik_int": cik_int,
                    "company_name": company,
                    "form": form,
                    "accession": acc,
                    "acceptance_datetime": acc_time
                })

    # --- Step 3: Build DataFrame ---
    print("🧱 Building Polars DataFrame...")
    df = pl.DataFrame(rows)

    # --- Step 4: Filter to only 8-K and 8-K/A ---
    df_8k = df.filter(pl.col("form").is_in(["8-K", "8-K/A"]))

    # --- Step 5: Keep only essential columns ---
    df_8k = df_8k.select(["accession", "cik_int", "company_name", "form", "acceptance_datetime"])

    # --- Step 6: Save to Parquet ---
    print(f"💾 Saving {df_8k.height:,} rows to {parquet_path}...")
    df_8k.write_parquet(parquet_path, compression="zstd")

    print("✅ Done.")
    return df_8k

