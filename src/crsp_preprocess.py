import os
import numpy as np
import pandas as pd
import wrds
from pathlib import Path
from dotenv import load_dotenv, find_dotenv


# ------------------------------------------------------------
# Directory setup
# ------------------------------------------------------------

THIS_FILE = Path(__file__).resolve()
REPO_ROOT = THIS_FILE.parent.parent
DATA_RAW = REPO_ROOT / "data" / "raw"
DATA_PREPROCESSED = REPO_ROOT / "data" / "preprocessed"

# Ensure directories exist
for d in [DATA_RAW, DATA_PREPROCESSED]:
    d.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------
# WRDS Connection
# ------------------------------------------------------------

def connect_to_wrds():
    """Connect securely to WRDS using credentials stored in a .env file."""
    load_dotenv(find_dotenv())
    wrds_user = os.getenv("WRDS_USERNAME")
    if wrds_user is None:
        raise ValueError("WRDS_USERNAME not found in environment variables.")
    print(f"Connecting to WRDS as {wrds_user}...")
    db = wrds.Connection(wrds_username=wrds_user, verbose=True)
    return db


# ------------------------------------------------------------
# CRSP Data Collection
# ------------------------------------------------------------

def collect_us_crsp_data(db):
    """Collects daily CRSP data for all US common stocks (2001-2024)."""
    output_path = DATA_RAW / "crsp_daily_us.csv"
    if output_path.exists():
        # Load existing data instead of querying WRDS again
        print(f"File already exists: {output_path}. Loading from disk...")
        return pd.read_csv(output_path, parse_dates=["date"])

    print("Collecting CRSP data from WRDS...")

    all_stocks = db.raw_sql(
        """
        SELECT a.permco, a.date, a.ret, a.prc, a.vol,
               n.comnam, n.shrcd, n.exchcd, n.ticker
        FROM crsp.dsf AS a
        JOIN crsp.stocknames AS n
          ON a.permco = n.permco
         AND a.date BETWEEN n.namedt AND n.nameenddt
        WHERE a.date BETWEEN '2001-01-01' AND '2024-12-31'
          AND n.shrcd IN (10, 11)           -- Common stocks
          AND n.exchcd BETWEEN 1 AND 3      -- NYSE, AMEX, NASDAQ
        ORDER BY a.date;
        """,
        date_cols=["date", "namedt", "nameenddt"]
    )

    print("Saving CRSP data to disk...")
    all_stocks.to_csv(output_path, index=False)
    print(f"Saved CRSP data to {output_path}")
    return all_stocks


def collect_sp500_crsp_data(db):
    """Collects CRSP data for S&P 500 constituents (2000-2024)."""
    output_path = DATA_RAW / "crsp_sp500_daily.csv"
    if output_path.exists():
        print(f"File already exists: {output_path}. Loading from disk...")
        return pd.read_csv(output_path, parse_dates=["date"])

    print("Collecting S&P 500 CRSP data from WRDS...")

    sp500 = db.raw_sql(
        """
        SELECT a.permco, a.start, a.ending, b.date, b.ret, n.comnam
        FROM crsp.msp500list AS a
        JOIN crsp.msf AS b
          ON a.permco = b.permco
         AND b.date BETWEEN a.start AND a.ending
        JOIN crsp.stocknames AS n
          ON a.permco = n.permco
         AND b.date BETWEEN n.namedt AND n.nameenddt
        WHERE b.date >= '2000-01-01'
        ORDER BY b.date;
        """,
        date_cols=["start", "ending", "date", "namedt", "nameenddt"]
    )

    print("Saving S&P 500 CRSP data to disk...")
    sp500.to_csv(output_path, index=False)
    print(f"Saved S&P 500 CRSP data to {output_path}")
    return sp500


# ------------------------------------------------------------
# Compustat Data Collection
# ------------------------------------------------------------

def collect_compustat(db):
    """Collects Compustat quarterly data with RDQ and maps it to CRSP PERMCOs."""
    output_path = DATA_RAW / "compustat_rdq_mapping.csv"
    if output_path.exists():
        print(f"File already exists: {output_path}. Loading from disk...")
        return pd.read_csv(output_path, parse_dates=["rdq"])

    print("Collecting Compustat data from WRDS...")

    compq = db.raw_sql("""
        SELECT gvkey, datadate, rdq, fyearq, fqtr, epspxq, epsfxq
        FROM comp.fundq
        WHERE rdq IS NOT NULL
          AND rdq BETWEEN '2001-01-01' AND '2024-12-31';
    """)

    ccm = db.raw_sql("""
        SELECT gvkey, lpermco AS permco, linktype, linkprim, linkdt, linkenddt
        FROM crsp.ccmxpf_linktable
        WHERE linktype IN ('LU','LC') AND linkprim IN ('P','C');
    """)

    # Convert date columns
    compq["datadate"] = pd.to_datetime(compq["datadate"])
    compq["rdq"] = pd.to_datetime(compq["rdq"])
    ccm["linkdt"] = pd.to_datetime(ccm["linkdt"])
    ccm["linkenddt"] = pd.to_datetime(ccm["linkenddt"]).fillna(pd.Timestamp("2099-12-31"))

    # Merge Compustat with CCM (valid RDQ link window)
    rdq_map = compq.merge(ccm, on="gvkey", how="inner")
    mask = (rdq_map["rdq"] >= rdq_map["linkdt"]) & (rdq_map["rdq"] <= rdq_map["linkenddt"])
    rdq_map = rdq_map.loc[mask, ["permco", "rdq", "fyearq", "fqtr", "epspxq", "epsfxq", "gvkey"]]
    rdq_map = rdq_map.dropna(subset=["permco"]).sort_values(["permco", "rdq"]).drop_duplicates(subset=["permco", "rdq"])

    print("Saving Compustat-CRSP RDQ mapping to disk...")
    rdq_map.to_csv(output_path, index=False)
    print(f"Saved Compustat-CRSP RDQ mapping to {output_path}")
    return rdq_map


# ------------------------------------------------------------
# Merging & Feature Engineering
# ------------------------------------------------------------

def merge_on_rdq(all_stocks, rdq_map):
    """Adds a binary 'on_rdq' flag to indicate whether a given date matches an RDQ."""
    output_path = DATA_PREPROCESSED / "crsp_with_rdq_flag.csv"
    if output_path.exists():
        print(f"File already exists: {output_path}. Loading from disk...")
        return pd.read_csv(output_path, parse_dates=["date"])

    print("Merging RDQ dates onto daily CRSP data...")

    merged = all_stocks.merge(
        rdq_map[["permco", "rdq"]].assign(on_rdq=1),
        left_on=["permco", "date"],
        right_on=["permco", "rdq"],
        how="left"
    ).drop(columns=["rdq"])

    merged["on_rdq"] = merged["on_rdq"].fillna(0).astype(np.int8)

    merged.to_csv(output_path, index=False)
    print(f"Saved merged dataset with RDQ flag to {output_path}")
    return merged


def construct_vol_missing_flag(all_stocks):
    """Adds a binary flag for missing trading volume ('vol')."""
    output_path = DATA_PREPROCESSED / "crsp_with_rdq_and_vol_flags.csv"
    if output_path.exists():
        print(f"File already exists: {output_path}. Loading from disk...")
        return pd.read_csv(output_path, parse_dates=["date"])

    print("Adding volume missing flag...")

    all_stocks["vol_missing_flag"] = all_stocks["vol"].isna().astype(np.int8)
    all_stocks.to_csv(output_path, index=False)
    print(f"Saved dataset with RDQ + volume flags to {output_path}")
    return all_stocks


# ------------------------------------------------------------
# Master Pipeline
# ------------------------------------------------------------

def crsp_preprocessing():
    """Main pipeline to collect, merge, and preprocess CRSP-Compustat data."""
    print("Starting data collection and preprocessing pipeline...")

    # if crsp_with_rdq_and_vol_flags.csv exists, skip processing
    output_path = DATA_PREPROCESSED / "crsp_with_rdq_and_vol_flags.csv"
    if output_path.exists():
        print(f"Preprocessed data already exists at {output_path}. Skipping processing.")
        return pd.read_csv(output_path, parse_dates=["date"])

    db = connect_to_wrds()

    # Step 1: Load or collect CRSP data
    all_stocks = collect_us_crsp_data(db)

    # Step 2: Load or collect Compustat RDQ mapping
    rdq_map = collect_compustat(db)

    # Step 3: Merge RDQ information
    all_stocks = merge_on_rdq(all_stocks, rdq_map)

    # Step 4: Add volume missing flag
    all_stocks = construct_vol_missing_flag(all_stocks)

    print("CRSP preprocessing completed successfully.")
    return all_stocks
