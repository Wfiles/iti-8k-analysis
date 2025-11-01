import wrds
from dotenv import load_dotenv, find_dotenv
import os
import numpy as np
import pandas as pd
from pathlib import Path

THIS_FILE = Path(__file__).resolve()
REPO_ROOT = THIS_FILE.parent.parent
DATA_DIR = REPO_ROOT / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)  # ensure folder exists


def collect_us_crsp_data(db):
    print("Collecting CRSP data from WRDS...")
    # Query CRSP data
    all_stocks = db.raw_sql(
        """
        SELECT a.permco, a.date, a.ret, a.prc, a.vol,
            n.comnam, n.shrcd, n.exchcd, n.ticker
        FROM crsp.dsf AS a
        JOIN crsp.stocknames AS n
        ON a.permco = n.permco
        AND a.date BETWEEN n.namedt AND n.nameenddt
        WHERE a.date BETWEEN '2001-01-01' AND '2024-12-31'
        AND n.shrcd IN (10, 11)           -- common stocks
        AND n.exchcd BETWEEN 1 AND 3      -- NYSE, AMEX, NASDAQ
        ORDER BY a.date;
        """,
        date_cols=["date", "namedt", "nameenddt"]
    )

    all_stocks.to_csv('./data/all_stocks_us.csv', index = False)

    return all_stocks

def collect_sp500_crsp_data(db):
    print("Collecting S&P 500 data from WRDS...")

    # Query CRSP data
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
    date_cols=['start', 'ending', 'date', 'namedt', 'nameenddt']
)


    sp500.to_csv('./data/sp500_us.csv', index = False)

    return sp500

def collect_compustat(db) : 
    print("Collecting Compustat data from WRDS...")
    
    # --- Compustat Quarterly with RDQ (earnings announcement dates) ---
    compq = db.raw_sql("""
        SELECT gvkey, datadate, rdq, fyearq, fqtr, epspxq, epsfxq
        FROM comp.fundq
        WHERE rdq IS NOT NULL
        AND rdq BETWEEN '2001-01-01' AND '2024-12-31'
    """)

    # --- CCM link table: GVKEY -> permco with valid windows ---
    ccm = db.raw_sql("""
        SELECT gvkey, lpermco AS permco, linktype, linkprim, linkdt, linkenddt
        FROM crsp.ccmxpf_linktable
        WHERE linktype IN ('LU','LC') AND linkprim IN ('P','C')
    """)

    # Dates to datetime
    for c in ["datadate","rdq"]:
        compq[c] = pd.to_datetime(compq[c])
    for c in ["linkdt","linkenddt"]:
        ccm[c] = pd.to_datetime(ccm[c])

    # Open-ended linkenddt -> future
    ccm["linkenddt"] = ccm["linkenddt"].fillna(pd.Timestamp("2099-12-31"))

    # Merge Compustat with CCM and keep only rows where RDQ falls inside the link window
    rdq_map = compq.merge(ccm, on="gvkey", how="inner")
    mask = (rdq_map["rdq"] >= rdq_map["linkdt"]) & (rdq_map["rdq"] <= rdq_map["linkenddt"])
    rdq_map = rdq_map.loc[mask, ["permco","rdq","fyearq","fqtr","epspxq","epsfxq","gvkey"]].dropna(subset=["permco"]).copy()

    # Deduplicate if multiple gvkeys/links collapse to same (permco, rdq)
    rdq_map = rdq_map.sort_values(["permco","rdq"]).drop_duplicates(subset=["permco","rdq"])
    return rdq_map

def merge_on_rdq(all_stocks, rdq_map):
    print("Merging RDQ onto daily stock data...")
    # Merge exact RDQ onto the daily panel
    all_stocks = all_stocks.merge(
        rdq_map[["permco","rdq"]].assign(on_rdq=1),
        left_on=["permco","date"], right_on=["permco","rdq"], how="left"
    ).drop(columns=["rdq"])

    all_stocks["on_rdq"] = all_stocks["on_rdq"].fillna(0).astype(np.int8)


    all_stocks.to_csv('./data/all_stocks_us_with_rdq.csv', index = False)

    return all_stocks


def construct_vol_missing_flag(all_stocks):
    print
    vol_missing_index = all_stocks[all_stocks['vol'].isna()].index

    # Create a flag column (1 = missing, 0 = not missing)
    all_stocks['vol_missing_flag'] = 0
    all_stocks.loc[vol_missing_index, 'vol_missing_flag'] = 1

    all_stocks.to_csv('./data/all_stocks_us_with_vol_flag.csv', index = False)

    return all_stocks

def connect_to_wrds():
    load_dotenv(find_dotenv())  # load .env file
    wrds_user = os.getenv("WRDS_USERNAME")
    db = wrds.Connection(wrds_username=wrds_user, verbose=True)
    return db

def construct_data():
    print("Starting data collection and processing...")
    db = connect_to_wrds()
    all_stocks = collect_us_crsp_data(db)
    all_stocks = pd.read_csv('./data/all_stocks_us.csv')
    all_stocks['date'] = pd.to_datetime(all_stocks['date'])
    rdq_map = collect_compustat(db)
    all_stocks = merge_on_rdq(all_stocks, rdq_map)
    all_stocks = construct_vol_missing_flag(all_stocks)
    return all_stocks
def main():
    construct_data()


if __name__ == "__main__":
    main()