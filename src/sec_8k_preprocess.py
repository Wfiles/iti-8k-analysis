import os
import requests
import zipfile
import orjson
import polars as pl
from tqdm import tqdm
import re
import requests
from bs4 import BeautifulSoup as bs
import pandas as pd
import unicodedata

from src.crsp_preprocess import connect_to_wrds


def download_zip(url: str, zip_path: str, force_download: bool = False) -> str:
    """Download ZIP from SEC if missing or forced."""
    os.makedirs(os.path.dirname(zip_path), exist_ok=True)
    if not os.path.exists(zip_path) or force_download:
        print("⬇️  Downloading ZIP from SEC...")
        headers = {"User-Agent": "DataScienceStudent/EPFL (matthias@example.com)"}
        with requests.get(url, headers=headers, stream=True) as r:
            r.raise_for_status()
            with open(zip_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        print("Download complete.")
    else:
        print("ZIP already exists locally, skipping download.")
    return zip_path

import zipfile
import orjson
import polars as pl
from tqdm import tqdm
from itertools import zip_longest

import zipfile
import orjson
import polars as pl
from tqdm import tqdm
from itertools import zip_longest

def parse_zip_batched(zip_path: str, only_8k: bool = True, batch_size: int = 200_000) -> pl.DataFrame:
    """
    Parse JSONs inside a ZIP into a Polars DataFrame using batching to avoid OOM.
    - only_8k: keep only 8-K / 8-K/A (dramatically reduces rows)
    - batch_size: number of rows per batch before converting to a DataFrame
    """
    batches = []
    rows = []

    with zipfile.ZipFile(zip_path, "r") as zf:
        json_files = [n for n in zf.namelist() if n.endswith(".json")]

        for name in tqdm(json_files, desc="Parsing SEC JSONs"):
            with zf.open(name) as f:
                data = orjson.loads(f.read())

            cik_raw = data.get("cik")
            cik_int = str(int(cik_raw)) if cik_raw else None
            company = data.get("name")

            recent = (data.get("filings", {}) or {}).get("recent", {}) or {}

            # Parallel arrays
            forms              = recent.get("form", []) or []
            accessions         = recent.get("accessionNumber", []) or []
            accept_ts          = recent.get("acceptanceDateTime", []) or []
            filing_dates       = recent.get("filingDate", []) or []
            report_dates       = recent.get("reportDate", []) or []
            acts               = recent.get("act", []) or []
            file_numbers       = recent.get("fileNumber", []) or []
            film_numbers       = recent.get("filmNumber", []) or []
            items_list         = recent.get("items", []) or []
            sizes              = recent.get("size", []) or []
            is_xbrl            = recent.get("isXBRL", []) or []
            is_inline_xbrl     = recent.get("isInlineXBRL", []) or []
            primary_doc        = recent.get("primaryDocument", []) or []
            primary_doc_descr  = recent.get("primaryDocDescription", []) or []

            for (form, acc, acc_time, fdate, rdate, act, fileno, filmno,
                 items, size, xbrl, ixbrl, pdoc, pdescr) in zip_longest(
                    forms, accessions, accept_ts, filing_dates, report_dates,
                    acts, file_numbers, film_numbers, items_list, sizes,
                    is_xbrl, is_inline_xbrl, primary_doc, primary_doc_descr,
                    fillvalue=None,
                 ):
                if only_8k and (form not in ("8-K", "8-K/A")):
                    continue

                # Keep URL components instead of full URL string (saves RAM)
                rows.append({
                    "cik_int": cik_int,
                    "company_name": company,
                    "form": form,
                    "accession": acc,
                    "filing_date": fdate,
                    "report_date": rdate,
                    "acceptance_datetime": acc_time,
                    "act": act,
                    "file_number": fileno,
                    "film_number": filmno,
                    "items": items,
                    "size": size,
                    "is_xbrl": xbrl,
                    "is_inline_xbrl": ixbrl,
                    "primary_document": pdoc,
                    "primary_doc_description": pdescr,
                })

                if len(rows) >= batch_size:
                    batches.append(pl.DataFrame(rows))
                    rows.clear()

    if rows:
        batches.append(pl.DataFrame(rows))
        rows.clear()

    # Concatenate batches and normalize types; rechunk consolidates memory
    df = pl.concat(batches, how="vertical_relaxed", rechunk=True) if batches else pl.DataFrame()

    # Type normalization (use strict=False to avoid errors on bad strings)
    if df.height > 0:
        df = (
            df.with_columns([
                pl.col("filing_date").str.to_date("%Y-%m-%d", strict=False),
                pl.col("report_date").str.to_date("%Y-%m-%d", strict=False),
                pl.col("acceptance_datetime").str.to_datetime("%Y-%m-%dT%H:%M:%SZ", strict=False),
            ])
            .with_columns(
                pl.col("acceptance_datetime").dt.date().alias("acceptance_date")
            )
        )

    return df


def process_filings(df: pl.DataFrame) -> pl.DataFrame:
    """Filter 8-K/8-K/A filings, keep essential columns, add accession_no_dash and url_txt."""
    df_8k = (
        df.filter(pl.col("form").is_in(["8-K", "8-K/A"]))
          .with_columns([
              pl.col("accession").str.replace_all("-", "").alias("accession_no_dash")
          ])
          .with_columns([
              ("https://www.sec.gov/Archives/edgar/data/"
               + pl.col("cik_int").cast(pl.Utf8)
               + "/"
               + pl.col("accession_no_dash")
               + "/"
               + pl.col("accession")
               + ".txt").alias("url_txt")
          ])
    )
    return df_8k


def load_8k_filings(
    url: str = "https://www.sec.gov/Archives/edgar/daily-index/bulkdata/submissions.zip",
    zip_path: str = "data/raw/submissions.zip",
    parquet_path: str = "data/preprocessed/submissions_8k.parquet",
    force_download: bool = False
) -> pl.DataFrame:
    """
    Download, parse, filter, and save SEC 8-K filings.

    If Parquet already exists and force_download=False, just reads and returns it.

    Returns:
        pl.DataFrame: Polars DataFrame with columns:
                      accession, cik_int, company_name, form,
                      acceptance_datetime, accession_no_dash, url_txt
    """
    os.makedirs(os.path.dirname(parquet_path), exist_ok=True)

    if os.path.exists(parquet_path) and not force_download:
        print(f" Parquet file exists at {parquet_path}. Reading...")
        return pl.read_parquet(parquet_path)

    # Download & parse
    zip_file = download_zip(url, zip_path, force_download)
    df_raw =  parse_zip_batched(zip_file)

    # Process & filter
    df_8k = process_filings(df_raw)

    # Save Parquet
    print(f"Saving {df_8k.height:,} rows to {parquet_path}...")
    df_8k.write_parquet(parquet_path, compression="zstd")
    print("Done.")
    return df_8k


def parse_8k_filing(link: str) -> pd.DataFrame:
    """
    Download and parse an SEC 8-K or 8-K/A filing text file.

    Args:
        link (str): Direct URL to the SEC filing TXT file.

    Returns:
        pd.DataFrame: Each row corresponds to an Item reported in the filing.
                      Columns: item, itemText, cik, conm (company name), edgar.link
                      Returns None if no items found.
    """

    # -------------------------------
    # Step 1: Download and clean text
    # -------------------------------
    def get_text(link: str) -> list[str]:
        """
        Retrieve filing text from SEC, normalize, and split by lines.
        """
        headers = {
            "User-Agent": "DataScience Student student@example.com",
            "Accept-Encoding": "gzip, deflate",
            "Host": "www.sec.gov"
        }
        page = requests.get(link, headers=headers)
        page.raise_for_status()
        html = bs(page.content, "lxml")
        text = html.get_text().replace(u'\xa0', ' ').replace("\t", " ").replace("\x92", "'").split("\n")

        # Check for SEC block message
        if any("our Request Originates from an Undeclared Automated Tool" in line for line in text):
            raise Exception("Blocked by SEC: Your Request Originates from an Undeclared Automated Tool")

        print(f"Downloaded filing from {link}")
        return text

    # -------------------------------
    # Step 2: Identify reported items
    # -------------------------------
    def get_items(text: list[str]) -> list[str]:
        """
        Extract all Item headers in the filing (e.g., 'Item 1.01', 'Item 2.02').
        """
        itemPattern = re.compile(r"^(Item\s[1-9][\.\d]*)", re.IGNORECASE)
        return [match.group(0) for line in text if (match := itemPattern.search(line.strip()))]

    # -------------------------------
    # Step 3: Extract text for each item
    # -------------------------------
    def get_data(file: list[str], items: list[str]) -> pd.DataFrame:
        """
        Map each detected Item to its corresponding text section.
        """
        text8k = []
        dataList = []
        stop = re.compile("SIGNATURE", re.IGNORECASE)
        companyCik = re.compile(r"(CENTRAL INDEX KEY:)([\s\d]+)", re.IGNORECASE)
        companyName = re.compile(r"(COMPANY CONFORMED NAME:)(.+)", re.IGNORECASE)
        control = 0
        itemPattern = re.compile("|".join(["^" + re.escape(i) for i in items]), re.IGNORECASE)
        cik = conm = None

        for line in file:
            if control == 0:
                # Extract CIK and company name from header
                if not cik and (match := companyCik.search(line)):
                    cik = match.group(2).strip()
                if not conm and (match := companyName.search(line)):
                    conm = match.group(2).strip()
                # Start of first item
                if itemPattern.search(line):
                    it = itemPattern.search(line).group(0)
                    text8k.append(re.sub(it, "", line))
                    control = 1
            else:
                # Collect text for each subsequent item
                if itemPattern.search(line):
                    dataList.append([it, "\n".join(text8k)])
                    it = itemPattern.search(line).group(0)
                    text8k = [re.sub(it, "", line)]
                elif stop.search(line):
                    dataList.append([it, "\n".join(text8k)])
                    break
                else:
                    text8k.append(line)

        if not dataList:
            return pd.DataFrame(columns=["item", "itemText", "cik", "conm", "edgar.link"])

        data = pd.DataFrame(dataList, columns=["item", "itemText"])
        data["cik"] = cik
        data["conm"] = conm
        data["edgar.link"] = link
        return data

    # -------------------------------
    # Step 4: Fallback extraction (if items not found)
    # -------------------------------
    def get_data_alternative(file: list[str]) -> pd.DataFrame:
        """
        Alternative method to extract items by scanning full text.
        """
        fullText = " ".join(file)
        fullText = unicodedata.normalize("NFKD", fullText).encode('ascii', 'ignore').decode('utf8')

        itemPattern = re.compile(r"(Item\s[1-9][\.\d]*)", re.IGNORECASE)
        items = itemPattern.findall(fullText)
        stop = re.compile("SIGNATURE", re.IGNORECASE)
        sig_match = stop.search(fullText)
        sig = sig_match.start() if sig_match else len(fullText)

        itemsStart = [fullText.find(i) for i in items] + [sig]
        dataList = [[items[n], fullText[itemsStart[n]:itemsStart[n+1]]] for n in range(len(items))]

        companyCik = re.compile(r"(CENTRAL INDEX KEY:)([\s\d]+)", re.IGNORECASE)
        companyName = re.compile(r"(COMPANY CONFORMED NAME:)(.+)", re.IGNORECASE)
        cik = companyCik.search(fullText).group(2).strip() if companyCik.search(fullText) else None
        conm = companyName.search(fullText).group(2).strip() if companyName.search(fullText) else None

        data = pd.DataFrame(dataList, columns=["item", "itemText"])
        data["cik"] = cik
        data["conm"] = conm
        data["edgar.link"] = link
        return data

    # -------------------------------
    # Step 5: Run pipeline
    # -------------------------------
    file = get_text(link)
    items = get_items(file)

    if items:
        df = get_data(file, items)
        if df.empty:
            df = get_data_alternative(file)
    else:
        df = get_data_alternative(file)
        if df.empty:
            print(f"No items found in filing: {link}")
            return None

    print(f"Parsed filing: {link} with {len(df)} items.")
    return df









def map_cik_to_permno(df: pl.DataFrame, cik_col: str = "cik", date_col: str = "date") -> pl.DataFrame:
    """
    Map each (CIK, date) pair from the input DataFrame to the corresponding CRSP PERMNO
    using the WRDS 'crsp.ccm_lookup' table. Allows specifying the column names for CIK and date.

    Parameters
    ----------
    df : pl.DataFrame
        Input Polars DataFrame.
    cik_col : str
        Name of the column containing CIK identifiers in df.
    date_col : str
        Name of the column containing dates in df.

    Returns
    -------
    pl.DataFrame
        Same as input DataFrame, with one additional column 'permno' containing the mapped CRSP ID.
    """

    # --- Connect to WRDS database ---
    db = connect_to_wrds()

    # --- Load the CCM lookup table ---
    ccm_lookup = db.get_table(
        library='crsp',
        table='ccm_lookup',
        columns=['cik', 'lpermno', 'linkdt', 'linkenddt']
    )

    # --- Drop missing rows ---
    ccm_lookup = ccm_lookup.dropna()

    # --- Convert to Polars and cast types ---
    ccm_lookup = (
        pl.from_pandas(ccm_lookup)
        .with_columns([
            pl.col("cik").cast(pl.Int64),
            pl.col("lpermno").cast(pl.Int64).alias("permno"),
            pl.col("linkdt").cast(pl.Date),
            pl.col("linkenddt").cast(pl.Date)
        ])
    )

    # --- Cast input DataFrame columns ---
    df = df.with_columns([
        pl.col(cik_col).cast(pl.Int64),
        pl.col(date_col).cast(pl.Date)
    ])

    # --- Join on CIK and filter by date range ---
    merged = (
        df.join(ccm_lookup, left_on=cik_col, right_on="cik", how="left")
        .filter((pl.col(date_col) >= pl.col("linkdt")) & (pl.col(date_col) <= pl.col("linkenddt")))
        .select(df.columns + ["permno"])
    )

    n_missing = merged.filter(pl.col("permno").is_null()).height
    print(f"Number of rows with missing permno: {n_missing}")

    return merged


def preprocess_sec_8k() -> pl.DataFrame:
    """Load, filter, map CIK to PERMNO, and return cleaned SEC 8-K filings DataFrame."""
    
    # Load 8-K filings
    df_8k = load_8k_filings()

    # Add days between report and filing, extract year, and filter
    df_8k = (
        df_8k
        # Compute days between report and filing
        .with_columns(
            ((pl.col("filing_date") - pl.col("report_date")).dt.total_seconds() / 86400)
            .alias("days_between_report_and_filing").cast(pl.Int32)
        )
        # Keep filings from 2004 onwards
        .filter(pl.col("report_date") >= pl.datetime(2004, 1, 1))
        # Extract report year
        .with_columns(pl.col("report_date").dt.year().alias("report_year"))
        # Keep filings where report date is 1-30 days before filing date
        .filter(pl.col("days_between_report_and_filing").is_between(1, 30))
    )
    
    # Cast cik_int to Int32
    df_8k = df_8k.with_columns(
    pl.col("cik_int").cast(pl.Int32)
    )

    # Map CIK to PERMNO
    df_8k_with_permno = map_cik_to_permno(df_8k, cik_col="cik_int", date_col="filing_date")

    # Keep only relevant columns
    df_8k_clean = df_8k_with_permno.select([
        "permno",
        "filing_date",
        "report_date",
        "report_year",
        "days_between_report_and_filing",
        "url_txt"
    ])

    return df_8k_clean