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
        print("✅ Download complete.")
    else:
        print("📦 ZIP already exists locally, skipping download.")
    return zip_path


def parse_zip(zip_path: str) -> pl.DataFrame:
    """Parse all JSONs in the ZIP and return a raw Polars DataFrame."""
    rows = []
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
            accessions = recent.get("accessionNumber", []) or []
            accept_ts = recent.get("acceptanceDateTime", []) or []

            for form, acc, acc_time in zip(forms, accessions, accept_ts):
                rows.append({
                    "cik_int": cik_int,
                    "company_name": company,
                    "form": form,
                    "accession": acc,
                    "acceptance_datetime": acc_time
                })

    return pl.DataFrame(rows)


def process_filings(df: pl.DataFrame) -> pl.DataFrame:
    """Filter 8-K/8-K/A filings, keep essential columns, add accession_no_dash and url_txt."""
    df_8k = (
        df.filter(pl.col("form").is_in(["8-K", "8-K/A"]))
          .select(["accession", "cik_int", "company_name", "form", "acceptance_datetime"])
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
        print(f"✅ Parquet file exists at {parquet_path}. Reading...")
        return pl.read_parquet(parquet_path)

    # Download & parse
    zip_file = download_zip(url, zip_path, force_download)
    df_raw = parse_zip(zip_file)

    # Process & filter
    df_8k = process_filings(df_raw)

    # Save Parquet
    print(f"💾 Saving {df_8k.height:,} rows to {parquet_path}...")
    df_8k.write_parquet(parquet_path, compression="zstd")
    print("✅ Done.")
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
