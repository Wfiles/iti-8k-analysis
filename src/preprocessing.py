import os
import glob
import pandas as pd
import re
import pytz
import tarfile
from tqdm import tqdm
import polars as pl
import wrds
from dotenv import load_dotenv, find_dotenv
from sentence_transformers import SentenceTransformer
import datetime as dt
from sklearn.decomposition import PCA


from src.merge_crsp_fnspid import add_permco_to_news_polars
# Enable tqdm support for pandas apply
tqdm.pandas()

# Regex pattern to extract the rest of the timestamp after the date
# Format: H:MM AM/PM EST/EDT/UTC
time_tz_pattern = re.compile(r"\s+(\d{1,2}:\d{2})\s*([AP]M)\s+(EST|EDT|UTC)$")

# Mapping of timezone abbreviations to numeric offsets for parsing
TZ_OFFSETS = {"EST": "-0500", "EDT": "-0400", "UTC": "+0000"}

def parse_reuters_timestamp(ts: str) -> pd.Timestamp:
    """
    Parse a Reuters timestamp string into a UTC datetime.

    Parameters:
    - ts: str, timestamp in format 'YYYYMMDD H:MM AM/PM EST/EDT/UTC'

    Returns:
    - pd.Timestamp in UTC if valid, otherwise pd.NaT
    """
    ts_str = str(ts).strip()
    ts_str = ts_str.replace("\u00A0", " ")  # Replace non-breaking spaces

    # Return NaT if timestamp is explicitly invalid
    if ts_str == "INVALID_DATE":
        return pd.NaT

    # Extract the date (first 8 digits)
    date_part = ts_str[:8]

    # Extract the time and timezone using regex
    match = time_tz_pattern.search(ts_str)
    if not match:
        return pd.NaT

    time_part, ampm, tz = match.groups()
    ts_formatted = f"{date_part} {time_part} {ampm} {TZ_OFFSETS[tz]}"

    # Parse the datetime and convert to UTC
    dt = pd.to_datetime(ts_formatted, format="%Y%m%d %I:%M %p %z", errors="coerce")
    return dt.tz_convert("UTC")

def load_and_parse_reuters(tar_path: str, save_path: str) -> pd.DataFrame:
    """
    Load processed CSV if it exists, otherwise extract TSV files from a tar.bz2,
    parse timestamps, and save the result as a CSV.

    Parameters:
    - tar_path: str, path to the raw .tar.bz2 file containing TSV files
    - save_path: str, path to save/load the processed CSV

    Returns:
    - pd.DataFrame with parsed UTC timestamps
    """
    # Load processed CSV if it exists
    if save_path and os.path.exists(save_path):
        print(f"Loading processed CSV from {save_path}")
        return pd.read_csv(save_path, parse_dates=["ts_parsed"])

    # Extract TSV files from tar.bz2
    print(f"Extracting {tar_path}")
    with tarfile.open(tar_path, "r:bz2") as tar:
        tar.extractall(path=os.path.dirname(tar_path))

    # After extraction, the folder 'reuters' already exists
    extracted_folder = os.path.join(os.path.dirname(tar_path), "reuters")

    # Find all TSV files directly in the extracted folder
    files = glob.glob(os.path.join(extracted_folder, "*.tsv"))
    if not files:
        raise FileNotFoundError(f"No TSV files found in {extracted_folder}")
    else:
        print(f"Found {len(files)} TSV files.")

    dfs = []
    # Read each TSV file with a progress bar
    for file in tqdm(files, desc="Reading TSV files"):
        df = pd.read_csv(file, sep="\t", names=["ts", "title", "href"], header=0)
        dfs.append(df)

    # Concatenate all DataFrames into a single DataFrame
    big_df = pd.concat(dfs, ignore_index=True)

    # Apply the timestamp parser to each row
    print("Parsing timestamps...")
    big_df["ts_parsed"] = big_df["ts"].progress_apply(parse_reuters_timestamp)

    # Sort by parsed timestamps and reset index
    big_df = big_df.sort_values("ts_parsed").reset_index(drop=True)

    # Save to CSV for future use
    if save_path:
        big_df.to_csv(save_path, index=False)
        print(f"Processed CSV saved to {save_path}")

    return big_df



def prepare_ITI_data(iti_csv_path: str) -> pd.DataFrame:
    """
    Load and prepare ITI data from a CSV file.

    Parameters:
    - iti_csv_path: str, path to the ITI CSV file

    Returns:
    - pd.DataFrame with 'date' as datetime and sorted
    """
    iti_df = pl.read_csv(iti_csv_path)
    iti_df = iti_df.with_columns(pl.col('date').cast(pl.Date))

    load_dotenv(find_dotenv())
    wrds_user = os.getenv("WRDS_USERNAME")
    db = wrds.Connection(wrds_username=wrds_user, verbose=False)

    mapping = db.raw_sql("""
        select permno, permco, namedt, nameendt
        from crsp.dsenames
    """, date_cols=["namedt", "nameendt"])

    mapping = pl.from_pandas(mapping) 

    mapping = mapping.with_columns(pl.col('namedt').cast(pl.Date), pl.col('nameendt').cast(pl.Date))
    iti_df = iti_df.with_columns(pl.col('date').cast(pl.Date))

    out = (
    iti_df.join(mapping, on="permno", how="inner")
         .filter((pl.col("date") >= pl.col("namedt")) &
                 (pl.col("date") <= pl.col("nameendt")))
    )
    return out.select([pl.col("date"), pl.col("ITI(13D)"), pl.col("ITI(impatient)"), pl.col("ITI(patient)"), pl.col("ITI(insider)"), pl.col("ITI(short)"), pl.col("permco")])

def process_news_returns(news_csv_path: str, all_stocks_csv_path: str) -> pl.DataFrame: 
    financial_news_df = pl.read_csv(
    news_csv_path,
    dtypes={
        "news_date": pl.String,
        "Article_title": pl.String,
        "ticker": pl.String,
        "Url": pl.String,
        "Publisher": pl.String,
        "Author": pl.String,   # price can be float32 to save RAM
        "Article": pl.String,      # read as string, then cast robustly
        "Lsa_summary": pl.String,
        "Luhn_summary": pl.String,
        "Textrank_summary": pl.String,
        "Lexrank_summary": pl.String,
        "permco": pl.Int32,
    },
    try_parse_dates=False,    # will parse "date", "namedt", "nameendt"
    null_values=["", "NA", "NaN", "null", "."],  # typical CSV missings
    )
    df_with_permco = add_permco_to_news_polars(financial_news_df)
    all_stocks = pl.read_csv(all_stocks_csv_path)
    all_stocks = all_stocks.with_columns(pl.col("date").cast(pl.Date))
    out = (df_with_permco.join(all_stocks, on=["permco", 'date'], how="right"))
    df = out.select('date', 'permco', 'ret', 'prc', 'vol', 'on_rdq', 'vol_missing_flag', 'comnam', 'Article_title')

    return df

def process_final_dataset(news_csv_path: str, all_stocks_csv_path: str, iti_csv_path: str = "data/ITIs.csv") -> pl.DataFrame:
    df = process_news_returns(news_csv_path, all_stocks_csv_path)
    iti_df = prepare_ITI_data(iti_csv_path)
    final_df = iti_df.join(df, on=['date', 'permco'], how='right')
    final_df = final_df.filter(pl.col('ITI(13D)').is_not_null() & pl.col('ITI(impatient)').is_not_null())

    final_df = final_df.filter(pl.col('date') >= pl.lit("2009-05-27").str.to_date())
    final_df.sort(['date', 'permco'])
    final_df.write_csv("data/processed/final_news_data.csv")
    return final_df


def embed_texts(df: pl.DataFrame, model_name: str = "all-MiniLM-L6-v2", batch_size: int = 32) -> pl.DataFrame:

    df_filtered = df.filter(pl.col('Article_title').is_not_null())
    model = SentenceTransformer(model_name)
    titles = df_filtered["Article_title"].to_list()


    batch_size = 64
    all_embeddings = []

    for i in tqdm(range(0, len(titles), batch_size), desc="Encoding headlines"):
        batch = titles[i:i+batch_size]
        emb = model.encode(batch, normalize_embeddings=True) 
        all_embeddings.extend(emb.tolist()) 


    df_emb = df_filtered.with_columns(pl.Series("embedding", all_embeddings))
    df_emb.write_parquet("data/processed/news_with_embeddings.parquet")

    return df_emb

def rolling_pca(df: pl.DataFrame, window_days=180, step_days=30, n_components=10):
    df = df.sort("date")
    start = df["date"].min()
    end = df["date"].max()
    
    dfs = []
    current = start + dt.timedelta(days=window_days)
    
    while current + dt.timedelta(days=step_days) <= end:
        train_start = current - dt.timedelta(days=window_days)
        train_end = current
        test_start = current
        test_end = current + dt.timedelta(days=step_days)

        # Train and test splits
        train_df = df.filter((pl.col("date") >= train_start) & (pl.col("date") < train_end))
        test_df = df.filter((pl.col("date") >= test_start) & (pl.col("date") < test_end))

        if train_df.height < n_components or test_df.height == 0:
            current += dt.timedelta(days=step_days)
            continue

        X_train = np.array(train_df["embedding"].to_list(), dtype=float)
        X_test = np.array(test_df["embedding"].to_list(), dtype=float)

        pca = PCA(n_components=n_components).fit(X_train)
        X_test_pca = pca.transform(X_test)

        test_df = test_df.with_columns(
            pl.Series("pca_embedding", [x.tolist() for x in X_test_pca]).cast(pl.List(pl.Float64))
        )

        dfs.append(test_df)
        current += dt.timedelta(days=step_days)

    return pl.concat(dfs)


def apply_rolling_pca() : 
    df = pl.read_parquet("data/processed/news_with_embeddings.parquet")
    df_pca = rolling_pca(df, window_days=180, step_days=30, n_components=10)
    df_pca = df_pca.drop("embedding")
    df_pca.write_parquet("data/processed/news_with_pca_embeddings.parquet")
    return df_pca.select(['date', 'permco', 'pca_embedding'])
