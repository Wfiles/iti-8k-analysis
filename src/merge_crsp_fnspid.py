import pandas as pd
from dotenv import load_dotenv, find_dotenv
import os
import wrds
import polars as pl

def add_permco_to_news(financial_news_df, nrows=None):
    """
    Takes a news DataFrame with columns ['Stock_symbol', 'Date', ...]
    and returns a DataFrame enriched with the 'permco' column from CRSP.

    Parameters:
    -----------
    financial_news_df : pd.DataFrame
        News DataFrame containing 'Stock_symbol' and 'Date'
    nrows : int, optional
        Number of rows to load from CSV for testing purposes

    Returns:
    --------
    pd.DataFrame
        News DataFrame with 'permco' added
    """
    # Load WRDS connection credentials from .env
    load_dotenv(find_dotenv())
    wrds_user = os.getenv("WRDS_USERNAME")
    db = wrds.Connection(wrds_username=wrds_user, verbose=False)

    # 1. Load CRSP stocknames table (permco <-> ticker mapping)
    stocknames = db.raw_sql("""
        SELECT permco, ticker
        FROM crsp.stocknames
    """)
    # Normalize ticker to uppercase and remove extra spaces
    stocknames['ticker'] = stocknames['ticker'].str.upper().str.strip()
    # Drop duplicate tickers to ensure one-to-one merge
    stocknames.drop_duplicates(subset="ticker", inplace=True)

    # 2. Prepare the news DataFrame
    news_df = financial_news_df.copy()
    if 'Date' not in news_df.columns or 'Stock_symbol' not in news_df.columns:
        raise ValueError("The DataFrame must contain 'Date' and 'Stock_symbol' columns")
    
    # Rename columns for consistency
    news_df.rename(columns={"Date": "news_date", "Stock_symbol": "ticker"}, inplace=True)
    news_df['ticker'] = news_df['ticker'].str.upper().str.strip()
    news_df['news_date'] = pd.to_datetime(news_df['news_date'])

    # 3. Merge news with stocknames to add permco
    merged = news_df.merge(stocknames, on="ticker", how="left")

    return merged

def add_permco_to_news_polars(financial_news_df, nrows=None) : 
    if isinstance(financial_news_df, pd.DataFrame):
        news = pl.from_pandas(financial_news_df)
    elif isinstance(financial_news_df, pl.DataFrame):
        news = financial_news_df.clone()
    else:
        raise TypeError("financial_news_df must be a pandas or Polars DataFrame")


    financial_news_df = financial_news_df.with_columns(
    pl.col("Date")
      .str.replace(pattern=" UTC", value="")
      .str.strptime(pl.Datetime, format="%Y-%m-%d %H:%M:%S", strict=False)
      .cast(pl.Date)  # mark as UTC without shifting
    )
    news = (
            financial_news_df.rename({"Stock_symbol": "ticker", "Date": "date"})
                .with_columns([
                    pl.col("ticker").str.to_uppercase().str.strip_chars(),
                ])
        )

    load_dotenv(find_dotenv())
    wrds_user = os.getenv("WRDS_USERNAME")
    db = wrds.Connection(wrds_username=wrds_user, verbose=False)

    # Note: your original used crsp.stocknames; we keep that for parity.
    # (For time-aware mapping, prefer crsp.dsenames and include date ranges.)
    stocknames_pd = db.raw_sql("""
        SELECT permco, ticker
        FROM crsp.stocknames
        WHERE ticker IS NOT NULL
    """)
    
    db.close()

    stocknames = (
        pl.from_pandas(stocknames_pd)
          .with_columns(pl.col("ticker").str.to_uppercase().str.strip_chars())
          .unique(subset=["ticker"], keep="first")
    )

   
    out = news.join(stocknames, on="ticker", how="left")

    return out

