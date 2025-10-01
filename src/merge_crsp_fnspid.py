import pandas as pd
from dotenv import load_dotenv, find_dotenv
import os
import wrds

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


