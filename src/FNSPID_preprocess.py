import os
import glob
import pandas as pd
import re
import tarfile
from tqdm import tqdm
import polars as pl
import wrds
from dotenv import load_dotenv, find_dotenv
from sentence_transformers import SentenceTransformer
import datetime as dt
from sklearn.decomposition import PCA
from src.merge_crsp_fnspid import add_permco_to_news_polars


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
