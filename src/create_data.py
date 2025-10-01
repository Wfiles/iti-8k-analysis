import pandas as pd
from dotenv import load_dotenv, find_dotenv
import os
import wrds
import polars as pl
from src.merge_crsp_fnspid import add_permco_to_news_polars

def create_news_data() : 
    financial_news_df = pl.read_csv(
    "data/raw/All_external.csv",
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

    all_stocks = pl.read_csv("data/all_stocks_us_with_vol_flag.csv")
    all_stocks = all_stocks.with_columns(pl.col("date").cast(pl.Date))
    out = (df_with_permco.join(all_stocks, on=["permco", 'date'], how="right"))
    df = out.select('date', 'permco', 'ret', 'prc', 'vol', 'on_rdq', 'vol_missing_flag', 'comnam', 'Article_title')