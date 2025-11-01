

from iti_preprocess import prepare_ITI_data
from src.FNSPID_preprocess import add_permco_to_news_polars
import polars as pl


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


def process_final_dataset(news_csv_path: str, all_stocks_csv_path: str, iti_csv_path: str ) -> pl.DataFrame:
    df = process_news_returns(news_csv_path, all_stocks_csv_path)
    iti_df = prepare_ITI_data(iti_csv_path)
    final_df = iti_df.join(df, on=['date', 'permco'], how='right')
    final_df = final_df.filter(pl.col('ITI(13D)').is_not_null() & pl.col('ITI(impatient)').is_not_null())

    final_df = final_df.filter(pl.col('date') >= pl.lit("2009-05-27").str.to_date())
    final_df.sort(['date', 'permco'])
    final_df.write_csv("data/processed/final_news_data.csv")
    return final_df
