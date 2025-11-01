import os
import pandas as pd
import polars as pl
import wrds
from dotenv import load_dotenv, find_dotenv


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