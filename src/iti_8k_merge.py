from src.sec_8k_preprocess import preprocess_sec_8k
from src.iti_preprocess import prepare_ITI_data
import polars as pl

def merge_8k_iti() -> pl.DataFrame:
    """Merge SEC 8-K filings with ITI data on PERMNO and report date."""
    
    # Preprocess SEC 8-K filings
    df_8k = preprocess_sec_8k()
    
    # Prepare ITI data
    df_iti = prepare_ITI_data()

    # Filter ITI data to only after the fisrt date of df_8k - 2 weeks

    first_8k_date = df_8k.select(pl.col("report_date").min() - pl.duration(weeks=2)).item()
    df_iti = df_iti.filter(pl.col("date") >= first_8k_date)
    

    # Merge on 'permno' and 'report_date'
    df_merged = df_iti.join(df_8k, left_on=['permno', 'date'], right_on=['permno', 'report_date'], how='left')
    return df_merged