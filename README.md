# news-based-asset-pricing

## Project Structure

```
.
├── data
│   ├── merged
│   │   └── crsp_iti_fnspid.csv          # final merged dataset with FNSPID news, CRSP prices, and ITI metrics
│   ├── preprocessed
│   │   ├── crsp_with_rdq_and_vol_flags.csv   # processed CRSP dataset with RDQ and volume flags
│   │   ├── crsp_with_rdq_flag.csv           # intermediate step for CRSP dataset
│   │   ├── financial_sentiment_analysis.csv # FNSPID dataset including sentiment scores
│   │   ├── gdelt_gkg_files                  # folder storing GDELT processed files
│   │   │   └── 20230115.parquet             # example GDELT file for a single day
│   │   └── submissions_8k.parquet           # processed SEC 8-K filings dataset
│   └── raw
│       ├── All_external.csv                  # raw input file for FNSPID dataset (to download)
│       ├── compustat_rdq_mapping.csv         # temporary file for CRSP dataset construction
│       ├── crsp_daily_us.csv                 # temporary file for CRSP dataset construction
│       ├── fnspid_crsp_with_sentiment.parquet # temporary intermediate FNSPID file
│       ├── gdelt_gkg_files                   # folder for temporary GDELT raw files
│       ├── ITIs.csv                           # raw input file for ITI dataset (to download)
│       └── submissions.zip                    # raw input file for SEC 8-K filings (to download)
├── FNSPID.ipynb        # analysis notebook for FNSPID dataset with ITI and CRSP
├── gdelt.ipynb         # analysis notebook for GDELT dataset
├── iti.ipynb           # analysis notebook for ITI dataset
├── LICENSE
├── outputs            # folder for generated plots and results
├── pdfs
│   ├── internet_appendix.pdf                     # appendix for ITI paper
│   ├── Semester_project_proposal.pdf            # project proposal
│   └── The_Journal_of_Finance_2024_BOGOUSSLAVSKY_Informed_Trading_Intensity.pdf # published ITI paper
├── README.md
├── sec_8k.ipynb       # analysis notebook for SEC 8-K filings
├── src
│   ├── CRSP_ITI_FNSPID_merge.py     # script to merge CRSP, ITI, and FNSPID datasets
│   ├── crsp_preprocess.py           # preprocessing for CRSP dataset
│   ├── FNSPID_preprocess.py         # preprocessing for FNSPID dataset
│   ├── gdelt_preprocess.py          # preprocessing for GDELT dataset
│   ├── iti_preprocess.py            # preprocessing for ITI dataset
│   ├── reuters_preprocess.py        # preprocessing for Reuters dataset
│   └── sec_8k_preprocess.py         # preprocessing for SEC 8-K filings

```