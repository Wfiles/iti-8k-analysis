# Insider Trading Intensity and Abnormal Returns Around 8-K Filings
## 📄 The main project report PDF is called `Project_report.pdf`
## Project Overview
This repository contains the Master Semester Project report for the Data Science program at EPFL (Autumn Semester 2025). Supervised by Prof. Pierre Collin-Dufresne, this research investigates how Informative Trading Intensity (ITI) interacts with corporate disclosures through SEC Form 8-K filings. The project aims to separate pre-disclosure informed trading from post-disclosure market reactions by analyzing both report dates and public filing dates.

## Authors
* **Matthias Wyss** 
* **William Jallot** 

## Methodology
The study combines market microstructure metrics with Natural Language Processing (NLP) to evaluate asset pricing and trading behavior:
* **Event Studies:** Conducted event studies to measure Cumulative Average Abnormal Returns (CAAR), Abnormal ITI, and Absolute Returns around 8-K disclosure dates.
* **Item Decomposition:** Separated 8-K filings by their specific item composition to assess how different corporate events (e.g., financial results vs. director appointments) drive trading behavior.
* **NLP & Sentiment Analysis:** Focused on free-text Item 8.01 ("Other Events") disclosures to compute sentiment scores. The team tested three approaches: direct FinBERT processing, chunk-based FinBERT averaging, and Mistral LLM summarization followed by FinBERT scoring.

## Key Findings
* **Information Leakage:** Abnormal ITI exhibits a pronounced spike immediately after the report date and begins to rise slightly before the event date, suggesting rumors or information leakage prior to public filing.
* **Event Heterogeneity:** Economically material disclosures—such as entry into definitive agreements (Item 1.01) and results of operations (Item 2.02) show strong increases in ITI prior to disclosure. In contrast, corporate governance updates (Items 5.02 and 5.07) show little to no abnormal ITI.
* **Volatility Spikes:** Absolute abnormal returns spike sharply at the disclosure date, indicating a sudden increase in market volatility and activity when new information is released.
* **Sentiment Predictability:** Combining Mistral-based summarization with FinBERT sentiment scoring successfully separates return dynamics, capturing economically relevant information better than simple raw-text approaches. 
* **ITI Nature:** While sentiment strongly impacts return direction, ITI behaves similarly across all sentiment groups, indicating that it is a measure of information intensity rather than a directional predictor of returns.

## Repository & Data
The datasets generated during this project—including the merged 8-K ITI dataset and the preprocessed NLP sentiment datasets—are publicly available in the project's GitHub repository at `https://github.com/Wfiles/iti-8k-analysis`.
## Project Structure

```
.
├── data
│   ├── merged
│   │   └── crsp_iti_fnspid.csv                 # final merged dataset with FNSPID news, CRSP prices, and ITI metrics
│   ├── preprocessed
│   │   ├── crsp_with_rdq_and_vol_flags.csv     # processed CRSP dataset with RDQ and volume flags
│   │   ├── crsp_with_rdq_flag.csv              # intermediate step for CRSP dataset
│   │   ├── financial_sentiment_analysis.csv    # FNSPID dataset including sentiment scores
│   │   ├── gdelt_gkg_files                     # folder storing GDELT processed files
│   │   │   └── 20230115.parquet                # example GDELT file for a single day
│   │   └── submissions_8k.parquet              # processed SEC 8-K filings dataset
│   └── raw
│       ├── All_external.csv                     # raw input file for FNSPID dataset (to download)
│       ├── compustat_rdq_mapping.csv            # temporary file for CRSP dataset construction
│       ├── crsp_daily_us.csv                    # temporary file for CRSP dataset construction
│       ├── fnspid_crsp_with_sentiment.parquet   # temporary intermediate FNSPID file
│       ├── gdelt_gkg_files                      # folder for temporary GDELT raw files
│       ├── ITIs.csv                             # raw input file for ITI dataset (to download)
│       └── submissions.zip                      # raw input file for SEC 8-K filings (to download)
├── src
│   ├── CRSP_ITI_FNSPID_merge.py     # script to merge CRSP, ITI, and FNSPID datasets
│   ├── crsp_preprocess.py           # preprocessing for CRSP dataset
│   ├── FNSPID_preprocess.py         # preprocessing for FNSPID dataset
│   ├── gdelt_preprocess.py          # preprocessing for GDELT dataset
│   ├── iti_preprocess.py            # preprocessing for ITI dataset
│   ├── reuters_preprocess.py        # preprocessing for Reuters dataset
│   └── sec_8k_preprocess.py         # preprocessing for SEC 8-K filings
├── outputs                                     # folder for generated plots and results
├── pdfs
│   ├── internet_appendix.pdf                    # appendix for ITI paper
│   ├── Semester_project_proposal.pdf            # project proposal
│   └── The_Journal_of_Finance_2024_BOGOUSSLAVSKY_Informed_Trading_Intensity.pdf # published ITI paper
├── FNSPID.ipynb        # analysis notebook for FNSPID dataset with ITI and CRSP
├── gdelt.ipynb         # analysis notebook for GDELT dataset
├── iti.ipynb           # analysis notebook for ITI dataset
├── sec_8k.ipynb        # analysis notebook for SEC 8-K filings
├── .env                # environment variables (for WRDS)
├── .gitignore          # git ignore file
├── README.md           # this README
└── LICENSE             # license file

```
