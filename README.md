# news-based-asset-pricing
# [cite_start]Enhancing News-Based Asset Pricing with Information-Driven Trading Signals [cite: 3]

## Overview
[cite_start]This repository outlines a Data Science Master's research project (COM-412) at EPFL for the Autumn Semester 2025[cite: 1, 2, 6, 10, 12]. [cite_start]The project aims to investigate whether incorporating information-driven trading (ITI) can enhance the predictive power of news-augmented asset pricing models[cite: 13].

## Methodology
* [cite_start]The project will unite two information channels by combining an embedding model, such as FinBERT, to create news embeddings with ITI scores[cite: 19].
* [cite_start]Each news signal will be weighted according to the level of informed trading that precedes it[cite: 19].
* [cite_start]The team will explore embedding the ITI metric directly into the representation so that a headline's impact is automatically scaled by the prevailing level of informed trading[cite: 20].

## Team & Responsibilities
* [cite_start]**Matthias Wyss**: Primarily handles the natural language processing component[cite: 5, 23]. [cite_start]This includes collecting financial news data, generating text embeddings using models such as FinBERT, and developing baseline pricing models that incorporate these embeddings[cite: 23].
* [cite_start]**William Jallot**: Focuses on the market microstructure dimension[cite: 5, 24]. [cite_start]This involves working with microstructural data to compute ITI scores based on established methodologies, or potentially extending or refining the metric[cite: 24].
* [cite_start]Both team members will collaborate closely on model integration, experimental design, and the incorporation of traditional factor models[cite: 26].

## Preliminary Data Sources
[cite_start]The analysis will utilize several datasets, which will be expanded as the project progresses[cite: 28, 29]:
* [cite_start]Reuteurs financial news from 2006 to 2013 [cite: 30]
* [cite_start]Bloomberg and Reuters dataset [cite: 31]
* [cite_start]FNSPID [cite: 32]
* [cite_start]Nifty [cite: 33]
* [cite_start]FinSen [cite: 34]
* [cite_start]SEntFiN [cite: 35]
* 
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
