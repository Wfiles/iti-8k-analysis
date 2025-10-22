import zipfile, orjson, polars as pl
from tqdm import tqdm

#Dowload zip file at https://www.sec.gov/search-filings/edgar-application-programming-interfaces, bulk data submissions.zip

zip_path = "../data/raw/submissions.zip"
rows = []

def to_str_list(x):
    if x is None:
        return []
    if isinstance(x, list):
        return [str(v) for v in x]
    # some feeds put "" instead of []:
    if x == "":
        return []
    return [str(x)]

def to_int_or_none(x):
    try:
        return int(x)
    except Exception:
        return None

with zipfile.ZipFile(zip_path, "r") as zf:
    for name in tqdm(zf.namelist(), desc="Parsing JSONs"):
        if not name.endswith(".json"):
            continue

        with zf.open(name) as f:
            data = orjson.loads(f.read())

        cik_raw = data.get("cik")
        cik_int = str(int(cik_raw)) if cik_raw else None
        company = data.get("name")
        tickers = to_str_list(data.get("tickers"))
        exchanges = to_str_list(data.get("exchanges"))
        sic = data.get("sic")
        sic_desc = data.get("sicDescription") or data.get("sic_description")

        recent = data.get("filings", {}).get("recent", {}) or {}
        forms      = recent.get("form", []) or []
        filing_dt  = recent.get("filingDate", []) or []
        accessions = recent.get("accessionNumber", []) or []
        prim_docs  = recent.get("primaryDocument", []) or []
        report_dt  = recent.get("reportDate", []) or []
        accept_ts  = recent.get("acceptanceDateTime", []) or []
        items      = recent.get("items", []) or []
        acts       = recent.get("act", []) or []
        sizes      = recent.get("size", []) or []
        file_no    = recent.get("fileNumber", []) or []
        film_no    = recent.get("filmNumber", []) or []
        prim_desc  = recent.get("primaryDocDescription", []) or []

        for i, (form, date, acc, doc) in enumerate(zip(forms, filing_dt, accessions, prim_docs)):
            if not form or not form.startswith("8-K"):
                continue

            item_val = items[i] if i < len(items) else None
            # sometimes 'items' is "", a list, or None → normalize to semicolon-joined string
            if isinstance(item_val, list):
                items_str = ";".join(map(str, item_val))
            elif item_val in (None, ""):
                items_str = None
            else:
                items_str = str(item_val)

            size_v = sizes[i] if i < len(sizes) else None
            size_int = to_int_or_none(size_v)

            acc_no_dash = acc.replace("-", "") if acc else ""
            base_dir = f"https://www.sec.gov/Archives/edgar/data/{cik_int}/{acc_no_dash}"

            rows.append({
                "cik": cik_raw,
                "cik_int": cik_int,
                "company_name": company,
                "tickers": ",".join(tickers),       # <- serialized to string
                "exchanges": ",".join(exchanges),   # <- serialized to string
                "sic": sic,
                "sic_description": sic_desc,
                "form": form,
                "filing_date": date,
                "report_date": report_dt[i] if i < len(report_dt) else None,
                "acceptance_datetime": accept_ts[i] if i < len(accept_ts) else None,
                "accession": acc,
                "primary_doc": doc,
                "primary_doc_description": prim_desc[i] if i < len(prim_desc) else None,
                "items": items_str,                 # <- serialized to string
                "act": acts[i] if i < len(acts) else None,
                "size_bytes": size_int,             # <- coerced to integer
                "file_number": file_no[i] if i < len(file_no) else None,
                "film_number": film_no[i] if i < len(film_no) else None,
                "url_html": f"{base_dir}/{doc}" if doc else None,
                "url_index": f"{base_dir}/index.html",
                "url_txt": f"{base_dir}.txt",
            })

# Build DataFrame safely
df = pl.DataFrame(rows)


df.write_parquet("sec_8k_filings_enriched.parquet", compression="zstd")
df.write_csv("sec_8k_filings_enriched.csv")
print(df.head())
print(f"Total 8-K rows: {df.height:,}")
