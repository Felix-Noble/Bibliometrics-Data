# 📚 AcademicAbstracts — Data Processing Pipeline

**Version:** 0.2.1  
**Author:** Felix Noble  

A modular, scalable **data processing pipeline** for collecting, integrating, and preparing bibliometric data from open-source academic databases (**OpenAlex**, **CrossRef**, **Semantic Scholar**) for downstream machine learning analysis.

---

## 🚀 Overview

The **data processing** part of this project handles the **entire ETL (Extract, Transform, Load)** workflow for academic bibliometrics:

1. **Scraping** — Retrieve raw metadata, abstracts, references, and citations from multiple APIs.
2. **Integration** — Merge and deduplicate data into a unified, partitioned Parquet database.
3. **Preprocessing** — Clean and standardize text, dates, and identifiers.
4. **Feature Extraction** — Generate yearly citation statistics and compute **semantic embeddings** for abstracts using `sentence-transformers`.
5. **Descriptive Analysis** — Produce histograms and scatter plots for exploratory data analysis.

The processed datasets are stored in **efficient, queryable Parquet format** for use in ML pipelines (e.g., the model training code in the analysis repo).

---

## ✨ Key Capabilities

### **1. Multi-Source Academic Data Scraping**
- **OpenAlex API** — Journal search, work retrieval, reference and citation expansion.
- **CrossRef API** — Journal ISSN lookup and publication metadata.
- **Semantic Scholar API** — Paper metadata, citation counts, and references.
- **Rate-limit aware** — Automatic polite waits and retry logic.
- **Batch processing** — Configurable buffer sizes for large-scale scraping.

### **2. Data Integration**
- Merge multiple CSV batches into a **single Dask DataFrame**.
- Deduplicate by `id_OpenAlex`.
- Partition and store as **Parquet** for fast downstream access.
- Schema enforcement for consistent typing.

### **3. Preprocessing**
- Convert publication dates to `datetime` and integer formats.
- Add yearly median citation statistics.
- Label papers as **above/below median citations** for their publication year.
- Clean abstracts and titles (remove problem characters, normalize whitespace).

### **4. Feature Extraction**
- Generate **semantic embeddings** for `title + abstract` using HuggingFace models (e.g., `all-MiniLM-L6-v2`).
- Store embeddings alongside metadata in the Parquet database.
- Fully parallelized with **Dask Distributed**.

### **5. Descriptive Statistics & Visualisation**
- Histograms for citation counts, median citations, and binary labels.
- Scatter plots of metrics vs. publication year.
- Outputs saved to `/figures` for inclusion in reports and README.

---

## ⚙️ Configuring the Pipeline

All modules are controlled via `config/config.toml`.  
This file defines **API settings, query terms, output locations, and processing parameters**.

### **Example `config.toml` Structure**
```toml
[pipeline]
journals = ["American Chemical Society", "Science", "Mind", "The Lancet", "Cell", "Psychological Review"]
integrated_db_loc = "~/data/abstracts_integrated"
features_db_loc = "~/data/abstracts_integrated/"

[pipeline.log]
file = "ERROR"
console = "INFO"

[scrape]
polite_email = "your.email@example.com"
OPENALEX_BASE_URL = "https://api.openalex.org"
CROSSREF_BASE_URL = "https://api.crossref.org"
S2_BASE_URL = "https://api.semanticscholar.org/graph/v1"
S2_API_KEY = "424242"
output_dir = "~/data/abstracts/"
buffer_size = 100
max_retry = 3
polite_wait = 1.0
fail_wait = 10.0
rate_limit_wait = 60.0
timeout_wait = 5.0
start_year = 2023
end_year = 2022

S2_queries = [
    "paperId", "externalIds", "url", "title", "abstract", "publicationDate",
    "journal", "isOpenAccess", "openAccessPdf", "citationCount", "referenceCount",
    "citations.paperId", "citations.externalIds"
]

OpenAlex_queries = [
    "publication_date", "id", "doi", "type", "open_access", "source",
    "cited_by_count", "cited_by_percentile_year_OpenAlex",
    "citation_normalized_percentile_OpenAlex", "referenced_works",
    "title", "abstract_inverted_index", "topics"
]

[scrape.log]
console = "INFO"
file = "ERROR"

[integration]
input_dir = ""
output_dir = ""
journal = ""

[integration.log]
console = "INFO"
file = "ERROR"

[preprocess.log]
file = "ERROR"
console = "INFO"

[feature_extraction]
HuggingFace_model_name = "all-MiniLM-L12-v2"
journal = "test"

[feature_extraction.log]
console = "INFO"
file = "ERROR"
```

---

## 🔄 How OpenAlex Queries Are Renamed

The pipeline automatically **maps raw OpenAlex API field names** to **more recognisable database column names** before saving to the final Parquet database.

| OpenAlex Query Term         | Final Database Column Name       | Description |
|-----------------------------|----------------------------------|-------------|
| `id`                        | `id_OpenAlex`                    | Unique OpenAlex work ID |
| `cited_by_count`            | `citation_count_OpenAlex`        | Total citations received |
| `referenced_count`          | `reference_count_OpenAlex`       | Number of references in the paper |
| `referenced_works`          | `referenced_works_OpenAlex`      | List of referenced work IDs |
| `abstract_inverted_index`   | `abstract_OpenAlex`               | Full abstract text (de-inverted) |
| `source`                    | `source` (dict)                  | Journal/source metadata |

This mapping is defined in the `OpenAlexHandler.format_names` dictionary in `scrape2.py` and ensures **consistent, human-readable column names** across the pipeline.

---

## 🛠️ Project Structure
