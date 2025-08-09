# Academic Abstracts Pipeline

## Overview
This repository contains a **modular, end-to-end data pipeline** for:
- **Scraping** academic metadata and abstracts from multiple scholarly APIs
- **Integrating** raw scraped data into a unified, deduplicated database
- **Extracting features** such as yearly citation statistics and semantic embeddings
- **Generating descriptive analytics** and visualizations

The system is designed for **scalability** (via Dask) and **extensibility** (modular config-driven architecture).

---

## Features

### 1. Config-Driven Architecture
- Centralized configuration in `config/config.toml`
- Strong **type and key validation** for configs (`utils/load_config.py`)
- Supports **multiple journals** and **year ranges**
- Easily switch between datasets without changing code

---

### 2. Data Scraping
**Implemented:**
- **OpenAlex API**:
  - Search by journal and year
  - Retrieve metadata, abstracts, citation counts, references, and topics
  - Handles polite API usage (rate limits, retries, adaptive wait times)
  - Saves results in **batch CSV files**
- **Reference Expansion**:
  - After initial scrape, fetches metadata for all referenced works
- **Duplicate Avoidance**:
  - Tracks already-scraped IDs to prevent redundant API calls

**Partially Implemented / Planned:**
- 🚧 **CrossRef API** integration for DOI discovery
- 🚧 **Semantic Scholar API** integration for additional metadata
- 🚧 **Citation Expansion** (fetch citing papers in addition to references)
- 🚧 **Parallelized multi-journal scraping** with distributed workers

---

### 3. Data Integration
**Implemented:**
- Reads all raw CSV batches for a journal
- Cleans and deduplicates records by `id_OpenAlex`
- Converts to **partitioned Parquet** format for efficient downstream processing
- Uses **Dask** for scalable processing

**Planned:**
- 🚧 Merge data from multiple APIs into a **single enriched record**
- 🚧 Automated schema alignment across sources

---

### 4. Feature Extraction
**Implemented:**
- **Date normalization** and extraction of `publication_year`
- **Yearly citation statistics**:
  - Median citations per year
  - Binary flag for "higher than median" citation count
- **Text cleaning** for abstracts and titles
- **Sentence embeddings**:
  - Uses HuggingFace `SentenceTransformer` models
  - Configurable model name
  - Stores embeddings as 384-dimensional float vectors
- **Caching** intermediate Dask DataFrames to disk for iterative processing

**Planned:**
- 🚧 Additional bibliometric features (e.g., h-index, citation velocity)
- 🚧 Topic modeling (LDA, BERTopic)

---

### 5. Descriptive Analytics
**Implemented:**
- Histograms for citation metrics
- Scatter plots of citation metrics vs. publication year
- Saves figures to `figures/` directory

**Planned:**
- 🚧 Interactive dashboards (Plotly/Dash)
- 🚧 Time-series trend analysis

---

## Tech Stack
- **Python 3.10+**
- **Dask** for scalable data processing
- **DuckDB** for fast local SQL queries
- **Pandas** for tabular data manipulation
- **SentenceTransformers** for embeddings
- **Matplotlib** for static visualizations
- **PyArrow** for Parquet I/O
- **TOML** for configuration

---

## Project Structure## 🔄 How OpenAlex Queries Are Renamed

The pipeline automatically **maps raw OpenAlex API field names** to **more recognisable database column names** before saving to the final Parquet database.

| OpenAlex Query Term         | Final Database Column Name       | Description |
|-----------------------------|----------------------------------|-------------|
| `id`                        | `id_OpenAlex`                    | Unique OpenAlex work ID |
| `cited_by_count`            | `citation_count_OpenAlex`        | Total citations received |
| `referenced_count`          | `reference_count_OpenAlex`       | Number of references in the paper |
| `referenced_works`          | `referenced_works_OpenAlex`      | List of referenced work IDs |
| `abstract_inverted_index`   | `abstract_OpenAlex`              | Full abstract text (de-inverted) |
| `source`                    | `source` (dict)                  | Journal/source metadata |

This mapping is defined in the `OpenAlexHandler.format_names` dictionary in `scrape.py` and ensures **consistent, human-readable column names** across the pipeline.

---

## 🛠️ Project Structure
~~~
src/
├── scrape2.py                 # OpenAlex scraping + reference expansion
├── data_integration_export.py # CSV → Parquet integration
├── feature_extraction.py      # Feature engineering + embeddings
├── descriptives.py            # Histograms & scatter plots
├── utils/
│     ├── load_config.py       # Config loading & validation
│     ├── setup_logging.py     # Logger setup
config/
└── config.toml                # Pipeline configuration
logs/                          # Runtime logs
figures/                       # Generated plots
~~~
---

## Example Workflow
```bash
# 1. Configure the Pipeline
# Edit config/config.toml to set journals, directories, and year range
# Example:
# [pipeline]
# journals = ["American Chemical Society", "Science"]
# scrape_output_dir = "~/data/abstracts"
# integrated_db_dir = "~/data/abstracts_integrated"
# features_db_dir = "~/data/abstracts_features"
# start_year = 2023
# end_year = 2022

# 2. Scrape Data from OpenAlex
# Fetches metadata, abstracts, and references for each journal/year
python src/scrape.py

# 3. Integrate Data
# Reads all CSV batches, cleans, deduplicates, and writes Parquet
python src/data_integration_export.py

# 4. Extract Features
# Adds yearly citation stats, cleans text, generates embeddings
python src/feature_extraction.py

# 5. Generate Descriptive Plots
# Produces histograms and scatter plots for citation metrics
python src/descriptives.py
```
---
