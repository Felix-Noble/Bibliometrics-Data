# data_integration_export.py
import pandas as pd
import dask.dataframe as dd
import shutil
import numpy as np
import os
from pathlib import Path
import logging
from typing import List
from dask.distributed import Client, LocalCluster

logger = logging.getLogger(Path(__file__).stem)

# --- Plan Implementation: Part 1 - Data Integration and Export Module ---
# This script is responsible for connecting to the 'dash DB' and exporting
# processed data into it. It covers the following parts of the plan:
# 1.1: Database Connection Manager (via get_db_engine)
# 1.2: Data Formatting (within export_to_dash_db)
# 1.3: Export Functionality (the export_to_dash_db function itself)
# 1.4: Idempotency (using the 'if_exists' parameter)

def export_csv_to_dask(input_dirs: List[Path], 
                       output_dir: Path, 
                       index_col: str ="id_OpenAlex", 
                       partition_col: str = "",
                       publication_date_col: str ="publication_date"):
    """
    Reads all CSV files from an input directory using Dask, and saves them
    as a single, partitioned Parquet dataset.

    Args:
        input_dirs : Paths to the directories containing source CSV files.
        output_dir : Path to the directory where the Parquet dataset will be saved.
        index_col : The name of the column to index by
        partition_col : The name of the column to partition the data by.
    """
    # --- 1. Setup: Ensure directories exist and handle potential errors ---
    client = Client(LocalCluster())
    logger.info(f"CLient Initialised | {client.dashboard_link}")
    
    if os.path.exists(output_dir):
        logger.warning(f"Warning: Output directory '{output_dir}' already exists.")
    
    # --- 2. Load all CSV filepaths ---
    csv_paths = []
    for path in input_dirs:
        csv_paths += list(path.glob("*.csv"))
        logger.info(f"Indexing all CSVs from: {path}")
        logger.debug(list(path.glob("*")))
        logger.debug(f"Path exists: {os.path.exists(path)} for Path: {path}")
    logger.debug(f"Found CSV paths: {csv_paths}")
    logger.info(f"Found {len(csv_paths)} files")

    # --- 3. Read CSV files and drop on invalid/duplicate id's ---
    try:
        ddf = dd.read_csv(csv_paths, 
                          dtype={'citation_count_OpenAlex': 'int32',
                                            "referenced_works_OpenAlex": 'object',
                                            "abstract_OpenAlex": 'string',
                                            "id_OpenAlex":'string',
                                            "source": "object",
                                            "publication_date":"object"})
        
        ddf = ddf.dropna(subset=[index_col])
        ddf = ddf.repartition(npartitions=64).persist()
       
        logger.info(f"Data loaded. Rows = {len(ddf)}")
        logger.info(f"Columns: {list(ddf.columns)}")
    
        ddf = ddf.drop_duplicates(subset = index_col, keep="first").persist()
        logger.info(f"Dropped duplicates... UNique Ids = {len(ddf)}")

        ddf = ddf.dropna(subset="id_OpenAlex")
        #ddf["publication_date_int"] = dd.to_datetime(ddf["publication_date"], format="%Y").astype(int)
#        logger.debug(ddf["publication_date_int"])
        #ddf = ddf.set_index(index_col).persist()
        #logger.info("Index set to id_OpenAlex")
    except Exception as e:
        logger.error(f"{e}")
        return
    
    # --- 4. Save data to parquet ---
    try:
        logger.info(f"\nWriting data to Parquet in '{output_dir}'...")

        ddf.to_parquet(
            output_dir,
            engine='pyarrow',
            write_index=True,
            overwrite=True,
            schema = {"id_OpenAlex": "string",
                "publication_date": "string",
                "referenced_works_OpenAlex": "string",
                }
        )
        logger.info(f"\n✅ Success! Data converted and saved to '{output_dir}'")
    except Exception as e:
        logger.error(f"An error occurred during Parquet conversion: {e}")

    finally:
        client.close()

## RUN ##
def run_main_data_integration_export(config):
  # placeholders dor config imports
    #TODO add client here and import at top
    journal = config["journal"] 
    output_dir = config["output_dir"]
    input_dir = config["input_dir"]
    os.makedirs(output_dir, exist_ok=True)
    directories = []

    input_dir_main = input_dir /  journal
    input_dir_ref = input_dir / journal / "references"
    input_dir_cit = input_dir / journal / "citations"
    
    directories.extend([input_dir_main, input_dir_cit, input_dir_ref])
    logger.debug(directories)
    output_dir_main = output_dir / journal

    # create main db
    export_csv_to_dask(directories, output_dir_main, 
                       publication_date_col="publication_date")

    ### do this for the reference dir also
if __name__ == "__main__":
    from utils.load_config import get_integration_config, get_pipeline_config
    from utils.setup_logging import setup_logger
    config = get_integration_config()
    pipeline_config = get_pipeline_config()
    config["output_dir"] = pipeline_config["integrated_db_loc"]

    setup_logger(logger, config["log"])

    run_main_data_integration_export(config)


