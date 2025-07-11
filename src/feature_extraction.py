# src/feature_extraction.py

####################################################################
## Accesses cleaned records in main dask db and extracts features ##
## (e.g. median reference count/year/decade, abstract embeddings) ##
####################################################################

import dask.dataframe as dd
import dask
import pandas as pd
import numpy as np
from pathlib import Path
import logging 
from sentence_transformers import SentenceTransformer
from dask.distributed import Client, LocalCluster, progress
from dask.diagnostics import ProgressBar

logger = logging.getLogger(Path(__file__).stem)

def calculate_groupby_stats(df):
    """
    This function is applied to each 'year' group in the Dask DataFrame.
    """
    # Isolate the column and drop NaNs for stats calculation
    citations = df["citation_count_OpenAlex"].dropna()

    # Handle cases where a year might have no valid citation data
    if citations.empty:
        median = np.nan
        mean = np.nan
        std = np.nan
    else:
        median = np.median(citations)
        mean = np.mean(citations)
        std = np.std(citations)

    # Create the new columns
    df["median_citation_count_year"] = median
    
    # Avoid division by zero if all citation counts in a year are the same
    if std > 0:
        df["normalised_citation_count_year"] = (df["citation_count_OpenAlex"] - mean) / std
    else:
        df["normalised_citation_count_year"] = 0.0 # Or np.nan, depending on desired output

    df["higher_than_median_year"] = (df["citation_count_OpenAlex"] > median).astype(int)
    
    return df#.drop(columns = ["publication_year"]) 

if __name__ == "__main__":
    # testing#
    from utils.load_config import get_feature_extract_config, get_pipeline_config
    from utils.setup_logging import setup_logger
    from tqdm.auto import tqdm
    import os
    ####################
    ## Install models ##
    ####################
   
    model_name = 'all-MiniLM-L6-v2'
    model_savedir = ".\\data\\SBERT"
    if not os.path.exists(os.path.join(model_savedir, model_name)):
        model = SentenceTransformer(model_name)
        os.makedirs(model_savedir, exist_ok=True)
        model.save(model_savedir)
        logger.info(f"Downloaded and saved Model {model}")
    else:
        model = SentenceTransformer(os.path.join(model_savedir,model_name))
        logger.info(f"Loaded Model {model}")

    pipeline_config = get_pipeline_config()
    config = get_feature_extract_config()
    setup_logger(logger, config["log"])

    client = Client(LocalCluster())
    logger.info(f"Initialised Client: {client}")
    model_future = client.scatter(model, broadcast=True)

    ddf = dd.read_parquet(pipeline_config["database_loc"])
    ddf["publication_date"] = dd.to_datetime(ddf["publication_date"])

    ddf = ddf.set_index("publication_date")
    ddf = ddf.dropna(subset = ["citation_count_OpenAlex"])
    #TODO: adapt this based on the size of the ddf on disk to ensure ~100MB per partition
    ddf = ddf.repartition(npartitions = 64)
    # meta = column structure of returned df
    #meta = ddf._meta.copy().drop(columns = ["publication_year"])
    #meta["median_citation_count_year"] = pd.Series(dtype='float64')
    #meta["normalised_citation_count_year"] = pd.Series(dtype='float64')
    #meta["higher_than_median_year"] = pd.Series(dtype='int32')

    #result_ddf = ddf.groupby('publication_year').apply(
    #        calculate_groupby_stats, 
    #        meta=meta,
    #        include_groups = False)
    #result_ddf = result_ddf.reset_index()
    yearly_stats = ddf.groupby("publication_year", observed=True).agg(
            median_citation_count_year = ('citation_count_OpenAlex', 'median'),
            mean_citation_count_year = ('citation_count_OpenAlex', 'mean'),
            std_citation_count_year = ('citation_count_OpenAlex', 'std')
            ).compute()

    result_ddf = dd.merge(ddf, yearly_stats, on = 'publication_year', how = 'left')
    del yearly_stats

    result_ddf = result_ddf.assign(
            normalised_citation_count_year = ( (result_ddf['citation_count_OpenAlex'] - result_ddf['mean_citation_count_year']) / result_ddf['std_citation_count_year'] ).fillna(0),
            higher_than_median_year = ( result_ddf['citation_count_OpenAlex'] > result_ddf['median_citation_count_year'] ).astype(int) 
            )

    # logger.debug(result_ddf.head)  
    logger.info("Results per year calculated")
    
    problem_chars = [")", "(", "[", "]", "}", "{"]
    def clean_func(text):
        temp = text.lower() # lower case all text
        # text = text.translate(str.maketrans('', '', string.punctuation))
        # temp = temp.translate(str.maketrans('', '', string.digits))
        
        for char in problem_chars:
            temp = temp.replace(char, " ") # remove problem chars 
        temp = " ".join(temp.split()) # remove extra white space 
        return temp
    
    # model_name = "all-MiniLM-L6-v2"
    def _get_embeddings_HuggingFace(texts: pd.Series, model: SentenceTransformer):
        # model = SentenceTransformer(model_name_or_path = model_name) # initialist SBERT model
        #TODO: add text cleaning to the worker jobs
        texts = texts.apply(clean_func)
        #texts = texts.fillna('').astype(str).tolist()
        text_list = texts.astype(str).tolist()
        vectors = model.encode(text_list, convert_to_numpy=True)
        return pd.DataFrame(vectors, columns = [f'embedding_{i}' for i in range(vectors.shape[1])], index=texts.index) 
    
    # clean_text = result_ddf["abstract_OpenAlex"].apply(clean_func, meta=meta)

    N_FEATURES = model.get_sentence_embedding_dimension()

    embedding_columns = [f'embedding_{i}' for i in range(N_FEATURES)]
    
    meta_df = pd.DataFrame(columns=embedding_columns, dtype=np.float32)
    del N_FEATURES, embedding_columns

    result_ddf = result_ddf.dropna(subset=["abstract_OpenAlex"]) 
    embeddings = result_ddf["abstract_OpenAlex"].map_partitions(
        _get_embeddings_HuggingFace, 
        model_future,
        meta=meta_df)
    
    result_ddf = result_ddf.merge(embeddings, left_index=True, right_index=True)
    
    logger.info("Computing Embeddings")
    result_ddf = result_ddf.persist() 
    progress(result_ddf) 
    result_ddf = result_ddf.reset_index()
    
    logger.info("Saving Results")
    result_ddf.to_parquet(
            pipeline_config["database_loc"].parent / "temp",
            engine='pyarrow',
            write_index=False,
            overwrite=True,
            partition_on=["publication_year"]
            )

    logger.info("Paraquet saved")

    logger.info("Closing Dask Client")
 
    client.close()

    quit()
    
