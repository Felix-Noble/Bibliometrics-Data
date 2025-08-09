# src/feature_extraction.py

####################################################################
## Accesses cleaned records in main dask db and extracts features ##
## (e.g. median reference count/year/decade, abstract embeddings) ##
####################################################################
from functools import cache
from os import read, wait
from os.path import join
import pyarrow as pa
import dask.dataframe as dd
import dask
import pandas as pd
import numpy as np
from pathlib import Path
import logging 
from sentence_transformers import SentenceTransformer
from dask.distributed import Client, LocalCluster, progress
from dask.diagnostics.progress import ProgressBar
import pyarrow.parquet as pq
import ast
import gc

from torch import embedding
logger = logging.getLogger(Path(__file__).stem)
dtype={'citation_count_OpenAlex': 'float64',
                                            "referenced_works_OpenAlex": 'string',
                                            "abstract_OpenAlex": 'string',
                                            "id_OpenAlex":'string',
                                            "publication_date":"string"}

# Turn Dask shuffle warning off
shuffle_logger = logging.getLogger("distributed.shuffle._scheduler_plugin")
shuffle_logger.setLevel(logging.ERROR)

def cache_index():
    idxs = ["A", "B"]
    idx = 0
    advance = yield idxs[idx]
    while True:
        if advance: 
            idx += 1
            if idx > len(idxs) - 1:
                idx = 0
        advance = yield idxs[idx]

# initialise indexer
cache_indexer = cache_index()
next(cache_indexer)
def read_ddf(path=f'/mnt/e/data/abstracts/cache/ddf/'): 
    path = os.path.join(path, cache_indexer.send(False))
    ddf = dd.read_parquet(path)
    return ddf 

def cache_ddf(ddf, path=f'/mnt/e/data/abstracts/cache/ddf/', npartitions=64):
    print("Index name:", ddf.index.name)
    path = os.path.join(path, cache_indexer.send(True))
    ddf = ddf.repartition(npartitions=npartitions)
    out = ddf.to_parquet(path, overwrite=True, write_index=True, compute=False, write_metadata_file=True)
    progress(client.compute(out))
    metadata = pq.read_metadata(os.path.join(path, "_metadata"))
    print(metadata.num_rows)
def convert_publication_date(ddf):
    ddf["publication_date"] = dd.to_datetime(ddf["publication_date"], format="%Y-%m-%d", errors="coerce")

    ddf = ddf.dropna(subset="publication_date")
    logger.info(f"Values converted, NA's Dropped | nrows ")
    ddf["publication_year"] = ddf["publication_date"].dt.year.astype(int)
    ddf["publication_date_int"] = ddf["publication_date"].astype(int)
    return ddf
    #

def add_yearly_stats(ddf):
    yearly_stats = ddf.groupby("publication_year")['citation_count_OpenAlex'].agg("median", meta=("median_citation_count_year", "float32"))
    yearly_stats = yearly_stats.reset_index()
    yearly_stats = yearly_stats.rename(columns= {'citation_count_OpenAlex':'median_citation_count_year'})
    #print(yearly_stats.columns)
    ddf = ddf.merge(yearly_stats, left_on='publication_year', right_on="publication_year", how='left')
    # Now, perform the final calculation on the merged data.
    ddf = ddf.assign(
        higher_than_median_year=(ddf['citation_count_OpenAlex'] > ddf['median_citation_count_year']).astype(int)
    )

    return ddf


def clean_func(text):

    problem_chars = [")", "(", "[", "]", "}", "{"]
    temp = text.lower() # lower case all text
    # text = text.translate(str.maketrans('', '', string.punctuation))
    # temp = temp.translate(str.maketrans('', '', string.digits))
    
    for char in problem_chars:
        temp = temp.replace(char, " ") # remove problem chars 
    temp = " ".join(temp.split()) # remove extra white space 
    return temp

def get_embeddings(ddf, model):

    
    # model_name = "all-MiniLM-L6-v2"
    def _get_embeddings_HuggingFace(texts: pd.Series, model: SentenceTransformer, embedding_cols: list):
        #texts = texts.apply(clean_func)
        #texts = texts.fillna('').astype(str).tolist()
        text_list = texts.astype(str).tolist()
        vectors = model.encode(text_list, convert_to_numpy=True)
        del text_list
        # CRITICAL CHECK: Ensure vectors is a valid 2D array before creating a DataFrame
        if not isinstance(vectors, np.ndarray) or vectors.ndim != 2 or vectors.shape[0] == 0:
            # If the output is empty or not a 2D array, return a correctly shaped empty DataFrame
            num_dimensions = model.get_sentence_embedding_dimension()
            return pd.DataFrame( columns=embedding_cols, index=texts.index)
        return pd.DataFrame(vectors, columns = embedding_cols, dtype=np.float32, index=texts.index) 
    
    N_FEATURES = model.get_sentence_embedding_dimension()

    embedding_cols = [f'embedding_{i}' for i in range(N_FEATURES)]
    
    meta_df = pd.DataFrame(columns=embedding_cols, dtype=np.float32)

    model_future = client.scatter(model, broadcast=True)
    embeddings = ddf["title_abstract"].map_partitions(
        _get_embeddings_HuggingFace, 
        model=model_future,
        embedding_cols = embedding_cols,
        meta=meta_df)
    
    del model_future
    client.run(gc.collect)
    return embeddings

def get_str_object_len(string):
    try : 
       return len(list(ast.literal_eval(string)))
    except Exception as e:
        return pd.NA

def replace_NA_empty_str(value):
    return "" if pd.isna(value) else str(value)

def replace_empty_str_NA(value):
    temp = " ".join(value.split())
    return pd.NA if len(temp) < 1 else value

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
    model_savedir = "/mnt/e/data/SBER/"
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
    journal = config["journal"]

    output_dir = pipeline_config["features_db_dir"] / journal
    os.environ['MALLOC_TRIM_THRESHOLD_'] = '1'
    
    client = Client(n_workers=4, threads_per_worker=1)
    #client.wait_for_workers(n_workers=N_WORKERS)
    logger.info(f"Initialised Client: {client.dashboard_link}")
    ddf = dd.read_parquet(pipeline_config["integrated_db_loc"] / journal)
    ddf["id_OpenAlex"] = ddf["id_OpenAlex"].apply(lambda x : np.int64(x.replace("https://openalex.org/W", "")), 
                                                meta=("id_OpenAlex","int64"))
     
    logger.info("Converting publication date")
    ddf = convert_publication_date(ddf)

    cache_ddf(ddf)
    ddf = read_ddf()

    ddf = add_yearly_stats(ddf)
    ddf.columns = ddf.columns.str.strip()
    cache_ddf(ddf)
    ddf = read_ddf()
    
    logger.info("Adding yearly stats")
    #ddf["referenced_works_count_OpenAlex"] = ddf["referenced_works_OpenAlex"].apply(
    #    get_str_object_len,
    #    meta = ('referenced_works_count_OpenAlex', 'float32') )

    title_str =  ddf["title"].apply(replace_NA_empty_str, meta = ("title_str", "string"))
    abstract_OpenAlex_str =  ddf["abstract_OpenAlex"].apply(replace_NA_empty_str, meta = ("abstract_OpenAlex_str", "string"))

    ddf["title_abstract"] = title_str + " " + abstract_OpenAlex_str
    del title_str, abstract_OpenAlex_str

    ddf["title_abstract"] = ddf["title_abstract"].apply(replace_empty_str_NA, meta=("title_abstract", "string"))
    
    ddf = ddf.dropna(subset = ["title_abstract"])
    ddf["title_abstract"] = ddf["title_abstract"].apply(clean_func, meta=("title_abstract", "string"))
    cache_ddf(ddf)
    ddf = read_ddf()

    ddf = ddf.set_index("id_OpenAlex") 
    #total_mem_gb = ddf.memory_usage(deep=True).sum().compute() / 1e9
    #npartitions = int(total_mem_gb // 0.05) + 1
    cache_ddf(ddf, npartitions=64)
    ddf = read_ddf()

    logger.info("Adding embeddings")
    embeddings = get_embeddings(ddf, model)
    ddf = ddf.merge(embeddings, left_index=True, right_index=True, how="left")

    cache_ddf(ddf)
    ddf = read_ddf()

    ddf.columns = ddf.columns.str.strip()
    #ddf = ddf.drop_duplicates(subset=["id_OpenAlex"])
    #ddf = ddf.set_index("id_OpenAlex")
    logger.info(f"Saving ddf | nrows ~ | cols: {ddf.columns}")

    #logger.info(f"N. Na's in referenced works col: {len(ddf['referenced_works_count_OpenAlex'].dropna().persist())}"
    out = ddf.to_parquet(
                output_dir / "parquet",
                engine='pyarrow',
                compute=False,
                write_index=True,
                overwrite=True,
                write_metadata_file=True,
                schema = {
                    "id_OpenAlex": "int64",
                    "publication_date": "timestamp[ns]",
                    "doi": "large_string",
                    "type": "large_string",
                    "open_access": "large_string",
                    "source": "large_string",
                    "citation_count_OpenAlex": "double",
                    "cited_by_percentile_year_OpenAlex": "double",
                    "citation_normalized_percentile_OpenAlex": "double",
                    "referenced_works_OpenAlex": "string",
                    "title": "large_string",
                    "abstract_OpenAlex": "large_string",
                    "topics": "large_string",
                    "publication_year": "int64",
                    "publication_date_int": "int64",
                    "median_citation_count_year": "double",
                    #"mean_citation_count_year":"double",
                    #"std_citation_count_year": "double",
                    #"normalised_citation_count_year": "double",
                    "higher_than_median_year": "int64",
                    "title_abstract": "string",
                    }
                )
    progress(client.compute(out), notebook=False)
    logger.info("Parquet saved")

    logger.info("Closing Dask Client")

    client.close()
