#scrape2.py
import requests
import time
import pandas as pd
from typing import List, Dict, Optional, Any, Tuple
from urllib.parse import quote_plus
from tqdm.auto import tqdm
import numpy as np
import os
import logging
import math
from pathlib import Path
from src.utils.setup_logging import setup_logger

## Logger ##
logger = logging.getLogger(Path(__file__).stem)

## Helpers ##
def _index_scraped_ids(output_dir: Path, id_types:Tuple[str] = ("doi", "id_OpenAlex")):
    """
    Returns a dict of all available id's in scraped data.
    """
    ids_found = {k:[] for k in id_types}

    logger.info(f"Searching {output_dir} for finished id's")
    logger.debug(f"Searching for {', '.join(id_types)}")

    for file in output_dir.glob("*.csv"):
        df = pd.read_csv(file, chunksize = 100)
        for chunk in df:
            for row in chunk.itertuples():
                for id in id_types:
                    ids_found[id].append(getattr(row, id))

    return {k:np.array(v) for k,v in ids_found.items()}

# --- Base Class for Progress Bars ---

class ProgressBarManager:
    """A base class to handle creating and updating tqdm progress bars."""
    def __init__(self):
        self.pbar = None

    def _setup_progress_bar(self, total: int, description: str):
        """Initializes a new progress bar."""
        self.pbar = tqdm(total=total, desc=description, unit="paper")

    def _update_progress_bar(self, n: int = 1):
        """Updates the progress bar."""
        if self.pbar:
            self.pbar.update(n)

    def _close_progress_bar(self):
        """Closes the progress bar and cleans up."""
        if self.pbar:
            self.pbar.close()
            self.pbar = None

class ResponseManager:
    def __init__(self):
        """A base class that manages reponse delays"""
        self.filler_response_time = 1000
        self.response_times = [self.filler_response_time]

    def _adapt_wait_time(self, response:Dict = {}, increase_factor: float = 1.5, decrease_factor: float = 0.9):
       
        response_time = response.get("meta", {}).get("db_response_time_ms", self.filler_response_time) 
        
        if self.response_times[-1] >= response_time:
            self.polite_wait = self.polite_wait * increase_factor
        else:
            self.polite_wait = self.polite_wait * decrease_factor
        self.response_times.append(response_time)

# --- Handler Classes for Each Data Source ---
# CROSSREF

def search_crossref(journal_name: str, 
                    publication_year: int, polite_email: str, 
                    max_results: int = 1000,) -> List[Dict]:
    global CROSSREF_BASE_URL
    """
    Searches CrossRef for works from a specific journal and year.

    Returns:
        A list of dicts, each containing 'doi'.
    """
    print(f"\n--- Searching CrossRef for '{journal_name}' ({publication_year}) ---")
    
    # 1. Find the journal's ISSN
    journal_search_url = f"{CROSSREF_BASE_URL}/journals"
    params = {'query.container-title': journal_name, 'rows': 1, 'mailto': POLITE_EMAIL}
    try:
        response = requests.get(journal_search_url, params=params)
        response.raise_for_status()
        journal_data = response.json()
        items = journal_data.get('message', {}).get('items', [])
        if not items or 'ISSN' not in items[0]:
            print(f"Error: Journal '{journal_name}' or its ISSN not found on CrossRef.")
            return []
        issn = items[0]['ISSN'][0]
        print(f"Found CrossRef Journal ISSN: {issn}")
    except requests.RequestException as e:
        print(f"Error finding journal on CrossRef: {e}")
        return []

    # 2. Fetch works using the ISSN and year
    works_url = f"{CROSSREF_BASE_URL}/journals/{issn}/works"
    results = []
    offset = 0
    while len(results) < max_results:
        try:
            params = {
                'filter': f"from-pub-date:{publication_year}-01-01,until-pub-date:{publication_year}-12-31",
                'rows': min(1000, max_results - len(results)), # Max rows per request is 1000
                'offset': offset,
                'mailto': polite_email
            }
            response = requests.get(works_url, params=params)
            response.raise_for_status()
            data = response.json().get('message', {})
            items = data.get('items', [])
            
            for work in items:
                results.append({'doi': work.get('DOI')})
            
            print(f"Retrieved {len(results)} works from CrossRef so far...")
            
            if not items or len(results) >= data.get('total-results', 0):
                break
            offset += len(items)
            time.sleep(0.1) # Be polite
        except requests.RequestException as e:
            print(f"Error fetching works from CrossRef: {e}")
            break

    print(f"Finished CrossRef search. Found {len(results)} works.")
    return results

# SEMANTIC SCHOLAR (S2)

def search_S2(journal_name: str, 
                            publication_year: int, 
                            S2_API_KEY: str = "",
                            max_results: int = 1000) -> List[Dict]:
    """
    Searches Semantic Scholar for works from a specific journal and year.

    Returns:
        A list of dicts, each containing 's2_paper_id' and 'doi'.
    """
    global S2_BASE_URL
    print(f"\n--- Searching Semantic Scholar for '{journal_name}' ({publication_year}) ---")
    
    search_url = f"{S2_BASE_URL}/paper/search"
    results = []
    offset = 0
    headers = {'x-api-key': S2_API_KEY} if S2_API_KEY else {}

    while len(results) < max_results:
        try:
            params = {
                'query': f'"{journal_name}"', # Use quotes for exact journal name match
                'year': str(publication_year),
                'offset': offset,
                'limit': 100, # Max limit per request
                'fields': 'externalIds'
            }
            response = requests.get(search_url, params=params, headers=headers)
            response.raise_for_status()
            data = response.json()
            
            papers = data.get('data', [])
            for paper in papers:
                 results.append({
                    's2_paper_id': paper.get('paperId'),
                    'doi': paper.get('externalIds', {}).get('DOI')
                })
            
            print(f"Retrieved {len(results)} works from Semantic Scholar so far...")

            if 'next' not in data or not papers:
                break
            offset = data['next']
            time.sleep(0.5) # Be polite to the API
        except requests.RequestException as e:
            print(f"Error fetching works from Semantic Scholar: {e}")
            break

    print(f"Finished Semantic Scholar search. Found {len(results)} works.")
    return results

def get_details_S2(paper_ids: List[str], query_terms: list,
                                 S2_API_KEY: str = "") -> Dict[str, Dict]:
    """
    Retrieves details from Semantic Scholar for a list of S2 Paper IDs or DOIs.

    Returns:
        A dictionary mapping the input ID to its details.
    """
    global S2_BASE_URL
    print(f"\n--- Getting details from Semantic Scholar for {len(paper_ids)} papers ---")
    details_map = {}
    
    # S2 batch endpoint is limited to 1000 IDs per call
    chunk_size = 500
    headers = {'x-api-key': S2_API_KEY} if S2_API_KEY else {}

    for i in range(0, len(paper_ids), chunk_size):
        chunk = paper_ids[i:i + chunk_size]
        print(f"  ...processing chunk {i//chunk_size + 1}")
        try:
            response = requests.post(
                f"{S2_BASE_URL}/paper/batch",
                headers=headers,
                params={'fields': ",".join(query_terms)},
                json={"ids": chunk}
            )
            response.raise_for_status()
            
            for paper_data in response.json():
                if paper_data: # API returns null for not-found papers
                    details_map[paper_data['paperId']] = {
                        's2_abstract': paper_data.get('abstract'),
                        's2_reference_count': paper_data.get('referenceCount'),
                        's2_citation_count': paper_data.get('citationCount'),
                        's2_citing_papers': [c['paperId'] for c in paper_data.get('citations', [])]
                    }
            time.sleep(1) # Be polite
        except requests.RequestException as e:
            print(f"An error occurred during Semantic Scholar batch request: {e}")
            
    print(f"Found details for {len(details_map)} papers on Semantic Scholar.")
    return details_map

class OpenAlexHandler(ProgressBarManager, ResponseManager):
    """Handles all interactions with the OpenAlex API."""    
    def __init__(self,
                 OPENALEX_BASE_URL: str,
                 polite_email,
                 max_retry: int = 5,
                 polite_wait: float = 0.1,
                 fail_wait: float = 10,
                 rate_limit_wait: float = 60,
                 timeout: float = 15,
                 ):
        
        ProgressBarManager.__init__(self)
        ResponseManager.__init__(self)

        self.OPENALEX_BASE_URL = OPENALEX_BASE_URL
        self.polite_email = polite_email
        self.polite_wait = polite_wait
        self.max_rety = max_retry
        self.fail_wait = fail_wait
        self.rate_limit_wait = rate_limit_wait
        self.timeout = timeout

        self.NA_VALUE = None
        self.headers = { # set headers to present as browser 
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/107.0.0.0 Safari/537.36'
        }

        self.format_funcs = {
                        "abstract_inverted_index":self.deinvert_abstract,
                        "source" : self.unpack_source,
                            }
        
        self.format_names = {
                        "id" : "id_OpenAlex",
                        "cited_by_count": "citation_count_OpenAlex",
                        "referenced_count" : "reference_count_OpenAlex",
                        "referenced_works": "referenced_works_OpenAlex",
                        "abstract_inverted_index": "abstract_OpenAlex"
        }
        self.errors = {
            404: [],
                  }
                
    def unpack_source(self, source):
        source_dict = {"id": source.get("id", self.NA_VALUE),
                       "name": source.get("display_name", self.NA_VALUE),
                       "is_open_access" : source.get("is_oa")}
        
        return source_dict
    
    def deinvert_abstract(self, inverted_abstract: Optional[Dict]) -> Optional[str]:
        """
        Converts an OpenAlex inverted abstract into a plain text string.
        See: https://docs.openalex.org/how-to-use-the-api/get-single-entities/get-single-works#the-abstract
        """
        if not inverted_abstract:
            return None

        # The abstract is a list of words. The index gives the position of each word.
        # We create a list of the correct size and fill it with the words.
        abstract_length = max([max(positions) for positions in inverted_abstract.values()]) + 1
        
        abstract_list = [''] * abstract_length
        try: 
            for word, positions in inverted_abstract.items():
                for pos in positions:
                    abstract_list[pos] = word
        except:
            print(abstract_length)
            print(inverted_abstract)
            quit()
            
        return ' '.join(abstract_list)

    def search_openalex(self, 
                        journal_name: str, 
                        publication_year: int, 
                        max_results: int = 1000) -> List[Dict]:
       
        """
        Searches OpenAlex for works from a specific journal and year.

        Returns:
            A list of dicts, each containing 'openalex_id' and 'doi'.
        """
        
        logger.info(f"--- Searching OpenAlex for '{journal_name}' ({publication_year}) ---")
        
        # 1. Find the journal's OpenAlex ID
        journal_search_url = f"{self.OPENALEX_BASE_URL}/journals?search={quote_plus(journal_name)}"
        try:
            response = requests.get(journal_search_url, params={'mailto': self.polite_email}, timeout=self.timeout)
            response.raise_for_status()
            journal_data = response.json()
            if not journal_data.get('results'):
                logger.error(f"Error: Journal '{journal_name}' not found on OpenAlex.")
                return []
            journal_id = journal_data['results'][0]['id'].split("/")[-1]
            logger.info(f"Found OpenAlex Journal ID: {journal_id}")
        except requests.RequestException as e:
            logger.error(f"Error finding journal on OpenAlex: {e}")
            return []

        # 2. Fetch works using the journal ID and year
        works_url = f"{self.OPENALEX_BASE_URL}/works"
        filters = f"host_venue.id:{journal_id},publication_year:{publication_year}"
        
        self._setup_progress_bar(total=max_results, description=f"Searching OpenAlex for {journal_name}-{publication_year}")
        results = []
        cursor = "*"
        
        while cursor and len(results) < max_results:
            try:
                params = {
                    'filter': filters,
                    'per_page': min(10, max_results - len(results)),
                    'cursor': cursor,
                    'mailto': self.polite_email
                }
                response = requests.get(works_url, params=params, headers=self.headers, timeout=self.timeout)
                response.raise_for_status()
                data = response.json()
                
                for work in data.get('results', []):
                    results.append({
                        self.format_names["id"]: work.get('id'),
                        'doi': work.get('doi')
                    })
                
                cursor = data['meta'].get('next_cursor')
                self._update_progress_bar()
                time.sleep(self.polite_wait)

            except requests.RequestException as e:
                logger.error(e)
                quit()
                break

        self._close_progress_bar()      
        logger.info(f"Finished OpenAlex search. Found {len(results)} works.")
        return results

    def setup_query_fetch(self, colnames):
        rows = []
        # TODO: make errors a class attribute 
        # TODO: check for rate limit error and wait 60. 

        for col in colnames:
            if col in self.format_names.keys():
                colnames[colnames.index(col)] = self.format_names[col]

        return rows, colnames

    def get_details_by_oaid(self, 
                            open_alex_ids: List[str],
                            query_terms: List[str],
                            ) -> pd.DataFrame:
        """
        Retrieves abstracts from OpenAlex for a list of OpenAlex IDs.

        Returns:
            A dictionary mapping OpenAlex ID to its abstract string.
        """
        colnames = ["id"] + query_terms
        rows, colnames = self.setup_query_fetch(colnames)

        logger.info(f"--- Getting abstracts from OpenAlex for {len(open_alex_ids)} papers ---")
        
        self._setup_progress_bar(total=len(open_alex_ids), description="Getting OpenAlex Details")

        for i, oaid in enumerate(open_alex_ids):
            logger.debug(f"{i}-{oaid}")
            tries = 0
            if not oaid or oaid is None or pd.isna(oaid): 
                continue
            while tries < self.max_rety:
                try:
                    
                    response = requests.get(f"{self.OPENALEX_BASE_URL}/works/{oaid}", params={'mailto': self.polite_email}, timeout=self.timeout)
                    
                    # specialised response error handling
                    if response.status_code == 429:
                        tqdm.write(f"--> Error | {response.status_code} | Rate limit, waiting for {self.rate_limit_wait}")
                        time.sleep(60)
                        self._adapt_wait_time(increase_factor = 4) 
                        tries += 1
                    
                    # generalised response error handling 
                    elif response.status_code in self.errors.keys():
                        tqdm.write(f"--> Error | {response.status_code} | Waiting for {self.rate_limit_wait}")
                        self.errors[response.status_code].append(oaid)
                        time.sleep(self.fail_wait)
                        tries += 1
                        continue
                
                    response.raise_for_status()
                    data = response.json()
                    df_row = [oaid]
                    for query in query_terms:
                        # query formatting logic 

                        query_result = data.get(query, self.NA_VALUE)
                        if query in self.format_funcs and query_result:
                            query_result = self.format_funcs[query](query_result)

                        df_row.append(query_result)

                    rows.append(df_row)
                    
                    tries = self.max_rety + 1
                    self._update_progress_bar()
                    time.sleep(self.polite_wait) # Be polite
                except requests.exceptions.Timeout:
                    tqdm.write("--> Search request timed out.")
                except requests.RequestException as e:
                    tries += 1
                    str_e = str(e)
                    tqdm.write(f"--> Try {tries} failed | {str_e} | response = {response}")
                    if str_e not in self.errors.keys():
                        self.errors[str_e] = []
                    self.errors[str_e].append(oaid)
                    time.sleep(self.fail_wait)

        self._close_progress_bar()
        logger.info(f"Found details for {len(rows)} papers on OpenAlex via OpenAlex ID.")
        if np.any([len(val) for val in self.errors.values()]):
            logger.error(self.errors)
        return pd.DataFrame(rows, columns=colnames)
    
    def get_details_by_doi(self, 
                            dois: List[str],
                            query_terms: List[str],
                            ) -> pd.DataFrame:
        """
        Retrieves details from OpenAlex for a list of DOIs.

        Returns:
            A dictionary mapping DOI to its details.
        """

        colnames = ["doi"] + query_terms
        rows, colnames = self.setup_query_fetch(colnames)

        logger.info(f"\n--- Getting details from OpenAlex for {len(dois)} DOIs ---")
        print(logger.level)
        print("DEBUG logger level")
        self._setup_progress_bar(total=len(dois), description="Getting OpenAlex Details")
        for i, doi in enumerate(dois):
            if not doi or doi is None or pd.isna(doi):
                continue
            tries = 0
            while tries < self.max_rety:
                
                # Format the DOI for the URL. Pass the raw DOI, not the full URL.
                url_doi = quote_plus(doi.replace("https://doi.org/", ""))
                work_url = f"{OPENALEX_BASE_URL}/works/doi:{url_doi}"
                try:

                    response = requests.get(work_url, params={'mailto': polite_email}, timeout=self.timeout)

                    if response.status_code in self.errors.keys():
                        self.errors[response.status_code].append(doi)
                        tqdm.write(f"--> Response Error: {response.status_code} for doi - {doi}")
                        break
                    data = response.json()

                    self._adapt_wait_time(data)
                    df_row = [doi]
                    for query in query_terms:
                        # query formatting logic 

                        query_result = data.get(query, self.NA_VALUE)
                        if query in self.format_funcs and query_result:
                            query_result = self.format_funcs[query](query_result)

                        df_row.append(query_result)

                    rows.append(df_row)
                    
                    tries = self.max_rety + 1
                    self._update_progress_bar()
                    time.sleep(self.polite_wait) # Be polite
                except requests.exceptions.Timeout:
                    tqdm.write("--> Search request timed out.")
                except requests.RequestException as e:
                    tries += 1
                    str_e = str(e)
                    if str_e not in self.errors.keys():
                        self.errors[str_e] = []
                    self.errors[str_e].append(doi)
                    time.sleep(self.fail_wait)

        self._close_progress_bar()
        logger.info(f"Found details for {len(rows)} papers on OpenAlex via DOI.")
        if np.any([len(val) for val in self.errors.values()]):
            logger.error(self.errors)

        return pd.DataFrame(rows, columns=colnames)

###################################
### --- Execution functions --- ###
###################################

def process_OpenAlex_from_OAids(scrape_config, oa_ids):
    setup_logger(logger, scrape_config["log"])

    CROSSREF_BASE_URL =scrape_config["CROSSREF_BASE_URL"]
    S2_BASE_URL = scrape_config["S2_BASE_URL"]
    S2_API_KEY = scrape_config["S2_API_KEY"]

    buffer_size = scrape_config["buffer_size"]
    start_year = scrape_config["start_year"]
    end_year = scrape_config["end_year"]
    output_dir = scrape_config["output_dir"]
    OA_query_terms = scrape_config["OpenAlex_queries"]
    
    OA = OpenAlexHandler(OPENALEX_BASE_URL = scrape_config["OPENALEX_BASE_URL"], 
                         polite_email = scrape_config["polite_email"], 
                         max_retry = scrape_config["max_retry"],
                         polite_wait = scrape_config["polite_wait"],
                         fail_wait = scrape_config["fail_wait"],
                         timeout = scrape_config["timeout_wait"],
                         )
    
    os.makedirs(output_dir, exist_ok=True)
    files = os.listdir(output_dir)
    n_files = len(files)
    temp_df = None
    batch_done = 0
    logger.info(f"Gathering details from OpenAlex for {len(oa_ids)} ids in {math.ceil(len(oa_ids) / buffer_size)} batches")
    ids_done = _index_scraped_ids(output_dir)
    logger.debug(f"Searching {output_dir} for existing scrape data")
    logger.debug(f"N. Ids before cut{len(oa_ids)}")
    oa_ids = [x for x in oa_ids if x not in ids_done["id_OpenAlex"]]
    logger.debug(f"N. Ids after cut{len(oa_ids)}")

    for batch_end_i in range(buffer_size, len(oa_ids), buffer_size):
        
        logger.info(f"Starting Batch {batch_done}")
        batch = OA.get_details_by_oaid(oa_ids[(batch_done*buffer_size):batch_end_i], OA_query_terms)
        batch.to_csv(output_dir / f"Batch{n_files+batch_done}.csv", index=False)
        batch_done += 1

    return True

if __name__ == "__main__":
    ### Testing ###

    dois = ["10.1587/TRANSINF.2014DAP0007", "10.6138/JIT.2015.16.3.20140918"]
    from ..utils.load_config import get_scrape_config
    scrape_config = get_scrape_config()
                    
    OPENALEX_BASE_URL = scrape_config["OPENALEX_BASE_URL"]
    CROSSREF_BASE_URL =scrape_config["CROSSREF_BASE_URL"]
    S2_BASE_URL = scrape_config["S2_BASE_URL"]
    S2_API_KEY = scrape_config["S2_API_KEY"]

    polite_email = scrape_config["polite_email"]
    buffer_size = scrape_config["buffer_size"]
    start_year = scrape_config["start_year"]
    end_year = scrape_config["end_year"]

    OA_query_terms = scrape_config["OpenAlex_queries"]

    OA = OpenAlexHandler(OPENALEX_BASE_URL, polite_email, polite_wait=0.1)
    # TODO: change the output dir here to a pipeline config option 

    for journal in scrape_config["journals"]:
        output_dir = scrape_config["output_dir"] / journal

        os.makedirs(output_dir, exist_ok=True)
        files = os.listdir(output_dir)
        n_files = len(files)
        temp_df = None
        batch_done = 0
        
        for year in np.linspace(start_year, end_year, np.abs(start_year - end_year) + 1, dtype=int):
            # IDs = OA.search_openalex(journal, year)
            try:
                DOIs = pd.read_csv(f"E:/Dropbox/4. Codebase/AcademicAbstracts/data/raw/{journal}/{year}.csv")["doi"].values
            except Exception as e:
                continue
            # OAIDs = [x["id_OpenAlex"] for x in IDs]
            # data = OA.get_details_by_oaid(OAIDs, OA_query_terms) 
            # TODO: add here a search for citation dois from S2

            data = OA.get_details_by_doi(DOIs, OA_query_terms)
            if temp_df is None:
                temp_df = data
            else:
                temp_df = pd.concat([temp_df, data], axis=0)
        
            if temp_df.shape[0] >= buffer_size or year == end_year:
                batch_done += 1
                temp_df.to_csv(output_dir / f"Batch{n_files+batch_done}.csv", index=False)
                temp_df = pd.DataFrame([], columns=temp_df.columns)
            
    

        