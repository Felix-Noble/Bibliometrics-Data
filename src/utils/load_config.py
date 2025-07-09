from pathlib import Path
from functools import lru_cache
import os

#########################
### --- Constants --- ###
#########################

## Colours ##
COL_BLUE = '\033[94m'  # Light blue
COL_DEFAULT = '\033[0m'  # Reset to default color

## Type Dicts ##
general_type_dict = {"log": {
                            "file": str,
                            "console": str
                            }
                    }

#########################
### --- Helpers --- ###
#########################

def find_project_root(marker_file_name: str = "pyproject.toml") -> Path:
    """
    Traverses up the directory tree from the current script's location
    to find the project root, identified by the presence of a marker file.

    Args:
        marker_file_name: The name of the file that marks the project root
                          (e.g., 'project.toml', 'pyproject.toml', '.git').

    Returns:
        A pathlib.Path object representing the project root directory.

    Raises:
        FileNotFoundError: If the marker file cannot be found by traversing up.
    """
    current_dir = Path(__file__).resolve().parent
    for parent in [current_dir, *current_dir.parents]:
        if (parent / marker_file_name).exists():
            return parent
    raise FileNotFoundError(f"Project root marker '{marker_file_name}' not found "
                            f"in {current_dir} or any parent directories.")

def type_check(dict_to_check, type_dict):
   # TODO: add auto conversion (i.e from int to float where possible)

    for (key, value) in type_dict.items():
        if isinstance(type_dict[key], type): # check for type instanced 
            if not isinstance(dict_to_check[key], value):
                return f"Expected type {value} for {key}, got {type(dict_to_check[key])}"
        elif isinstance(type_dict[key], tuple): # check for array-like (0) filled with type (1)
            if not isinstance(dict_to_check[key], type_dict[key][0]) and all(isinstance(item,  type_dict[key][1]) for item in dict_to_check[key]):
                return f"Expected type {type_dict[key][0]} filled with {type_dict[key][1]} for {key}, got {type(dict_to_check[key])} filled {[type(x) for x in dict_to_check[key]]}"
        elif isinstance(type_dict[key], dict): # recursively process nested dictionaries 
            type_check(dict_to_check[key], type_dict[key])
        else:
            raise ValueError(f"Unexpected type contained in type dict:  {type_dict}")
        
    return False

def dict_key_check(dict_to_check, dict_to_compare):
    for (key, value) in dict_to_compare.items():
        if not isinstance(value, dict):
            if key not in dict_to_check.keys():
                return key
        else:
            dict_key_check(dict_to_check[key], value)
    return False

def key_check(step_config, type_dict):
    missing_in_config = dict_key_check(step_config, type_dict) # check that all type dict keys are in config keys 
    missing_in_type = dict_key_check(type_dict, step_config) # check that all config keys are in the type dict

    if missing_in_config:
        return f"Expected {missing_in_config} to be in config keys. {COL_BLUE}Check config.toml{COL_DEFAULT}"
    if missing_in_type:
        return f"Expected {missing_in_type} to be in type_dict keys. {COL_BLUE}Update type_dict in config.py{COL_DEFAULT}"

def config_init_check(config, type_dict, step_name):
    key_error = key_check(config, type_dict)
    if key_error:
        raise ValueError(f"{step_name} | {key_error}")
    type_error = type_check(config, type_dict)
    if type_error:
        raise TypeError(f"{step_name} | {type_error}")
    
############################
### --- Load configs --- ###
############################

@lru_cache()
def _load_config(path: Path = None) -> dict:
    """
    Loads and caches the TOML configuration from the specified path.
    """
    if path is None:
        path = find_project_root() / "config" / "config.toml"
    
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    try:
        import toml
        return toml.load(path)
    except ImportError:
        import tomli
        with open(path, "rb") as f:
            return tomli.load(f)

def get_pipeline_config():
    cfg = _load_config()
    
    step_name = cfg["pipeline"]["names"]["pipeline"]
    pipeline_config = cfg[step_name]
    type_dict = {"journals": (list, str),
                 "names": {"pipeline": str,
                           "scrape": str,
                           "preprocess": str,
                           "feature_extraction": str}}
    # Add univesally applicable type dicts to specific dict 
    type_dict = type_dict | general_type_dict
    config_init_check(pipeline_config, type_dict, step_name)

    return pipeline_config

def get_scrape_config():
    cfg = _load_config()
    step_name = cfg["pipeline"]["names"]["scrape"]
    scrape_config = cfg["scrape"]
    type_dict = {"polite_email": str,
                 "OPENALEX_BASE_URL": str,
                 "CROSSREF_BASE_URL": str,
                 "S2_BASE_URL": str,
                 "S2_API_KEY": str,
                 "output_dir": str,
                 "buffer_size": int,
                 "max_retry": int,
                 "polite_wait": float,
                 "fail_wait": float,
                 "rate_limit_wait" : float,
                 "timeout_wait": float,
                 "start_year": int,
                 "end_year": int,
                 "S2_queries" : (list, str),
                 "OpenAlex_queries": (list, str)
                 }

    type_dict = type_dict | general_type_dict
    config_init_check(scrape_config, type_dict, step_name)

    # Critical Checks # 
    if scrape_config["start_year"] - scrape_config["end_year"] == 0:
        raise ValueError(f'No difference between start_year : {scrape_config["start_year"]} and end_year : {scrape_config["end_year"]} check config')
    
    # Type alterations #
    for key in ["output_dir"]:
        scrape_config[key] = Path(scrape_config[key])

    return cfg["scrape"]

def get_preprocess_config():
    cfg = _load_config()
    step_name = cfg["pipeline"]["names"]["preprocess"]
    prep_config = cfg[step_name]

    # Type checks 
    type_dict = {"input_dir": str,
                 "output_dir": str,
                 "problem_chars": list,
                 "chunk_size": int,
                 "buffer_size": int,
                 "data_cols" : list,
                 "cols_to_clean": list}
    # Add univesally applicable type dicts to specific dict 
    type_dict = type_dict | general_type_dict
    config_init_check(prep_config, type_dict, step_name)

    # Type of contents check
    for array_name in ["problem_chars", "data_cols", "cols_to_clean"]:
        for (i,char) in enumerate(prep_config[array_name]):
            if not isinstance(char, str):
                raise TypeError(f'Expected string contents for {array_name} array, got {type(char)} at index {i}')
    
    # correct contents checks
    for colname in prep_config["cols_to_clean"]:
        if colname not in prep_config["data_cols"]:
            raise ValueError(f'Clean column {colname} not in data_cols : {prep_config["data_cols"]}')

    # Exists checks 
    if not os.path.exists(prep_config["input_dir"]):
            raise FileNotFoundError(f'Input dir does not exist: {prep_config["input_data"]}')

    # Return filepaths as Path types
    for key in ["input_dir", "output_dir"]:
        prep_config[key] = Path(prep_config[key])
    
    return prep_config

def get_extract_config():
    cfg = _load_config()
    extraction_config = cfg["data"]["feature_extraction"]
    step_name = cfg["pipeline"]["names"]["feature_extraction"]

    type_dict = {"input_file": str,
                 "output_dir": str,
                 "attributes": list,
                 "buffer_size": int,
                 "report_interval": int,
                 "HuggingFace_model_name": str
                 }
    type_dict = type_dict | general_type_dict
    config_init_check(extraction_config, type_dict, step_name)

    # Return filepaths as Path types
    for key in ["input_file", "output_dir"]:
        extraction_config[key] = Path(extraction_config[key])
    return extraction_config