## Bibliometrics Data ##

## Aim 
Collect bibliometrics data from open source databases, (e.g. OpenAlex, CrossRef, Semantic Scholar), organise and compare results across sources, export heirarchichal dash databases for each journal.

## Use
All execution is handled by main.py, which contains all pipeline level logic (directory assignment, order of execution):

# Config
All modules are controled through the config/config.toml file

All config entries are checked for types via internal type dicts in config.py. The load_config util will raise Value errors if a config setting is loaded that no type entry is available for. 
# Scrape
The scrape module is responseible for sourcing all information from each source, it dumps the stored data into batch{n}.csv files. 

# Export (upcoming)
This module will export the dumped csv files into dash databases (DB). Journal specific entries will be stored in a DB indexed by date. The references/citations of these entries will be indexed by doi in a seperate DB

