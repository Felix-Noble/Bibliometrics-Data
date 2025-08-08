from src.utils.load_config import get_pipeline_config, get_scrape_config
from src.utils.setup_logging import setup_logger

from src.scrape import process_OpenAlex_from_OAids
from src.scrape import search_OpenAlex_journal_year

from pathlib import Path
import pandas as pd
import sys
import ast
import numpy as np
import logging

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    journals_long =  ["American Chemical Society", "Science","PLOS ONE", "Scientific Reports","Nature Communications","Science Advances","Cell","The Lancet","The New England Journal of Medicine (NEJM)","Journal of the American Medical Association (JAMA)","The BMJ","Science Signaling","Nature Reviews Materials","eLife""Cell","Nature Medicine","Nature Biotechnology","Nature Genetics","Nature Methods","Journal of Biological Chemistry","The EMBO Journal","Journal of Cell Biology","Molecular Cell","Immunity","Cancer Cell","Neuron","Trends in Biochemical Sciences","Annual Review of Biochemistry","Current Biology","The Journal of Clinical Investigation","Genes & Development","PLOS Biology","Nature Structural & Molecular Biology","Molecular Biology and Evolution", "Frontiers in Microbiology","Journal of Neuroscience","Cell Metabolism","Nature Immunology","Cell Stem Cell""Journal of the American Chemical Society (JACS)","Angewandte Chemie","Chemical Reviews","Chemical Society Reviews","Nature Chemistry","Accounts of Chemical Research","Green Chemistry","Journal of Organic Chemistry","Organic Letters","Chemical Science","ACS Nano","Nano Letters","Journal of Physical Chemistry Letters","ACS Central Science","Nature Catalysis""Physical Review Letters","Nature Physics","Reviews of Modern Physics","The Astrophysical Journal","Nature Materials","Advanced Materials","Physical Review B","Physical Review D","Monthly Notices of the Royal Astronomical Society","Astronomy & Astrophysics","Nature Astronomy","Physical Review X","Nature Photonics""The Lancet","The New England Journal of Medicine (NEJM)","Journal of the American Medical Association (JAMA)","Nature Medicine","The BMJ","Annals of Internal Medicine","Journal of Clinical Oncology","Circulation","European Heart Journal","Gut","Hepatology","The Journal of Allergy and Clinical Immunology","American Journal of Respiratory and Critical Care Medicine","The Journal of Experimental Medicine","Cancer Research","Clinical Cancer Research","The Lancet Oncology","The Lancet Infectious Diseases","The Lancet Neurology","JAMA Internal Medicine","The Lancet Global Health","Annals of Oncology","Nature Reviews Drug Discovery""American Economic Review","Quarterly Journal of Economics","Econometrica","Journal of Political Economy","American Political Science Review","American Journal of Political Science","American Sociological Review","American Journal of Sociology","Journal of Personality and Social Psychology","Psychological Science","Psychological Bulletin","Psychological Review","Journal of Finance","Journal of Financial Economics","Administrative Science Quarterly","Academy of Management Journal","Academy of Management Review","Organization Science","Journal of Marketing","Journal of Marketing Research","Journal of Consumer Research","Management Science","Strategic Management Journal","Foreign Affairs","International Security","History of Political Economy","The American Historical Review","Past & Present","The Journal of Modern History","Critical Inquiry","PMLA (Publications of the Modern Language Association of America)","Representations","Signs: Journal of Women in Culture and Society","Journal of Economic Perspectives","Political Analysis","Mind","Ethics","The Philosophical Review""Nature Electronics","IEEE Transactions on Pattern Analysis and Machine Intelligence","IEEE Transactions on Industrial Electronics","IEEE Journal on Selected Areas in Communications","Communications of the ACM","Journal of Machine Learning Research","IEEE Transactions on Information Theory","IEEE Transactions on Automatic Control","IEEE Transactions on Power Electronics","Advanced Functional Materials","IEEE Transactions on Fuzzy Systems","IEEE Transactions on Communications","Journal of the ACM","Proceedings of the Conference on Neural Information Processing Systems (NeurIPS)",  "Proceedings of the International Conference on Machine Learning (ICML)" "Nature Geoscience", "Nature Climate Change","Earth-Science Reviews","Geophysical Research Letters","Journal of Climate","Water Research","Environmental Science & Technology","Global Change Biology","Ecology Letters","Annual Review of Earth and Planetary Sciences","Nature Energy","Energy & Environmental Science","The Cryosphere", "Frontiers in Ecology and the Environment"]
    pipeline_config = get_pipeline_config()
    scrape_config = get_scrape_config()
    pipeline_config["journals"] = journals_long
    scrape_output_dir = pipeline_config["scrape_output_dir"]
    # setup_logger(logger, scrape_config["log"])

    """start_year = scrape_config["start_year"]
    end_year = scrape_config["end_year"]
    for journal in pipeline_config["journals"]:
        scrape_config["journal"] = journal
        for year in np.linspace(start_year, end_year, abs(end_year-start_year)+1, dtype=np.int16):
            print(year)
            scrape_config["year"] = year
            search_OpenAlex_journal_year(scrape_config)
    
"""

    for journal in pipeline_config["journals"]:
        ids = pd.Series([])
        finished = [] 
        for file in list((scrape_output_dir / journal / "references").glob("*.csv"))[0:]:
            print(f"Processing file: {file}")
            file = pd.read_csv(file)
            finished += file["id_OpenAlex"].dropna().to_list()
            ref_lists = file["referenced_works_OpenAlex"].dropna().apply(ast.literal_eval)
            # TODO: add empty ref lists as None instead of [] so they are dropped with .dropna()
            ref_lists = [pd.Series(ls) for ls in ref_lists if len(ls) > 0]
            if len(ref_lists) > 1:
                ref_lists = pd.concat([pd.Series(ls) for ls in ref_lists if len(ls) > 0])
            
                ids = pd.concat([ids, ref_lists])
        ids = list(set(ids) - set(finished))
        scrape_config["output_dir"] = scrape_output_dir / journal / "references"

        process_OpenAlex_from_OAids(scrape_config, ids)

