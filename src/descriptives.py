#src/descriptives.py
import os
import duckdb
import matplotlib.pyplot  as plt
from utils.load_config import get_pipeline_config
from utils.setup_logging import setup_logger
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    data_config = get_pipeline_config()
    #client = Client(LocalCluster())
    #print(client.dashboard_link)
    journal = "ACS"

    scatter_cols = ["citation_count_OpenAlex", "median_citation_count_year",
                     "higher_than_median_year",
                    ]
    hist_cols = scatter_cols 
    all_cols = hist_cols + ["abstract_OpenAlex"]
    print("Script started")
    db_dir = data_config["features_db_loc"] /journal
    db_files = [os.path.join(db_dir,x) for x in os.listdir(os.path.expanduser(db_dir)) if "parq" in x]
    #ddf = ddf.dropna(subset=["abstract_OpenAlex"])
    embedding_cols= [f"embedding_{x}" for x in range(384)]

    for col in hist_cols:
        print(f"Starting hist for {col}")
        data = duckdb.sql(f"""
                        SELECT {col}
                        FROM read_parquet({db_files})

                          """).df()
        plt.figure()
        try:
            plt.hist(data, bins=200, color='skyblue', edgecolor='black')
            plt.title(f'Histogram of {col} | N = {len(data)}')
            plt.xlabel(col)
            plt.ylabel('Frequency')
            plt.grid(True, linestyle='--', alpha=0.6)
            # plt.xlim((min(data)*1.02, max(data)*(1.02)))
            plt.savefig(Path(__file__).parent.parent / "figures" / f"hist-{col}.png")
            print(f"Saved hist for {col}")

        except Exception as e:
            print(f"Error for {col} | {e}")

        plt.close()
        del data

    for col in scatter_cols:
        print(f"Starting scatter for {col}")
        data = duckdb.sql(f"""
                        SELECT {col}, publication_year
                        FROM read_parquet({db_files})

                          """).df()      
        
        plt.figure()
        try:
            plt.scatter(
            data["publication_year"], 
            data[col], 
            alpha=0.5, 
            s=10 
                )
            plt.title(f'Scatter of {col} vs publication year| N = {len(data)}')
            plt.xlabel('publication_year')
            plt.ylabel(col)
            plt.grid(True, linestyle='--', alpha=0.6)
            # plt.ylim((min(data[col])*1.02, max(data[col])*(2.02)))
            plt.savefig(Path(__file__).parent.parent / "figures" / f"scatter-{col} vs publication year.png")
            print(f"Saved scatter for {col}")

        except Exception as e:
            print(f"Error for {col} | {e}")

        plt.close()

    
