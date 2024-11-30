from glob import glob

import pandas as pd


def concat_csvs(file_glob: str) -> pd.DataFrame:
    """Combine all CSVs matching the glob pattern into one DataFrame"""
    files = glob(file_glob)
    if not files:
        raise FileNotFoundError(f"No files found matching {file_glob}")
    return pd.concat([pd.read_csv(file) for file in files], ignore_index=True)
