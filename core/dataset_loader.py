"""
Download and manage HuggingFace datasets locally.
"""

import logging
from pathlib import Path

import pandas as pd
from datasets import load_dataset

import config

logger = logging.getLogger(__name__)


def download_dataset(
    repo_id: str,
    split: str = "train",
    limit: int | None = None,
) -> pd.DataFrame:
    """
    Download a dataset from HuggingFace Hub.

    Args:
        repo_id: HuggingFace dataset ID (e.g. 'WithinUsAI/claude_mythos_distilled_25k')
        split: Dataset split ('train', 'validation', 'test')
        limit: Optional row limit (e.g. 1000 for first 1000 rows). None = full dataset.

    Returns:
        pandas DataFrame with the dataset contents.
    """
    logger.info("Downloading '%s' (split=%s, limit=%s)...", repo_id, split, limit)

    ds = load_dataset(repo_id, split=split)

    if limit and limit > 0:
        ds = ds.select(range(min(limit, len(ds))))

    df = ds.to_pandas()
    logger.info("Downloaded %d rows, %d columns.", len(df), len(df.columns))
    return df


def save_dataset(df: pd.DataFrame, name: str) -> Path:
    """
    Save a DataFrame as CSV and Parquet under output/datasets/.

    Args:
        df: DataFrame to save.
        name: Dataset identifier (used for filename, e.g. 'claude_mythos_distilled_25k_train')

    Returns:
        Path to the saved CSV file.
    """
    safe_name = name.replace("/", "_").replace("\\", "_")
    csv_path = config.DATASETS_DIR / f"{safe_name}.csv"
    parquet_path = config.DATASETS_DIR / f"{safe_name}.parquet"

    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    df.to_parquet(parquet_path, index=False)
    logger.info("Dataset saved: %s (CSV), %s (Parquet)", csv_path, parquet_path)
    return csv_path


def load_local_dataset(path: Path | str) -> pd.DataFrame:
    """Load a previously saved dataset from disk (CSV or Parquet)."""
    path = Path(path)
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path, encoding="utf-8-sig")


def list_saved_datasets() -> list[dict]:
    """List all saved datasets under output/datasets/ with metadata."""
    results = []
    for f in sorted(config.DATASETS_DIR.iterdir()):
        if f.suffix == ".csv":
            try:
                df = pd.read_csv(f, nrows=5, encoding="utf-8-sig")
                total = sum(1 for _ in open(f, encoding="utf-8-sig")) - 1  # -1 for header
                results.append({
                    "path": str(f),
                    "name": f.stem,
                    "columns": list(df.columns),
                    "rows": total,
                })
            except Exception as e:
                logger.warning("Could not read %s: %s", f, e)
    return results
