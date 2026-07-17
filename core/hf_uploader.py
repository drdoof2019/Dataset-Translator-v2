"""
HuggingFace Hub upload utilities.
"""

import logging
from pathlib import Path

import pandas as pd
from huggingface_hub import HfApi, login
from datasets import Dataset

import config

logger = logging.getLogger(__name__)


def test_token(token: str | None = None) -> tuple[bool, str]:
    """Validate a HuggingFace token. Returns (success, username_or_error)."""
    token = token or config.HF_TOKEN
    if not token:
        return False, "No HF token configured."
    try:
        api = HfApi(token=token)
        user_info = api.whoami()
        return True, user_info.get("name", "unknown")
    except Exception as e:
        return False, str(e)


def upload_dataset(
    csv_path: str | Path,
    repo_name: str,
    hf_token: str | None = None,
    private: bool = False,
) -> tuple[bool, str]:
    """
    Upload a CSV file as a HuggingFace dataset.

    Args:
        csv_path: Path to the translated CSV file.
        repo_name: Repository name (e.g. 'my-translated-dataset').
                   Will be prefixed with the authenticated user's name.
        hf_token: HF write token (falls back to config).
        private: Whether to create a private repo.

    Returns:
        (success, message_or_url)
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        return False, f"File not found: {csv_path}"

    token = hf_token or config.HF_TOKEN
    if not token:
        return False, "No HuggingFace token provided."

    try:
        login(token=token)
        api = HfApi(token=token)
        user_info = api.whoami()
        username = user_info["name"]
        repo_id = f"{username}/{repo_name}"

        logger.info("Loading CSV into HF Dataset: %s", csv_path)
        hf_dataset = Dataset.from_csv(str(csv_path), encoding="utf-8-sig")

        logger.info("Pushing to Hub: %s (private=%s)", repo_id, private)
        hf_dataset.push_to_hub(repo_id, private=private)

        url = f"https://huggingface.co/datasets/{repo_id}"
        logger.info("Upload successful: %s", url)
        return True, url

    except Exception as e:
        logger.error("Upload failed: %s", e)
        return False, str(e)


def upload_readme(
    readme_path: str | Path,
    repo_id: str,
    hf_token: str | None = None,
) -> tuple[bool, str]:
    """Upload a README.md file to an existing HF repo."""
    readme_path = Path(readme_path)
    if not readme_path.exists():
        return False, f"README not found: {readme_path}"

    token = hf_token or config.HF_TOKEN
    try:
        api = HfApi(token=token)
        api.upload_file(
            path_or_fileobj=str(readme_path),
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="dataset",
        )
        return True, "README.md uploaded successfully."
    except Exception as e:
        return False, str(e)


def generate_readme(
    df: pd.DataFrame,
    repo_id: str,
    source_dataset: str,
    language: str = "tr",
) -> str:
    """Generate a basic README.md content for the translated dataset."""
    # Column analysis table
    col_analysis = "| Column | Type | Non-Null | Unique |\n"
    col_analysis += "|---|---|---|---|\n"
    for col in df.columns:
        col_analysis += f"| {col} | {df[col].dtype} | {df[col].count()} | {df[col].nunique()} |\n"

    sample = df.head(3).to_markdown(index=False)

    return f"""---
language: {language}
license: mit
---

# Dataset Card for {repo_id}

## Dataset Description

This dataset is the **{language}** translation of [{source_dataset}](https://huggingface.co/datasets/{source_dataset}).
Translated using a local LLM via LM Studio.

## Dataset Structure

### Data Fields

{col_analysis}

### Data Sample

{sample}

## Usage

```python
from datasets import load_dataset

dataset = load_dataset("{repo_id}")
print(dataset)
```

## Source

Original dataset: [{source_dataset}](https://huggingface.co/datasets/{source_dataset})
"""
