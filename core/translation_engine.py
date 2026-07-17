"""
Cell-by-cell translation engine with checkpoint/resume support.
"""

import logging
from pathlib import Path
from threading import Event

import pandas as pd

import config
from translator import Translator
from core.progress_manager import ProgressManager

logger = logging.getLogger(__name__)


def _output_path(dataset_id: str, split: str) -> Path:
    safe = dataset_id.replace("/", "_").replace("\\", "_")
    return config.TRANSLATED_DIR / f"{safe}_{split}_translated.csv"


def _translate_value(
    translator: Translator,
    value,
    json_mode: str,
    source_lang: str,
    target_lang: str,
    translate_keys: list[str] | None,
    skip_keys: list[str] | None,
    max_tokens: int = 4096,
    on_leaf_translate=None,
) -> str:
    """
    Translate a single cell value, routing to translate_json() when appropriate.

    json_mode:
        "auto"   — detect JSON structure, use translate_json if found
        "force"  — always attempt translate_json (falls back to plain if not JSON)
        "off"    — always use plain translate_cell
    """
    text = str(value) if pd.notna(value) else ""
    if not text.strip():
        return text

    if json_mode == "off":
        return translator.translate_cell(text, source_lang, target_lang, max_tokens=max_tokens,
                                          on_leaf_translate=on_leaf_translate)

    if json_mode == "force":
        return translator.translate_json(text, translate_keys, skip_keys, source_lang, target_lang,
                                         max_tokens=max_tokens, on_leaf_translate=on_leaf_translate)

    # auto mode
    if Translator.is_json_cell(text):
        return translator.translate_json(text, translate_keys, skip_keys, source_lang, target_lang,
                                         max_tokens=max_tokens, on_leaf_translate=on_leaf_translate)
    return translator.translate_cell(text, source_lang, target_lang, max_tokens=max_tokens,
                                     on_leaf_translate=on_leaf_translate)


def translate_dataset(
    df: pd.DataFrame,
    dataset_id: str,
    split: str,
    columns_to_translate: list[str],
    source_lang: str = "en",
    target_lang: str = "tr",
    translator: Translator | None = None,
    start_row: int | None = None,
    stop_event: Event | None = None,
    on_progress=None,
    json_mode: str = "auto",
    translate_keys: list[str] | None = None,
    skip_keys: list[str] | None = None,
    max_tokens: int = 4096,
    on_leaf_translate=None,
) -> Path:
    """
    Translate selected columns of a DataFrame cell-by-cell.

    Saves progress after EVERY ROW (crash-safe). Supports resume via ProgressManager.

    Args:
        df: Source DataFrame (full dataset, not filtered).
        dataset_id: HuggingFace dataset identifier (for state key).
        split: Split name (for state key & output filename).
        columns_to_translate: List of column names to translate.
        source_lang: Source language code.
        target_lang: Target language code.
        translator: Translator instance (created from config if None).
        start_row: Override start row (default: resume from checkpoint).
        stop_event: threading.Event — when set, stops after current row.
        on_progress: Callback(current_row, total_rows, current_cell) for live updates.
        json_mode: "auto" | "force" | "off" — how to handle JSON-structured cells.
        translate_keys: JSON keys whose values should be translated (default: config).
        skip_keys: JSON keys to never touch (default: config).

    Returns:
        Path to the translated CSV output file.
    """
    if translator is None:
        translator = Translator()

    total = len(df)
    pm = ProgressManager(dataset_id, split, columns_to_translate, total)
    out_path = _output_path(dataset_id, split)

    # Determine start row
    if start_row is not None:
        effective_start = start_row
    else:
        effective_start = pm.get_resume_row()

    is_first_write = (effective_start == 0)

    # Write header on first run
    if is_first_write:
        df.head(0).to_csv(out_path, index=False, mode="w", header=True, encoding="utf-8-sig")

    logger.info("Starting translation: rows %d-%d, columns %s (json_mode=%s)",
                effective_start, total - 1, columns_to_translate, json_mode)

    for row_idx in range(effective_start, total):
        # Check stop signal BEFORE processing this row
        if stop_event and stop_event.is_set():
            logger.info("Stop requested. Halting at row %d.", row_idx)
            break

        row = df.iloc[row_idx].copy()

        for col in columns_to_translate:
            if stop_event and stop_event.is_set():
                break

            cell_value = row[col]
            if on_progress:
                on_progress(row_idx, total, f"Row {row_idx}, Column: {col}")

            translated = _translate_value(
                translator, cell_value, json_mode, source_lang, target_lang,
                translate_keys, skip_keys, max_tokens,
                on_leaf_translate=on_leaf_translate,
            )
            row[col] = translated

            if on_progress:
                on_progress(row_idx, total, f"Row {row_idx}, Column: {col}")

        # Write the entire translated row to CSV (append, no header)
        row_df = row.to_frame().T
        row_df.to_csv(out_path, index=False, mode="a", header=False, encoding="utf-8-sig")

        # Update checkpoint
        pm.save_progress(row_idx)

        if on_progress:
            on_progress(row_idx + 1, total, f"Row {row_idx} done ({row_idx + 1}/{total})")

    logger.info("Translation finished. Output: %s", out_path)
    return out_path


def get_output_path(dataset_id: str, split: str) -> Path:
    """Return the expected output path for a given dataset/split."""
    return _output_path(dataset_id, split)


def read_translated_csv(path: Path | str, nrows: int = 5) -> pd.DataFrame:
    """Read and preview the translated CSV."""
    return pd.read_csv(path, nrows=nrows, encoding="utf-8-sig")
