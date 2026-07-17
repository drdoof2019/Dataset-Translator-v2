"""
Dataset Translator — Gradio Web App

Translates HuggingFace datasets using a local LLM via LM Studio.
Supports resume, progress tracking, and HuggingFace upload.
"""

import asyncio
import json
import logging
import sys
import threading
from pathlib import Path

import gradio as gr
import pandas as pd
from datasets import get_dataset_split_names

import config
from translator import Translator
from core.dataset_loader import download_dataset, save_dataset
from core.progress_manager import ProgressManager, list_pending_jobs
from core.translation_engine import translate_dataset, get_output_path
from core.hf_uploader import test_token, upload_dataset, generate_readme

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ── Shared translation state (module-level, single-user local app) ──
_trans: dict = {
    "running": False,
    "current_row": 0,
    "total_rows": 0,
    "status": "Idle",
    "current_cell": "",
    "output_path": None,
    "error": None,
    "stop_event": threading.Event(),
    "current_source_text": "",
    "current_translated_text": "",
}

# ── Global reset helper ──

def global_reset_fn():
    """Delete all state, translated, and dataset files for a full clean slate."""
    import shutil
    for d in [config.STATE_DIR, config.TRANSLATED_DIR, config.DATASETS_DIR]:
        if d.exists():
            for f in d.iterdir():
                if f.is_file() and f.name != ".gitkeep":
                    f.unlink()
                    logger.info("Deleted: %s", f)
    # Reset _trans completely
    _trans["running"] = False
    _trans["current_row"] = 0
    _trans["total_rows"] = 0
    _trans["status"] = "Idle"
    _trans["current_cell"] = ""
    _trans["output_path"] = None
    _trans["error"] = None
    _trans["stop_event"].clear()
    _trans["current_source_text"] = ""
    _trans["current_translated_text"] = ""
    logger.info("Global reset complete.")


# ═══════════════════════════════════════════════════════════════════════
# Event handlers
# ═══════════════════════════════════════════════════════════════════════

def fetch_splits_fn(dataset_id):
    """Fetch available splits for a HuggingFace dataset."""
    if not dataset_id or not dataset_id.strip():
        return gr.Dropdown(choices=["train"], value="train"), "Enter a dataset ID first."
    try:
        splits = get_dataset_split_names(dataset_id.strip(), token=config.HF_TOKEN or None)
        if not splits:
            splits = ["train"]
        default = splits[0] if "train" not in splits else "train"
        return gr.Dropdown(choices=splits, value=default), f"✅ Found splits: {', '.join(splits)}"
    except Exception as e:
        logger.warning("Could not fetch splits for '%s': %s", dataset_id, e)
        return gr.Dropdown(choices=["train"], value="train"), f"⚠️ Could not fetch splits (using 'train'): {e}"


def on_cell_inspector(evt: gr.SelectData):
    """Pretty-print a clicked cell value using gr.JSON."""
    try:
        val = evt.value
        if val is None:
            return gr.JSON(value={"info": "Empty cell"}, label="Cell Content")
        text = str(val)
        try:
            parsed = json.loads(text)
            # gr.JSON accepts dicts/lists directly — renders as interactive tree
            return gr.JSON(value=parsed, label="Cell Content (JSON)")
        except (json.JSONDecodeError, TypeError):
            # Plain text — wrap in a dict so gr.JSON can render it
            return gr.JSON(value={"text": text}, label="Cell Content (Plain Text)")
    except Exception:
        return gr.JSON(value={"error": str(evt.value) if evt else "unknown"}, label="Cell Content")


def analyze_columns_fn(dataset_path):
    """Analyze columns for JSON structure and show translate/skip key info."""
    if not dataset_path or not Path(dataset_path).exists():
        return "⚠️ No dataset loaded."
    try:
        df = pd.read_csv(dataset_path, nrows=5, encoding="utf-8-sig")
        translate_set = set(config.JSON_TRANSLATE_KEYS)
        skip_set = set(config.JSON_SKIP_KEYS)

        text_cols = df.select_dtypes(include=["object"]).columns.tolist()
        if not text_cols:
            return "No text columns detected."

        sections = []
        for col in text_cols:
            json_count = 0
            all_keys: set = set()
            for val in df[col]:
                if pd.isna(val):
                    continue
                try:
                    parsed = json.loads(str(val))
                    if isinstance(parsed, (dict, list)):
                        json_count += 1
                        if isinstance(parsed, list):
                            for item in parsed:
                                if isinstance(item, dict):
                                    all_keys.update(item.keys())
                        elif isinstance(parsed, dict):
                            all_keys.update(parsed.keys())
                except (json.JSONDecodeError, TypeError):
                    pass

            if json_count == 0:
                sections.append(f"**`{col}`** — plain text")
            else:
                to_translate = sorted(all_keys & translate_set)
                to_skip = sorted(all_keys & skip_set)
                unknown = sorted(all_keys - translate_set - skip_set)

                parts = [f"**`{col}`** — JSON detected ({json_count}/{len(df)} rows)"]
                if to_translate:
                    parts.append(f"  🟢 Translate: `{', '.join(to_translate)}`")
                if to_skip:
                    parts.append(f"  🔴 Skip: `{', '.join(to_skip)}`")
                if unknown:
                    parts.append(f"  ⚪ Untouched (not in any list): `{', '.join(unknown)}`")
                sections.append("\n".join(parts))

        return "\n\n".join(sections)
    except Exception as e:
        return f"❌ Analysis error: {e}"


def update_col_counter(selected):
    """Update the column selection counter text."""
    if not selected:
        return "📌 No columns selected"
    return f"📌 **{len(selected)}** column(s) selected: {', '.join(selected)}"


def test_lm_fn(lm_url, lm_model, system_prompt, message_template):
    """Test LM Studio connection."""
    t = Translator(base_url=lm_url, model=lm_model,
                   system_prompt=system_prompt, message_template=message_template)
    ok, msg = t.test_connection()
    return f"✅ {msg}" if ok else f"❌ {msg}"


def test_hf_fn(hf_token):
    """Test HuggingFace token."""
    ok, msg = test_token(hf_token)
    return f"✅ Logged in as: {msg}" if ok else f"❌ {msg}"


def download_fn(dataset_id, split, limit, confirm=False):
    """Download a HuggingFace dataset and save locally.

    If there are pending/in-progress translations and confirm=False,
    returns a warning requiring a second click. On confirm=True (or no
    pending jobs), performs global reset then downloads.
    """
    empty = "⚠️ No dataset loaded."
    _blank8 = (None, gr.CheckboxGroup(choices=[]), None, "", "", empty, "📌 No columns selected", empty)

    if not dataset_id or not dataset_id.strip():
        return ("❌ Please enter a dataset ID.", *_blank8, False)

    # Check for pending/in-progress jobs
    if not confirm:
        has_jobs = _trans["running"]
        if not has_jobs and config.STATE_DIR.exists():
            has_jobs = any(config.STATE_DIR.glob("*.json"))
        if has_jobs:
            return (
                "⚠️ Pending translation will be lost! Click Download again to confirm.",
                *_blank8, True  # set confirm flag
            )

    dataset_id = dataset_id.strip()
    split = split or "train"
    # Treat 0 and None as "download all"
    row_limit = int(limit) if limit and int(limit) > 0 else None

    # Global reset before downloading new dataset
    if _trans["running"]:
        _trans["stop_event"].set()
    global_reset_fn()

    try:
        safe_name = f"{dataset_id}_{split}".replace("/", "_")
        local_csv = config.DATASETS_DIR / f"{safe_name}.csv"

        if local_csv.exists():
            df = pd.read_csv(local_csv, encoding="utf-8-sig")
            cached_rows = len(df)
            if row_limit and cached_rows > row_limit:
                df = df.head(row_limit)
                save_dataset(df, safe_name)  # overwrite cache so translation reads correct row count
                status = f"📂 Cache had {cached_rows} rows — re-sliced to {len(df)}"
            else:
                status = f"📂 Loaded from cache: {local_csv.name} ({len(df)} rows)"
        else:
            df = download_dataset(dataset_id, split, row_limit)
            save_dataset(df, f"{dataset_id}_{split}")
            status = f"⬇️ Downloaded & saved ({len(df)} rows)"

        preview = df.head(10)
        columns = list(df.columns)
        col_analysis = analyze_columns_fn(str(local_csv))
        return (
            status,
            preview,
            gr.CheckboxGroup(choices=columns, value=[]),
            str(local_csv),
            dataset_id,
            split,
            col_analysis,
            "📌 No columns selected",
            col_analysis,
            False,  # reset confirm flag
        )
    except Exception as e:
        logger.exception("Download failed")
        err = f"⚠️ Download failed: {e}"
        return (f"❌ Error: {e}", None,
                gr.CheckboxGroup(choices=[]), None, "", "",
                err, "📌 No columns selected", err, False)


def select_text_cols_fn(dataset_path):
    """Auto-select text (object) columns."""
    if not dataset_path:
        return gr.CheckboxGroup(choices=[])
    try:
        df = pd.read_csv(dataset_path, nrows=5, encoding="utf-8-sig")
        text_cols = df.select_dtypes(include=["object"]).columns.tolist()
        return gr.CheckboxGroup(choices=list(df.columns), value=text_cols)
    except Exception:
        return gr.CheckboxGroup(choices=[])


def update_info_fn(dataset_path, dataset_id, split, columns, src_lang, tgt_lang):
    """Build a summary string for the Translate tab."""
    if not dataset_path or not Path(dataset_path).exists():
        return "⚠️ No dataset loaded. Go to the Download tab first."
    if not columns:
        return "⚠️ No columns selected. Go to the Download tab to select columns."

    try:
        total = len(pd.read_csv(dataset_path, encoding="utf-8-sig"))
        info = (
            f"📊 Dataset: {dataset_id}  |  Split: {split}  |  Rows: {total}\n"
            f"📝 Columns: {', '.join(columns)}\n"
            f"🌐 Language: {src_lang} → {tgt_lang}"
        )
        pm = ProgressManager(dataset_id, split, columns, total)
        if pm.has_pending():
            info += f"\n⏸️ Pending job: {pm.completed_rows}/{total} rows done — will resume from row {pm.completed_rows}"
        return info
    except Exception as e:
        return f"Error: {e}"


def start_translate_fn(dataset_path, dataset_id, split, columns, src_lang, tgt_lang,
                       lm_url, lm_model, json_mode, translate_keys_str, skip_keys_str,
                       system_prompt="", message_template="", max_tokens=4096):
    """Start (or resume) translation in a background thread."""
    if not dataset_path or not Path(dataset_path).exists():
        return "❌ No dataset loaded."
    if not columns:
        return "❌ No columns selected."
    if _trans["running"]:
        return "⚠️ Translation already running."

    _trans["stop_event"].clear()
    _trans["running"] = True
    _trans["error"] = None
    _trans["current_row"] = 0
    _trans["current_cell"] = ""
    _trans["output_path"] = None
    _trans["current_source_text"] = ""
    _trans["current_translated_text"] = ""

    def _run():
        try:
            translator = Translator(base_url=lm_url, model=lm_model,
                                    system_prompt=system_prompt, message_template=message_template)

            # Pre-flight: verify model is loaded and can translate
            _trans["status"] = "Checking LM Studio model..."
            ok, msg = translator.pre_flight_check()
            if not ok:
                _trans["status"] = f"❌ Model check failed: {msg}"
                _trans["error"] = msg
                _trans["running"] = False
                return

            df = pd.read_csv(dataset_path, encoding="utf-8-sig")
            _trans["total_rows"] = len(df)
            _trans["status"] = "Running..."

            def on_progress(current, total, cell):
                _trans["current_row"] = current
                _trans["total_rows"] = total
                _trans["current_cell"] = cell

            def on_leaf_translate(source_text, translated_text):
                _trans["current_source_text"] = source_text
                _trans["current_translated_text"] = translated_text

            # Parse JSON keys from comma-separated strings
            t_keys = [k.strip() for k in translate_keys_str.split(",") if k.strip()] or None
            s_keys = [k.strip() for k in skip_keys_str.split(",") if k.strip()] or None

            output = translate_dataset(
                df=df,
                dataset_id=dataset_id,
                split=split,
                columns_to_translate=columns,
                source_lang=src_lang,
                target_lang=tgt_lang,
                translator=translator,
                stop_event=_trans["stop_event"],
                on_progress=on_progress,
                json_mode=json_mode,
                translate_keys=t_keys,
                skip_keys=s_keys,
                max_tokens=max_tokens,
                on_leaf_translate=on_leaf_translate,
            )
            _trans["output_path"] = str(output)
            _trans["status"] = "Completed!"
        except Exception as e:
            logger.exception("Translation error")
            _trans["error"] = str(e)
            _trans["status"] = f"Error: {e}"
        finally:
            _trans["running"] = False

    threading.Thread(target=_run, daemon=True).start()
    return "▶️ Translation started..."


def stop_translate_fn():
    """Signal the translation thread to stop after the current row."""
    if not _trans["running"]:
        return "Nothing is running."
    _trans["stop_event"].set()
    return "⏹️ Stop requested. Finishing current row..."


def clear_progress_fn(dataset_id, split, columns, dataset_path):
    """Full reset: delete state, translated CSV, dataset CSV, reset _trans."""
    _no = gr.update()  # shorthand: leave component unchanged
    _blank_8 = (_no, _no, _no, _no, _no, _no, _no, _no)

    if not dataset_id:
        return ("No dataset selected.", *_blank_8)

    # If translation is running, auto-stop and warn
    if _trans["running"]:
        _trans["stop_event"].set()
        return ("⚠️ Translation is stopping... Click Clear again once it halts.", *_blank_8)

    try:
        # 1. Delete progress state JSON
        pm = ProgressManager(dataset_id, split or "train", columns or [], 0)
        pm.clear()

        # 2. Delete translated CSV
        safe_name = f"{dataset_id}_{split}".replace("/", "_")
        translated_csv = config.TRANSLATED_DIR / f"{safe_name}_translated.csv"
        if translated_csv.exists():
            translated_csv.unlink()
            logger.info("Deleted translated CSV: %s", translated_csv)

        # 3. Delete dataset CSV + parquet
        if dataset_path:
            for suffix in (".csv", ".parquet"):
                p = Path(dataset_path).with_suffix(suffix)
                if p.exists():
                    p.unlink()
                    logger.info("Deleted dataset file: %s", p)

        # 4. Reset _trans completely
        _trans["running"] = False
        _trans["current_row"] = 0
        _trans["total_rows"] = 0
        _trans["status"] = "Idle"
        _trans["current_cell"] = ""
        _trans["output_path"] = None
        _trans["error"] = None
        _trans["stop_event"].clear()
        _trans["current_source_text"] = ""
        _trans["current_translated_text"] = ""

        # Returns: status, translate_info, status_box, progress_bar,
        #          output_path, upload_csv_path, dataset_path_s, download_status
        return ("🗑️ All cleared: progress, translated file, and dataset removed.",
                "", "Idle", 0, "", "", None, "")
    except Exception as e:
        logger.exception("Clear progress error")
        return (f"Error during clear: {e}", *_blank_8)


def poll_progress():
    """Called by gr.Timer every 2 s to update the Translate tab UI.

    Returns (status, progress, output_path, upload_csv_path,
             start_btn, stop_btn, clear_btn, source_preview, translated_preview).
    """
    _u = gr.update  # shorthand
    running = _trans["running"]

    if running:
        row = _trans["current_row"]
        total = _trans["total_rows"]
        pct = row / total if total > 0 else 0
        status = f"🔄 Row {row}/{total} ({pct:.0%}) — {_trans['current_cell']}"
        return (status, pct, _u(), _u(),
                _u(visible=False), _u(visible=True), _u(interactive=False),
                _trans.get("current_source_text", ""),
                _trans.get("current_translated_text", ""))

    status = _trans["status"]
    total = _trans["total_rows"]
    row = _trans["current_row"]

    if _trans.get("error"):
        return (f"❌ {status}", (row / total if total else 0), "", _u(),
                _u(visible=True), _u(visible=False), _u(interactive=True), "", "")

    if status == "Completed!":
        out = _trans.get("output_path", "") or ""
        return (f"✅ Completed ({total} rows)", 1.0, out, out,
                _u(visible=True), _u(visible=False), _u(interactive=True), "", "")

    return (status, (row / total if total else 0), _u(), _u(),
            _u(visible=True), _u(visible=False), _u(interactive=True), "", "")


def gen_readme_fn(csv_path, repo_id, source_dataset, language):
    """Generate a README.md preview."""
    if not csv_path or not Path(csv_path).exists():
        return "❌ No CSV file found."
    try:
        df = pd.read_csv(csv_path, nrows=5, encoding="utf-8-sig")
        full_df = pd.read_csv(csv_path, encoding="utf-8-sig")
        content = generate_readme(full_df, repo_id or "dataset", source_dataset or "unknown", language or "tr")
        # Save alongside CSV
        readme_path = Path(csv_path).parent / "README.md"
        readme_path.write_text(content, encoding="utf-8")
        return content
    except Exception as e:
        return f"❌ Error: {e}"


def upload_fn(csv_path, repo_name, private, hf_token):
    """Upload translated dataset to HuggingFace Hub."""
    if not csv_path or not Path(csv_path).exists():
        return "❌ No CSV file found."
    if not repo_name or not repo_name.strip():
        return "❌ Please enter a repo name."
    try:
        success, result = upload_dataset(csv_path, repo_name.strip(), hf_token, private)
        if success:
            # Also upload README if exists
            readme_path = Path(csv_path).parent / "README.md"
            if readme_path.exists():
                _, username = test_token(hf_token)
                repo_id = f"{username}/{repo_name.strip()}"
                from core.hf_uploader import upload_readme
                upload_readme(readme_path, repo_id, hf_token)
            return f"✅ Upload successful!\n{result}"
        return f"❌ Upload failed: {result}"
    except Exception as e:
        return f"❌ Error: {e}"


# ═══════════════════════════════════════════════════════════════════════
# Auto-detect pending jobs on startup
# ═══════════════════════════════════════════════════════════════════════

def detect_pending_jobs_fn():
    """Scan output/state/ for incomplete jobs and auto-load the most recent one.

    Returns 12 values matching download_btn outputs + hf_dataset_id + preview_df:
    (download_status, preview_df, column_selector, dataset_path, dataset_id, split,
     col_analysis, col_counter, translate_col_analysis, pending_confirm,
     hf_dataset_id_update)
    """
    _no = gr.update()
    _empty = (None, gr.CheckboxGroup(choices=[]), None, "", "", "No pending jobs.",
              "📌 No columns selected", "", _no, False, "")

    try:
        jobs = list_pending_jobs()
        if not jobs:
            return _empty

        # Pick the most recent by timestamp
        job = max(jobs, key=lambda j: j.get("timestamp") or "")
        dataset_id = job["dataset_id"]
        split = job.get("split", "train")
        columns = job.get("columns", [])
        completed = job.get("completed_rows", 0)
        total = job.get("total_rows", 0)

        # Find the matching dataset CSV
        safe_name = f"{dataset_id}_{split}".replace("/", "_")
        dataset_csv = config.DATASETS_DIR / f"{safe_name}.csv"
        if not dataset_csv.exists():
            return _empty

        # Load preview
        df = pd.read_csv(dataset_csv, nrows=10, encoding="utf-8-sig")
        all_columns = list(pd.read_csv(dataset_csv, nrows=0, encoding="utf-8-sig").columns)
        col_analysis = analyze_columns_fn(str(dataset_csv))
        status = (
            f"⏸️ Resumed pending job: {dataset_id}/{split}\n"
            f"   {completed}/{total} rows done — will resume from row {completed}"
        )
        col_counter = f"📌 {len(columns)} column(s) selected: {', '.join(columns)}"

        _trans["status"] = f"Pending: {completed}/{total} rows done. Press Start to resume."

        return (
            status,                                          # download_status
            df,                                              # preview_df
            gr.CheckboxGroup(choices=all_columns, value=columns),  # column_selector
            str(dataset_csv),                                # dataset_path_s
            dataset_id,                                      # dataset_id_s
            split,                                           # split_s
            col_analysis,                                    # col_analysis_md
            col_counter,                                     # col_counter
            col_analysis,                                    # translate_col_analysis
            False,                                           # pending_confirm_s
            dataset_id,                                      # hf_dataset_id
        )
    except Exception as e:
        logger.warning("Error detecting pending jobs: %s", e)
        return _empty


# ═══════════════════════════════════════════════════════════════════════
# Gradio app layout
# ═══════════════════════════════════════════════════════════════════════

def create_app() -> gr.Blocks:
    with gr.Blocks(title="Dataset Translator") as app:

        # ── Session state ──
        dataset_path_s = gr.State(None)
        dataset_id_s = gr.State("")
        split_s = gr.State("train")
        columns_s = gr.State([])
        pending_confirm_s = gr.State(False)

        gr.Markdown(
            "# 🌐 Dataset Translator\n"
            "Translate HuggingFace datasets using a **local LLM** via LM Studio.  \n"
            "Download → Select columns → Translate → Upload."
        )

        with gr.Tabs():
            # ────────────────── Tab 1: Config ──────────────────
            with gr.Tab("⚙️ Config"):
                gr.Markdown("### LM Studio Settings")
                with gr.Row():
                    lm_url = gr.Textbox(
                        label="Base URL",
                        value=config.LM_STUDIO_BASE_URL,
                        info="LM Studio server address",
                    )
                    lm_model = gr.Textbox(
                        label="Model Name",
                        value=config.LM_STUDIO_MODEL,
                        info="Must match LM Studio's loaded model",
                    )
                test_lm_btn = gr.Button("🔌 Test LM Studio Connection")
                lm_result = gr.Textbox(label="Result", interactive=False)

                gr.Markdown("---\n### HuggingFace Token")
                hf_token_input = gr.Textbox(
                    label="HF Token (write access)",
                    value=config.HF_TOKEN,
                    type="password",
                )
                test_hf_btn = gr.Button("🔑 Test HF Token")
                hf_result = gr.Textbox(label="Result", interactive=False)

                gr.Markdown("---\n### Default Languages")
                with gr.Row():
                    source_lang = gr.Dropdown(
                        label="Source Language",
                        choices=["en", "de", "fr", "es", "ru", "ar", "zh", "ja", "ko", "it", "pt"],
                        value=config.DEFAULT_SOURCE_LANG,
                    )
                    target_lang = gr.Dropdown(
                        label="Target Language",
                        choices=["tr", "en", "de", "fr", "es", "ru", "ar", "zh", "ja"],
                        value=config.DEFAULT_TARGET_LANG,
                    )

                gr.Markdown("---\n### Translation Prompt")
                gr.Markdown(
                    "**General models** (default): system prompt with translation instructions, template `{text}`.  \n"
                    "**TranslateGemma**: clear system prompt, switch template to `{src} to {tgt}: {text}`."
                )
                with gr.Row():
                    system_prompt_input = gr.Textbox(
                        label="System Prompt",
                        value=config.SYSTEM_PROMPT,
                        placeholder="e.g. You are a professional translator. Translate from {source} to {target}. Output only the translation.",
                        info="Leave empty for TranslateGemma. Use {source} and {target} as language placeholders.",
                        lines=3,
                        scale=3,
                    )
                    message_template_input = gr.Textbox(
                        label="Message Template",
                        value=config.MESSAGE_TEMPLATE,
                        info="Use {src}, {tgt}, {text} placeholders.",
                        scale=1,
                    )
                with gr.Row():
                    preset_tg_btn = gr.Button("🏷️ Preset: TranslateGemma", scale=1)
                    preset_general_btn = gr.Button("🤖 Preset: General Model", scale=1)

                gr.Markdown("---\n### Token Limit")
                max_tokens_input = gr.Number(
                    label="Max Tokens",
                    value=config.MAX_TOKENS,
                    precision=0,
                    minimum=128,
                    maximum=32768,
                    info="Max tokens for LLM response. Texts exceeding this will be auto-chunked at sentence boundaries.",
                )

            # ────────────────── Tab 2: Download ──────────────────
            with gr.Tab("📥 Download"):
                gr.Markdown("### Download a HuggingFace Dataset")
                with gr.Row():
                    hf_dataset_id = gr.Textbox(
                        label="Dataset ID",
                        placeholder="e.g. WithinUsAI/claude_mythos_distilled_25k",
                        scale=3,
                    )
                    fetch_splits_btn = gr.Button("🔍 Fetch Splits", scale=1)
                with gr.Row():
                    split_input = gr.Dropdown(
                        label="Split",
                        choices=["train"],
                        value="train",
                        scale=1,
                    )
                    row_limit = gr.Number(
                        label="Row Limit (0 = full)",
                        value=0,
                        precision=0,
                        minimum=0,
                        scale=1,
                    )
                split_status = gr.Textbox(label="Splits", interactive=False, visible=False)
                download_btn = gr.Button("⬇️ Download Dataset", variant="primary")
                download_status = gr.Textbox(label="Status", interactive=False)
                preview_df = gr.Dataframe(label="Preview (first 10 rows)", interactive=False)

                gr.Markdown("### 🔍 Cell Inspector")
                gr.Markdown("Click any cell in the preview above to see its content rendered as interactive JSON.")
                cell_inspector = gr.JSON(value={"info": "Click a cell above"}, label="Cell Content")

                gr.Markdown("### 📊 Column Analysis")
                col_analysis_md = gr.Markdown("No dataset loaded yet.")

                gr.Markdown("### Select Columns to Translate")
                column_selector = gr.CheckboxGroup(label="Columns", choices=[])
                with gr.Row():
                    select_text_btn = gr.Button("📝 Select Text Columns")
                    select_all_btn = gr.Button("✅ Select All")
                    clear_cols_btn = gr.Button("🗑️ Clear Selection")
                col_counter = gr.Markdown("📌 No columns selected")

            # ────────────────── Tab 3: Translate ──────────────────
            with gr.Tab("🌐 Translate"):
                gr.Markdown("### Translation Settings")
                translate_info = gr.Textbox(
                    label="Summary",
                    interactive=False,
                    lines=4,
                    elem_id="translate-status",
                )
                gr.Markdown("### 📊 Column Analysis (from Download tab)")
                translate_col_analysis = gr.Markdown("No dataset loaded yet.")

                gr.Markdown("### JSON Cell Handling")
                with gr.Row():
                    json_mode_radio = gr.Radio(
                        label="JSON Mode",
                        choices=["auto", "force", "off"],
                        value="auto",
                        info="auto= detect JSON cells; force= treat all cells as JSON; off= plain text only",
                        scale=1,
                    )
                with gr.Row():
                    translate_keys_input = gr.Textbox(
                        label="Translate Keys",
                        value=", ".join(config.JSON_TRANSLATE_KEYS),
                        info="Comma-separated JSON keys to translate",
                        scale=1,
                    )
                    skip_keys_input = gr.Textbox(
                        label="Skip Keys",
                        value=", ".join(config.JSON_SKIP_KEYS),
                        info="Comma-separated JSON keys to leave untouched",
                        scale=1,
                    )

                gr.Markdown("### Controls")
                with gr.Row():
                    start_btn = gr.Button("▶️ Start Translation", variant="primary", scale=2)
                    stop_btn = gr.Button("⏹️ Stop", variant="stop", scale=1, visible=False)
                    clear_btn = gr.Button("🗑️ Clear Progress", scale=1)

                start_result = gr.Textbox(label="Action", interactive=False)

                gr.Markdown("### Progress")
                status_box = gr.Textbox(label="Status", interactive=False)
                progress_bar = gr.Slider(
                    label="Progress",
                    minimum=0,
                    maximum=1,
                    value=0,
                    interactive=False,
                )
                output_path_display = gr.Textbox(label="Output File", interactive=False)

                gr.Markdown("### 🔤 Live Translation Preview")
                with gr.Row():
                    source_preview = gr.Textbox(
                        label="🔤 Source (original)",
                        interactive=False,
                        lines=6,
                        max_lines=20,
                    )
                    translated_preview = gr.Textbox(
                        label="🌐 Translated",
                        interactive=False,
                        lines=6,
                        max_lines=20,
                    )

            # ────────────────── Tab 4: Upload ──────────────────
            with gr.Tab("📤 Upload"):
                gr.Markdown("### Upload to HuggingFace Hub")
                upload_csv_path = gr.Textbox(
                    label="Translated CSV Path",
                    placeholder="Auto-filled after translation",
                )
                with gr.Row():
                    repo_name = gr.Textbox(label="Repo Name", placeholder="my-translated-dataset", scale=2)
                    private_repo = gr.Checkbox(label="Private", value=False, scale=1)
                upload_hf_token = gr.Textbox(
                    label="HF Token",
                    value=config.HF_TOKEN,
                    type="password",
                )

                gr.Markdown("### README.md")
                readme_source = gr.Textbox(
                    label="Source Dataset",
                    placeholder="e.g. WithinUsAI/claude_mythos_distilled_25k",
                )
                readme_lang = gr.Textbox(label="Language Code", value="tr")
                gen_readme_btn = gr.Button("📄 Generate README Preview")
                readme_preview = gr.Textbox(label="README Preview", lines=15, interactive=False)

                gr.Markdown("---")
                upload_btn = gr.Button("🚀 Upload to HuggingFace", variant="primary")
                upload_result = gr.Textbox(label="Result", interactive=False, lines=3)

        # ── Timer: poll progress every 2 seconds (+ button visibility) ──
        timer = gr.Timer(2)
        timer.tick(
            fn=poll_progress,
            outputs=[status_box, progress_bar, output_path_display, upload_csv_path,
                     start_btn, stop_btn, clear_btn, source_preview, translated_preview],
        )

        # ═══════════════════════════════════════════════════════
        # Event bindings
        # ═══════════════════════════════════════════════════════

        # Config tab
        test_lm_btn.click(fn=test_lm_fn,
                          inputs=[lm_url, lm_model, system_prompt_input, message_template_input],
                          outputs=[lm_result])
        test_hf_btn.click(fn=test_hf_fn, inputs=[hf_token_input], outputs=[hf_result])

        # Download tab
        fetch_splits_btn.click(
            fn=fetch_splits_fn,
            inputs=[hf_dataset_id],
            outputs=[split_input, split_status],
        )
        download_btn.click(
            fn=download_fn,
            inputs=[hf_dataset_id, split_input, row_limit, pending_confirm_s],
            outputs=[download_status, preview_df, column_selector,
                     dataset_path_s, dataset_id_s, split_s,
                     col_analysis_md, col_counter, translate_col_analysis,
                     pending_confirm_s],
        )
        preview_df.select(
            fn=on_cell_inspector,
            outputs=[cell_inspector],
        )
        select_text_btn.click(
            fn=select_text_cols_fn,
            inputs=[dataset_path_s],
            outputs=[column_selector],
        )
        def _select_all(p):
            if not p:
                return gr.CheckboxGroup(choices=[])
            cols = list(pd.read_csv(p, nrows=0, encoding="utf-8-sig").columns)
            return gr.CheckboxGroup(choices=cols, value=cols)

        select_all_btn.click(
            fn=_select_all,
            inputs=[dataset_path_s],
            outputs=[column_selector],
        )
        clear_cols_btn.click(
            fn=lambda: gr.CheckboxGroup(choices=[], value=[]),
            outputs=[column_selector],
        )

        # Store selected columns in state + update counter whenever checkbox changes
        def _on_col_change(cols, all_choices):
            """Update columns state and counter text."""
            selected = cols or []
            counter = update_col_counter(selected)
            return selected, counter

        column_selector.change(
            fn=_on_col_change,
            inputs=[column_selector, column_selector],
            outputs=[columns_s, col_counter],
        )

        # Translate tab — info updates when columns or languages change
        column_selector.change(
            fn=update_info_fn,
            inputs=[dataset_path_s, dataset_id_s, split_s, column_selector, source_lang, target_lang],
            outputs=[translate_info],
        )
        source_lang.change(
            fn=update_info_fn,
            inputs=[dataset_path_s, dataset_id_s, split_s, column_selector, source_lang, target_lang],
            outputs=[translate_info],
        )
        target_lang.change(
            fn=update_info_fn,
            inputs=[dataset_path_s, dataset_id_s, split_s, column_selector, source_lang, target_lang],
            outputs=[translate_info],
        )

        start_btn.click(
            fn=start_translate_fn,
            inputs=[dataset_path_s, dataset_id_s, split_s, columns_s,
                    source_lang, target_lang, lm_url, lm_model,
                    json_mode_radio, translate_keys_input, skip_keys_input,
                    system_prompt_input, message_template_input, max_tokens_input],
            outputs=[start_result],
        )
        # Preset buttons
        preset_tg_btn.click(
            fn=lambda: ("", "{src} to {tgt}: {text}"),
            outputs=[system_prompt_input, message_template_input],
        )
        preset_general_btn.click(
            fn=lambda: (config._DEFAULT_SYSTEM_PROMPT, "{text}"),
            outputs=[system_prompt_input, message_template_input],
        )
        stop_btn.click(fn=stop_translate_fn, outputs=[start_result])
        clear_btn.click(
            fn=clear_progress_fn,
            inputs=[dataset_id_s, split_s, columns_s, dataset_path_s],
            outputs=[start_result, translate_info, status_box, progress_bar,
                     output_path_display, upload_csv_path, dataset_path_s, download_status],
        )

        # Upload tab
        gen_readme_btn.click(
            fn=gen_readme_fn,
            inputs=[upload_csv_path, repo_name, readme_source, readme_lang],
            outputs=[readme_preview],
        )
        upload_btn.click(
            fn=upload_fn,
            inputs=[upload_csv_path, repo_name, private_repo, upload_hf_token],
            outputs=[upload_result],
        )

        # ── Auto-resume: detect pending jobs on app load ──
        app.load(
            fn=detect_pending_jobs_fn,
            outputs=[download_status, preview_df, column_selector,
                     dataset_path_s, dataset_id_s, split_s,
                     col_analysis_md, col_counter, translate_col_analysis,
                     pending_confirm_s, hf_dataset_id],
        )

    return app


# ═══════════════════════════════════════════════════════════════════════
# Windows asyncio fix — suppress WinError 10054 noise
# ═══════════════════════════════════════════════════════════════════════

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())


class _ConnResetFilter(logging.Filter):
    """Suppress ConnectionResetError from asyncio on Windows."""

    def filter(self, record: logging.LogRecord) -> bool:
        return not (
            "ConnectionResetError" in record.getMessage()
            or "_call_connection_lost" in record.getMessage()
        )


logging.getLogger("asyncio").addFilter(_ConnResetFilter())

# ═══════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    app = create_app()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        theme=gr.themes.Soft(),
    )
