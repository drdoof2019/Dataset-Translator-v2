# 🌐 Dataset Translator

A Gradio web application that translates [HuggingFace](https://huggingface.co) datasets using a **local LLM** via [LM Studio](https://lmstudio.ai). No cloud API keys required for translation — everything runs locally.

## ✨ Features

- **Local LLM Translation** — Uses LM Studio's OpenAI-compatible endpoint; no external API costs
- **HuggingFace Integration** — Download datasets by ID, upload translated results back to the Hub
- **Cell-by-Cell Translation** — Translates individual cell values, preserving dataset structure
- **JSON Cell Support** — Intelligently translates JSON-structured cells (e.g., multi-turn conversations) while skipping non-translatable keys like `role`, `id`, `timestamp`
- **Auto-Resume** — Checkpoint after every row; interrupted translations resume from where they stopped
- **Auto-Detect Pending Jobs** — On startup, automatically loads the most recent incomplete translation
- **Auto-Chunking** — Splits long texts into token-safe chunks (configurable via `MAX_TOKENS`)
- **Live Preview** — Real-time source/translated text preview during translation
- **Column Analysis** — Detects JSON structure and classifies keys as translatable / skip / untouched
- **Cell Inspector** — Click any preview cell to view its content as pretty-printed JSON
- **README Generator** — Auto-generates a HuggingFace dataset card for uploaded repos

## 📋 Requirements

- **Python 3.10+**
- **[LM Studio](https://lmstudio.ai)** with a loaded translation model (e.g., `translategemma-12b-it`)
- Internet connection (only for dataset download/upload)

## 🚀 Quick Start

### 1. Install dependencies

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure environment

Create a `.env` file in the project root:

```ini
LM_STUDIO_BASE_URL=http://127.0.0.1:1234/v1
LM_STUDIO_MODEL=translategemma-12b-it
LM_STUDIO_API_KEY=lm-studio
HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxx
DEFAULT_SOURCE_LANG=en
DEFAULT_TARGET_LANG=tr
MAX_TOKENS=4096
```

| Variable | Description | Default |
|---|---|---|
| `LM_STUDIO_BASE_URL` | LM Studio server address | `http://127.0.0.1:1234/v1` |
| `LM_STUDIO_MODEL` | Model name loaded in LM Studio | `translategemma-12b-it` |
| `LM_STUDIO_API_KEY` | API key (LM Studio default: `lm-studio`) | `lm-studio` |
| `HF_TOKEN` | HuggingFace write-access token | *(required for upload)* |
| `DEFAULT_SOURCE_LANG` | Default source language code | `en` |
| `DEFAULT_TARGET_LANG` | Default target language code | `tr` |
| `MAX_TOKENS` | Max tokens per chunk (auto-chunking threshold) | `4096` |
| `SYSTEM_PROMPT` | Custom system prompt (overrides built-in default) | *(built-in general preset)* |
| `MESSAGE_TEMPLATE` | User message format (`{src}`, `{tgt}`, `{text}`) | `{text}` |

### 3. Start LM Studio

1. Open LM Studio and download a translation model (e.g., `translategemma-12b-it`)
2. Go to the **Local Server** tab, select the model, and click **Start Server**
3. Verify the server is running at `http://127.0.0.1:1234`

### 4. Run the app

```powershell
.venv\Scripts\activate
python app.py
```

Open `http://localhost:7860` in your browser.

## 📖 Usage

The app has four tabs:

### ⚙️ Config
- Set LM Studio URL, model name, and API key
- Test LM Studio connection
- Enter and validate HuggingFace token
- Set source/target languages
- Choose system prompt preset (General models / TranslateGemma)

### 📥 Download
- Enter a HuggingFace dataset ID (e.g., `WithinUsAI/claude_mythos_distilled_25k`)
- Click **Fetch Splits** to auto-detect available splits
- Set row limit (`0` = all rows)
- Download and preview the first 10 rows
- Auto-analyze columns for JSON structure
- Select which text columns to translate

### 🌐 Translate
- Review translation settings (dataset, columns, languages)
- Choose JSON mode: `auto` / `force` / `off`
- Configure JSON Translate Keys and Skip Keys
- Click **Start Translation** — progress updates every 2 seconds
- Use **Stop** to halt after the current row
- Use **Clear Progress** to reset and start over

### 📤 Upload
- Set repo name and visibility (private/public)
- Generate a README.md dataset card preview
- Upload the translated CSV + README to HuggingFace Hub

## 🔧 JSON Cell Translation

Some datasets contain JSON-structured cells (e.g., conversation data). This tool handles them intelligently:

```json
[
  {"role": "user", "content": "What is AI?"},
  {"role": "assistant", "content": "AI is artificial intelligence..."}
]
```

- **Translate Keys** — values of these keys get translated: `content`, `text`, `summary`, `description`, `question`, `answer`, `input`, `output`, `response`, `message`, `instruction`, `completion`
- **Skip Keys** — these keys and their sub-trees are never touched: `role`, `id`, `name`, `type`, `model`, `source`, `lang`, `language`, `timestamp`, `created_at`, `updated_at`, `index`, `metadata`

| Mode | Behavior |
|---|---|
| `auto` (default) | Detect JSON automatically; use JSON translation for JSON cells, plain text otherwise |
| `force` | Treat all cells as JSON; non-JSON cells fall back to plain translation |
| `off` | Disable JSON detection; translate all cells as plain text |

## 🔄 Resume Mechanism

- Progress is checkpointed to `output/state/` after every row
- On app restart, the most recent incomplete job is **auto-detected and loaded**
- Press **Start Translation** to resume from the last completed row
- Checkpoint files: `output/state/<dataset_id>_<split>_<columns_hash>.json`

## 📁 Project Structure

```
dataset_translation_v2/
├── app.py                     # Gradio web app (main entry point)
├── config.py                  # Configuration loader (.env → settings)
├── translator.py              # LLM translation class (LM Studio client)
├── requirements.txt           # Python dependencies
├── KULLANIM.md                # Turkish usage guide
├── README.md                  # This file
├── core/
│   ├── __init__.py
│   ├── dataset_loader.py      # HuggingFace dataset download/save
│   ├── progress_manager.py    # Checkpoint tracking + pending job detection
│   ├── translation_engine.py  # Cell-by-cell translation engine
│   ├── tokenizer_utils.py     # Token counting + auto-chunking
│   └── hf_uploader.py         # HuggingFace Hub upload
└── output/                    # Generated at runtime (gitignored)
    ├── datasets/              # Downloaded datasets (CSV + Parquet)
    ├── translated/            # Translated datasets
    └── state/                 # Checkpoint files
```

## 🏗️ Architecture

```
┌─────────────┐     ┌──────────────┐     ┌─────────────────┐
│   Gradio    │────▶│  Translator  │────▶│   LM Studio     │
│   UI        │     │  (app.py)    │     │   (local LLM)   │
│  (app.py)   │◀────│              │◀────│                 │
└─────────────┘     └──────┬───────┘     └─────────────────┘
                           │
                    ┌──────▼───────┐
                    │ Translation  │
                    │   Engine     │
                    │ (cell-by-cell)│
                    └──────┬───────┘
                           │
              ┌────────────┼────────────┐
              ▼            ▼            ▼
        ┌──────────┐ ┌──────────┐ ┌──────────┐
        │ Dataset  │ │ Progress  │ │   HF     │
        │ Loader   │ │ Manager   │ │ Uploader │
        └──────────┘ └──────────┘ └──────────┘
```

- **[`app.py`](app.py)** — Gradio UI with 4 tabs, live polling, and auto-resume on startup
- **[`translator.py`](translator.py)** — Wraps LM Studio's OpenAI-compatible chat endpoint; handles single translation, JSON walking, and streaming
- **[`core/translation_engine.py`](core/translation_engine.py)** — Iterates rows/cells, invokes [`Translator`](translator.py), manages progress callbacks
- **[`core/progress_manager.py`](core/progress_manager.py)** — Per-dataset JSON checkpoints; [`list_pending_jobs()`](core/progress_manager.py:104) scans for incomplete work
- **[`core/dataset_loader.py`](core/dataset_loader.py)** — Downloads from HuggingFace, saves as CSV + Parquet, caches locally
- **[`core/tokenizer_utils.py`](core/tokenizer_utils.py)** — Token counting and auto-chunking for long texts
- **[`core/hf_uploader.py`](core/hf_uploader.py)** — Uploads translated CSV + generated README to HuggingFace Hub

## 🛠️ Troubleshooting

| Problem | Solution |
|---|---|
| LM Studio connection error | Verify LM Studio is running and model is loaded; use **Test LM Studio Connection** button |
| HuggingFace upload error | Ensure token has **write** access; validate with **Test HF Token** button |
| Translation too slow | Try a smaller/faster model; select fewer columns; use Row Limit to test a subset |
| Memory error | Use Row Limit to process in batches; check LM Studio model memory settings |
| Poor translation quality | Try a different model (e.g., `translategemma-12b-it`, `gemma-2`); check JSON mode settings |
| ConnectionResetError on Windows | Already handled — the app suppresses asyncio Proactor pipe errors automatically |

## 📝 License

This project is provided as-is for research and educational purposes.

## 📚 Additional Documentation

- **[`KULLANIM.md`](KULLANIM.md)** — Detailed Turkish usage guide (kullanım kılavuzu)