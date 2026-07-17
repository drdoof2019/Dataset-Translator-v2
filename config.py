"""
Configuration module — loads .env and exposes application settings.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# --- LM Studio ---
LM_STUDIO_BASE_URL = os.getenv("LM_STUDIO_BASE_URL", "http://127.0.0.1:1234/v1")
LM_STUDIO_MODEL = os.getenv("LM_STUDIO_MODEL", "translategemma-12b-it")
LM_STUDIO_API_KEY = os.getenv("LM_STUDIO_API_KEY", "lm-studio")

# --- Translation prompt ---
# Default: General model preset with detailed translation instructions.
# Use {source} and {target} placeholders for language names.
_DEFAULT_SYSTEM_PROMPT = (
    "Above, you see a text in {source}. Please translate it to {target}. "
    "Do not print the original text, just the translation.\n\n"
    "Follow the following instructions:\n\n"
    "Ensure the translation accurately reflects the original text's meaning.\n"
    "The translation should have correct grammar, including proper sentence structure, "
    "verb conjugation, punctuation, and the correct use of articles.\n"
    "The translation should read naturally and fluently as if originally written in the "
    "target language. Avoid awkward phrasing or literal translations that sound unnatural.\n"
    "Pay special attention to proper nouns and specific terms. Names of people, places, "
    "organizations, and other terms that should not be translated must be handled with care "
    "to maintain their original meaning and recognition.\n"
    "Ensure that the translation maintains the original text's tone and style."
)
SYSTEM_PROMPT = os.getenv("SYSTEM_PROMPT", "").strip() or _DEFAULT_SYSTEM_PROMPT

# Message template: how the user message is formatted.
# Use {src}, {tgt}, {text} placeholders.
# General models: "{text}"  |  TranslateGemma: "{src} to {tgt}: {text}"
MESSAGE_TEMPLATE = os.getenv("MESSAGE_TEMPLATE", "{text}")

# --- HuggingFace ---
HF_TOKEN = os.getenv("HF_TOKEN", "")

# --- Translation defaults ---
DEFAULT_SOURCE_LANG = os.getenv("DEFAULT_SOURCE_LANG", "en")
DEFAULT_TARGET_LANG = os.getenv("DEFAULT_TARGET_LANG", "tr")
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "4096"))

# --- JSON cell translation ---
# Keys whose string values should be translated inside JSON cells
JSON_TRANSLATE_KEYS = ["content", "text", "summary", "description", "question", "answer",
                       "input", "output", "response", "message", "instruction", "completion"]

# Keys to NEVER touch (and their entire sub-trees)
JSON_SKIP_KEYS = ["role", "id", "name", "type", "model", "source", "lang", "language",
                  "timestamp", "created_at", "updated_at", "index", "metadata"]

# --- Directory paths ---
BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "output"
DATASETS_DIR = OUTPUT_DIR / "datasets"
TRANSLATED_DIR = OUTPUT_DIR / "translated"
STATE_DIR = OUTPUT_DIR / "state"

# Ensure directories exist
for d in [OUTPUT_DIR, DATASETS_DIR, TRANSLATED_DIR, STATE_DIR]:
    d.mkdir(parents=True, exist_ok=True)
