"""
Local LLM translator using LM Studio's OpenAI-compatible API.

Uses a translation-specific model (e.g. TranslateGemma) with a preset
that expects the message format: "{source} to {target}: {text}"
"""

import json
import time
import logging
from openai import OpenAI, APIError, APITimeoutError, APIConnectionError

import config
from core.tokenizer_utils import estimate_tokens, chunk_text

logger = logging.getLogger(__name__)

class Translator:
    """Wraps LM Studio's OpenAI-compatible chat endpoint for translation."""

    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        api_key: str | None = None,
        system_prompt: str | None = None,
        message_template: str | None = None,
    ):
        self.base_url = base_url or config.LM_STUDIO_BASE_URL
        self.model = model or config.LM_STUDIO_MODEL
        self.api_key = api_key or config.LM_STUDIO_API_KEY
        self.system_prompt = system_prompt if system_prompt is not None else config.SYSTEM_PROMPT
        self.message_template = message_template if message_template is not None else config.MESSAGE_TEMPLATE
        self.client = OpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
        )

    # ------------------------------------------------------------------
    # Core translation
    # ------------------------------------------------------------------

    def _build_messages(
        self,
        text: str,
        source_lang_code: str,
        target_lang_code: str,
    ) -> list[dict]:
        """Build the messages list (system + user) for the chat API."""
        messages = []
        # System prompt (optional)
        if self.system_prompt:
            system_content = self.system_prompt.replace("{source}", source_lang_code).replace("{target}", target_lang_code)
            messages.append({"role": "system", "content": system_content})
        # User message from template
        user_content = self.message_template.format(
            src=source_lang_code, tgt=target_lang_code, text=text,
        )
        messages.append({"role": "user", "content": user_content})
        return messages

    def translate(
        self,
        text: str,
        source_lang_code: str = "en",
        target_lang_code: str = "tr",
        max_tokens: int = 4096,
        chunk_enabled: bool = True,
    ) -> str:
        """
        Translate a single text string via the configured prompt + template.

        If the text exceeds the token limit, it is split into chunks at
        sentence boundaries, each chunk is translated separately, and the
        results are concatenated.

        Uses system prompt + message template to build the API request.
        Raises RuntimeError on model loading / 400 errors so callers can abort.
        """
        # Chunk if text is too large
        if chunk_enabled:
            chunks = chunk_text(text, max_tokens)
            if len(chunks) > 1:
                logger.info("Text chunked into %d parts for translation", len(chunks))
                translated_parts = []
                for i, chunk in enumerate(chunks):
                    part_result = self._translate_single(
                        chunk, source_lang_code, target_lang_code, max_tokens
                    )
                    translated_parts.append(part_result)
                return " ".join(translated_parts)

        return self._translate_single(text, source_lang_code, target_lang_code, max_tokens)

    def _translate_single(
        self,
        text: str,
        source_lang_code: str,
        target_lang_code: str,
        max_tokens: int,
    ) -> str:
        """Send a single translation request to the LLM (no chunking)."""
        messages = self._build_messages(text, source_lang_code, target_lang_code)

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.0,
                max_tokens=max_tokens,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            )
            msg = response.choices[0].message
            content = (msg.content or "").strip()
            if not content:
                finish = response.choices[0].finish_reason
                reason_tok = getattr(response.usage, "completion_tokens_details", None)
                reason_n = getattr(reason_tok, "reasoning_tokens", "?") if reason_tok else "?"
                logger.warning(
                    "Model returned empty content (finish_reason=%s, reasoning_tokens=%s/%s). "
                    "The model may be spending all tokens on internal reasoning. "
                    "Increase max_tokens or use a translation-specific model (e.g. translategemma).",
                    finish, reason_n, response.usage.completion_tokens,
                )
            return content
        except APIError as e:
            err_msg = str(e)
            # Model loading failures or invalid model — raise so the UI can stop
            if "400" in err_msg or "Failed to load model" in err_msg or "invalid_request_error" in err_msg:
                raise RuntimeError(
                    f"LM Studio model error: {e}\n\n"
                    f"Model '{self.model}' could not be loaded. "
                    f"Please check LM Studio — make sure the model is loaded and the server is running."
                ) from e
            logger.error("Translation API error: %s", e)
            return ""
        except (APITimeoutError, APIConnectionError) as e:
            logger.error("Translation API error: %s", e)
            return ""

    def translate_cell(
        self,
        text: str,
        source_lang_code: str = "en",
        target_lang_code: str = "tr",
        max_tokens: int = 512,
        max_retries: int = 3,
        retry_delay: float = 2.0,
        chunk_enabled: bool = True,
        on_leaf_translate=None,
    ) -> str:
        """
        Translate a single cell value with retry logic.

        Returns the original text unchanged when:
        - The value is empty / NaN / None
        - The value is not a string
        - All retries fail
        """
        # Skip non-translatable values
        if not isinstance(text, str) or not text.strip():
            return text

        for attempt in range(max_retries):
            result = self.translate(text, source_lang_code, target_lang_code, max_tokens, chunk_enabled=chunk_enabled)
            if result:
                if on_leaf_translate:
                    on_leaf_translate(text, result)
                return result
            if attempt < max_retries - 1:
                wait = retry_delay * (attempt + 1)
                logger.warning(
                    "Translation attempt %d/%d failed, retrying in %.1fs...",
                    attempt + 1, max_retries, wait,
                )
                time.sleep(wait)

        # All retries exhausted — return original
        logger.error("Translation failed after %d retries, keeping original.", max_retries)
        return text

    # ------------------------------------------------------------------
    # JSON-aware translation (multiturn conversations, nested structures)
    # ------------------------------------------------------------------

    @staticmethod
    def is_json_cell(text: str) -> bool:
        """Return True if the text looks like a valid JSON object or array."""
        if not isinstance(text, str):
            return False
        stripped = text.strip()
        if not stripped:
            return False
        try:
            parsed = json.loads(stripped)
            return isinstance(parsed, (dict, list))
        except (json.JSONDecodeError, ValueError):
            return False

    def _walk_json(
        self,
        obj,
        translate_keys: set[str],
        skip_keys: set[str],
        source_lang: str,
        target_lang: str,
        max_tokens: int,
        on_leaf_translate=None,
    ):
        """
        Recursively walk a JSON-structured object.
        Translates string values whose parent key is in translate_keys.
        Keys in skip_keys are left untouched (including their sub-trees).
        """
        if isinstance(obj, dict):
            result = {}
            for k, v in obj.items():
                if k in skip_keys:
                    result[k] = v
                elif isinstance(v, str):
                    if k in translate_keys:
                        result[k] = self.translate_cell(v, source_lang, target_lang, max_tokens,
                                                        on_leaf_translate=on_leaf_translate)
                    else:
                        result[k] = v
                elif isinstance(v, (dict, list)):
                    result[k] = self._walk_json(v, translate_keys, skip_keys, source_lang, target_lang, max_tokens,
                                                on_leaf_translate=on_leaf_translate)
                else:
                    result[k] = v
            return result
        elif isinstance(obj, list):
            return [
                self._walk_json(item, translate_keys, skip_keys, source_lang, target_lang, max_tokens,
                                on_leaf_translate=on_leaf_translate)
                for item in obj
            ]
        else:
            return obj

    def translate_json(
        self,
        text: str,
        translate_keys: list[str] | None = None,
        skip_keys: list[str] | None = None,
        source_lang_code: str = "en",
        target_lang_code: str = "tr",
        max_tokens: int = 512,
        on_leaf_translate=None,
    ) -> str:
        """
        Translate text values inside a JSON-structured cell.

        Recursively walks the JSON tree and translates every string value
        whose parent key is in *translate_keys*.  Keys in *skip_keys* (and
        their entire sub-trees) are preserved as-is.

        Falls back to translate_cell() if the text is not valid JSON.

        Args:
            text: Raw cell value (may be a JSON string).
            translate_keys: Keys whose string values should be translated.
                            Defaults to config.JSON_TRANSLATE_KEYS.
            skip_keys: Keys to leave untouched. Defaults to config.JSON_SKIP_KEYS.
            source_lang_code: Source language.
            target_lang_code: Target language.
            max_tokens: Max tokens per LLM call.
        """
        if translate_keys is None:
            translate_keys = config.JSON_TRANSLATE_KEYS
        if skip_keys is None:
            skip_keys = config.JSON_SKIP_KEYS

        translate_set = set(translate_keys)
        skip_set = set(skip_keys)

        # Try to parse as JSON
        try:
            parsed = json.loads(text.strip())
        except (json.JSONDecodeError, ValueError):
            # Not JSON — fall back to plain translation
            return self.translate_cell(text, source_lang_code, target_lang_code, max_tokens,
                                       on_leaf_translate=on_leaf_translate)

        if not isinstance(parsed, (dict, list)):
            return self.translate_cell(text, source_lang_code, target_lang_code, max_tokens,
                                       on_leaf_translate=on_leaf_translate)

        translated = self._walk_json(parsed, translate_set, skip_set, source_lang_code, target_lang_code, max_tokens,
                                     on_leaf_translate=on_leaf_translate)

        try:
            return json.dumps(translated, ensure_ascii=False, indent=2)
        except (TypeError, ValueError) as e:
            logger.error("JSON serialization failed after translation: %s", e)
            return text

    def translate_stream(
        self,
        text: str,
        source_lang_code: str = "en",
        target_lang_code: str = "tr",
        max_tokens: int = 512,
    ) -> str:
        """Stream-variant: prints tokens as they arrive, returns full text."""
        messages = self._build_messages(text, source_lang_code, target_lang_code)

        try:
            stream = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.0,
                max_tokens=max_tokens,
                stream=True,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            )
            full_response = ""
            for chunk in stream:
                delta = chunk.choices[0].delta
                piece = delta.content or ""
                if piece:
                    print(piece, end="", flush=True)
                    full_response += piece
            return full_response.strip()
        except (APIError, APITimeoutError, APIConnectionError) as e:
            logger.error("Streaming translation error: %s", e)
            return ""

    # ------------------------------------------------------------------
    # Connection helpers
    # ------------------------------------------------------------------

    def test_connection(self) -> tuple[bool, str]:
        """Test connectivity to LM Studio. Returns (success, message)."""
        try:
            models = self.client.models.list()
            model_ids = [m.id for m in models.data]
            if self.model not in model_ids:
                return False, (
                    f"Connected but model '{self.model}' not found! "
                    f"Available models: {model_ids}\n"
                    f"Load the model in LM Studio or update the model name in Config."
                )
            return True, f"Connected. Model '{self.model}' is available. All models: {model_ids}"
        except Exception as e:
            return False, f"Connection failed: {e}"

    def pre_flight_check(self) -> tuple[bool, str]:
        """Send a tiny test translation to verify the model actually works.

        Raises/returns an error if the model can't generate output.
        """
        try:
            result = self.translate("Hello", "en", "tr", max_tokens=256)
            if result:
                return True, f"Pre-flight OK. Test translation: '{result}'"
            return False, "Model returned empty response. The model may not be loaded properly in LM Studio."
        except RuntimeError as e:
            return False, str(e)
        except Exception as e:
            return False, f"Pre-flight check failed: {e}"
