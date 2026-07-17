"""
Token estimation and text chunking utilities for LLM translation.

Uses a simple heuristic (avg 4 chars/token for English) to estimate token count
without requiring tiktoken or other heavy dependencies.
"""

import re
import logging

logger = logging.getLogger(__name__)

# Average characters per token (rough heuristic for English/mixed text)
CHARS_PER_TOKEN = 4


def estimate_tokens(text: str) -> int:
    """Estimate token count for text using a simple char-based heuristic."""
    if not text:
        return 0
    return max(1, len(text) // CHARS_PER_TOKEN)


def _split_at_boundaries(text: str) -> list[str]:
    """
    Split text at sentence boundaries (., !, ?, double newline).
    Falls back to newline, then semicolons, then commas, then word boundaries.
    """
    # Try sentence-level split first
    for pattern in [
        r'(?<=[.!?])\s+',          # sentence endings followed by whitespace
        r'\n\n+',                    # double newline (paragraph break)
        r'\n+',                      # single newline
        r'(?<=[;:])\s+',            # semicolons/colons
        r'(?<=,)\s+',               # commas
    ]:
        parts = re.split(pattern, text)
        parts = [p for p in parts if p.strip()]
        if len(parts) > 1:
            return parts

    # Fallback: word-level split
    words = text.split()
    if len(words) <= 1:
        return [text]

    # Group words into halves as a last resort
    mid = len(words) // 2
    return [" ".join(words[:mid]), " ".join(words[mid:])]


def chunk_text(text: str, max_tokens: int) -> list[str]:
    """
    Split text into chunks that each fit within max_tokens.

    Uses a safety margin (0.75) to account for prompt overhead (system prompt,
    message template, etc.). Splits at sentence boundaries when possible;
    falls back to word boundaries if a single sentence exceeds the limit.

    Args:
        text: The text to chunk.
        max_tokens: Max tokens per chunk.

    Returns:
        List of text chunks. Returns [text] if it already fits.
    """
    if not text or not text.strip():
        return [text]

    # Safety margin: reserve 25% for prompt/overhead
    effective_limit = max_tokens * 0.75
    estimated = estimate_tokens(text)

    if estimated <= effective_limit:
        return [text]

    # Recursive splitting
    parts = _split_at_boundaries(text)
    chunks: list[str] = []
    current_chunk = ""

    for part in parts:
        candidate = (current_chunk + " " + part).strip() if current_chunk else part
        if estimate_tokens(candidate) <= effective_limit:
            current_chunk = candidate
        else:
            if current_chunk:
                chunks.append(current_chunk)
            # If the part itself is too large, recursively split it
            if estimate_tokens(part) > effective_limit:
                sub_parts = _split_at_boundaries(part)
                if len(sub_parts) == 1:
                    # Can't split further — include as-is (best effort)
                    logger.warning("Single chunk exceeds token limit: %d tokens (limit %.0f)",
                                   estimate_tokens(part), effective_limit)
                    chunks.append(part)
                else:
                    chunks.extend(chunk_text(part, max_tokens))
                current_chunk = ""
            else:
                current_chunk = part

    if current_chunk:
        chunks.append(current_chunk)

    logger.debug("Chunked text into %d parts (estimated %d tokens, limit %d)",
                 len(chunks), estimated, max_tokens)
    return chunks
