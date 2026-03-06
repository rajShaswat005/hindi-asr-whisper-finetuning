"""
Hindi Text Normalization Module.

Cleans and normalises Hindi (Devanagari) transcripts for ASR training:
- Strip leading / trailing whitespace
- Collapse multiple spaces
- Remove non-Devanagari punctuation and Latin characters (configurable)
- Normalize Unicode (NFC)
- Remove Devanagari numerals (replaced with verbal equivalents or stripped)
- Strip invisible / zero-width characters
"""

from __future__ import annotations

import re
import unicodedata
from typing import Optional

# Unicode ranges
_DEVANAGARI_RANGE = r"\u0900-\u097F"
_DEVANAGARI_EXTENDED = r"\uA8E0-\uA8FF"  # Devanagari Extended block

# Devanagari digit map (optional: replace with empty string to strip)
_DEVANAGARI_DIGITS: dict[str, str] = {
    "०": "0",
    "१": "1",
    "२": "2",
    "३": "3",
    "४": "4",
    "५": "5",
    "६": "6",
    "७": "7",
    "८": "8",
    "९": "9",
}

# Zero-width and invisible characters
_INVISIBLE_PATTERN = re.compile(
    r"[\u200B-\u200D\uFEFF\u00AD\u2028\u2029\u00A0]"
)

# Collapse runs of whitespace
_WHITESPACE_PATTERN = re.compile(r"\s+")

# Characters allowed in cleaned Hindi text (Devanagari + ASCII space + comma/period)
_ALLOWED_PATTERN = re.compile(
    rf"[^{_DEVANAGARI_RANGE}{_DEVANAGARI_EXTENDED}\s]"
)


def normalize_unicode(text: str) -> str:
    """Apply NFC Unicode normalization."""
    return unicodedata.normalize("NFC", text)


def remove_invisible_chars(text: str) -> str:
    """Remove zero-width and other invisible Unicode characters."""
    return _INVISIBLE_PATTERN.sub("", text)


def replace_devanagari_digits(text: str, replacement: str = "") -> str:
    """
    Replace Devanagari digit characters.

    Parameters
    ----------
    text:
        Input Hindi text.
    replacement:
        String to substitute each Devanagari digit with.
        Pass ``""`` to strip digits, or provide ASCII equivalents by using
        :func:`devanagari_digits_to_ascii` instead.

    Returns
    -------
    Text with Devanagari digits replaced.
    """
    for deva, _ in _DEVANAGARI_DIGITS.items():
        text = text.replace(deva, replacement)
    return text


def devanagari_digits_to_ascii(text: str) -> str:
    """Convert Devanagari digit characters to their ASCII equivalents."""
    for deva, ascii_digit in _DEVANAGARI_DIGITS.items():
        text = text.replace(deva, ascii_digit)
    return text


def remove_non_devanagari(text: str) -> str:
    """
    Remove characters that are not Devanagari script or whitespace.

    This strips Latin characters, punctuation, and other scripts that
    should not appear in clean Hindi transcripts.
    """
    return _ALLOWED_PATTERN.sub("", text)


def collapse_whitespace(text: str) -> str:
    """Collapse multiple consecutive whitespace characters into a single space."""
    return _WHITESPACE_PATTERN.sub(" ", text).strip()


def normalize_text(
    text: str,
    remove_non_devanagari_chars: bool = True,
    convert_digits: bool = True,
) -> str:
    """
    Full normalization pipeline for Hindi ASR transcripts.

    Pipeline order:
    1. Unicode NFC normalization
    2. Remove invisible / zero-width characters
    3. Optionally convert / remove Devanagari digits
    4. Optionally remove non-Devanagari characters
    5. Collapse whitespace and strip

    Parameters
    ----------
    text:
        Raw Hindi transcript string.
    remove_non_devanagari_chars:
        When True, strip all non-Devanagari characters (except whitespace).
    convert_digits:
        When True, convert Devanagari digits to ASCII digits.
        When False, strip Devanagari digits entirely.

    Returns
    -------
    Normalized transcript string.
    """
    text = normalize_unicode(text)
    text = remove_invisible_chars(text)

    if convert_digits:
        text = devanagari_digits_to_ascii(text)
    else:
        text = replace_devanagari_digits(text, replacement="")

    if remove_non_devanagari_chars:
        text = remove_non_devanagari(text)

    text = collapse_whitespace(text)
    return text


def batch_normalize(
    texts: list[str],
    remove_non_devanagari_chars: bool = True,
    convert_digits: bool = True,
) -> list[str]:
    """
    Normalize a batch of Hindi transcripts.

    Parameters
    ----------
    texts:
        List of raw transcript strings.
    remove_non_devanagari_chars:
        Forwarded to :func:`normalize_text`.
    convert_digits:
        Forwarded to :func:`normalize_text`.

    Returns
    -------
    List of normalized strings.
    """
    return [
        normalize_text(t, remove_non_devanagari_chars=remove_non_devanagari_chars, convert_digits=convert_digits)
        for t in texts
    ]
