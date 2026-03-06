"""
Tests for Hindi text normalization.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data.text_normalization import (
    batch_normalize,
    collapse_whitespace,
    devanagari_digits_to_ascii,
    normalize_text,
    normalize_unicode,
    remove_invisible_chars,
    remove_non_devanagari,
    replace_devanagari_digits,
)


class TestNormalizeUnicode:
    def test_nfc_applied(self):
        # Decomposed form (NFD) vs composed form (NFC)
        text_nfd = "\u0928\u092E\u0938\u094D\u0924\u0947"  # नमस्ते (already NFC-compatible)
        result = normalize_unicode(text_nfd)
        assert isinstance(result, str)

    def test_identity_for_nfc_text(self):
        text = "नमस्ते दुनिया"
        assert normalize_unicode(text) == text


class TestRemoveInvisibleChars:
    def test_removes_zero_width_space(self):
        text = "नमस्\u200Bते"
        assert remove_invisible_chars(text) == "नमस्ते"

    def test_removes_bom(self):
        text = "\uFEFFनमस्ते"
        assert remove_invisible_chars(text) == "नमस्ते"

    def test_preserves_normal_text(self):
        text = "नमस्ते दुनिया"
        assert remove_invisible_chars(text) == text


class TestDevanagariDigits:
    def test_replace_with_empty(self):
        text = "३ बजे"
        result = replace_devanagari_digits(text, replacement="")
        assert result == " बजे"

    def test_to_ascii(self):
        text = "१२३"
        result = devanagari_digits_to_ascii(text)
        assert result == "123"

    def test_mixed_digits(self):
        text = "नमस्ते ४५६"
        result = devanagari_digits_to_ascii(text)
        assert result == "नमस्ते 456"

    def test_no_digits_unchanged(self):
        text = "नमस्ते"
        result = devanagari_digits_to_ascii(text)
        assert result == text


class TestRemoveNonDevanagari:
    def test_removes_latin(self):
        text = "Hello नमस्ते World"
        result = remove_non_devanagari(text)
        # Latin characters removed, Devanagari and whitespace preserved
        assert "Hello" not in result
        assert "World" not in result
        assert "नमस्ते" in result

    def test_preserves_devanagari_and_spaces(self):
        text = "नमस्ते दुनिया"
        result = remove_non_devanagari(text)
        assert result == "नमस्ते दुनिया"

    def test_removes_punctuation(self):
        text = "नमस्ते, दुनिया!"
        result = remove_non_devanagari(text)
        assert "," not in result
        assert "!" not in result


class TestCollapseWhitespace:
    def test_multiple_spaces_collapsed(self):
        assert collapse_whitespace("a  b   c") == "a b c"

    def test_newlines_collapsed(self):
        assert collapse_whitespace("a\n\nb") == "a b"

    def test_strips_leading_trailing(self):
        assert collapse_whitespace("  hello  ") == "hello"

    def test_tabs_collapsed(self):
        assert collapse_whitespace("a\t\tb") == "a b"


class TestNormalizeText:
    def test_basic_hindi(self):
        text = "  नमस्ते   दुनिया  "
        result = normalize_text(text)
        assert result == "नमस्ते दुनिया"

    def test_converts_devanagari_digits_by_default(self):
        text = "१२३"
        result = normalize_text(text, remove_non_devanagari_chars=False)
        assert result == "123"

    def test_strips_devanagari_digits_when_not_converting(self):
        text = "१२३ नमस्ते"
        result = normalize_text(text, convert_digits=False)
        # Devanagari digits stripped, Latin chars are fine without remove_non_devanagari
        assert "१" not in result
        assert "नमस्ते" in result

    def test_removes_invisible_chars(self):
        text = "नमस्\u200Bते"
        result = normalize_text(text)
        assert "\u200B" not in result

    def test_empty_string(self):
        assert normalize_text("") == ""

    def test_only_whitespace(self):
        assert normalize_text("   ") == ""


class TestBatchNormalize:
    def test_processes_multiple_strings(self):
        texts = ["  नमस्ते  ", "Hello नमस्ते", "१२३"]
        results = batch_normalize(texts, remove_non_devanagari_chars=False, convert_digits=True)
        assert len(results) == 3
        assert results[0] == "नमस्ते"
        assert results[2] == "123"

    def test_empty_list(self):
        assert batch_normalize([]) == []
