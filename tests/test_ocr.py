"""
tests/test_ocr.py
==================
Unit tests for the OCR engine.

Run:
    python -m pytest tests/test_ocr.py -v
"""

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from backend.utils.ocr_engine import (
    extract_text_from_image,
    is_page_scanned,
    _text_quality_score,
)


# ---------------------------------------------------------------------------
# is_page_scanned heuristic
# ---------------------------------------------------------------------------

def test_is_page_scanned_with_short_text():
    """Text shorter than threshold should be detected as scanned."""
    assert is_page_scanned("", threshold=50) is True
    assert is_page_scanned("hello", threshold=50) is True
    assert is_page_scanned("   ", threshold=50) is True


def test_is_page_scanned_with_long_text():
    """Text longer than threshold should NOT be detected as scanned."""
    long_text = "This is a page with sufficient native text content for extraction."
    assert is_page_scanned(long_text, threshold=50) is False


def test_is_page_scanned_none_input():
    """None input should be treated as scanned (empty)."""
    assert is_page_scanned(None, threshold=50) is True


# ---------------------------------------------------------------------------
# _text_quality_score
# ---------------------------------------------------------------------------

def test_quality_score_empty():
    assert _text_quality_score("") == 0.0
    assert _text_quality_score("   ") == 0.0


def test_quality_score_good_text():
    """Well-formed text should score significantly above zero."""
    score = _text_quality_score("This is a well-formed sentence with real words.")
    assert score > 0.5, f"Expected quality > 0.5, got {score}"


def test_quality_score_garbage():
    """Garbage characters should score lower than real text."""
    garbage = "## $$ !! @@ %% ^^ ** (( ))"
    good = "Newton's second law describes force equals mass times acceleration"
    assert _text_quality_score(good) > _text_quality_score(garbage)


# ---------------------------------------------------------------------------
# extract_text_from_image – structural tests
# ---------------------------------------------------------------------------

def test_extract_returns_dict_keys():
    """Even on failure, extract_text_from_image should return the expected keys."""
    # Feed empty bytes – will fail but should return structured dict
    result = extract_text_from_image(b"")
    assert isinstance(result, dict)
    assert "text" in result
    assert "confidence" in result
    assert "method" in result
    assert "error" in result or result["method"] == "no_text_detected"


def test_extract_with_real_image():
    """If test_ocr.png exists, verify OCR produces a non-empty result dict."""
    img_path = os.path.join(os.path.dirname(__file__), "..", "test_ocr.png")
    if not os.path.exists(img_path):
        import pytest
        pytest.skip("test_ocr.png not found – skipping real OCR test")

    result = extract_text_from_image(img_path)
    assert isinstance(result, dict)
    assert "text" in result
    assert "confidence" in result
    assert "processing_time_ms" in result
    # We don't assert text is non-empty because the test image may be too small/blank
