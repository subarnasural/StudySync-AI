from dotenv import load_dotenv; load_dotenv()

import sys
import os
import json
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from scripts.query_data import query_rag

DATASET_PATH = os.path.join(os.path.dirname(__file__), "eval_dataset.json")


def _load_dataset():
    with open(DATASET_PATH, encoding="utf-8") as f:
        return json.load(f)


def _assert_rag_matches_keywords(question: str, expected_keywords: list[str], min_hits: int = 1):
    result = query_rag(question)
    # query_rag returns a dict: {answer, sources, source_details, chunks_used, processing_time_ms}
    response_text = result.get("answer", "")
    normalized = response_text.lower()
    hits = [kw for kw in expected_keywords if kw.lower() in normalized]

    assert response_text.strip(), f"Empty RAG response for question: {question}"
    assert len(hits) >= min_hits, (
        f"Question: {question}\n"
        f"Expected at least {min_hits} keyword hit(s), found {len(hits)}.\n"
        f"Expected keywords: {expected_keywords}\n"
        f"Response: {response_text}"
    )

    # Verify response structure
    assert "sources" in result, "Response missing 'sources' field"
    assert "chunks_used" in result, "Response missing 'chunks_used' field"


def test_perceptron_response_contains_expected_terms():
    sample = _load_dataset()[0]
    _assert_rag_matches_keywords(sample["question"], sample["relevant_keywords"], min_hits=1)


def test_backprop_response_contains_expected_terms():
    sample = _load_dataset()[1]
    _assert_rag_matches_keywords(sample["question"], sample["relevant_keywords"], min_hits=1)
