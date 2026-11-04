import json
import pytest

from src.evidence_detector import _extract_json, review_evidence


def test_extracts_plain_json():
    assert _extract_json('{"claims": [], "overall_note": "ok"}') == {
        "claims": [], "overall_note": "ok"
    }


def test_extracts_fenced_json():
    assert _extract_json('```json\n{"claims": []}\n```') == {"claims": []}


def test_rejects_invalid_json():
    with pytest.raises(ValueError):
        _extract_json("not JSON")


def test_empty_pages_do_not_call_api():
    result = review_evidence([], image_results=[])
    assert result["claims"] == []
    assert "No crawled" in result["overall_note"]
