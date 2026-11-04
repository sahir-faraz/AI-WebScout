from src.crawler import _normalise, _same_host


def test_normalise_removes_fragment_and_tracking_parameters():
    value = _normalise("https://example.com/page?utm_source=news&id=3#section")
    assert value == "https://example.com/page?id=3"


def test_same_host_compares_hostnames():
    assert _same_host("https://docs.example.com/a", "docs.example.com")
    assert not _same_host("https://other.example.com/a", "docs.example.com")
