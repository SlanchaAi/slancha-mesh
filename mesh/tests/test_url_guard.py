"""SSRF guard for node-supplied node_url (#98)."""

from __future__ import annotations

import pytest

from mesh.url_guard import NodeUrlError, validate_node_url


@pytest.mark.parametrize("url", [
    "http://node-a.ts.net:8003",        # MagicDNS name
    "https://10.0.0.5:8004/v1",          # private LAN
    "http://192.168.1.20:11434",         # private LAN
    "http://100.64.0.9:8003",            # tailnet CGNAT
    "http://127.0.0.1:11434",            # loopback — legit single-box (allowed by default)
])
def test_legit_node_urls_pass(url):
    assert validate_node_url(url) == url


@pytest.mark.parametrize("url", [
    "http://169.254.169.254/latest/meta-data/",   # cloud IMDS (link-local)
    "http://169.254.169.254:80/",
    "file:///etc/passwd",                          # non-http scheme
    "gopher://internal/",
    "ftp://10.0.0.1/",
    "http://0.0.0.0/",                             # unspecified
    "http://[::]/",
    "no-scheme-host",
])
def test_ssrf_and_bad_scheme_rejected(url):
    with pytest.raises(NodeUrlError):
        validate_node_url(url)


def test_block_loopback_when_flag_set(monkeypatch):
    assert validate_node_url("http://127.0.0.1:8003")  # default: allowed
    monkeypatch.setenv("SLANCHA_NODE_URL_BLOCK_LOOPBACK", "1")
    with pytest.raises(NodeUrlError):
        validate_node_url("http://127.0.0.1:8003")


def test_block_private_when_flag_set(monkeypatch):
    assert validate_node_url("http://10.0.0.5:8003")  # default: allowed
    monkeypatch.setenv("SLANCHA_NODE_URL_BLOCK_PRIVATE", "1")
    with pytest.raises(NodeUrlError):
        validate_node_url("http://10.0.0.5:8003")


def test_loaded_model_ingest_rejects_poisoned_node_url():
    """The node-supplied ingest model refuses a poisoned node_url at the boundary."""
    from pydantic import ValidationError

    from mesh.models import LoadedModel

    base = dict(specialist_id="qwen2.5-coder-7b-q4-ollama", model_id="qwen2.5-coder:7b",
                loaded_at="2026-06-03T00:00:00Z")
    LoadedModel(**base, node_url="http://node-a.ts.net:8003")          # valid passes
    LoadedModel(**base, node_url=None)                                  # None ok
    for bad in ("http://169.254.169.254/", "file:///etc/passwd"):
        with pytest.raises(ValidationError):
            LoadedModel(**base, node_url=bad)
