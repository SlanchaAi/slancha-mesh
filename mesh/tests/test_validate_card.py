"""Tests for the SpecialistCard TOML linter."""

from __future__ import annotations

from pathlib import Path

import pytest

from mesh.validate_card import (
    KNOWN_CAPABILITIES,
    KNOWN_DOMAINS,
    main,
    validate_paths,
)


def _write(path: Path, content: str) -> Path:
    path.write_text(content)
    return path


# Minimal valid TOML — every other test extends/breaks this.
_GOOD_TOML = """
model_id = "vendor/model"
specialist_id = "{stem}"
domain = "general"
difficulty_tiers = ["medium"]
languages = ["en"]
required_backend = "vllm"
storage_gb = 10.0
runtime_gb = 12.0
min_vram_gb = 8.0
context_window = 8192
n_layers = 32
capabilities = ["streaming"]
"""


def test_clean_card_no_findings(tmp_path):
    f = _write(tmp_path / "clean-card.toml", _GOOD_TOML.format(stem="clean-card"))
    report = validate_paths([f])
    assert report.findings == [], report.findings


def test_runtime_lt_storage_errors(tmp_path):
    bad = _GOOD_TOML.format(stem="bad").replace(
        "runtime_gb = 12.0", "runtime_gb = 8.0"
    )
    f = _write(tmp_path / "bad.toml", bad)
    report = validate_paths([f])
    codes = [x.code for x in report.findings]
    assert "RUNTIME_LT_STORAGE" in codes
    assert report.has_errors


def test_unknown_capability_warns(tmp_path):
    bad = _GOOD_TOML.format(stem="bad-cap").replace(
        'capabilities = ["streaming"]', 'capabilities = ["streming"]'  # typo
    )
    f = _write(tmp_path / "bad-cap.toml", bad)
    report = validate_paths([f])
    assert any(x.code == "UNKNOWN_CAPABILITY" for x in report.findings)
    # warning, not error
    assert not report.has_errors


def test_unknown_domain_warns(tmp_path):
    bad = _GOOD_TOML.format(stem="d").replace(
        'domain = "general"', 'domain = "wrtiing"'
    )
    f = _write(tmp_path / "d.toml", bad)
    report = validate_paths([f])
    assert any(x.code == "UNKNOWN_DOMAIN" for x in report.findings)


def test_unknown_language_warns(tmp_path):
    bad = _GOOD_TOML.format(stem="lang").replace(
        'languages = ["en"]', 'languages = ["engish"]'
    )
    f = _write(tmp_path / "lang.toml", bad)
    report = validate_paths([f])
    assert any(x.code == "UNKNOWN_LANGUAGE_TAG" for x in report.findings)


def test_filename_id_mismatch_warns(tmp_path):
    """File 'a.toml' declaring specialist_id='b' surfaces the mismatch."""
    bad = _GOOD_TOML.format(stem="b")  # specialist_id=b
    f = _write(tmp_path / "a.toml", bad)
    report = validate_paths([f])
    assert any(x.code == "FILENAME_ID_MISMATCH" for x in report.findings)


def test_quality_router_observed_hardcoded_errors(tmp_path):
    """Operator setting router_observed directly is a hard error."""
    bad = _GOOD_TOML.format(stem="q") + 'quality_router_observed = 4.5\n'
    f = _write(tmp_path / "q.toml", bad)
    report = validate_paths([f])
    assert any(x.code == "QUALITY_OBSERVED_HARDCODED" for x in report.findings)
    assert report.has_errors


def test_pydantic_validation_error_surfaces(tmp_path):
    """Missing required field → Pydantic validation error → reported."""
    bad = _GOOD_TOML.format(stem="missing").replace(
        'required_backend = "vllm"\n', ""
    )
    f = _write(tmp_path / "missing.toml", bad)
    report = validate_paths([f])
    assert any(x.code == "PYDANTIC_VALIDATION" for x in report.findings)
    assert report.has_errors


def test_toml_parse_error_surfaces(tmp_path):
    f = _write(tmp_path / "garbage.toml", "this is = not [ valid")
    report = validate_paths([f])
    assert any(x.code == "TOML_PARSE" for x in report.findings)
    assert report.has_errors


def test_duplicate_specialist_id_across_files(tmp_path):
    a = _write(tmp_path / "a.toml", _GOOD_TOML.format(stem="a"))
    b_text = _GOOD_TOML.format(stem="b").replace(
        'specialist_id = "b"', 'specialist_id = "a"'  # collide with file a
    )
    b = _write(tmp_path / "b.toml", b_text)
    report = validate_paths([a, b])
    assert any(x.code == "DUPLICATE_SPECIALIST_ID" for x in report.findings)
    assert report.has_errors


def test_file_not_found_errors(tmp_path):
    report = validate_paths([tmp_path / "does-not-exist.toml"])
    assert any(x.code == "FILE_NOT_FOUND" for x in report.findings)


def test_main_returns_0_for_clean_catalog(tmp_path, capsys):
    _write(tmp_path / "ok.toml", _GOOD_TOML.format(stem="ok"))
    rc = main([str(tmp_path / "ok.toml")])
    assert rc == 0
    assert "OK" in capsys.readouterr().out


def test_main_returns_1_on_error(tmp_path, capsys):
    bad = _GOOD_TOML.format(stem="bad").replace(
        "runtime_gb = 12.0", "runtime_gb = 8.0"
    )
    _write(tmp_path / "bad.toml", bad)
    rc = main([str(tmp_path / "bad.toml")])
    assert rc == 1


def test_main_strict_promotes_warnings_to_errors(tmp_path):
    """Warning-only TOML returns 0 in normal mode, 1 in --strict."""
    bad = _GOOD_TOML.format(stem="warn").replace(
        'capabilities = ["streaming"]', 'capabilities = ["typo-only"]'
    )
    _write(tmp_path / "warn.toml", bad)
    assert main([str(tmp_path / "warn.toml")]) == 0
    assert main([str(tmp_path / "warn.toml"), "--strict"]) == 1


# ── Run against the actual on-disk catalog ─────────────────────────────────


def test_real_catalog_has_known_capabilities_and_domains():
    """Sanity check: KNOWN_* sets cover the real catalog's authored values.

    If this test fails, EITHER the catalog is using a new capability /
    domain that should be added to the canonical set (extend KNOWN_*),
    OR it's a typo bug that the linter just caught.
    """
    from mesh.catalog import load_catalog

    cards = load_catalog()
    unknown_caps = set()
    unknown_domains = set()
    for c in cards:
        for cap in c.capabilities:
            if cap not in KNOWN_CAPABILITIES:
                unknown_caps.add(cap)
        if c.domain not in KNOWN_DOMAINS:
            unknown_domains.add(c.domain)

    # If anything surfaces, fail loudly with a hint.
    assert not unknown_caps, (
        f"catalog uses capability values not in KNOWN_CAPABILITIES: {unknown_caps}. "
        f"Either extend the set in mesh/validate_card.py or fix the catalog typos."
    )
    assert not unknown_domains, (
        f"catalog uses domain values not in KNOWN_DOMAINS: {unknown_domains}. "
        f"Either extend the set or fix the catalog typos."
    )
