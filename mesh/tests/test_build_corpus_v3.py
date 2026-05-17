"""Tests for the pure-function helpers in mesh.scripts.build_corpus_v3.

The HF-streaming code paths are not unit-tested (network-bound); we cover
the deterministic classifier + signals shaping below. iter_source +
build_corpus are exercised via smoke runs in CI.
"""

from __future__ import annotations

from mesh.scripts.build_corpus_v3 import (
    classify_difficulty,
    classify_domain,
    classify_language,
    route_class_from_difficulty,
    signals_for,
)


# ---------------------------------------------------------------------------
# Domain classifier
# ---------------------------------------------------------------------------


def test_classify_domain_code_fenced_block():
    assert classify_domain("Here is code:\n```python\nx = 1\n```") == "code"


def test_classify_domain_code_def_keyword():
    assert classify_domain("def foo(): pass") == "code"


def test_classify_domain_code_sql():
    assert classify_domain("SELECT id FROM users WHERE x = 1") == "code"


def test_classify_domain_math_keyword():
    assert classify_domain("Calculate the integral of x^2 dx.") == "math"


def test_classify_domain_math_latex_marker():
    assert classify_domain("Show that \\frac{a}{b} simplifies.") == "math"


def test_classify_domain_translate_signal():
    assert classify_domain("Please translate this paragraph to French.") == "multilingual"


def test_classify_domain_reasoning_keyword():
    assert classify_domain("Why does the sky appear blue at noon?") == "reasoning"


def test_classify_domain_default_is_general():
    assert classify_domain("Tell me about the Roman Empire.") == "general"


def test_classify_domain_empty_text_is_general():
    assert classify_domain("") == "general"


# ---------------------------------------------------------------------------
# Language classifier (cheap ASCII-ratio heuristic)
# ---------------------------------------------------------------------------


def test_classify_language_english_text():
    assert classify_language("The quick brown fox jumps over the lazy dog.") == "en"


def test_classify_language_non_ascii_heavy():
    assert classify_language("これは日本語のテキストです。これは日本語のテキストです。") == "other"


def test_classify_language_empty_defaults_to_en():
    assert classify_language("") == "en"


# ---------------------------------------------------------------------------
# Difficulty classifier
# ---------------------------------------------------------------------------


def test_classify_difficulty_easy():
    assert classify_difficulty("hi") == "easy"


def test_classify_difficulty_medium():
    assert classify_difficulty("x" * 400) == "medium"


def test_classify_difficulty_hard():
    assert classify_difficulty("x" * 1200) == "hard"


# ---------------------------------------------------------------------------
# Route class derivation
# ---------------------------------------------------------------------------


def test_route_class_mapping():
    assert route_class_from_difficulty("easy") == "hot_interactive"
    assert route_class_from_difficulty("medium") == "standard"
    assert route_class_from_difficulty("hard") == "batch"


# ---------------------------------------------------------------------------
# signals_for composes the above
# ---------------------------------------------------------------------------


def test_signals_for_returns_full_schema():
    sig = signals_for("def hello(): pass")
    assert set(sig.keys()) == {
        "domain", "difficulty", "language", "needs_tools", "route_class",
    }
    assert sig["domain"] == "code"
    assert sig["language"] == "en"
    assert sig["needs_tools"] is False
    # short → easy → hot_interactive
    assert sig["difficulty"] == "easy"
    assert sig["route_class"] == "hot_interactive"


def test_signals_for_long_math_prompt():
    text = "Compute the integral of x^2 from 0 to 1, " * 30
    sig = signals_for(text)
    assert sig["domain"] == "math"
    assert sig["difficulty"] == "hard"
    assert sig["route_class"] == "batch"
