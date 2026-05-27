"""SpecialistCard TOML linter — catches authoring bugs at edit time.

Caught here (instead of at test_each_card_is_specialist_card time):
- Pydantic validation errors (missing required fields, wrong types)
- runtime_gb < storage_gb (the invariant that broke when demo-model-v1
  landed with runtime_gb=48 storage_gb=54)
- Unknown capability strings — typos like "tooluse" instead of "tools"
- Unknown domains — typos like "wrtiing" instead of "writing"
- Unknown languages — keeps the language tag pool curated
- Duplicate specialist_id across files
- File-stem-vs-specialist_id mismatch — easy copy-paste bug

Usage:
    python -m mesh.validate_card                                # lint all in mesh/catalog/
    python -m mesh.validate_card mesh/catalog/demo-model-v2.toml  # one file
    python -m mesh.validate_card --all                          # equivalent to no-args
    python -m mesh.validate_card --strict                       # also fail on warnings

Exit codes:
    0  all good
    1  errors found
    2  bad invocation (file missing, etc.)

For CI: drop the `python -m mesh.validate_card` call before pytest in
your pyproject.toml's pre-commit hook or workflow.
"""

from __future__ import annotations

import argparse
import sys
import tomllib
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from pydantic import ValidationError

from mesh.models import SpecialistCard

# Canonical sets — extend when new vocabulary lands. Out-of-set values
# emit warnings (strict mode promotes to errors) rather than hard errors
# so additive vocab doesn't break old TOMLs.
KNOWN_CAPABILITIES = {
    "streaming",
    "system_prompt",
    "tools",
    "json_mode",
    "json_schema",
    "vision",
    "seed",
    "parallel_tool_calls",
    "reasoning",
    "cache_control",
}

KNOWN_DOMAINS = {
    "writing",
    "code",
    "math",
    "reasoning",
    "general",
    "multilingual",
    "creative",
    "summarization",
    "tool_use",
}

KNOWN_LANGUAGE_TAGS = {
    # Common ISO 639-1 codes the catalog has used. Open for additions
    # — flag unknown ones so typos like "engish" don't slip through.
    "en", "es", "fr", "de", "it", "pt", "nl", "ru", "zh", "ja", "ko",
    "ar", "hi", "bn", "ur", "tr", "vi", "id", "th", "pl", "uk", "cs",
    "sv", "no", "da", "fi", "el", "he", "fa",
}


@dataclass(frozen=True)
class Finding:
    path: Path
    severity: str  # "error" | "warning"
    code: str      # short stable identifier — caller can grep
    message: str


@dataclass
class _Report:
    findings: list[Finding] = field(default_factory=list)

    def error(self, path: Path, code: str, message: str) -> None:
        self.findings.append(Finding(path=path, severity="error", code=code, message=message))

    def warning(self, path: Path, code: str, message: str) -> None:
        self.findings.append(Finding(path=path, severity="warning", code=code, message=message))

    @property
    def has_errors(self) -> bool:
        return any(f.severity == "error" for f in self.findings)

    @property
    def has_warnings(self) -> bool:
        return any(f.severity == "warning" for f in self.findings)


# ── Per-card checks ─────────────────────────────────────────────────────────


def _check_runtime_vs_storage(card: SpecialistCard, path: Path, report: _Report) -> None:
    if card.runtime_gb < card.storage_gb:
        report.error(
            path,
            "RUNTIME_LT_STORAGE",
            f"runtime_gb ({card.runtime_gb}) < storage_gb ({card.storage_gb}); "
            f"runtime budget must include weights",
        )


def _check_domain(card: SpecialistCard, path: Path, report: _Report) -> None:
    if card.domain not in KNOWN_DOMAINS:
        report.warning(
            path,
            "UNKNOWN_DOMAIN",
            f"domain={card.domain!r} not in canonical set; "
            f"close matches: {sorted(KNOWN_DOMAINS)[:5]}...",
        )


def _check_capabilities(card: SpecialistCard, path: Path, report: _Report) -> None:
    for cap in card.capabilities:
        if cap not in KNOWN_CAPABILITIES:
            report.warning(
                path,
                "UNKNOWN_CAPABILITY",
                f"capability={cap!r} not in canonical set; "
                f"typo? known: {sorted(KNOWN_CAPABILITIES)}",
            )


def _check_languages(card: SpecialistCard, path: Path, report: _Report) -> None:
    for lang in card.languages:
        if lang not in KNOWN_LANGUAGE_TAGS:
            report.warning(
                path,
                "UNKNOWN_LANGUAGE_TAG",
                f"language={lang!r} not in canonical set; typo?",
            )


def _check_specialist_id_matches_filename(
    card: SpecialistCard, path: Path, report: _Report
) -> None:
    """File `qwen3-math-7b-q4.toml` should declare specialist_id `qwen3-math-7b-q4`.

    Loose check — file stem vs specialist_id, case-sensitive. Catches
    accidental copy-paste where someone duplicates a TOML and forgets to
    rename the specialist_id field.
    """
    if path.stem != card.specialist_id:
        report.warning(
            path,
            "FILENAME_ID_MISMATCH",
            f"file stem {path.stem!r} != specialist_id {card.specialist_id!r}; "
            f"intentional? if not, rename the file or fix the field",
        )


def _check_quality_consistency(card: SpecialistCard, path: Path, report: _Report) -> None:
    """quality.router_observed should never be set directly in a TOML.

    That field is written by the Phase 6 probe service; an operator
    setting it manually means tests + audits will be lying to consumers.
    """
    if card.quality_router_observed is not None:
        report.error(
            path,
            "QUALITY_OBSERVED_HARDCODED",
            f"quality_router_observed should be NULL at authoring time "
            f"(was {card.quality_router_observed}); it's populated by "
            f"`python -m mesh.quality_probe`. Use quality_node_self_reported "
            f"if you mean self-reported.",
        )


def _check_one(path: Path) -> tuple[SpecialistCard | None, _Report]:
    """Validate a single TOML. Returns (card_or_None, findings)."""
    report = _Report()

    if not path.exists():
        report.error(path, "FILE_NOT_FOUND", f"no such file: {path}")
        return None, report
    if path.suffix != ".toml":
        report.error(path, "NOT_TOML", f"file is not *.toml: {path}")
        return None, report

    try:
        with path.open("rb") as f:
            data = tomllib.load(f)
    except tomllib.TOMLDecodeError as exc:
        report.error(path, "TOML_PARSE", f"TOML parse error: {exc}")
        return None, report

    try:
        card = SpecialistCard(**data)
    except ValidationError as exc:
        for err in exc.errors():
            loc = ".".join(str(p) for p in err.get("loc", []))
            report.error(
                path,
                "PYDANTIC_VALIDATION",
                f"{loc}: {err['msg']} (got: {err.get('input')!r})",
            )
        return None, report

    _check_runtime_vs_storage(card, path, report)
    _check_domain(card, path, report)
    _check_capabilities(card, path, report)
    _check_languages(card, path, report)
    _check_specialist_id_matches_filename(card, path, report)
    _check_quality_consistency(card, path, report)

    return card, report


def validate_paths(paths: Iterable[Path]) -> _Report:
    """Validate every TOML at the given paths + cross-file duplicate-ID check."""
    aggregate = _Report()
    ids_seen: dict[str, Path] = {}

    for path in paths:
        card, rep = _check_one(path)
        aggregate.findings.extend(rep.findings)
        if card is not None:
            prior = ids_seen.get(card.specialist_id)
            if prior is not None:
                aggregate.error(
                    path,
                    "DUPLICATE_SPECIALIST_ID",
                    f"specialist_id={card.specialist_id!r} also declared in {prior}",
                )
            else:
                ids_seen[card.specialist_id] = path

    return aggregate


# ── CLI ─────────────────────────────────────────────────────────────────────


_DEFAULT_CATALOG = Path(__file__).parent / "catalog"


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("files", nargs="*", help="TOMLs to validate. Default: every *.toml in mesh/catalog/")
    p.add_argument("--all", action="store_true", help="Explicitly validate the whole catalog")
    p.add_argument("--strict", action="store_true", help="Promote warnings to errors (CI-friendly)")
    return p


def _format_finding(f: Finding, *, strict: bool) -> str:
    sev = f.severity.upper()
    if strict and f.severity == "warning":
        sev = "WARNING-AS-ERROR"
    return f"  [{sev}] {f.code} {f.path}: {f.message}"


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    if args.files and not args.all:
        paths = [Path(f) for f in args.files]
    else:
        paths = sorted(_DEFAULT_CATALOG.glob("*.toml"))

    if not paths:
        print(f"no TOMLs to validate (looked in {_DEFAULT_CATALOG})", file=sys.stderr)
        return 2

    report = validate_paths(paths)
    if not report.findings:
        print(f"OK — {len(paths)} card(s) clean")
        return 0

    by_severity = defaultdict(int)
    for f in report.findings:
        by_severity[f.severity] += 1
        print(_format_finding(f, strict=args.strict), file=sys.stderr)

    err_count = by_severity["error"]
    warn_count = by_severity["warning"]
    print(
        f"\nSummary: {err_count} error(s), {warn_count} warning(s) across {len(paths)} file(s)",
        file=sys.stderr,
    )

    if err_count > 0 or (args.strict and warn_count > 0):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
