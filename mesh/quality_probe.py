"""Phase 6 — router-observed quality probe service.

Pluggable scaffold: a CLI / library that periodically queries each
registered specialist with a probe set + scores the responses + writes
the result back into the registry as `quality.router_observed`. Drift
detection emits `mesh.quality.drift` when |delta| > threshold.

Scoring is intentionally pluggable. The default `StubScorer` is a
placeholder so the substrate runs end-to-end on day 1; real scorers
(LLM-as-judge against a reference set, held-out eval replay, hand-rated
cache lookup) plug in via the `Scorer` protocol without touching the
runner.

Runs as a CLI module (no scheduler infrastructure in v0.1):
    python -m mesh.quality_probe --base-url http://localhost:8088 \
        --token $SLANCHA_NODE_TOKEN

Operators wire it up to cron / systemd-timer for periodic execution.
Frequency: hourly is plenty for v0.1 — sample_count grows slowly,
trends emerge over days.
"""

from __future__ import annotations

import json
import logging
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterable, Literal, Protocol

logger = logging.getLogger(__name__)


# ── Probe set ───────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ProbePrompt:
    """A single probe to send to a specialist.

    `domain` matches the SpecialistCard `domain` so probes get routed to
    the right specialist.  `reference` is the gold answer if the scorer
    needs one (LLM-judge with reference; held-out eval lookup); None
    means the scorer must work without a reference (stylometric only).
    """

    prompt_id: str
    domain: str
    text: str
    reference: str | None = None


# A tiny default probe set so the substrate doesn't require operator
# config on day 1. Real deployments plug in their own corpus.
DEFAULT_PROBE_SET: tuple[ProbePrompt, ...] = (
    ProbePrompt(
        prompt_id="general-001",
        domain="general",
        text="Briefly explain why explicit fallback chains beat silent retries in a router.",
    ),
    ProbePrompt(
        prompt_id="writing-001",
        domain="writing",
        text="Write one sentence about a city in winter.",
    ),
    ProbePrompt(
        prompt_id="code-001",
        domain="code",
        text="In Python, given a list of dicts, return the dict whose 'score' key is highest.",
    ),
)


# ── Scorer protocol + default stub ──────────────────────────────────────────


class Scorer(Protocol):
    """Interface a scoring backend must satisfy."""

    def score(self, probe: ProbePrompt, response_text: str) -> float:
        """Return a quality score on a 0..5 scale.

        Implementers MUST clamp to [0, 5]. Higher = better. The default
        stub uses response length + non-empty heuristic; real impls
        plug in LLM-judge / held-out eval / human cache.
        """
        ...


class StubScorer:
    """Placeholder scorer — substrate-only.

    Score = min(5.0, max(0.0, ln(len(response_text) + 1)))
    Bounded, monotonic on response length, doesn't crash on empty text.
    DO NOT mistake this for a quality signal — it's a working sentinel
    so the substrate has something to write while the real scorer is
    being built.
    """

    def score(self, probe: ProbePrompt, response_text: str) -> float:
        if not response_text:
            return 0.0
        import math

        raw = math.log(len(response_text.strip()) + 1)
        return max(0.0, min(5.0, raw))


class ScorerError(Exception):
    """Raised when the LLM-judge scorer cannot produce a usable score.

    Transport failure, non-2xx, missing/malformed content, or no
    parseable integer in the judge reply all surface as this. Unlike the
    never-raise probe path, a judge fault is NOT a specialist's 0 — the
    caller must be able to tell "specialist answered badly" from "scorer
    broke".
    """


class LocalJudgeScorer:
    """LLM-as-judge scorer against a local OpenAI-compatible endpoint.

    Sends a short deterministic judge prompt (temperature 0) asking the
    model to rate, as a single integer 0–5, how well a probe response
    answers the probe — judged against `probe.reference` when one is set.
    Parses the first integer from the reply and clamps to [0.0, 5.0].

    Satisfies the `Scorer` protocol structurally. Scoring failures raise
    `ScorerError`; the module never raises at import (httpx is imported
    lazily inside `score`, matching the repo's other HTTP call sites).
    """

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        api_key: str | None = None,
        timeout_s: float = 30.0,
    ) -> None:
        self.base_url = base_url
        self.model = model
        self.api_key = api_key
        self.timeout_s = timeout_s

    def _judge_messages(self, probe: ProbePrompt, response_text: str) -> list[dict]:
        """Build the deterministic judge chat messages."""
        system = (
            "You are a strict grader. Rate, as a single INTEGER from 0 to 5, "
            "how well the RESPONSE answers the PROMPT (0=worst, 5=best). "
            "Reply with ONLY the integer, no words."
        )
        parts = [f"PROMPT:\n{probe.text}"]
        if probe.reference:
            parts.append(f"REFERENCE ANSWER:\n{probe.reference}")
        parts.append(f"RESPONSE:\n{response_text}")
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": "\n\n".join(parts)},
        ]

    def score(self, probe: ProbePrompt, response_text: str) -> float:
        import httpx

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        try:
            resp = httpx.post(
                f"{self.base_url.rstrip('/')}/v1/chat/completions",
                json={
                    "model": self.model,
                    "messages": self._judge_messages(probe, response_text),
                    "temperature": 0,
                },
                headers=headers,
                timeout=self.timeout_s,
            )
        except (httpx.HTTPError, ConnectionError) as exc:
            raise ScorerError(f"judge request failed: {type(exc).__name__}") from exc
        if not 200 <= resp.status_code < 300:
            raise ScorerError(f"judge returned HTTP {resp.status_code}")
        try:
            content = resp.json()["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise ScorerError("judge response missing assistant content") from exc
        if not isinstance(content, str):
            raise ScorerError("judge content is not text")
        m = re.search(r"\d+", content)
        if m is None:
            raise ScorerError(f"no integer in judge reply: {content!r}")
        return max(0.0, min(5.0, float(int(m.group()))))


# ── Probe observation ──────────────────────────────────────────────────────


ObservationSource = Literal["synthetic", "shadow_traffic", "real_traffic"]


@dataclass(frozen=True)
class QualityObservation:
    """One observation written back to the registry.

    `score` is the aggregate over `sample_count` probe results; the
    registry-side handler updates SpecialistCard.quality_router_observed
    + quality_sample_count + last_evaluated_at + observation_source.
    """

    specialist_id: str
    score: float
    sample_count: int
    observation_source: ObservationSource
    observed_at: datetime


# ── Drift detection ─────────────────────────────────────────────────────────

DEFAULT_DRIFT_THRESHOLD = 0.5  # 0.5 points on a 5-point scale


@dataclass(frozen=True)
class DriftEvent:
    """Drift detected when |observed - prior| > threshold."""

    specialist_id: str
    prior_score: float
    new_score: float
    delta: float
    direction: Literal["up", "down"]
    threshold: float


def detect_drift(
    *,
    prior: float | None,
    current: float,
    threshold: float = DEFAULT_DRIFT_THRESHOLD,
    specialist_id: str = "",
) -> DriftEvent | None:
    """Return a DriftEvent when the delta exceeds threshold; else None.

    First observation (prior=None) is never drift — drift is a
    longitudinal concept.
    """
    if prior is None:
        return None
    delta = current - prior
    if abs(delta) <= threshold:
        return None
    return DriftEvent(
        specialist_id=specialist_id,
        prior_score=prior,
        new_score=current,
        delta=delta,
        direction="up" if delta > 0 else "down",
        threshold=threshold,
    )


# ── Probe runner ────────────────────────────────────────────────────────────


class ProbeRunner:
    """Drive one round of probes against a set of specialists.

    Takes a base URL for the mesh registry, fetches /registry to enumerate
    active specialists + their node URLs, sends probes to each, scores
    responses, returns observations. Does NOT write back — that's the
    caller's job (cli.write_observation posts to /v1/admin/quality_observation).

    Separation lets unit tests stub each phase independently.
    """

    def __init__(
        self,
        *,
        scorer: Scorer | None = None,
        probe_set: Iterable[ProbePrompt] = DEFAULT_PROBE_SET,
        http_timeout_s: float = 30.0,
    ) -> None:
        self.scorer = scorer or StubScorer()
        self.probe_set = list(probe_set)
        self.http_timeout_s = http_timeout_s

    def _send_chat(self, node_url: str, model: str, prompt: str) -> str:
        """OpenAI-compatible /v1/chat/completions call.

        Returns the assistant message text; "" on failure. Failure does
        NOT raise — the probe round continues with a 0 score for that
        attempt so a single dead specialist doesn't blank the whole
        observation set.
        """
        body = json.dumps(
            {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.7,
                "max_tokens": 256,
            }
        ).encode("utf-8")
        req = urllib.request.Request(
            f"{node_url.rstrip('/')}/chat/completions",
            method="POST",
            data=body,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=self.http_timeout_s) as resp:  # noqa: S310
                payload = json.loads(resp.read())
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            logger.warning("probe call failed: %s", type(exc).__name__)
            return ""
        try:
            return payload["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError):
            return ""

    def probe_one(
        self,
        *,
        specialist_id: str,
        model_id: str,
        node_url: str,
        domain: str,
        observed_at: datetime | None = None,
        observation_source: ObservationSource = "synthetic",
    ) -> QualityObservation:
        """Probe one specialist, return aggregated observation.

        Sends every probe in the set whose `domain` matches the
        specialist's domain (or the universal `general` bucket as
        fallback). Scores each response, averages, returns one
        observation.
        """
        applicable = [p for p in self.probe_set if p.domain in (domain, "general")]
        if not applicable:
            applicable = list(self.probe_set)  # last-resort: probe everything

        scores: list[float] = []
        for probe in applicable:
            response = self._send_chat(node_url=node_url, model=model_id, prompt=probe.text)
            scores.append(self.scorer.score(probe, response))

        n = max(len(scores), 1)
        mean = sum(scores) / n
        return QualityObservation(
            specialist_id=specialist_id,
            score=round(mean, 3),
            sample_count=len(scores),
            observation_source=observation_source,
            observed_at=observed_at or datetime.now(timezone.utc),
        )


# ── HTTP helpers for CLI ────────────────────────────────────────────────────


def write_observation(
    *,
    registry_base_url: str,
    token: str,
    obs: QualityObservation,
    http_timeout_s: float = 10.0,
) -> dict | None:
    """POST one observation to /v1/admin/quality_observation.

    Caller catches the dict response (which carries any DriftEvent
    detected by the registry) so a probe-round caller can log + alert.
    Returns None on transport failure.
    """
    body = json.dumps(
        {
            "specialist_id": obs.specialist_id,
            "score": obs.score,
            "sample_count": obs.sample_count,
            "observation_source": obs.observation_source,
            "observed_at": obs.observed_at.isoformat(),
        }
    ).encode("utf-8")
    req = urllib.request.Request(
        f"{registry_base_url.rstrip('/')}/quality_observation",
        method="POST",
        data=body,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {token}",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=http_timeout_s) as resp:  # noqa: S310
            return json.loads(resp.read())
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        logger.warning("write_observation failed: %s", type(exc).__name__)
        return None


# ── CLI entry point ─────────────────────────────────────────────────────────


def _main(argv: list[str] | None = None) -> int:
    """python -m mesh.quality_probe"""
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True, help="Mesh registry base URL")
    parser.add_argument("--token", required=True, help="SLANCHA_NODE_TOKEN bearer")
    parser.add_argument(
        "--observation-source",
        choices=["synthetic", "shadow_traffic", "real_traffic"],
        default="synthetic",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO)

    # Fetch /registry to get active specialists.
    req = urllib.request.Request(
        f"{args.base_url.rstrip('/')}/registry",
        headers={"Authorization": f"Bearer {args.token}"},
    )
    try:
        with urllib.request.urlopen(req, timeout=30.0) as resp:  # noqa: S310
            payload = json.loads(resp.read())
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        # This is a cron/systemd-timer entry point; an unhandled traceback
        # here is noise with no signal. The two helpers below (_send_chat,
        # write_observation) already guard their network calls — match them.
        logger.error("failed to fetch registry at %s: %s", args.base_url, exc)
        return 1

    snapshot = payload.get("snapshot") or {}
    catalog = snapshot.get("catalog", {})
    bindings = snapshot.get("specialists", {})

    runner = ProbeRunner()
    if isinstance(runner.scorer, StubScorer):
        logger.warning(
            "StubScorer in use: quality scores reflect response LENGTH, not "
            "answer quality. These values are written to the registry and feed "
            "routing decisions — wire a real Scorer (e.g. LocalJudgeScorer) "
            "before relying on quality_router_observed."
        )
    for specialist_id, card in catalog.items():
        nodes = bindings.get(specialist_id, [])
        if not nodes:
            continue  # Specialist registered but not currently bound to any node.
        node_url = nodes[0].get("node_url")
        if not node_url:
            continue
        obs = runner.probe_one(
            specialist_id=specialist_id,
            model_id=card.get("model_id", specialist_id),
            node_url=node_url,
            domain=card.get("domain", "general"),
            observation_source=args.observation_source,
        )
        resp_data = write_observation(
            registry_base_url=args.base_url,
            token=args.token,
            obs=obs,
        )
        logger.info(
            "probed %s score=%.3f n=%d drift=%s",
            specialist_id,
            obs.score,
            obs.sample_count,
            resp_data.get("drift") if isinstance(resp_data, dict) else None,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
