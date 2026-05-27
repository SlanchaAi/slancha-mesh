"""Grade captured real-traffic replay entries — P0 live-wiring.

Connects the GRADING primitive (a mesh.quality_probe `Scorer`) to its two
sinks:

  (a) the registry's per-specialist quality field, via
      MeshRegistry.record_quality_observation, and
  (b) the replay corpus's per-entry quality label
      (ReplayEntry.judge_score / judge_source).

Unlike ProbeRunner — which sends *synthetic* probes to live specialists —
this grades traffic ALREADY captured into the TrafficReplayStore: no
network probe, just score the stored oracle_response against its prompt.

Pure orchestration over injected collaborators (store, scorer, registry):
unit-tests with fakes, no HTTP, no daemon, no hot-path change. The store's
ReplayEntry is a mutable dataclass and recent() returns live references, so
the per-entry label is stamped in place; judge_* fields are disjoint from
anything TrafficReplayStore.add() mutates, so this is safe alongside the
serving path for an offline batch.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from mesh.quality_probe import ObservationSource, ProbePrompt, Scorer, ScorerError
from mesh.replay_store import TrafficReplayStore

if TYPE_CHECKING:  # avoid import cycle + let tests inject a fake registry
    from mesh.registry import MeshRegistry

logger = logging.getLogger(__name__)


def grade_replay_corpus(
    store: TrafficReplayStore,
    scorer: Scorer,
    registry: "MeshRegistry",
    *,
    observation_source: ObservationSource = "real_traffic",
    judge_source: str | None = None,
    domain: str | None = None,
    limit: int | None = None,
) -> dict:
    """Grade ungraded replay entries and write to both quality sinks.

    For each recent entry that lacks a judge_score:
      1. wrap (prompt_text, domain) into a ProbePrompt and score the
         stored oracle_response via ``scorer.score(...)``,
      2. stamp judge_score + judge_source back onto the entry (sink b),
      3. accumulate the score under the entry's served_by_specialist.

    After grading, write one quality observation per specialist — the mean
    over that specialist's freshly-graded entries — to the registry (sink a).

    A ``ScorerError`` on any entry is logged + skipped: a broken judge never
    aborts the batch and never counts as a 0 (matching the probe path's
    "judge fault != specialist's 0" contract). Entries with no
    served_by_specialist are still graded + stamped (sink b) but cannot
    contribute to a per-specialist registry observation (sink a).

    `limit` bounds how many most-recent (optionally domain-filtered) entries
    are considered; None means the whole store. `judge_source` defaults to
    the scorer's class name.

    Returns::

        {
          "specialists": {sid: {"score": float, "sample_count": int}},
          "graded": int,                 # entries scored + stamped this run
          "skipped_no_specialist": int,  # graded but no specialist to credit
          "errors": [{"prompt_hash": str, "error": str}, ...],
        }
    """
    judge_source = judge_source or type(scorer).__name__
    n = limit if limit is not None else len(store)
    entries = store.recent(n=n, domain=domain)

    by_specialist: dict[str, list[float]] = {}
    errors: list[dict] = []
    graded = 0
    skipped_no_specialist = 0

    for entry in entries:
        if entry.judge_score is not None:
            continue  # already graded — idempotent re-runs skip it
        probe = ProbePrompt(
            prompt_id=entry.prompt_hash,
            domain=entry.domain,
            text=entry.prompt_text,
            reference=None,  # captured traffic carries no gold reference
        )
        try:
            score = scorer.score(probe, entry.oracle_response)
        except ScorerError as exc:
            logger.warning("grade skipped for %s: %s", entry.prompt_hash, exc)
            errors.append({"prompt_hash": entry.prompt_hash, "error": str(exc)})
            continue

        # Sink (b): per-entry quality label, stamped in place on the live
        # entry reference returned by recent().
        entry.judge_score = score
        entry.judge_source = judge_source
        graded += 1

        if entry.served_by_specialist is None:
            skipped_no_specialist += 1
            continue
        by_specialist.setdefault(entry.served_by_specialist, []).append(score)

    # Sink (a): one registry observation per specialist, mean over its
    # freshly-graded entries.
    specialists: dict[str, dict] = {}
    for specialist_id, scores in by_specialist.items():
        mean = round(sum(scores) / len(scores), 3)
        registry.record_quality_observation(
            specialist_id=specialist_id,
            score=mean,
            sample_count=len(scores),
            observation_source=observation_source,
        )
        specialists[specialist_id] = {"score": mean, "sample_count": len(scores)}

    return {
        "specialists": specialists,
        "graded": graded,
        "skipped_no_specialist": skipped_no_specialist,
        "errors": errors,
    }
