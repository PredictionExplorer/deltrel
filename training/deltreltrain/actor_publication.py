"""Rate-limited visibility of durable completed-game and policy-only samples."""

from __future__ import annotations

from collections.abc import Callable, Mapping
import time


class PublicationProgress:
    def __init__(
        self,
        *,
        metadata: Mapping[str, object],
        base_games: int,
        base_samples: int,
        base_policy_only_samples: int = 0,
        base_policy_first_samples: int = 0,
        base_enriched_samples: int = 0,
        base_written_samples: int | None = None,
        base_evaluator_rows: int,
        base_wall_seconds: float,
        task_started: float,
        evaluator_rows: Callable[[], int],
        heartbeat: Callable[..., None],
        emit: Callable[[dict], None],
        interval_seconds: float = 1.0,
    ) -> None:
        self.metadata = dict(metadata)
        self.base_games = base_games
        self.base_samples = base_samples
        self.base_policy_only_samples = base_policy_only_samples
        self.base_policy_first_samples = base_policy_first_samples
        self.base_enriched_samples = base_enriched_samples
        self.base_written_samples = (
            base_samples if base_written_samples is None else base_written_samples
        )
        self.base_evaluator_rows = base_evaluator_rows
        self.base_wall_seconds = base_wall_seconds
        self.task_started = task_started
        self.evaluator_rows = evaluator_rows
        self.heartbeat = heartbeat
        self.emit = emit
        self.interval_seconds = interval_seconds
        self.games = self.samples = self.policy_only_samples = 0
        self.policy_first_samples = self.enriched_samples = self.written_samples = 0
        self._emitted_enriched_samples = self._emitted_written_samples = 0
        self._emitted_policy_first_samples = 0
        self._emitted_policy_only_samples = 0
        self._emitted_games = self._emitted_samples = 0
        self._last_emit: float | None = None

    def _cumulative(self, now: float) -> dict[str, object]:
        return {
            "cumulative_games": self.base_games + self.games,
            "cumulative_samples": self.base_samples + self.samples,
            "cumulative_policy_only_samples": self.base_policy_only_samples
            + self.policy_only_samples,
            "cumulative_policy_first_samples": self.base_policy_first_samples
            + self.policy_first_samples,
            "cumulative_enriched_samples": self.base_enriched_samples
            + self.enriched_samples,
            "cumulative_written_samples": self.base_written_samples
            + self.written_samples,
            "pending_policy_rows": self.base_policy_first_samples
            + self.policy_first_samples
            - self.base_enriched_samples
            - self.enriched_samples,
            "cumulative_evaluator_rows": self.base_evaluator_rows
            + self.evaluator_rows(),
            "cumulative_batch_wall_seconds": self.base_wall_seconds
            + now
            - self.task_started,
        }

    def progress(self, **fields) -> None:
        now = time.monotonic()
        if fields.get("phase") in (
            "selfplay_completed",
            "selfplay_policy_salvaged",
            "selfplay_policy_published",
        ):
            games, samples = fields["completed_games"], fields["persisted_decisions"]
            policy_only = fields.get(
                "retained_incomplete_decisions",
                fields.get("salvaged_policy_decisions", self.policy_only_samples),
            )
            policy_published = fields.get("policy_published_decisions", policy_only)
            enriched = fields.get("enriched_decisions", self.enriched_samples)
            written = fields.get("replay_written_decisions", samples)
            if (
                type(games) is not int
                or type(samples) is not int
                or games < self.games
                or samples < self.samples
                or type(policy_only) is not int
                or policy_only < self.policy_only_samples
                or policy_only > samples
                or (
                    "retained_incomplete_decisions" not in fields
                    and policy_only - self.policy_only_samples > samples - self.samples
                )
                or any(
                    type(value) is not int or value < previous
                    for value, previous in (
                        (policy_published, self.policy_first_samples),
                        (enriched, self.enriched_samples),
                        (written, self.written_samples),
                    )
                )
                or not 0 <= enriched <= policy_published <= samples <= written
                or policy_only > policy_published - enriched
            ):
                raise ValueError("durable publication counters moved backwards")
            self.games, self.samples = games, samples
            self.policy_only_samples = policy_only
            self.policy_first_samples, self.enriched_samples, self.written_samples = (
                policy_published,
                enriched,
                written,
            )
            self._flush(now, force=False)
        self.heartbeat(**(fields | self._cumulative(now)))

    def _flush(self, now: float, *, force: bool) -> None:
        if (
            self.games == self._emitted_games
            and self.samples == self._emitted_samples
            and self.enriched_samples == self._emitted_enriched_samples
            and self.written_samples == self._emitted_written_samples
            and self.policy_only_samples == self._emitted_policy_only_samples
        ):
            return
        if (
            not force
            and self._last_emit is not None
            and now - self._last_emit < self.interval_seconds
        ):
            return
        self.emit(
            self.metadata
            | self._cumulative(now)
            | {
                "record_kind": "publication",
                "timestamp_ns": time.time_ns(),
                "published_games": self.games - self._emitted_games,
                "published_samples": self.samples - self._emitted_samples,
                "published_policy_only_samples": self.policy_only_samples
                - self._emitted_policy_only_samples,
                "published_policy_first_samples": self.policy_first_samples
                - self._emitted_policy_first_samples,
                "published_enriched_samples": self.enriched_samples
                - self._emitted_enriched_samples,
                "published_written_samples": self.written_samples
                - self._emitted_written_samples,
                "published_task_policy_only_samples": self.policy_only_samples,
                "published_task_games": self.games,
                "published_task_samples": self.samples,
                "task_elapsed_seconds": now - self.task_started,
            }
        )
        self._emitted_games, self._emitted_samples = self.games, self.samples
        self._emitted_policy_only_samples = self.policy_only_samples
        self._emitted_policy_first_samples = self.policy_first_samples
        self._emitted_enriched_samples = self.enriched_samples
        self._emitted_written_samples = self.written_samples
        self._last_emit = now

    def finish(self) -> None:
        self._flush(time.monotonic(), force=True)
