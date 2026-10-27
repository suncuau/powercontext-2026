# Copyright (c) 2026 OceanBase.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Rerank Memory candidates through one batch-scored HTTP endpoint."""

from __future__ import annotations

import math
from typing import Any

import httpx

from powercontext.builtin.artifacts.memory.models import MemoryHit
from powercontext.builtin.artifacts.memory.reranking import MemoryRerankDecision
from powercontext.builtin.inference import InferenceUsage

DEFAULT_HTTP_RERANK_POLICY_ID = "http-cross-encoder-v1"
DEFAULT_HTTP_RERANK_BATCH_SIZE = 32
_SCORE_KEYS = frozenset({"rank", "score"})


class HTTPMemoryReranker:
    """Score the whole coarse pool in bounded batches and order it globally."""

    def __init__(
        self,
        endpoint: str,
        *,
        client: httpx.AsyncClient,
        batch_size: int = DEFAULT_HTTP_RERANK_BATCH_SIZE,
        policy_id: str = DEFAULT_HTTP_RERANK_POLICY_ID,
    ) -> None:
        if not isinstance(endpoint, str) or not endpoint.strip():
            raise ValueError("rerank endpoint must be a non-empty string")  # noqa: TRY003
        if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size < 1:
            raise ValueError("rerank batch size must be a positive integer")  # noqa: TRY003
        self.policy_id = policy_id
        self._endpoint = endpoint
        self._client = client
        self._batch_size = batch_size

    async def rerank(
        self,
        query: str,
        candidates: tuple[MemoryHit, ...],
        limit: int,
        /,
    ) -> MemoryRerankDecision:
        """Return the globally best original ranks without changing hit identity."""

        _validate_bounds(len(candidates), limit)
        scores: dict[int, float] = {}
        requests = 0
        for start in range(0, len(candidates), self._batch_size):
            batch = candidates[start : start + self._batch_size]
            ranks = tuple(start + offset + 1 for offset in range(len(batch)))
            payload = {
                "query": query,
                "candidates": [
                    {"rank": rank, "text": candidate.text} for rank, candidate in zip(ranks, batch, strict=True)
                ],
            }
            response = await self._client.post(self._endpoint, json=payload)
            response.raise_for_status()
            requests += 1
            scores.update(_read_batch_scores(response, ranks))

        ordered = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
        return MemoryRerankDecision(
            selected_ranks=tuple(rank for rank, _score in ordered[:limit]),
            usage=InferenceUsage(requests=requests),
        )


def _read_batch_scores(response: httpx.Response, ranks: tuple[int, ...]) -> dict[int, float]:
    try:
        payload = response.json()
    except ValueError as error:
        raise ValueError(f"rerank response is not valid JSON: {error}") from error  # noqa: TRY003

    if not isinstance(payload, dict) or set(payload) != {"scores"}:
        raise ValueError("rerank response must contain exactly a scores field")  # noqa: TRY003
    entries = payload["scores"]
    if not isinstance(entries, list):
        raise ValueError("rerank scores must be a list")  # noqa: TRY003, TRY004

    scores: dict[int, float] = {}
    for entry in entries:
        rank, score = _read_entry(entry)
        if rank in scores:
            raise ValueError(f"rerank response repeats rank {rank}")  # noqa: TRY003
        scores[rank] = score

    if set(scores) != set(ranks):
        raise ValueError("rerank response must cover exactly the ranks it was given")  # noqa: TRY003
    return scores


def _read_entry(entry: Any) -> tuple[int, float]:
    if not isinstance(entry, dict) or set(entry) != _SCORE_KEYS:
        raise ValueError("each rerank score must contain exactly rank and score")  # noqa: TRY003
    rank = entry["rank"]
    score = entry["score"]
    if not isinstance(rank, int) or isinstance(rank, bool):
        raise ValueError("rerank ranks must be integers")  # noqa: TRY003, TRY004
    if not isinstance(score, (int, float)) or isinstance(score, bool):
        raise ValueError("rerank scores must be numbers")  # noqa: TRY003, TRY004
    numeric = float(score)
    if not math.isfinite(numeric):
        raise ValueError("rerank scores must be finite")  # noqa: TRY003
    return rank, numeric


def _validate_bounds(candidate_count: int, limit: int) -> None:
    if candidate_count < 1:
        raise ValueError("rerank requires at least one candidate")  # noqa: TRY003
    if limit < 1 or limit > candidate_count:
        raise ValueError("rerank limit must be between one and the candidate count")  # noqa: TRY003


__all__ = [
    "DEFAULT_HTTP_RERANK_BATCH_SIZE",
    "DEFAULT_HTTP_RERANK_POLICY_ID",
    "HTTPMemoryReranker",
]
