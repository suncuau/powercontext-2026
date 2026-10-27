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

"""Coalesce concurrent embedding requests and cache their vectors.

Callers that need the same text within one coalescing window share a single
provider request: the batch is dispatched once, the per-call order is restored
on the way out, and a cached vector is never charged to the provider again.
"""

from __future__ import annotations

import asyncio
from collections import OrderedDict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from itertools import count
from typing import Any, Protocol

from powercontext.builtin.inference.models import EmbeddingResult, InferenceUsage

DEFAULT_MAX_BATCH_SIZE = 32
DEFAULT_COALESCE_MS = 10
DEFAULT_CACHE_SIZE = 256


class EmbeddingProvider(Protocol):
    """The provider shape this wrapper batches without changing."""

    profile: Any

    async def embed(self, texts: Sequence[str], /) -> EmbeddingResult: ...


@dataclass(frozen=True, slots=True)
class _BatchValue:
    """One resolved vector together with the provider request that produced it."""

    vector: tuple[float, ...]
    batch_id: int
    usage: InferenceUsage | None


@dataclass(frozen=True, slots=True)
class _Waiter:
    """One pending future per (text, caller) pair."""

    future: asyncio.Future[_BatchValue]


class CoalescingEmbeddingModel:
    """Batch concurrent ``embed`` calls and serve repeated texts from an LRU cache."""

    def __init__(
        self,
        model: EmbeddingProvider,
        *,
        max_batch_size: int = DEFAULT_MAX_BATCH_SIZE,
        coalesce_ms: int = DEFAULT_COALESCE_MS,
        cache_size: int = DEFAULT_CACHE_SIZE,
    ) -> None:
        if not isinstance(max_batch_size, int) or isinstance(max_batch_size, bool) or max_batch_size < 1:
            raise ValueError("max_batch_size must be a positive integer")  # noqa: TRY003
        if not isinstance(coalesce_ms, int) or isinstance(coalesce_ms, bool) or coalesce_ms < 0:
            raise ValueError("coalesce_ms must be a non-negative integer")  # noqa: TRY003
        if not isinstance(cache_size, int) or isinstance(cache_size, bool) or cache_size < 0:
            raise ValueError("cache_size must be a non-negative integer")  # noqa: TRY003
        self._model = model
        self.profile = model.profile
        self._max_batch_size = max_batch_size
        self._coalesce_seconds = coalesce_ms / 1000
        self._cache_size = cache_size
        self._cache: OrderedDict[str, tuple[float, ...]] = OrderedDict()
        self._pending: OrderedDict[str, list[asyncio.Future[_BatchValue]]] = OrderedDict()
        self._flush_handle: asyncio.Handle | None = None
        self._running: set[asyncio.Task[None]] = set()
        self._batch_ids = count(1)

    async def embed(self, texts: Sequence[str], /) -> EmbeddingResult:
        """Return one vector per input text, in input order."""

        values = tuple(texts)
        if not values:
            return EmbeddingResult(vectors=(), usage=InferenceUsage(requests=0))

        unique = tuple(dict.fromkeys(values))
        resolved: dict[str, tuple[float, ...]] = {}
        missing: list[str] = []
        for text in unique:
            cached = self._cache.get(text)
            if cached is None:
                missing.append(text)
            else:
                self._cache.move_to_end(text)
                resolved[text] = cached

        requests = 0
        input_tokens = 0
        output_tokens = 0
        if missing:
            futures = self._subscribe(missing)
            settled = await asyncio.gather(*(asyncio.shield(futures[text]) for text in missing))
            charged: set[int] = set()
            for text, value in zip(missing, settled, strict=True):
                resolved[text] = value.vector
                if value.batch_id in charged:
                    continue
                charged.add(value.batch_id)
                usage = value.usage
                if usage is not None:
                    input_tokens += usage.input_tokens or 0
                    output_tokens += usage.output_tokens or 0
            requests = len(charged)

        usage = (
            InferenceUsage(requests=0)
            if requests == 0
            else InferenceUsage(requests=requests, input_tokens=input_tokens, output_tokens=output_tokens)
        )
        return EmbeddingResult(vectors=tuple(resolved[text] for text in values), usage=usage)

    def _subscribe(self, texts: Iterable[str]) -> dict[str, asyncio.Future[_BatchValue]]:
        loop = asyncio.get_running_loop()
        futures: dict[str, asyncio.Future[_BatchValue]] = {}
        for text in texts:
            future: asyncio.Future[_BatchValue] = loop.create_future()
            self._pending.setdefault(text, []).append(future)
            futures[text] = future
        self._schedule()
        return futures

    def _schedule(self) -> None:
        if self._flush_handle is not None:
            return
        loop = asyncio.get_running_loop()
        if self._coalesce_seconds <= 0:
            self._flush_handle = loop.call_soon(self._flush)
        else:
            self._flush_handle = loop.call_later(self._coalesce_seconds, self._flush)

    def _flush(self) -> None:
        self._flush_handle = None
        if not self._pending:
            return
        batch: list[str] = []
        waiters: dict[str, list[asyncio.Future[_BatchValue]]] = {}
        while self._pending and len(batch) < self._max_batch_size:
            text, futures = self._pending.popitem(last=False)
            batch.append(text)
            waiters[text] = futures
        task = asyncio.create_task(self._run_batch(tuple(batch), waiters))
        self._running.add(task)
        task.add_done_callback(self._running.discard)
        if self._pending:
            self._flush_handle = asyncio.get_running_loop().call_soon(self._flush)

    async def _run_batch(self, batch: tuple[str, ...], waiters: dict[str, list[asyncio.Future[_BatchValue]]]) -> None:
        batch_id = next(self._batch_ids)
        try:
            result = await self._model.embed(batch)
        except Exception as error:
            self._fail(waiters, error)
            return
        vectors = tuple(result.vectors)
        if len(vectors) != len(batch):
            self._fail(waiters, ValueError("embedding provider returned the wrong number of vectors"))
            return
        for text, vector in zip(batch, vectors, strict=True):
            self._remember(text, vector)
            settled = _BatchValue(vector=tuple(vector), batch_id=batch_id, usage=result.usage)
            for future in waiters[text]:
                if not future.done():
                    future.set_result(settled)

    def _remember(self, text: str, vector: Sequence[float]) -> None:
        if self._cache_size <= 0:
            return
        self._cache[text] = tuple(vector)
        self._cache.move_to_end(text)
        while len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)

    @staticmethod
    def _fail(waiters: dict[str, list[asyncio.Future[_BatchValue]]], error: Exception) -> None:
        for futures in waiters.values():
            for future in futures:
                if not future.done():
                    future.set_exception(error)


__all__ = [
    "DEFAULT_CACHE_SIZE",
    "DEFAULT_COALESCE_MS",
    "DEFAULT_MAX_BATCH_SIZE",
    "CoalescingEmbeddingModel",
    "EmbeddingProvider",
]
