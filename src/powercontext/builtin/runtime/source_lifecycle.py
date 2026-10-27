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

"""Source lifecycle events over the evidence dependency graph.

Each Scope owns a directed acyclic graph whose roots are captured Sources and
whose derived nodes are Memory entries and Topic Memory artifacts. One event
turns one Source on or off; availability then propagates through every
dependent node according to that node's ``all``/``any`` policy.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from powercontext.artifacts import ArtifactRef
from powercontext.builtin.artifacts.memory.models import MemoryCitation
from powercontext.builtin.persistence.source_lifecycle import StoredLifecycleNode
from powercontext.builtin.runtime.models import ReactivateMemoryEntryRequest, RetireMemoryEntryRequest
from powercontext.sources.models import SourceRef

LifecyclePolicy = Literal["all", "any"]
NonEmptyText = Annotated[str, str]


class SourceLifecycleError(ValueError):
    """Raised when a lifecycle registration, event, or reference violates its contract."""

    def __init__(self, detail: str) -> None:
        self.detail = detail
        super().__init__(f"source lifecycle error: {detail}")


class SourceLifecycleEvent(BaseModel):
    """One durable availability change for exactly one registered Source."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    node_id: NonEmptyText
    revision: int = Field(gt=0, strict=True)
    sequence: int = Field(gt=0, strict=True)
    available: bool = Field(strict=True)
    reason: NonEmptyText

    @field_validator("node_id", "reason")
    @classmethod
    def reject_blank_text(cls, value: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("lifecycle text must be a non-empty, non-blank string")  # noqa: TRY003
        return value


@dataclass(slots=True)
class LifecycleNode:
    """One registered node of a Scope's evidence dependency graph."""

    node_id: str
    kind: Literal["source", "memory", "topic"]
    reference: object
    parents: tuple[str, ...]
    policy: LifecyclePolicy
    revision: int
    sequence: int
    available: bool
    reason: str = ""
    current_reference: object | None = None


class SourceLifecycleGraph:
    """Validate one Scope's dependency graph and propagate Source availability."""

    def __init__(self) -> None:
        self._nodes: dict[str, LifecycleNode] = {}

    @property
    def nodes(self) -> tuple[LifecycleNode, ...]:
        return tuple(self._nodes.values())

    def register_source(self, node_id: str, reference: object, /, *, revision: int) -> LifecycleNode:
        """Register one captured Source root that starts available."""

        _require_node_id(node_id)
        if not isinstance(revision, int) or isinstance(revision, bool) or revision < 1:
            raise SourceLifecycleError("source revision must be a positive integer")  # noqa: TRY003
        return self._register(
            LifecycleNode(
                node_id=node_id,
                kind="source",
                reference=reference,
                parents=(),
                policy="all",
                revision=revision,
                sequence=0,
                available=True,
            )
        )

    def register_memory(
        self,
        node_id: str,
        reference: object,
        parents: Sequence[str],
        policy: LifecyclePolicy,
        /,
    ) -> LifecycleNode:
        """Register one Memory entry derived from existing nodes."""

        return self._register_derived(node_id, "memory", reference, parents, policy)

    def register_topic(
        self,
        node_id: str,
        reference: object,
        parents: Sequence[str],
        policy: LifecyclePolicy,
        /,
    ) -> LifecycleNode:
        """Register one Topic Memory artifact derived from existing nodes."""

        return self._register_derived(node_id, "topic", reference, parents, policy)

    def apply(self, event: SourceLifecycleEvent, /) -> list[str]:
        """Apply one Source event and return every available node id, sorted."""

        node = self._nodes.get(event.node_id)
        if node is None or node.kind != "source":
            raise SourceLifecycleError(f"{event.node_id!r} is not a registered Source")  # noqa: TRY003
        if event.revision != node.revision:
            raise SourceLifecycleError("event revision does not match the registered Source")  # noqa: TRY003
        if event.sequence < node.sequence:
            raise SourceLifecycleError("event sequence must increase")  # noqa: TRY003
        if event.sequence == node.sequence:
            if event.available == node.available and event.reason == node.reason:
                return list(self.active_nodes())
            raise SourceLifecycleError("event sequence was reused with different content")  # noqa: TRY003
        node.sequence = event.sequence
        node.available = event.available
        node.reason = event.reason
        self._recompute()
        return list(self.active_nodes())

    def resolve(self, node_id: str, /, *, purpose: str) -> object:
        """Return the original reference for one node and one stated purpose."""

        node = self._nodes.get(node_id)
        if node is None:
            raise SourceLifecycleError(f"unknown lifecycle node {node_id!r}")  # noqa: TRY003
        if purpose == "audit":
            return node.reference
        if purpose == "generation":
            if not node.available:
                raise SourceLifecycleError(f"{node_id!r} is not available for generation")  # noqa: TRY003
            return node.reference
        raise SourceLifecycleError(f"unknown lifecycle purpose {purpose!r}")  # noqa: TRY003

    def active_nodes(self) -> tuple[str, ...]:
        """Return every available node id in deterministic order."""

        return tuple(sorted(node_id for node_id, node in self._nodes.items() if node.available))

    def _register(self, node: LifecycleNode) -> LifecycleNode:
        existing = self._nodes.get(node.node_id)
        if existing is not None:
            if _same_binding(existing, node):
                return existing
            raise SourceLifecycleError(f"{node.node_id!r} is already registered with a different binding")  # noqa: TRY003
        self._nodes[node.node_id] = node
        return node

    def _register_derived(
        self,
        node_id: str,
        kind: Literal["memory", "topic"],
        reference: object,
        parents: Sequence[str],
        policy: LifecyclePolicy,
    ) -> LifecycleNode:
        _require_node_id(node_id)
        if policy not in ("all", "any"):
            raise SourceLifecycleError("derived policy must be 'all' or 'any'")  # noqa: TRY003
        resolved = tuple(parents)
        if not resolved:
            raise SourceLifecycleError("derived nodes need at least one parent")  # noqa: TRY003
        if len(set(resolved)) != len(resolved):
            raise SourceLifecycleError("derived nodes must not repeat a parent")  # noqa: TRY003
        if node_id in resolved:
            raise SourceLifecycleError("a node cannot depend on itself")  # noqa: TRY003
        for parent in resolved:
            _require_node_id(parent)
            if parent not in self._nodes:
                raise SourceLifecycleError(f"unknown lifecycle parent {parent!r}")  # noqa: TRY003
        return self._register(
            LifecycleNode(
                node_id=node_id,
                kind=kind,
                reference=reference,
                parents=resolved,
                policy=policy,
                revision=1,
                sequence=0,
                available=_supports(resolved, policy, self._nodes),
            )
        )

    def _recompute(self) -> None:
        for node in self._nodes.values():
            if node.kind == "source":
                continue
            node.available = _supports(node.parents, node.policy, self._nodes)


def _supports(parents: Sequence[str], policy: LifecyclePolicy, nodes: dict[str, LifecycleNode]) -> bool:
    states = [nodes[parent].available for parent in parents]
    return all(states) if policy == "all" else any(states)


def _same_binding(left: LifecycleNode, right: LifecycleNode) -> bool:
    return (
        left.kind == right.kind
        and left.parents == right.parents
        and left.policy == right.policy
        and left.revision == right.revision
        and left.reference == right.reference
    )


def _require_node_id(value: object) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise SourceLifecycleError("node ids must be non-empty, trimmed strings")  # noqa: TRY003
    return value


_REFERENCE_MODELS: dict[str, type[BaseModel]] = {
    "SourceRef": SourceRef,
    "MemoryCitation": MemoryCitation,
    "ArtifactRef": ArtifactRef,
}


class LifecycleApplication:
    """Expose the Scope-bound Source lifecycle service."""

    def __init__(self, runtime: Any) -> None:
        self._runtime = runtime

    def for_scope(self, scope_id: str, /) -> ScopedLifecycleApplication:
        return ScopedLifecycleApplication(self._runtime, scope_id)


class ScopedLifecycleApplication:
    """Register one Scope's evidence graph and propagate Source availability."""

    def __init__(self, runtime: Any, scope_id: str, /) -> None:
        self._runtime = runtime
        self.scope_id = _require_node_id(scope_id)

    async def register_source(self, node_id: str, reference: SourceRef, /, *, revision: int) -> None:
        """Register one captured Source root after proving it is readable."""

        await self._records().get_source(reference.source_type, reference.source_id)
        await self._mutate(lambda graph: graph.register_source(node_id, reference, revision=revision))

    async def register_memory(
        self,
        node_id: str,
        reference: MemoryCitation,
        parents: Sequence[str],
        policy: LifecyclePolicy,
    ) -> None:
        """Register one Memory entry after proving its citation is readable."""

        from powercontext.builtin.runtime.models import GetMemoryEntryRequest

        await self._memory().get(GetMemoryEntryRequest(citation=reference))
        await self._mutate(lambda graph: graph.register_memory(node_id, reference, parents, policy))

    async def register_topic(
        self,
        node_id: str,
        reference: ArtifactRef,
        parents: Sequence[str],
        policy: LifecyclePolicy,
    ) -> None:
        """Register one Topic Memory revision after proving it is readable."""

        await self._records().get_artifact_revision(reference.family, reference.artifact_id, reference.revision)
        await self._mutate(lambda graph: graph.register_topic(node_id, reference, parents, policy))

    async def apply(self, event: SourceLifecycleEvent, /) -> list[str]:
        """Apply one event, reconcile native state, and return every available node."""

        return await self._mutate(lambda graph: graph.apply(event))

    async def unavailable_topic_keys(self) -> frozenset[tuple[str, str, int]]:
        """Return the Topic Memory revisions that lifecycle state withdraws."""

        graph = await self._load()
        return frozenset(
            (node.reference.family, node.reference.artifact_id, node.reference.revision)
            for node in graph.nodes
            if node.kind == "topic"
            and not node.available
            and isinstance(node.reference, ArtifactRef)
        )

    async def resolve(self, node_id: str, /, *, purpose: str) -> object:
        """Return the original reference for one node and one stated purpose."""

        return (await self._load()).resolve(node_id, purpose=purpose)

    async def active_nodes(self) -> tuple[str, ...]:
        """Return every available node id in deterministic order."""

        return (await self._load()).active_nodes()

    def _records(self) -> Any:
        return self._runtime.records.for_scope(self.scope_id)

    def _memory(self) -> Any:
        return self._runtime.memory.for_scope(self.scope_id)

    async def _mutate(self, mutation: Callable[[SourceLifecycleGraph], object]) -> list[str]:
        graph = await self._load()
        mutation(graph)
        await self._reconcile_memories(graph)
        await self._persist(graph)
        return list(graph.active_nodes())

    async def _load(self) -> SourceLifecycleGraph:
        rows = await self._runtime._records().load_lifecycle_nodes(self.scope_id)
        graph = SourceLifecycleGraph()
        for row in rows:
            reference = _REFERENCE_MODELS[row.reference_model].model_validate(row.reference)
            if row.kind == "source":
                graph.register_source(row.node_id, reference, revision=row.revision)
            elif row.kind == "memory":
                graph.register_memory(row.node_id, reference, row.parents, row.policy)
            else:
                graph.register_topic(row.node_id, reference, row.parents, row.policy)
        _restore_source_states(graph, rows)
        for row in rows:
            if row.current_reference is not None and row.current_reference_model is not None:
                graph._nodes[row.node_id].current_reference = _REFERENCE_MODELS[
                    row.current_reference_model
                ].model_validate(row.current_reference)
        return graph

    async def _persist(self, graph: SourceLifecycleGraph) -> None:
        rows = tuple(_stored_node(node) for node in graph.nodes)
        await self._runtime._records().replace_lifecycle_nodes(self.scope_id, rows)

    async def _reconcile_memories(self, graph: SourceLifecycleGraph) -> None:
        memory = self._memory()
        page = await memory.list(include_inactive=True)
        states = {row.entry.entry_id: row.state for row in page.entries}
        current_ref = page.memory_ref
        for node in graph.nodes:
            if node.kind != "memory":
                continue
            citation = node.current_reference if node.current_reference is not None else node.reference
            if not isinstance(citation, MemoryCitation):
                raise SourceLifecycleError(f"{node.node_id!r} lost its Memory citation")  # noqa: TRY003
            if current_ref is not None and citation.memory_ref != current_ref:
                # Every retire/reactivate publishes a new Memory revision, so the
                # next node must cite the current head rather than its stale one.
                citation = citation.model_copy(update={"memory_ref": current_ref})
                node.current_reference = citation
            state = states.get(citation.entry_id)
            if node.available and state != "active":
                result = await memory.reactivate(
                    ReactivateMemoryEntryRequest(citation=citation, reason="source lifecycle")
                )
            elif not node.available and state == "active":
                result = await memory.retire(RetireMemoryEntryRequest(citation=citation, reason="source lifecycle"))
            else:
                continue
            current_ref = result.memory_ref
            states[citation.entry_id] = "active" if node.available else "inactive"
            if result.entry is not None:
                node.current_reference = result.entry.citation.model_copy(update={"memory_ref": current_ref})


def _restore_source_states(graph: SourceLifecycleGraph, rows: Sequence[StoredLifecycleNode]) -> None:
    for row in rows:
        if row.kind != "source":
            continue
        node = graph._nodes[row.node_id]
        node.sequence = row.sequence
        node.available = row.available
        node.reason = row.reason
    graph._recompute()


def _stored_node(node: LifecycleNode) -> StoredLifecycleNode:
    reference = node.reference
    model_name = type(reference).__name__
    if model_name not in _REFERENCE_MODELS:
        raise SourceLifecycleError(f"unsupported lifecycle reference {model_name!r}")  # noqa: TRY003
    return StoredLifecycleNode(
        node_id=node.node_id,
        kind=node.kind,
        parents=node.parents,
        policy=node.policy,
        revision=node.revision,
        sequence=node.sequence,
        available=node.available,
        reason=node.reason,
        reference_model=model_name,
        reference=reference.model_dump(mode="json"),
        current_reference_model=None if node.current_reference is None else type(node.current_reference).__name__,
        current_reference=None
        if node.current_reference is None
        else node.current_reference.model_dump(mode="json"),
    )


__all__ = [
    "LifecycleApplication",
    "LifecycleNode",
    "LifecyclePolicy",
    "ScopedLifecycleApplication",
    "SourceLifecycleError",
    "SourceLifecycleEvent",
    "SourceLifecycleGraph",
]
