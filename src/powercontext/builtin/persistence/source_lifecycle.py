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

"""Persist one Scope's Source lifecycle dependency graph."""

from __future__ import annotations

import json

from pydantic import BaseModel, ConfigDict, JsonValue, TypeAdapter
from sqlalchemy import delete, insert, select
from sqlalchemy.ext.asyncio import AsyncConnection

from powercontext.builtin.persistence.errors import InvalidRepositoryArgumentError
from powercontext.builtin.persistence.tables import SOURCE_LIFECYCLE_NODES_TABLE
from powercontext.limits import MAX_SCOPE_ID_LENGTH, MAX_SOURCE_ID_LENGTH

_JSON_OBJECT = TypeAdapter(dict[str, JsonValue])


class StoredLifecycleNode(BaseModel):
    """One durable lifecycle node, independent of its runtime reference type."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    node_id: str
    kind: str
    parents: tuple[str, ...]
    policy: str
    revision: int
    sequence: int
    available: bool
    reason: str
    reference_model: str
    reference: dict[str, JsonValue]
    current_reference_model: str | None = None
    current_reference: dict[str, JsonValue] | None = None


class SourceLifecycleNodeRepository:
    """Load and replace the whole graph of one Scope in one transaction."""

    async def load_scope(self, connection: AsyncConnection, scope_id: str, /) -> tuple[StoredLifecycleNode, ...]:
        _require_scope_id(scope_id)
        table = SOURCE_LIFECYCLE_NODES_TABLE
        rows = (
            await connection.execute(
                select(table)
                .where(table.c.scope_id == scope_id)
                .order_by(table.c.ordinal, table.c.node_id)
            )
        ).mappings()
        return tuple(
            StoredLifecycleNode(
                node_id=row["node_id"],
                kind=row["kind"],
                parents=tuple(json.loads(row["parents"])),
                policy=row["policy"],
                revision=int(row["revision"]),
                sequence=int(row["sequence"]),
                available=bool(row["available"]),
                reason=row["reason"],
                reference_model=row["reference_model"],
                reference=_JSON_OBJECT.validate_python(json.loads(row["reference"])),
                current_reference_model=row["current_reference_model"],
                current_reference=None
                if row["current_reference"] is None
                else _JSON_OBJECT.validate_python(json.loads(row["current_reference"])),
            )
            for row in rows
        )

    async def replace_scope(
        self,
        connection: AsyncConnection,
        scope_id: str,
        nodes: tuple[StoredLifecycleNode, ...],
        /,
    ) -> None:
        _require_scope_id(scope_id)
        table = SOURCE_LIFECYCLE_NODES_TABLE
        await connection.execute(delete(table).where(table.c.scope_id == scope_id))
        for ordinal, node in enumerate(nodes):
            _require_node(node)
            await connection.execute(
                insert(table).values(
                    scope_id=scope_id,
                    node_id=node.node_id,
                    ordinal=ordinal,
                    kind=node.kind,
                    parents=json.dumps(list(node.parents)),
                    policy=node.policy,
                    revision=node.revision,
                    sequence=node.sequence,
                    available=node.available,
                    reason=node.reason,
                    reference_model=node.reference_model,
                    reference=json.dumps(node.reference, sort_keys=True),
                    current_reference_model=node.current_reference_model,
                    current_reference=None
                    if node.current_reference is None
                    else json.dumps(node.current_reference, sort_keys=True),
                )
            )


def _require_scope_id(value: object) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise InvalidRepositoryArgumentError("scope_id", "must be a non-empty trimmed string")
    if len(value) > MAX_SCOPE_ID_LENGTH:
        raise InvalidRepositoryArgumentError("scope_id", "must not exceed the scope id length limit")
    return value


def _require_node(node: StoredLifecycleNode) -> None:
    if not node.node_id or node.node_id != node.node_id.strip() or len(node.node_id) > MAX_SOURCE_ID_LENGTH:
        raise InvalidRepositoryArgumentError("node_id", "must be a bounded trimmed identity")
    if node.kind not in {"source", "memory", "topic"}:
        raise InvalidRepositoryArgumentError("kind", "must be source, memory, or topic")
    if node.policy not in {"all", "any"}:
        raise InvalidRepositoryArgumentError("policy", "must be all or any")


__all__ = ["SourceLifecycleNodeRepository", "StoredLifecycleNode"]
