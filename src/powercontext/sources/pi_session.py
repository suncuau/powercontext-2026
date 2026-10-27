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

"""Capture the visible events of one selected Pi session branch.

A Pi v3 session archive is a JSONL node graph: every record points at its
parent, and a leaf plus its ancestors define the history path a user was
looking at. This module turns that path into immutable per-event Sources and
provides a connector that imports only the events a previous run has not yet
captured.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import quote

from pydantic import BaseModel, JsonValue, field_validator, model_validator

from powercontext.errors import InvalidSourceReferenceError
from powercontext.limits import MAX_SOURCE_ID_LENGTH
from powercontext.sources.connectors import (
    ConnectorRunCompletion,
    ConnectorRunSession,
    ConnectorRunStatus,
    ConnectorSubmissionStatus,
)
from powercontext.sources.definitions import AdapterSourceDefinition
from powercontext.sources.models import Source, SourceMaterialization
from powercontext.sources.projections import TEXT_EVIDENCE_PROJECTION_KEY, TextEvidence

PI_EVENT_SOURCE_NAME = "pi-event"
PI_SESSION_CONNECTOR_NAME = "pi-session-history"
PI_SESSION_VERSION = "1"
PI_SESSION_FORMAT_VERSION = 3
CHECKPOINT_VERSION = 1

_MESSAGE = "message"
_TOOL_RESULT_ROLE = "toolResult"
_BASH_ROLE = "bashExecution"
_TEXT_BLOCK = "text"
_TOOL_CALL_BLOCK = "toolCall"


class PiSessionInput(BaseModel):
    """One complete Pi v3 JSONL archive and an optional selected leaf."""

    content: bytes
    leaf_id: str | None = None


class PiEventInput(BaseModel):
    """One visible event kept verbatim from a Pi session archive."""

    session_id: str
    event_id: str
    parent_id: str | None
    event_type: str
    role: str
    content: str
    record: dict[str, JsonValue]


class PiEventSource(Source):
    """A captured Pi session event that is usable as Artifact evidence."""

    source_id: str
    session_id: str
    event_id: str
    parent_id: str | None
    event_type: str
    role: str
    content: str
    record: dict[str, JsonValue]

    @field_validator("source_id")
    @classmethod
    def validate_source_id(cls, value: str) -> str:
        _require_source_id(value)
        return value

    @model_validator(mode="after")
    def validate_identity(self) -> PiEventSource:
        if self.name != self.source_id:
            raise ValueError("event Sources must be named by their identity")  # noqa: TRY003
        return self


class PiEventSourceAdapter:
    """Resolve and read event Sources for the ``pi-event`` Definition."""

    input_class = PiEventInput
    name = PI_EVENT_SOURCE_NAME
    source_class = PiEventSource

    async def resolve(self, value: PiEventInput, /) -> PiEventSource:
        source_id = pi_event_source_id(value.session_id, value.event_id)
        return PiEventSource(
            name=source_id,
            source_id=source_id,
            materialization=SourceMaterialization.CAPTURED,
            session_id=value.session_id,
            event_id=value.event_id,
            parent_id=value.parent_id,
            event_type=value.event_type,
            role=value.role,
            content=value.content,
            record=value.record,
        )

    async def read(self, source: PiEventSource, /) -> PiEventInput:
        return PiEventInput(
            session_id=source.session_id,
            event_id=source.event_id,
            parent_id=source.parent_id,
            event_type=source.event_type,
            role=source.role,
            content=source.content,
            record=source.record,
        )


class PiEventTextEvidenceProjection:
    """Expose one captured event as standard text evidence."""

    name = TEXT_EVIDENCE_PROJECTION_KEY.name
    version = TEXT_EVIDENCE_PROJECTION_KEY.version
    source_class = PiEventSource
    output_class: type[BaseModel] = TextEvidence

    def project(self, source: PiEventSource, /) -> TextEvidence:
        return TextEvidence(
            source_type=PI_EVENT_SOURCE_NAME,
            source_id=source.source_id,
            content=source.content,
            metadata={
                "session_id": source.session_id,
                "event_id": source.event_id,
                "parent_id": source.parent_id,
                "event_type": source.event_type,
                "role": source.role,
            },
        )


def parse_pi_session(value: PiSessionInput, /) -> tuple[PiEventInput, ...]:
    """Return the visible events of the selected branch, ordered root to leaf."""

    graph = _load_graph(value.content)
    return _branch_events(graph, value.leaf_id)


def pi_event_source_id(session_id: str, event_id: str, /) -> str:
    """Build the stable, delimiter-safe identity of one captured event."""

    encoded_session = quote(_require_text("session_id", session_id), safe="")
    encoded_event = quote(_require_text("event_id", event_id), safe="")
    source_id = f"pi:{encoded_session}:{encoded_event}"
    _require_source_id(source_id)
    return source_id


@dataclass(frozen=True, slots=True)
class _SessionGraph:
    """Validated archive nodes keyed by record identity."""

    session_id: str
    nodes: dict[str, dict[str, Any]]
    parents: dict[str, str | None]
    order: tuple[str, ...]

    def leaf(self, leaf_id: str | None) -> str:
        if leaf_id is not None:
            _require_text("leaf_id", leaf_id)
            if leaf_id not in self.nodes:
                raise ValueError(f"leaf {leaf_id!r} is not part of the session archive")  # noqa: TRY003
            return leaf_id
        if not self.order:
            raise ValueError("session archive contains no records after its header")  # noqa: TRY003
        return self.order[-1]

    def path(self, leaf_id: str) -> tuple[str, ...]:
        node_id: str | None = leaf_id
        path: list[str] = []
        while node_id is not None:
            path.append(node_id)
            node_id = self.parents[node_id]
        path.reverse()
        return tuple(path)


def _load_graph(content: bytes) -> _SessionGraph:
    records = _decode_records(content)
    header = records[0]
    if header.get("type") != "session" or header.get("version") != PI_SESSION_FORMAT_VERSION:
        raise ValueError("session archive must start with a version 3 session record")  # noqa: TRY003
    session_id = _require_text("session_id", header.get("id"))
    nodes: dict[str, dict[str, Any]] = {}
    parents: dict[str, str | None] = {}
    order: list[str] = []
    for record in records[1:]:
        event_id = _require_text("event_id", record.get("id"))
        if event_id in nodes:
            raise ValueError(f"duplicate event id {event_id!r} in session archive")  # noqa: TRY003
        parent_id = record.get("parentId")
        if parent_id is not None:
            _require_text("parent_id", parent_id)
            if parent_id not in nodes:
                raise ValueError(f"event {event_id!r} references unknown parent {parent_id!r}")  # noqa: TRY003
        nodes[event_id] = record
        parents[event_id] = parent_id
        order.append(event_id)
    return _SessionGraph(session_id=session_id, nodes=nodes, parents=parents, order=tuple(order))


def _decode_records(content: bytes) -> list[dict[str, Any]]:
    try:
        text = bytes(content).decode("utf-8-sig")
    except UnicodeDecodeError as error:
        raise ValueError("session archive must be strict UTF-8 with an optional leading BOM") from error  # noqa: TRY003

    records: list[dict[str, Any]] = []
    for number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"session archive line {number} is not valid JSON") from error  # noqa: TRY003
        if not isinstance(record, dict):
            raise ValueError(f"session archive line {number} is not a JSON object")  # noqa: TRY003, TRY004
        records.append(record)
    if not records:
        raise ValueError("session archive is empty")  # noqa: TRY003
    return records


def _branch_events(graph: _SessionGraph, leaf_id: str | None) -> tuple[PiEventInput, ...]:
    events: list[PiEventInput] = []
    for node_id in graph.path(graph.leaf(leaf_id)):
        event = _event_for(graph.session_id, graph.nodes[node_id])
        if event is not None:
            events.append(event)
    return tuple(events)


def _event_for(session_id: str, record: dict[str, Any]) -> PiEventInput | None:
    event_type = record.get("type")
    if event_type == _MESSAGE:
        rendered = _message_event(record.get("message"))
    elif event_type == "compaction" or event_type == "branch_summary":
        summary = record.get("summary")
        body = summary if isinstance(summary, str) else ""
        rendered = (str(event_type), body) if body else None
    elif event_type == "custom_message":
        body = _render_text_blocks(record.get("content"))
        rendered = ("custom", body) if body else None
    else:
        # Control nodes keep their place in the graph but carry no text.
        rendered = None
    if rendered is None:
        return None
    role, body = rendered
    return PiEventInput(
        session_id=session_id,
        event_id=_require_text("event_id", record.get("id")),
        parent_id=record.get("parentId"),
        event_type=str(event_type),
        role=role,
        content=f"{role}: {body}",
        record=record,
    )


def _message_event(message: object) -> tuple[str, str] | None:
    if not isinstance(message, dict):
        return None
    role = message.get("role")
    if not isinstance(role, str) or not role:
        return None
    if role == _TOOL_RESULT_ROLE:
        name = _string_or_empty(message.get("toolName"))
        call_id = _string_or_empty(message.get("toolCallId"))
        body = f"[tool_result {name} {call_id}] {_render_text_blocks(message.get('content'))}"
        return role, body
    if role == _BASH_ROLE:
        body = _render_bash(message)
        return (role, body) if body else None
    body = _render_content(message.get("content"))
    return (role, body) if body else None


def _render_bash(message: dict[str, Any]) -> str:
    command = _string_or_empty(message.get("command"))
    output = _string_or_empty(message.get("output"))
    if not command and not output:
        return ""
    return f"$ {command}\n{output}" if output else f"$ {command}"


def _render_content(content: object) -> str:
    if isinstance(content, str):
        return content
    return _render_blocks(content, tool_calls=True)


def _render_text_blocks(content: object) -> str:
    if isinstance(content, str):
        return content
    return _render_blocks(content, tool_calls=False)


def _render_blocks(content: object, *, tool_calls: bool) -> str:
    if not isinstance(content, list):
        return ""
    rendered: list[str] = []
    for block in content:
        if isinstance(block, str):
            text = block
        elif isinstance(block, dict) and block.get("type") == _TEXT_BLOCK:
            text = _string_or_empty(block.get("text"))
        elif isinstance(block, dict) and tool_calls and block.get("type") == _TOOL_CALL_BLOCK:
            text = _render_tool_call(block)
        else:
            # Thinking, image, and other non-visible blocks never become text.
            continue
        if text:
            rendered.append(text)
    return "\n".join(rendered)


def _render_tool_call(block: dict[str, Any]) -> str:
    name = _string_or_empty(block.get("name"))
    call_id = _string_or_empty(block.get("id"))
    arguments = block.get("arguments")
    if arguments is None:
        arguments = {}
    payload = json.dumps(arguments, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return f"[tool_call {name} {call_id}] {payload}"


class PiSessionConnector:
    """Import the events of one selected Pi branch without re-reading history."""

    name = PI_SESSION_CONNECTOR_NAME
    version = PI_SESSION_VERSION
    source_definitions = frozenset({PI_EVENT_SOURCE_NAME})

    def __init__(self, path: str | Path, leaf_id: str | None = None) -> None:
        self.path = Path(path)
        self.leaf_id = leaf_id

    async def run(self, session: ConnectorRunSession, /) -> ConnectorRunCompletion:
        graph = _load_graph(self.path.read_bytes())
        events = _branch_events(graph, self.leaf_id)
        state = _checkpoint_state(session.checkpoint)
        imported = state.setdefault(graph.session_id, {})

        pending: list[tuple[PiEventInput, str]] = []
        for event in events:
            digest = _record_digest(event.record)
            known = imported.get(event.event_id)
            if known is None:
                pending.append((event, digest))
            elif known != digest:
                raise ValueError(f"event {event.event_id!r} changed after it was imported")  # noqa: TRY003

        for event, digest in pending:
            outcome = await session.submit(event.event_id, PI_EVENT_SOURCE_NAME, event)
            if outcome.status is ConnectorSubmissionStatus.ACCEPTED:
                imported[event.event_id] = digest

        return ConnectorRunCompletion(
            status=ConnectorRunStatus.COMPLETE,
            checkpoint={"version": CHECKPOINT_VERSION, "sessions": state},
        )


def _checkpoint_state(checkpoint: JsonValue | None) -> dict[str, dict[str, str]]:
    if checkpoint is None:
        return {}
    if not isinstance(checkpoint, dict):
        raise ValueError("connector checkpoint must be a JSON object")  # noqa: TRY003, TRY004
    if checkpoint.get("version") != CHECKPOINT_VERSION:
        raise ValueError("connector checkpoint has an unsupported version")  # noqa: TRY003
    sessions = checkpoint.get("sessions")
    if not isinstance(sessions, dict):
        raise ValueError("connector checkpoint must contain a sessions object")  # noqa: TRY003, TRY004
    state: dict[str, dict[str, str]] = {}
    for session_id, imported in sessions.items():
        if not isinstance(session_id, str) or not isinstance(imported, dict):
            raise ValueError("connector checkpoint sessions must map ids to objects")  # noqa: TRY003, TRY004
        state[session_id] = {
            str(event_id): str(digest) for event_id, digest in imported.items() if isinstance(event_id, str)
        }
    return state


def _record_digest(record: dict[str, JsonValue]) -> str:
    encoded = json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


def _string_or_empty(value: object) -> str:
    return value if isinstance(value, str) else ""


def _require_text(field: str, value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"session archive {field} must be a non-empty string")  # noqa: TRY003
    return value


def _require_source_id(value: object) -> str:
    if not isinstance(value, str) or not value:
        raise InvalidSourceReferenceError("source_id", "must be a non-empty string")
    if value != value.strip():
        raise InvalidSourceReferenceError("source_id", "must not contain leading or trailing whitespace")
    if len(value) > MAX_SOURCE_ID_LENGTH:
        raise InvalidSourceReferenceError("source_id", f"must not exceed {MAX_SOURCE_ID_LENGTH} characters")
    return value


PI_EVENT_SOURCE_ADAPTER = PiEventSourceAdapter()
PI_EVENT_SOURCE_DEFINITION = AdapterSourceDefinition(
    PI_EVENT_SOURCE_ADAPTER,
    version=PI_SESSION_VERSION,
    projections=(PiEventTextEvidenceProjection(),),
)

__all__ = [
    "PI_EVENT_SOURCE_ADAPTER",
    "PI_EVENT_SOURCE_DEFINITION",
    "PI_EVENT_SOURCE_NAME",
    "PI_SESSION_CONNECTOR_NAME",
    "PI_SESSION_VERSION",
    "PiEventInput",
    "PiEventSource",
    "PiEventSourceAdapter",
    "PiEventTextEvidenceProjection",
    "PiSessionConnector",
    "PiSessionInput",
    "parse_pi_session",
    "pi_event_source_id",
]
