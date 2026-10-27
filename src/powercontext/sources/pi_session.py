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

A Pi v3 session is a JSONL graph linked through ``parentId``. The selected leaf
and its ancestors form the historical path this module exposes as Sources. Every
visible event keeps its complete original record; graph control nodes only
preserve connectivity between evidence events.
"""

from __future__ import annotations

import hashlib
import json
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
)
from powercontext.sources.definitions import AdapterSourceDefinition
from powercontext.sources.models import Source, SourceMaterialization
from powercontext.sources.projections import TEXT_EVIDENCE_PROJECTION_KEY, TextEvidence

PI_EVENT_SOURCE_NAME = "pi-event"
PI_EVENT_SOURCE_VERSION = "1"
PI_SESSION_CONNECTOR_NAME = "pi-session-history"
PI_SESSION_CONNECTOR_VERSION = "1"
IDENTITY_PREFIX = "pi"
CHECKPOINT_VERSION = 1


class PiSessionInput(BaseModel):
    """One complete, offline Pi v3 JSONL session archive."""

    content: bytes
    leaf_id: str | None = None


class PiEventInput(BaseModel):
    """One visible event on the selected branch, with its complete record."""

    session_id: str
    event_id: str
    parent_id: str | None
    event_type: str
    role: str
    content: str
    record: dict[str, JsonValue]


class PiEventSource(Source):
    """A captured session event usable as textual Artifact evidence."""

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
    """Resolve and read Pi event Sources for the ``pi-event`` Definition."""

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
    """Expose one visible event as ordered JSON text evidence."""

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
    """Parse the complete archive and return the selected branch's events.

    Every record is validated before any event is returned, so malformed
    archives never yield a partial branch.
    """

    document = _parse_document(value.content, value.leaf_id)
    return document.events


def pi_event_source_id(session_id: str, event_id: str, /) -> str:
    """Build the delimiter-safe identity of one captured Pi event."""

    _require_identity_part("session_id", session_id)
    _require_identity_part("event_id", event_id)
    readable = f"{IDENTITY_PREFIX}:{quote(session_id, safe='')}:{quote(event_id, safe='')}"
    if len(readable) <= MAX_SOURCE_ID_LENGTH:
        _require_source_id(readable)
        return readable
    digest = hashlib.sha256(f"{session_id}\x00{event_id}".encode()).hexdigest()
    source_id = f"{IDENTITY_PREFIX}:{digest}"
    _require_source_id(source_id)
    return source_id


class PiSessionConnector:
    """Incrementally capture one Pi session file through the Connector contract."""

    name = PI_SESSION_CONNECTOR_NAME
    version = PI_SESSION_CONNECTOR_VERSION
    source_definitions = frozenset({PI_EVENT_SOURCE_NAME})

    def __init__(self, path: str | Path, leaf_id: str | None = None) -> None:
        self._path = Path(path)
        self._leaf_id = leaf_id

    async def run(self, session: ConnectorRunSession, /) -> ConnectorRunCompletion:
        document = _parse_document(self._path.read_bytes(), self._leaf_id)
        imported = _restore_checkpoint(session.checkpoint, document.session_id)
        hashes = {event_id: _record_fingerprint(record) for event_id, record in document.records.items()}
        for event_id, fingerprint in imported.items():
            if event_id not in hashes:
                raise ValueError(f"imported Pi event {event_id!r} is missing from the session")  # noqa: TRY003
            if hashes[event_id] != fingerprint:
                raise ValueError(f"imported Pi event {event_id!r} was rewritten")  # noqa: TRY003
        for event in document.events:
            if event.event_id in imported:
                continue
            await session.submit(event.event_id, PI_EVENT_SOURCE_NAME, event)
            imported[event.event_id] = hashes[event.event_id]
        checkpoint = {
            "version": CHECKPOINT_VERSION,
            "session_id": document.session_id,
            "events": dict(sorted(imported.items())),
        }
        return ConnectorRunCompletion(status=ConnectorRunStatus.COMPLETE, checkpoint=checkpoint)


class _ParsedSession:
    """Validated graph plus the selected branch's visible events."""

    __slots__ = ("session_id", "records", "events")

    def __init__(
        self,
        session_id: str,
        records: dict[str, dict[str, JsonValue]],
        events: tuple[PiEventInput, ...],
    ) -> None:
        self.session_id = session_id
        self.records = records
        self.events = events


def _parse_document(content: bytes, leaf_id: str | None) -> _ParsedSession:
    text = _decode_document(content)
    rows = _read_rows(text)
    if not rows:
        raise ValueError("Pi session is empty")  # noqa: TRY003
    header = rows[0]
    if not isinstance(header, dict):
        raise ValueError("Pi session header must be a JSON object")  # noqa: TRY003
    if header.get("type") != "session":
        raise ValueError("Pi session must start with a session record")  # noqa: TRY003
    if header.get("version") != 3:
        raise ValueError("Pi session must use version 3")  # noqa: TRY003
    session_id = header.get("id")
    if not isinstance(session_id, str) or not session_id:
        raise ValueError("Pi session id must be a non-empty string")  # noqa: TRY003

    records: dict[str, dict[str, JsonValue]] = {}
    parents: dict[str, str | None] = {}
    order: list[str] = []
    seen: set[str] = set()
    for record in rows[1:]:
        if not isinstance(record, dict):
            raise ValueError("Pi session records must be JSON objects")  # noqa: TRY003
        event_id = record.get("id")
        if not isinstance(event_id, str) or not event_id:
            raise ValueError("Pi session record id must be a non-empty string")  # noqa: TRY003
        if event_id in seen:
            raise ValueError(f"duplicate Pi session record id {event_id!r}")  # noqa: TRY003
        parent_id = record.get("parentId")
        if parent_id is not None:
            if not isinstance(parent_id, str) or parent_id not in seen:
                raise ValueError(f"Pi session record {event_id!r} has an unknown parent")  # noqa: TRY003
        seen.add(event_id)
        records[event_id] = record
        parents[event_id] = parent_id
        order.append(event_id)

    if leaf_id is None:
        if not order:
            return _ParsedSession(session_id, records, ())
        leaf_id = order[-1]
    elif leaf_id not in records:
        raise ValueError(f"Pi session leaf {leaf_id!r} does not exist")  # noqa: TRY003

    chain = _ancestor_chain(leaf_id, parents)
    events: list[PiEventInput] = []
    for event_id in chain:
        event = _event_from_record(session_id, records[event_id])
        if event is not None:
            events.append(event)
    return _ParsedSession(session_id, records, tuple(events))


def _ancestor_chain(leaf_id: str, parents: dict[str, str | None]) -> list[str]:
    chain: list[str] = []
    current: str | None = leaf_id
    while current is not None:
        chain.append(current)
        current = parents.get(current)
    chain.reverse()
    return chain


def _event_from_record(session_id: str, record: dict[str, JsonValue]) -> PiEventInput | None:
    event_type = record.get("type")
    if not isinstance(event_type, str):
        return None
    if event_type == "message":
        message = record.get("message")
        if not isinstance(message, dict):
            return None
        role = message.get("role")
        if not isinstance(role, str) or not role:
            return None
        body = _message_body(role, message)
    elif event_type in {"compaction", "branch_summary"}:
        summary = record.get("summary")
        if not isinstance(summary, str) or not summary:
            return None
        role = event_type
        body = summary
    elif event_type == "custom_message":
        body = _visible_text(record.get("content"))
        if not body:
            return None
        role = "custom"
    else:
        return None
    if not body:
        return None
    return PiEventInput(
        session_id=session_id,
        event_id=_record_id(record),
        parent_id=_parent_id(record),
        event_type=event_type,
        role=role,
        content=f"{role}: {body}",
        record=record,
    )


def _message_body(role: str, message: dict[str, Any]) -> str:
    if role == "toolResult":
        tool_name = message.get("toolName")
        tool_call_id = message.get("toolCallId")
        name = tool_name if isinstance(tool_name, str) and tool_name else "unknown"
        call_id = tool_call_id if isinstance(tool_call_id, str) else ""
        return f"[tool_result {name} {call_id}] {_visible_text(message.get('content'))}"
    if role == "bashExecution":
        command = message.get("command")
        output = message.get("output")
        command_text = command if isinstance(command, str) else ""
        output_text = _visible_text(output) if isinstance(output, list) else (
            output if isinstance(output, str) else ""
        )
        return f"$ {command_text}\n{output_text}"
    return _message_parts(message.get("content"))


def _visible_text(content: object) -> str:
    """Join visible text blocks, leaving thinking, images and controls out."""

    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    parts: list[str] = []
    for block in content:
        if not isinstance(block, dict):
            continue
        block_type = block.get("type")
        if block_type == "text" and isinstance(block.get("text"), str):
            parts.append(block["text"])
    return "\n".join(parts)


def _message_parts(content: object) -> str:
    """Render one message body, including its tool calls in block order."""

    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    parts: list[str] = []
    for block in content:
        if not isinstance(block, dict):
            continue
        block_type = block.get("type")
        if block_type == "text" and isinstance(block.get("text"), str):
            parts.append(block["text"])
        elif block_type == "toolCall":
            name = block.get("name")
            call_id = block.get("id")
            name_text = name if isinstance(name, str) and name else "unknown"
            call_id_text = call_id if isinstance(call_id, str) else ""
            arguments = _compact_json(block.get("arguments", {}))
            parts.append(f"[tool_call {name_text} {call_id_text}] {arguments}")
    return "\n".join(parts)


def _compact_json(value: object) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    except (TypeError, ValueError):
        return json.dumps(str(value), ensure_ascii=False)


def _record_fingerprint(record: dict[str, JsonValue]) -> str:
    encoded = json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode()).hexdigest()


def _restore_checkpoint(checkpoint: JsonValue | None, session_id: str) -> dict[str, str]:
    if not isinstance(checkpoint, dict):
        return {}
    if checkpoint.get("version") != CHECKPOINT_VERSION:
        return {}
    if checkpoint.get("session_id") != session_id:
        return {}
    events = checkpoint.get("events")
    if not isinstance(events, dict):
        return {}
    restored: dict[str, str] = {}
    for event_id, fingerprint in events.items():
        if isinstance(event_id, str) and isinstance(fingerprint, str):
            restored[event_id] = fingerprint
    return restored


def _decode_document(content: bytes) -> str:
    try:
        return bytes(content).decode("utf-8-sig")
    except UnicodeDecodeError as error:
        raise ValueError("Pi session must be strict UTF-8 with an optional leading BOM") from error  # noqa: TRY003


def _read_rows(text: str) -> list[object]:
    rows: list[object] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as error:
            raise ValueError(f"malformed Pi session JSON on line {line_number}: {error}") from error  # noqa: TRY003
    return rows


def _record_id(record: dict[str, JsonValue]) -> str:
    event_id = record.get("id")
    return event_id if isinstance(event_id, str) else ""


def _parent_id(record: dict[str, JsonValue]) -> str | None:
    parent_id = record.get("parentId")
    return parent_id if isinstance(parent_id, str) else None


def _require_identity_part(field: str, value: object) -> str:
    if not isinstance(value, str) or not value:
        raise InvalidSourceReferenceError(field, "must be a non-empty string")
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
    version=PI_EVENT_SOURCE_VERSION,
    projections=(PiEventTextEvidenceProjection(),),
)

__all__ = [
    "PI_EVENT_SOURCE_ADAPTER",
    "PI_EVENT_SOURCE_DEFINITION",
    "PI_EVENT_SOURCE_NAME",
    "PI_EVENT_SOURCE_VERSION",
    "PI_SESSION_CONNECTOR_NAME",
    "PI_SESSION_CONNECTOR_VERSION",
    "PiEventInput",
    "PiEventSource",
    "PiEventSourceAdapter",
    "PiEventTextEvidenceProjection",
    "PiSessionConnector",
    "PiSessionInput",
    "parse_pi_session",
    "pi_event_source_id",
]
