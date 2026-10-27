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

"""Deterministic context packs and prefix-reusing incremental deltas.

A client that already holds the previous pack only needs the shared prefix
count and the appended suffix: the receiver rebuilds the target pack from its
own copy of the base pack, so context transported between turns stays byte for
byte identical to a full ``prepare``.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any

PACK_SCHEMA = "powercontext.context-pack.v1"
DELTA_SCHEMA = "powercontext.context-pack-delta.v1"
DEFAULT_CHUNK_BYTES = 512

_PACK_FIELDS = frozenset({"schema", "items", "digest", "byte_count"})
_DELTA_FIELDS = frozenset({"schema", "base_digest", "target_digest", "prefix_items", "appended"})
_DIGEST_PATTERN = re.compile(r"sha256:[0-9a-f]{64}")
_CONTINUATION_MASK = 0xC0
_CONTINUATION_VALUE = 0x80


class ContextPackError(ValueError):
    """Raised when a context pack or delta violates the wire contract."""


@dataclass(frozen=True, slots=True)
class ContextPack:
    """One complete, checksummed context body split into transport chunks."""

    items: tuple[bytes, ...]
    digest: str
    byte_count: int

    @classmethod
    def build(cls, items: tuple[bytes, ...] | list[bytes], /) -> ContextPack:
        """Build a pack from already-chunked bytes."""

        chunks = _validate_chunks(items, "items")
        digest = _pack_digest(chunks)
        return cls(items=chunks, digest=digest, byte_count=_pack_byte_count(chunks, digest))

    @classmethod
    def from_content(cls, content: str, /, *, chunk_bytes: int = DEFAULT_CHUNK_BYTES) -> ContextPack:
        """Split one UTF-8 body into deterministic, boundary-safe chunks."""

        if not isinstance(content, str):
            raise ContextPackError("context content must be a string")  # noqa: TRY003
        if not isinstance(chunk_bytes, int) or isinstance(chunk_bytes, bool) or chunk_bytes < 1:
            raise ContextPackError("chunk size must be a positive integer")  # noqa: TRY003
        return cls.build(_split_content(content, chunk_bytes))

    @classmethod
    def from_bytes(cls, payload: bytes, /) -> ContextPack:
        """Decode and verify one canonical pack document."""

        decoded = _load_document(payload, _PACK_FIELDS, PACK_SCHEMA)
        items = _decode_chunks(decoded["items"], "items")
        pack = cls.build(items)
        if decoded["digest"] != pack.digest:
            raise ContextPackError("context pack digest does not match its items")  # noqa: TRY003
        if not isinstance(decoded["byte_count"], int) or isinstance(decoded["byte_count"], bool):
            raise ContextPackError("context pack byte_count must be an integer")  # noqa: TRY003
        if decoded["byte_count"] != pack.byte_count:
            raise ContextPackError("context pack byte_count does not match its encoding")  # noqa: TRY003
        return pack

    def to_bytes(self) -> bytes:
        """Return the canonical wire encoding of this pack."""

        return _canonical_json({
            "schema": PACK_SCHEMA,
            "items": [_encode_chunk(item) for item in self.items],
            "digest": self.digest,
            "byte_count": self.byte_count,
        })


@dataclass(frozen=True, slots=True)
class ContextPackDelta:
    """The suffix one target pack adds after a retained base prefix."""

    base_digest: str
    target_digest: str
    prefix_items: int
    appended: tuple[bytes, ...]

    @classmethod
    def between(cls, base: ContextPack, target: ContextPack, /) -> ContextPackDelta:
        """Keep the longest common chunk prefix and append the target suffix."""

        if not isinstance(base, ContextPack) or not isinstance(target, ContextPack):
            raise ContextPackError("delta endpoints must be context packs")  # noqa: TRY003
        prefix = 0
        limit = min(len(base.items), len(target.items))
        while prefix < limit and base.items[prefix] == target.items[prefix]:
            prefix += 1
        return cls(
            base_digest=base.digest,
            target_digest=target.digest,
            prefix_items=prefix,
            appended=target.items[prefix:],
        )

    @classmethod
    def from_bytes(cls, payload: bytes, /) -> ContextPackDelta:
        """Decode and verify one canonical delta document."""

        decoded = _load_document(payload, _DELTA_FIELDS, DELTA_SCHEMA)
        prefix = decoded["prefix_items"]
        if not isinstance(prefix, int) or isinstance(prefix, bool) or prefix < 0:
            raise ContextPackError("delta prefix_items must be a non-negative integer")  # noqa: TRY003
        base_digest = _validate_digest(decoded["base_digest"], "base_digest")
        target_digest = _validate_digest(decoded["target_digest"], "target_digest")
        delta = cls(
            base_digest=base_digest,
            target_digest=target_digest,
            prefix_items=prefix,
            appended=_decode_chunks(decoded["appended"], "appended"),
        )
        return delta

    @property
    def byte_count(self) -> int:
        """Return the wire size of this delta, protocol fields included."""

        return len(self.to_bytes())

    def to_bytes(self) -> bytes:
        """Return the canonical wire encoding of this delta."""

        return _canonical_json({
            "schema": DELTA_SCHEMA,
            "base_digest": self.base_digest,
            "target_digest": self.target_digest,
            "prefix_items": self.prefix_items,
            "appended": [_encode_chunk(item) for item in self.appended],
        })


def apply_context_pack_delta(base: ContextPack, delta: ContextPackDelta, /) -> ContextPack:
    """Rebuild the target pack from a verified base and delta."""

    if not isinstance(base, ContextPack) or not isinstance(delta, ContextPackDelta):
        raise ContextPackError("delta application requires a context pack and a delta")  # noqa: TRY003
    if base.digest != delta.base_digest:
        raise ContextPackError("delta was computed against a different base pack")  # noqa: TRY003
    if delta.prefix_items > len(base.items):
        raise ContextPackError("delta retains more chunks than the base pack holds")  # noqa: TRY003
    rebuilt = ContextPack.build((*base.items[: delta.prefix_items], *delta.appended))
    if rebuilt.digest != delta.target_digest:
        raise ContextPackError("rebuilt pack does not match the delta target digest")  # noqa: TRY003
    return rebuilt


def _split_content(content: str, chunk_bytes: int) -> tuple[bytes, ...]:
    data = content.encode("utf-8")
    chunks: list[bytes] = []
    start = 0
    while start < len(data):
        end = min(start + chunk_bytes, len(data))
        while end < len(data) and (data[end] & _CONTINUATION_MASK) == _CONTINUATION_VALUE:
            end -= 1
        if end == start:
            raise ContextPackError("content cannot be split within the requested chunk size")  # noqa: TRY003
        chunks.append(data[start:end])
        start = end
    return tuple(chunks)


def _pack_digest(items: tuple[bytes, ...]) -> str:
    unsigned = {"schema": PACK_SCHEMA, "items": [_encode_chunk(item) for item in items]}
    return "sha256:" + hashlib.sha256(_canonical_json(unsigned)).hexdigest()


def _pack_byte_count(items: tuple[bytes, ...], digest: str) -> int:
    count = 0
    while True:
        encoded = _canonical_json({
            "schema": PACK_SCHEMA,
            "items": [_encode_chunk(item) for item in items],
            "digest": digest,
            "byte_count": count,
        })
        if len(encoded) == count:
            return count
        count = len(encoded)


def _canonical_json(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _encode_chunk(item: bytes) -> str:
    return base64.b64encode(item).decode("ascii")


def _decode_chunks(value: Any, field: str) -> tuple[bytes, ...]:
    if not isinstance(value, list):
        raise ContextPackError(f"{field} must be a list of base64 strings")  # noqa: TRY003
    chunks: list[bytes] = []
    for entry in value:
        if not isinstance(entry, str):
            raise ContextPackError(f"{field} must contain base64 strings")  # noqa: TRY003
        try:
            chunks.append(base64.b64decode(entry, validate=True))
        except (binascii.Error, ValueError) as error:
            raise ContextPackError(f"{field} contains an invalid base64 chunk") from error  # noqa: TRY003
    return tuple(chunks)


def _validate_chunks(items: object, field: str) -> tuple[bytes, ...]:
    if isinstance(items, (bytes, bytearray, memoryview)) or not isinstance(items, (tuple, list)):
        raise ContextPackError(f"{field} must be a tuple of bytes")  # noqa: TRY003
    chunks: list[bytes] = []
    for item in items:
        if not isinstance(item, (bytes, bytearray, memoryview)):
            raise ContextPackError(f"{field} must contain bytes")  # noqa: TRY003
        chunks.append(bytes(item))
    return tuple(chunks)


def _load_document(payload: bytes, fields: frozenset[str], schema: str) -> dict[str, Any]:
    if not isinstance(payload, (bytes, bytearray, memoryview)):
        raise ContextPackError("wire payload must be bytes")  # noqa: TRY003
    try:
        decoded = json.loads(bytes(payload).decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ContextPackError("wire payload is not valid UTF-8 JSON") from error  # noqa: TRY003
    if not isinstance(decoded, dict):
        raise ContextPackError("wire payload must be a JSON object")  # noqa: TRY003
    if set(decoded) != fields:
        raise ContextPackError("wire payload must contain exactly the declared fields")  # noqa: TRY003
    if decoded.get("schema") != schema:
        raise ContextPackError("wire payload schema is not supported")  # noqa: TRY003
    return decoded


def _validate_digest(value: object, field: str) -> str:
    if not isinstance(value, str) or not _DIGEST_PATTERN.fullmatch(value):
        raise ContextPackError(f"{field} must be a sha256 digest")  # noqa: TRY003
    return value


__all__ = [
    "DEFAULT_CHUNK_BYTES",
    "DELTA_SCHEMA",
    "PACK_SCHEMA",
    "ContextPack",
    "ContextPackDelta",
    "ContextPackError",
    "apply_context_pack_delta",
]
