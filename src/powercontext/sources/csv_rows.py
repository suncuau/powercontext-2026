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

"""Row-level CSV Sources with stable identity and standard text evidence.

A CSV document is captured as one Source per record. Each row keeps the exact
strings read from the document, so a retrieved citation can be resolved back to
the complete original record even after the file is reordered or the process
restarts.
"""

from __future__ import annotations

import csv
import io
import json
from urllib.parse import quote

from pydantic import BaseModel, field_validator, model_validator

from powercontext.errors import InvalidSourceReferenceError
from powercontext.limits import MAX_SOURCE_ID_LENGTH
from powercontext.sources.definitions import AdapterSourceDefinition
from powercontext.sources.models import Source, SourceMaterialization
from powercontext.sources.projections import TEXT_EVIDENCE_PROJECTION_KEY, TextEvidence

CSV_ROW_SOURCE_NAME = "csv-row"
CSV_ROW_SOURCE_VERSION = "1"
IDENTITY_PREFIX = "csv"


class CSVDocumentInput(BaseModel):
    """One complete, offline CSV document to capture row by row."""

    document_id: str
    primary_key: str
    content: bytes


class CSVRowInput(BaseModel):
    """One parsed CSV record, addressed by header and primary-key value."""

    document_id: str
    primary_key: str
    primary_key_value: str
    columns: tuple[str, ...]
    values: tuple[str, ...]


class CSVRowSource(Source):
    """A captured row that is usable as textual Artifact evidence."""

    source_id: str
    document_id: str
    primary_key: str
    primary_key_value: str
    columns: tuple[str, ...]
    values: tuple[str, ...]

    @field_validator("source_id")
    @classmethod
    def validate_source_id(cls, value: str) -> str:
        _require_source_id(value)
        return value

    @model_validator(mode="after")
    def validate_identity(self) -> CSVRowSource:
        if self.name != self.source_id:
            raise ValueError("row Sources must be named by their identity")  # noqa: TRY003
        return self


class CSVRowSourceAdapter:
    """Resolve and read row Sources for the ``csv-row`` Definition."""

    input_class = CSVRowInput
    name = CSV_ROW_SOURCE_NAME
    source_class = CSVRowSource

    async def resolve(self, value: CSVRowInput, /) -> CSVRowSource:
        source_id = csv_row_source_id(value.document_id, value.primary_key_value)
        return CSVRowSource(
            name=source_id,
            source_id=source_id,
            materialization=SourceMaterialization.CAPTURED,
            document_id=value.document_id,
            primary_key=value.primary_key,
            primary_key_value=value.primary_key_value,
            columns=value.columns,
            values=value.values,
        )

    async def read(self, source: CSVRowSource, /) -> CSVRowInput:
        return CSVRowInput(
            document_id=source.document_id,
            primary_key=source.primary_key,
            primary_key_value=source.primary_key_value,
            columns=source.columns,
            values=source.values,
        )


class CSVRowTextEvidenceProjection:
    """Expose one row as ordered JSON without coupling consumers to the Source."""

    name = TEXT_EVIDENCE_PROJECTION_KEY.name
    version = TEXT_EVIDENCE_PROJECTION_KEY.version
    source_class = CSVRowSource
    output_class: type[BaseModel] = TextEvidence

    def project(self, source: CSVRowSource, /) -> TextEvidence:
        row = _row_value(source)
        return TextEvidence(
            source_type=CSV_ROW_SOURCE_NAME,
            source_id=source.source_id,
            content=_serialize_row(row),
            metadata={
                "document_id": source.document_id,
                "primary_key": source.primary_key,
                "primary_key_value": source.primary_key_value,
                "columns": list(source.columns),
                "row": row,
            },
        )


def parse_csv_document(value: CSVDocumentInput, /) -> tuple[CSVRowInput, ...]:
    """Parse one offline CSV document into validated row inputs.

    The whole document is decoded and validated before any row is returned; a
    malformed document raises ``ValueError`` instead of yielding partial rows.
    """

    document_id = _require_document_id(value.document_id)
    primary_key = _require_primary_key(value.primary_key)
    text = _decode_document(value.content)
    reader = csv.reader(io.StringIO(text, newline=""), strict=True)

    try:
        header = next(reader, None)
        if header is None:
            raise ValueError("CSV document is empty")  # noqa: TRY003
        columns = _validate_header(header)
        key_index = _key_index(columns, primary_key)
        rows = _read_records(reader, document_id, primary_key, columns, key_index)
    except csv.Error as error:
        raise ValueError(f"malformed CSV document: {error}") from error  # noqa: TRY003
    return tuple(rows)


def csv_row_source_id(document_id: str, primary_key_value: str, /) -> str:
    """Build the delimiter-safe identity of one captured CSV row."""

    encoded_document = quote(_require_document_id(document_id), safe="")
    encoded_key = quote(_require_primary_key_value(primary_key_value), safe="")
    source_id = f"{IDENTITY_PREFIX}:{encoded_document}:{encoded_key}"
    _require_source_id(source_id)
    return source_id


def _serialize_row(row: dict[str, str]) -> str:
    """Serialize one ordered row exactly as the standard projection requires."""

    return json.dumps(row, ensure_ascii=False, separators=(",", ":"))


def _read_records(
    reader: csv.reader,
    document_id: str,
    primary_key: str,
    columns: tuple[str, ...],
    key_index: int,
) -> list[CSVRowInput]:
    rows: list[CSVRowInput] = []
    seen: set[str] = set()
    for record in reader:
        if not record:
            # A physically empty line is not a record.
            continue
        if len(record) != len(columns):
            raise ValueError(  # noqa: TRY003
                f"CSV record {len(rows) + 1} has {len(record)} fields, expected {len(columns)}"
            )
        primary_key_value = record[key_index]
        _require_primary_key_value(primary_key_value)
        if primary_key_value in seen:
            raise ValueError(f"duplicate primary key value {primary_key_value!r}")  # noqa: TRY003
        seen.add(primary_key_value)
        rows.append(
            CSVRowInput(
                document_id=document_id,
                primary_key=primary_key,
                primary_key_value=primary_key_value,
                columns=columns,
                values=tuple(record),
            )
        )
    return rows


def _validate_header(header: list[str]) -> tuple[str, ...]:
    if not header:
        raise ValueError("CSV header must contain at least one column")  # noqa: TRY003
    columns: list[str] = []
    for field in header:
        if not field.strip():
            raise ValueError(f"CSV header column {field!r} must not be blank")  # noqa: TRY003
        if field in columns:
            raise ValueError(f"duplicate CSV header column {field!r}")  # noqa: TRY003
        columns.append(field)
    return tuple(columns)


def _key_index(columns: tuple[str, ...], primary_key: str) -> int:
    try:
        return columns.index(primary_key)
    except ValueError:
        raise ValueError(f"primary key {primary_key!r} is not a CSV header column") from None  # noqa: TRY003


def _row_value(source: CSVRowSource) -> dict[str, str]:
    if len(source.columns) != len(source.values):
        raise ValueError("row Source must carry one value per column")  # noqa: TRY003
    return dict(zip(source.columns, source.values, strict=True))


def _decode_document(content: bytes) -> str:
    try:
        return bytes(content).decode("utf-8-sig")
    except UnicodeDecodeError as error:
        raise ValueError(  # noqa: TRY003
            "CSV content must be strict UTF-8 with an optional leading BOM"
        ) from error


def _require_document_id(value: object) -> str:
    if not isinstance(value, str) or not value:
        raise InvalidSourceReferenceError("document_id", "must be a non-empty string")
    if value != value.strip():
        raise InvalidSourceReferenceError("document_id", "must not contain leading or trailing whitespace")
    return value


def _require_primary_key(value: object) -> str:
    if not isinstance(value, str) or not value:
        raise InvalidSourceReferenceError("primary_key", "must be a non-empty string")
    return value


def _require_primary_key_value(value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise InvalidSourceReferenceError("primary_key_value", "must be a non-empty, non-blank string")
    return value


def _require_source_id(value: object) -> str:
    if not isinstance(value, str) or not value:
        raise InvalidSourceReferenceError("source_id", "must be a non-empty string")
    if value != value.strip():
        raise InvalidSourceReferenceError("source_id", "must not contain leading or trailing whitespace")
    if len(value) > MAX_SOURCE_ID_LENGTH:
        raise InvalidSourceReferenceError("source_id", f"must not exceed {MAX_SOURCE_ID_LENGTH} characters")
    return value


CSV_ROW_SOURCE_ADAPTER = CSVRowSourceAdapter()
CSV_ROW_SOURCE_DEFINITION = AdapterSourceDefinition(
    CSV_ROW_SOURCE_ADAPTER,
    version=CSV_ROW_SOURCE_VERSION,
    projections=(CSVRowTextEvidenceProjection(),),
)

__all__ = [
    "CSV_ROW_SOURCE_ADAPTER",
    "CSV_ROW_SOURCE_DEFINITION",
    "CSV_ROW_SOURCE_NAME",
    "CSV_ROW_SOURCE_VERSION",
    "CSVDocumentInput",
    "CSVRowInput",
    "CSVRowSource",
    "CSVRowSourceAdapter",
    "CSVRowTextEvidenceProjection",
    "csv_row_source_id",
    "parse_csv_document",
]
