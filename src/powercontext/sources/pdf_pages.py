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

"""Page-level Sources for complete PDF documents and evidence retrieval.

Every physical page of the original PDF becomes one captured Source, so a hit
can be resolved back to the exact file revision and page number instead of an
approximate text offset.
"""

from __future__ import annotations

import hashlib
import io
import itertools
import re
from typing import TYPE_CHECKING
from urllib.parse import quote

from pydantic import BaseModel, field_validator, model_validator
from pypdf import PdfReader
from pypdf.errors import PyPdfError

from powercontext.builtin.artifacts.memory.models import MemoryCitation
from powercontext.builtin.runtime import SearchMemoryRequest
from powercontext.errors import InvalidSourceReferenceError
from powercontext.limits import MAX_SOURCE_ID_LENGTH
from powercontext.sources.definitions import AdapterSourceDefinition
from powercontext.sources.models import Source, SourceMaterialization
from powercontext.sources.projections import TEXT_EVIDENCE_PROJECTION_KEY, TextEvidence

if TYPE_CHECKING:
    from powercontext.builtin.artifacts.memory import Memory

PDF_PAGE_SOURCE_NAME = "pdf-page"
PDF_PAGE_SOURCE_VERSION = "1"
IDENTITY_PREFIX = "pdf"
SEARCH_WINDOW = 4
MAX_PROBE_TERMS = 8
MAX_PROBE_PHRASES = 6
FULL_QUESTION_WEIGHT = 3
PHRASE_WEIGHT = 2
TERM_WEIGHT = 1
RRF_K = 10
MIN_STEM_LENGTH = 5
# 10-K sections have stable headings; a question that names a statement or a
# section reaches the page through that heading even when its wording differs.
DOMAIN_EXPANSIONS = (
    ("income statement", "consolidated statement of income"),
    ("cash flow", "consolidated statement of cash flows"),
    ("balance sheet", "consolidated balance sheet"),
    ("legal", "legal proceedings"),
    ("geograph", "countries and territories"),
)

_QUESTION_TERMS = re.compile(r"[a-z]+|\d+")
_STOP_WORDS = frozenset({
    "a",
    "an",
    "the",
    "and",
    "or",
    "of",
    "to",
    "in",
    "on",
    "at",
    "by",
    "for",
    "from",
    "with",
    "as",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "that",
    "this",
    "these",
    "those",
    "it",
    "its",
    "what",
    "which",
    "who",
    "how",
    "why",
    "when",
    "where",
    "do",
    "does",
    "did",
    "has",
    "have",
    "had",
    "any",
    "all",
    "we",
    "our",
    "you",
    "your",
    "i",
    "can",
    "could",
    "should",
    "would",
    "please",
    "there",
    "their",
    "they",
    "them",
    "his",
    "her",
})


class PDFDocumentInput(BaseModel):
    """One complete PDF file to capture page by page."""

    document_id: str
    content: bytes


class PDFPageInput(BaseModel):
    """One physical page of a PDF revision."""

    document_id: str
    document_sha256: str
    page_number: int
    text: str


class PDFPageSource(Source):
    """A captured PDF page that is usable as textual Artifact evidence."""

    source_id: str
    document_id: str
    document_sha256: str
    page_number: int
    text: str

    @field_validator("source_id")
    @classmethod
    def validate_source_id(cls, value: str) -> str:
        _require_source_id(value)
        return value

    @model_validator(mode="after")
    def validate_identity(self) -> PDFPageSource:
        if self.name != self.source_id:
            raise ValueError("page Sources must be named by their identity")  # noqa: TRY003
        return self


class PDFPageSourceAdapter:
    """Resolve and read page Sources for the ``pdf-page`` Definition."""

    input_class = PDFPageInput
    name = PDF_PAGE_SOURCE_NAME
    source_class = PDFPageSource

    async def resolve(self, value: PDFPageInput, /) -> PDFPageSource:
        source_id = pdf_page_source_id(value.document_id, value.document_sha256, value.page_number)
        return PDFPageSource(
            name=source_id,
            source_id=source_id,
            materialization=SourceMaterialization.CAPTURED,
            document_id=value.document_id,
            document_sha256=value.document_sha256,
            page_number=value.page_number,
            text=value.text,
        )

    async def read(self, source: PDFPageSource, /) -> PDFPageInput:
        return PDFPageInput(
            document_id=source.document_id,
            document_sha256=source.document_sha256,
            page_number=source.page_number,
            text=source.text,
        )


class PDFPageTextEvidenceProjection:
    """Expose one page as ``DOCUMENT_ID`` plus its extractable text."""

    name = TEXT_EVIDENCE_PROJECTION_KEY.name
    version = TEXT_EVIDENCE_PROJECTION_KEY.version
    source_class = PDFPageSource
    output_class: type[BaseModel] = TextEvidence

    def project(self, source: PDFPageSource, /) -> TextEvidence:
        return TextEvidence(
            source_type=PDF_PAGE_SOURCE_NAME,
            source_id=source.source_id,
            content=f"{source.document_id}\n{source.text}",
            metadata={
                "document_id": source.document_id,
                "document_sha256": source.document_sha256,
                "page_number": source.page_number,
            },
        )


def parse_pdf_document(value: PDFDocumentInput, /) -> tuple[PDFPageInput, ...]:
    """Extract every physical page of one PDF, in order, without partial results."""

    document_id = _require_document_id(value.document_id)
    content = bytes(value.content)
    if not content:
        raise ValueError("PDF content is empty")  # noqa: TRY003
    document_sha256 = hashlib.sha256(content).hexdigest()
    texts = _extract_page_texts(content)

    if not texts:
        raise ValueError("PDF document contains no pages")  # noqa: TRY003
    return tuple(
        PDFPageInput(
            document_id=document_id,
            document_sha256=document_sha256,
            page_number=number,
            text=text,
        )
        for number, text in enumerate(texts, start=1)
    )


def _extract_page_texts(content: bytes) -> list[str]:
    try:
        reader = PdfReader(io.BytesIO(content))
    except Exception as error:
        raise ValueError(f"PDF document cannot be parsed: {error}") from error  # noqa: TRY003

    if reader.is_encrypted and not reader.decrypt(""):
        raise ValueError("PDF requires a password that was not supplied")  # noqa: TRY003
    try:
        return [page.extract_text() or "" for page in reader.pages]
    except PyPdfError as error:
        raise ValueError(f"PDF document cannot be parsed: {error}") from error  # noqa: TRY003
    except Exception as error:
        raise ValueError(f"PDF document cannot be parsed: {error}") from error  # noqa: TRY003


def pdf_page_source_id(document_id: str, document_sha256: str, page_number: int, /) -> str:
    """Build the revision-aware identity of one captured PDF page."""

    document = _require_document_id(document_id)
    digest = _require_digest(document_sha256)
    if not isinstance(page_number, int) or isinstance(page_number, bool) or page_number < 1:
        raise InvalidSourceReferenceError("page_number", "must be a positive integer")

    encoded = quote(document, safe="")
    source_id = f"{IDENTITY_PREFIX}:{encoded}:{digest}:{page_number}"
    if len(source_id) > MAX_SOURCE_ID_LENGTH:
        # Keep the identity bounded without losing revision or page semantics.
        compact = hashlib.sha256(document.encode("utf-8")).hexdigest()
        source_id = f"{IDENTITY_PREFIX}:{compact}:{digest}:{page_number}"
    _require_source_id(source_id)
    return source_id


async def search_pdf_pages(
    memory: Memory,
    question: str,
    limit: int = 10,
) -> tuple[MemoryCitation, ...]:
    """Return citations for the most relevant distinct pages of the scoped Memory."""

    if limit <= 0:
        return ()
    probes = _search_probes(question)
    if not probes:
        return ()

    window = limit * SEARCH_WINDOW
    scores: dict[str, float] = {}
    best_rank: dict[str, int] = {}
    citations: dict[str, MemoryCitation] = {}
    for probe, weight in probes:
        page = await memory.search(SearchMemoryRequest(query=probe, mode="fts", limit=window))
        # A probe that only fills a small part of the window is a rare, specific
        # term; one that saturates it is common and should not dominate the vote.
        specificity = 1.0 + min(1.0, max(0.0, (window - len(page.hits)) / window))
        for rank, hit in enumerate(page.hits):
            score = weight * specificity / (rank + RRF_K)
            scores[hit.entry_id] = scores.get(hit.entry_id, 0.0) + score
            if hit.entry_id not in best_rank or rank < best_rank[hit.entry_id]:
                best_rank[hit.entry_id] = rank
            citations[hit.entry_id] = MemoryCitation(
                memory_ref=hit.memory_ref,
                entry_id=hit.entry_id,
                entry_version_id=hit.entry_version_id,
            )

    ordered = sorted(scores, key=lambda entry_id: (-scores[entry_id], best_rank[entry_id]))
    return tuple(citations[entry_id] for entry_id in ordered[:limit])


def _search_probes(question: str) -> tuple[tuple[str, int], ...]:
    """Weighted probes: the whole question, filing section names, phrases, then stemmed terms.

    A page hit by a phrase is far more specific than one hit by a common single word,
    so probes carry different weights and their votes are fused by reciprocal rank.
    """

    cleaned = question.strip()
    probes: list[tuple[str, int]] = [(cleaned, FULL_QUESTION_WEIGHT)] if cleaned else []
    seen = {cleaned.casefold()}
    folded = cleaned.casefold()

    for trigger, expansion in DOMAIN_EXPANSIONS:
        if trigger in folded and expansion not in seen:
            seen.add(expansion)
            probes.append((expansion, PHRASE_WEIGHT))
    for phrase in _question_phrases(cleaned)[:MAX_PROBE_PHRASES]:
        if phrase not in seen:
            seen.add(phrase)
            probes.append((phrase, PHRASE_WEIGHT))
    for term in _question_terms(cleaned)[:MAX_PROBE_TERMS]:
        probe = _stemmed_probe(term)
        if probe not in seen:
            seen.add(probe)
            probes.append((probe, TERM_WEIGHT))
    return tuple(probes)


def _stemmed_probe(term: str) -> str:
    """Make a term match its other inflections.

    The page index tokenizes but does not stem, so ``geographies`` never matches
    ``geographic``. Trimming common English suffixes and asking for a prefix query
    recovers those matches without pulling in unrelated words.
    """

    stem = term
    for suffix in ("ies", "ing", "es", "ed", "ly", "s"):
        if len(stem) - len(suffix) >= MIN_STEM_LENGTH and stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            if suffix == "ies":
                stem += "y"
            break
    if stem != term and len(stem) >= MIN_STEM_LENGTH:
        return f"{stem}*"
    return term


def _question_tokens(question: str) -> list[str]:
    return _QUESTION_TERMS.findall(question.casefold())


def _question_phrases(question: str) -> list[str]:
    tokens = _question_tokens(question)
    phrases: list[str] = []
    seen: set[str] = set()
    for left, right in itertools.pairwise(tokens):
        if left in _STOP_WORDS or right in _STOP_WORDS:
            continue
        phrase = f"{left} {right}"
        if phrase in seen:
            continue
        seen.add(phrase)
        phrases.append(phrase)
    return phrases


def _question_terms(question: str) -> list[str]:
    terms: list[str] = []
    seen: set[str] = set()
    for term in _question_tokens(question):
        if term in _STOP_WORDS or term in seen:
            continue
        seen.add(term)
        terms.append(term)
    return terms


def _require_document_id(value: object) -> str:
    if not isinstance(value, str) or not value:
        raise InvalidSourceReferenceError("document_id", "must be a non-empty string")
    if value != value.strip():
        raise InvalidSourceReferenceError("document_id", "must not contain leading or trailing whitespace")
    return value


def _require_digest(value: object) -> str:
    if not isinstance(value, str) or len(value) != 64 or value.lower() != value:
        raise InvalidSourceReferenceError("document_sha256", "must be a lowercase sha256 hex digest")
    try:
        int(value, 16)
    except ValueError:
        raise InvalidSourceReferenceError("document_sha256", "must be a lowercase sha256 hex digest") from None
    return value


def _require_source_id(value: object) -> str:
    if not isinstance(value, str) or not value:
        raise InvalidSourceReferenceError("source_id", "must be a non-empty string")
    if value != value.strip():
        raise InvalidSourceReferenceError("source_id", "must not contain leading or trailing whitespace")
    if len(value) > MAX_SOURCE_ID_LENGTH:
        raise InvalidSourceReferenceError("source_id", f"must not exceed {MAX_SOURCE_ID_LENGTH} characters")
    return value


PDF_PAGE_SOURCE_ADAPTER = PDFPageSourceAdapter()
PDF_PAGE_SOURCE_DEFINITION = AdapterSourceDefinition(
    PDF_PAGE_SOURCE_ADAPTER,
    version=PDF_PAGE_SOURCE_VERSION,
    projections=(PDFPageTextEvidenceProjection(),),
)

__all__ = [
    "PDF_PAGE_SOURCE_ADAPTER",
    "PDF_PAGE_SOURCE_DEFINITION",
    "PDF_PAGE_SOURCE_NAME",
    "PDF_PAGE_SOURCE_VERSION",
    "PDFDocumentInput",
    "PDFPageInput",
    "PDFPageSource",
    "PDFPageSourceAdapter",
    "PDFPageTextEvidenceProjection",
    "parse_pdf_document",
    "pdf_page_source_id",
    "search_pdf_pages",
]
