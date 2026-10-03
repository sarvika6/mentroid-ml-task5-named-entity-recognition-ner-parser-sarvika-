"""High-level EntityExtractor facade.

This module owns the public API that the CLI and application code should use.
It wraps a :class:`~ner_parser.backends.BaseNERBackend` and adds:

- Single-document extraction via :meth:`EntityExtractor.extract`.
- Bulk extraction via :meth:`EntityExtractor.extract_batch` (uses
  ``nlp.pipe()`` under the hood when the spaCy backend is active).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from ner_parser.backends import BaseNERBackend, SpacyBackend
from ner_parser.config import DEFAULT_SPACY_MODEL, SPACY_LABEL_MAP
from ner_parser.models import ArticleResult

if TYPE_CHECKING:
    pass  # future: protocol-based type narrowing

logger = logging.getLogger(__name__)


class EntityExtractor:
    """Facade for running NER over one or many article texts.

    Instantiate once and call :meth:`extract` or :meth:`extract_batch`
    repeatedly without reloading the underlying model.

    Args:
        backend: A concrete :class:`~ner_parser.backends.BaseNERBackend`
            instance.  Defaults to a :class:`~ner_parser.backends.SpacyBackend`
            loaded with ``en_core_web_sm``.

    Example::

        from ner_parser.extractor import EntityExtractor

        extractor = EntityExtractor()
        result = extractor.extract("Google opened a new office in Bengaluru last Monday.")
        for entity in result.entities:
            print(entity)
    """

    def __init__(self, backend: BaseNERBackend | None = None) -> None:
        self._backend: BaseNERBackend = backend or SpacyBackend(
            model_name=DEFAULT_SPACY_MODEL,
            label_map=SPACY_LABEL_MAP,
        )
        logger.info(
            "EntityExtractor initialised with backend '%s'.",
            self._backend.name,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def backend(self) -> BaseNERBackend:
        """The underlying NER backend in use.

        Returns:
            The :class:`~ner_parser.backends.BaseNERBackend` instance.
        """
        return self._backend

    def extract(self, text: str, article_id: int = 0) -> ArticleResult:
        """Extract named entities from a single text.

        Args:
            text: Raw article text. May be empty (returns an empty result).
            article_id: Optional integer identifier stored in the result.

        Returns:
            An :class:`~ner_parser.models.ArticleResult` with all
            deduplicated entities found.
        """
        results = self._backend.extract_batch([text])
        result = results[0]
        # Patch the article_id if the caller supplied a non-zero value
        if article_id != result.article_id:
            from dataclasses import replace  # noqa: PLC0415

            result = replace(result, article_id=article_id)
        return result

    def extract_batch(self, texts: list[str]) -> list[ArticleResult]:
        """Extract named entities from multiple texts in one pass.

        Uses the backend's batching mechanism (``nlp.pipe`` for spaCy) for
        efficiency.

        Args:
            texts: List of raw article strings.  Empty entries are handled
                gracefully and produce empty :class:`ArticleResult` objects.

        Returns:
            A list of :class:`~ner_parser.models.ArticleResult`, one per
            input text and in the same order.

        Raises:
            ValueError: If *texts* is empty.
        """
        if not texts:
            raise ValueError("'texts' must contain at least one element.")
        logger.info("Processing batch of %d texts.", len(texts))
        return self._backend.extract_batch(texts)
