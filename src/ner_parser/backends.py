"""Backend abstraction for NER model providers.

Implements the *Strategy* design pattern so the extraction engine can be
swapped between spaCy and HuggingFace without touching business logic.

Classes:
    BaseNERBackend: Abstract base class defining the common interface.
    SpacyBackend:   spaCy-based backend (default).
    HuggingFaceBackend: HuggingFace *transformers* pipeline backend.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any

from ner_parser.config import (
    HF_AGGREGATION_STRATEGY,
    HF_LABEL_MAP,
    HF_MODEL_NAME,
    OUTPUT_CATEGORIES,
    PIPE_BATCH_SIZE,
    SPACY_LABEL_MAP,
)
from ner_parser.models import ArticleResult, Entity

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _deduplicate(entities: list[Entity]) -> list[Entity]:
    """Deduplicate entities case-insensitively, preserving first-seen order.

    Args:
        entities: Raw list of :class:`~ner_parser.models.Entity` objects,
            possibly containing duplicates.

    Returns:
        A new list with duplicates removed. The first occurrence is kept.
    """
    seen: set[tuple[str, str]] = set()
    result: list[Entity] = []
    for ent in entities:
        key = (ent.text.lower(), ent.label)
        if key not in seen:
            seen.add(key)
            result.append(ent)
    return result


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------


class BaseNERBackend(ABC):
    """Abstract base class for NER backends.

    Subclasses must implement :meth:`extract_batch` which accepts a list of
    plain-text strings and returns a corresponding list of
    :class:`~ner_parser.models.ArticleResult` objects.
    """

    #: Human-readable name shown in logs and comparison tables.
    name: str = "base"

    #: The set of output categories this backend can predict.
    #: Evaluate will skip any category absent from this set and render
    #: the cell as ``n/a`` rather than ``0.00``.
    supported_categories: frozenset[str] = frozenset(OUTPUT_CATEGORIES)

    @abstractmethod
    def extract_batch(self, texts: list[str]) -> list[ArticleResult]:
        """Process a list of texts and extract named entities.

        Args:
            texts: Input texts. Empty strings are handled gracefully.

        Returns:
            One :class:`~ner_parser.models.ArticleResult` per input text,
            in the same order.
        """
        raise NotImplementedError


# ---------------------------------------------------------------------------
# spaCy backend
# ---------------------------------------------------------------------------


class SpacyBackend(BaseNERBackend):
    """NER backend powered by a spaCy language pipeline.

    The model is loaded **once** at construction time to avoid repeated I/O.

    Args:
        model_name: spaCy model identifier (e.g. ``"en_core_web_sm"``).
        label_map: Mapping from spaCy label strings to canonical categories.
        batch_size: Number of documents per :meth:`nlp.pipe` batch.

    Raises:
        OSError: If the model is not installed. The error message includes the
            exact ``python -m spacy download`` command needed.
    """

    def __init__(
        self,
        model_name: str = "en_core_web_sm",
        label_map: dict[str, str] | None = None,
        batch_size: int = PIPE_BATCH_SIZE,
    ) -> None:
        self.model_name = model_name
        self.label_map: dict[str, str] = label_map if label_map is not None else SPACY_LABEL_MAP
        self.batch_size = batch_size
        self.name = f"spacy:{model_name}"
        self._nlp = self._load_model()

    def _load_model(self) -> Any:  # noqa: ANN401
        """Load the spaCy model, raising a clear error if it is missing.

        Returns:
            A loaded spaCy ``Language`` object.

        Raises:
            OSError: If the model package is not installed.
        """
        try:
            import spacy  # noqa: PLC0415

            nlp = spacy.load(self.model_name)
            logger.info("Loaded spaCy model '%s'.", self.model_name)
            return nlp
        except OSError as exc:
            msg = (
                f"spaCy model '{self.model_name}' not found.\n"
                f"Install it with:\n\n"
                f"    python -m spacy download {self.model_name}\n"
            )
            raise OSError(msg) from exc

    def extract_batch(self, texts: list[str]) -> list[ArticleResult]:
        """Extract named entities from *texts* using :meth:`nlp.pipe`.

        Args:
            texts: List of raw article strings.

        Returns:
            List of :class:`~ner_parser.models.ArticleResult`, one per text.
        """
        results: list[ArticleResult] = []

        # Use empty string sentinel so nlp.pipe never sees None
        safe_texts = [t if t.strip() else "" for t in texts]

        docs = self._nlp.pipe(safe_texts, batch_size=self.batch_size)
        for idx, doc in enumerate(docs):
            raw = texts[idx]
            if not raw.strip():
                logger.warning("Article %d is empty or whitespace-only.", idx)
                results.append(ArticleResult(article_id=idx, source=raw))
                continue

            entities: list[Entity] = []
            for ent in doc.ents:
                canonical = self.label_map.get(ent.label_)
                if canonical is None:
                    continue
                entities.append(
                    Entity(
                        text=ent.text,
                        label=canonical,
                        start=ent.start_char,
                        end=ent.end_char,
                    )
                )

            deduped = _deduplicate(entities)
            logger.debug(
                "Article %d: %d raw entities → %d after dedup.",
                idx,
                len(entities),
                len(deduped),
            )
            results.append(
                ArticleResult(article_id=idx, source=raw, entities=deduped)
            )

        return results


# ---------------------------------------------------------------------------
# HuggingFace backend
# ---------------------------------------------------------------------------


class HuggingFaceBackend(BaseNERBackend):
    """NER backend powered by a HuggingFace *transformers* token-classification
    pipeline.

    .. note::
        The default model (``dslim/bert-base-NER``) does **not** produce
        ``DATE`` tags.  Only ``LOC`` and ``ORG`` (and ``PER``) are available.

    Args:
        model_name: HuggingFace model identifier.
        aggregation_strategy: Token aggregation strategy passed to the
            HuggingFace pipeline (e.g. ``"simple"``).
        label_map: Mapping from HuggingFace label strings to canonical names.

    Raises:
        ImportError: If the ``transformers`` package is not installed.
    """

    #: dslim/bert-base-NER is trained on CoNLL-2003, which has no DATE class.
    #: Declaring this prevents the evaluator from reporting 0.00 for Dates
    #: and misleading the reader into thinking the model attempted and failed.
    supported_categories: frozenset[str] = frozenset({"Locations", "Organizations"})

    def __init__(
        self,
        model_name: str = HF_MODEL_NAME,
        aggregation_strategy: str = HF_AGGREGATION_STRATEGY,
        label_map: dict[str, str] | None = None,
    ) -> None:
        self.model_name = model_name
        self.aggregation_strategy = aggregation_strategy
        self.label_map: dict[str, str] = label_map if label_map is not None else HF_LABEL_MAP
        self.name = f"hf:{model_name}"
        self._pipeline = self._load_pipeline()

    def _load_pipeline(self) -> Any:  # noqa: ANN401
        """Initialise the HuggingFace NER pipeline.

        Returns:
            A HuggingFace ``Pipeline`` object configured for token
            classification.

        Raises:
            ImportError: If ``transformers`` is not available.
        """
        try:
            from transformers import pipeline as hf_pipeline  # noqa: PLC0415

            pipe = hf_pipeline(
                "ner",
                model=self.model_name,
                aggregation_strategy=self.aggregation_strategy,
            )
            logger.info(
                "Loaded HuggingFace model '%s' (strategy='%s').",
                self.model_name,
                self.aggregation_strategy,
            )
            return pipe
        except ImportError as exc:
            msg = (
                "The 'transformers' package is required for the HuggingFace backend.\n"
                "Install it with:  pip install transformers torch\n"
            )
            raise ImportError(msg) from exc

    def extract_batch(self, texts: list[str]) -> list[ArticleResult]:
        """Extract named entities from *texts* using the HuggingFace pipeline.

        Args:
            texts: List of raw article strings.

        Returns:
            List of :class:`~ner_parser.models.ArticleResult`, one per text.
        """
        results: list[ArticleResult] = []

        for idx, raw in enumerate(texts):
            if not raw.strip():
                logger.warning("Article %d is empty or whitespace-only.", idx)
                results.append(ArticleResult(article_id=idx, source=raw))
                continue

            raw_ents: list[dict[str, Any]] = self._pipeline(raw)
            entities: list[Entity] = []

            for ent in raw_ents:
                # HF tag format with aggregation: e.g. "ORG", "LOC"
                hf_label: str = ent.get("entity_group", ent.get("entity", ""))
                canonical = self.label_map.get(hf_label)
                if canonical is None:
                    continue
                entities.append(
                    Entity(
                        text=ent["word"],
                        label=canonical,
                        start=int(ent["start"]),
                        end=int(ent["end"]),
                    )
                )

            deduped = _deduplicate(entities)
            logger.debug(
                "Article %d (HF): %d raw entities → %d after dedup.",
                idx,
                len(entities),
                len(deduped),
            )
            results.append(
                ArticleResult(article_id=idx, source=raw, entities=deduped)
            )

        return results
