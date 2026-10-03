"""Data model definitions for the NER pipeline.

Uses :mod:`dataclasses` so the structures remain lightweight, serialisable,
and easy to convert to/from JSON.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True, order=True)
class Entity:
    """A single named entity extracted from a piece of text.

    Attributes:
        text: Surface form of the entity as it appears in the source text.
        label: Canonical category label (e.g. ``"Dates"``, ``"Locations"``).
        start: Character offset of the first character (inclusive).
        end: Character offset one past the last character (exclusive).
    """

    text: str
    label: str
    start: int
    end: int

    def to_dict(self) -> dict[str, int | str]:
        """Return a JSON-serialisable dictionary representation.

        Returns:
            A dict with keys ``text``, ``label``, ``start``, ``end``.
        """
        return {
            "text": self.text,
            "label": self.label,
            "start": self.start,
            "end": self.end,
        }


@dataclass
class ArticleResult:
    """Aggregated NER results for one news article.

    Attributes:
        article_id: Zero-based index of the article in the input batch.
        source: Raw source text of the article.
        entities: All deduplicated entities found, preserving first-seen order.
    """

    article_id: int
    source: str
    entities: list[Entity] = field(default_factory=list)

    # ------------------------------------------------------------------
    # Convenience accessors
    # ------------------------------------------------------------------

    def by_category(self, category: str) -> list[Entity]:
        """Return entities that belong to *category*.

        Args:
            category: Canonical category name (e.g. ``"Dates"``).

        Returns:
            A list of :class:`Entity` objects filtered by the given label.
        """
        return [e for e in self.entities if e.label == category]

    def categories(self) -> set[str]:
        """Return the set of distinct category labels present in this result.

        Returns:
            A set of category name strings.
        """
        return {e.label for e in self.entities}

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serialisable dictionary representation.

        Returns:
            A dict with keys ``article_id``, ``source``, and ``entities``.
        """
        return {
            "article_id": self.article_id,
            "source": self.source,
            "entities": [e.to_dict() for e in self.entities],
        }
