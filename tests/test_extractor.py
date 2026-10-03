"""Pytest test suite for the NER parser pipeline.

Tests cover:
- Label mapping configuration.
- Entity deduplication (case-insensitive, first-seen order preserved).
- Empty text handling.
- Batch extraction behaviour.
- CSV / JSON / table formatting smoke tests.
- evaluate_backend metric correctness (via FakeBackend stub).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

# evaluate.py lives at the project root, not inside the src package.
# Insert the root so it is importable as a plain module.
sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluate import _micro_average, evaluate_backend  # noqa: E402

from ner_parser.backends import BaseNERBackend, _deduplicate
from ner_parser.config import OUTPUT_CATEGORIES, SPACY_LABEL_MAP
from ner_parser.extractor import EntityExtractor
from ner_parser.formatting import format_csv, format_json, format_table
from ner_parser.models import ArticleResult, Entity

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def extractor() -> EntityExtractor:
    """Return a shared EntityExtractor backed by en_core_web_sm."""
    return EntityExtractor()


@pytest.fixture
def sample_result() -> ArticleResult:
    """Minimal ArticleResult for formatting tests."""
    return ArticleResult(
        article_id=0,
        source="Google opened an office in London last Monday.",
        entities=[
            Entity(text="Google", label="Organizations", start=0, end=6),
            Entity(text="London", label="Locations", start=26, end=32),
            Entity(text="last Monday", label="Dates", start=33, end=44),
        ],
    )


# ---------------------------------------------------------------------------
# Config: label mapping
# ---------------------------------------------------------------------------


class TestLabelMapping:
    """Verify that SPACY_LABEL_MAP covers all required source labels."""

    def test_date_mapped(self) -> None:
        assert SPACY_LABEL_MAP["DATE"] == "Dates"

    def test_gpe_mapped_to_locations(self) -> None:
        assert SPACY_LABEL_MAP["GPE"] == "Locations"

    def test_loc_mapped_to_locations(self) -> None:
        assert SPACY_LABEL_MAP["LOC"] == "Locations"

    def test_org_mapped_to_organizations(self) -> None:
        assert SPACY_LABEL_MAP["ORG"] == "Organizations"

    def test_output_categories_are_canonical(self) -> None:
        """Every value in SPACY_LABEL_MAP should be in OUTPUT_CATEGORIES."""
        for canonical in SPACY_LABEL_MAP.values():
            assert canonical in OUTPUT_CATEGORIES


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------


class TestDeduplication:
    """Unit tests for the _deduplicate helper."""

    def test_removes_exact_duplicate(self) -> None:
        entities = [
            Entity("Apple", "Organizations", 0, 5),
            Entity("Apple", "Organizations", 50, 55),
        ]
        result = _deduplicate(entities)
        assert len(result) == 1
        assert result[0].start == 0  # first-seen preserved

    def test_removes_case_insensitive_duplicate(self) -> None:
        entities = [
            Entity("apple", "Organizations", 0, 5),
            Entity("Apple", "Organizations", 20, 25),
            Entity("APPLE", "Organizations", 40, 45),
        ]
        result = _deduplicate(entities)
        assert len(result) == 1

    def test_different_labels_not_deduplicated(self) -> None:
        """Same text with different labels are distinct entities."""
        entities = [
            Entity("Amazon", "Organizations", 0, 6),
            Entity("Amazon", "Locations", 0, 6),
        ]
        result = _deduplicate(entities)
        assert len(result) == 2

    def test_preserves_insertion_order(self) -> None:
        entities = [
            Entity("Paris", "Locations", 10, 15),
            Entity("Rome", "Locations", 30, 34),
            Entity("paris", "Locations", 50, 55),  # duplicate
        ]
        result = _deduplicate(entities)
        assert [e.text for e in result] == ["Paris", "Rome"]

    def test_empty_list(self) -> None:
        assert _deduplicate([]) == []


# ---------------------------------------------------------------------------
# Empty text handling
# ---------------------------------------------------------------------------


class TestEmptyText:
    """Extractor should handle empty / whitespace-only text gracefully."""

    def test_empty_string_returns_empty_result(self, extractor: EntityExtractor) -> None:
        result = extractor.extract("")
        assert result.entities == []

    def test_whitespace_only_returns_empty_result(self, extractor: EntityExtractor) -> None:
        result = extractor.extract("   \n\t  ")
        assert result.entities == []

    def test_batch_with_mixed_empty(self, extractor: EntityExtractor) -> None:
        texts = ["", "Apple is headquartered in Cupertino.", ""]
        results = extractor.extract_batch(texts)
        assert len(results) == 3
        assert results[0].entities == []
        assert results[2].entities == []


# ---------------------------------------------------------------------------
# Batch extraction
# ---------------------------------------------------------------------------


class TestBatchExtraction:
    """Tests for extract_batch correctness and ordering."""

    def test_returns_same_count_as_input(self, extractor: EntityExtractor) -> None:
        texts = [
            "Google is based in Mountain View.",
            "The UN met in Geneva yesterday.",
            "Tesla reported profits last quarter.",
        ]
        results = extractor.extract_batch(texts)
        assert len(results) == len(texts)

    def test_article_ids_match_order(self, extractor: EntityExtractor) -> None:
        texts = ["First article.", "Second article.", "Third article."]
        results = extractor.extract_batch(texts)
        for i, r in enumerate(results):
            assert r.article_id == i

    def test_raises_on_empty_list(self, extractor: EntityExtractor) -> None:
        with pytest.raises(ValueError, match="at least one element"):
            extractor.extract_batch([])

    def test_known_org_extracted(self, extractor: EntityExtractor) -> None:
        """A clearly labelled ORG should appear in results."""
        results = extractor.extract_batch(["Microsoft announced layoffs last year."])
        orgs = [e.text for e in results[0].by_category("Organizations")]
        assert "Microsoft" in orgs

    def test_known_location_extracted(self, extractor: EntityExtractor) -> None:
        results = extractor.extract_batch(["The summit was held in Paris."])
        locs = [e.text for e in results[0].by_category("Locations")]
        assert "Paris" in locs

    def test_known_date_extracted(self, extractor: EntityExtractor) -> None:
        results = extractor.extract_batch(["The event occurred on January 5, 2023."])
        dates = [e.text for e in results[0].by_category("Dates")]
        assert any("January" in d for d in dates)

    def test_sanity_check_sentence(self, extractor: EntityExtractor) -> None:
        """Canonical diagnostic sentence — setup check, not accuracy benchmark.

        If this test fails the model is not loaded or wired incorrectly.
        If labels differ from expected it indicates model version drift; the
        test documents what en_core_web_sm 3.8.0 predicts on this sentence.
        """
        text = "Apple opened an office in Chennai on 4 November 2025."
        result = extractor.extract(text)

        by_label: dict[str, list[str]] = {}
        for ent in result.entities:
            by_label.setdefault(ent.label, []).append(ent.text)

        # All three must be present — if any is missing the pipeline is broken.
        assert "Organizations" in by_label, (
            f"Expected 'Apple' tagged ORG, got labels: {list(by_label)}"
        )
        assert "Locations" in by_label, (
            f"Expected 'Chennai' tagged GPE→Locations, got labels: {list(by_label)}"
        )
        assert "Dates" in by_label, (
            f"Expected '4 November 2025' tagged DATE, got labels: {list(by_label)}"
        )

        # Confirm the specific surface forms.
        assert "Apple" in by_label["Organizations"]
        assert "Chennai" in by_label["Locations"]
        assert any("November" in d for d in by_label["Dates"])

        # Slice invariant: text[start:end] must equal entity text for every entity.
        for ent in result.entities:
            assert text[ent.start : ent.end] == ent.text, (
                f"Offset mismatch: text[{ent.start}:{ent.end}]="
                f"{text[ent.start:ent.end]!r} != {ent.text!r}"
            )

    def test_character_offsets_within_bounds(self, extractor: EntityExtractor) -> None:
        """Offsets are in-bounds and text[start:end] == ent.text for one sentence."""
        text = "Amazon opened a centre in Bengaluru last week."
        result = extractor.extract(text)
        for ent in result.entities:
            assert ent.start >= 0
            assert ent.end <= len(text)
            assert text[ent.start : ent.end] == ent.text, (
                f"Offset mismatch: text[{ent.start}:{ent.end}]="
                f"{text[ent.start:ent.end]!r} != {ent.text!r}"
            )

    def test_offsets_across_all_sample_articles(self, extractor: EntityExtractor) -> None:
        """text[start:end] == ent.text holds for every entity in all 5 sample articles.

        This is the key invariant: character offsets produced by the pipeline
        must index back into the source string without adjustment.
        """
        import json
        from pathlib import Path

        articles_path = Path(__file__).parent.parent / "data" / "sample_articles.json"
        if not articles_path.exists():
            pytest.skip("data/sample_articles.json not found — skipping data-file test")

        articles: list[str] = json.loads(articles_path.read_text(encoding="utf-8"))
        results = extractor.extract_batch(articles)

        failures: list[str] = []
        for result in results:
            text = result.source
            for ent in result.entities:
                sliced = text[ent.start : ent.end]
                if sliced != ent.text:
                    failures.append(
                        f"Article {result.article_id}: "
                        f"text[{ent.start}:{ent.end}]={sliced!r} != {ent.text!r}"
                    )

        assert not failures, (
            f"{len(failures)} offset mismatch(es):\n" + "\n".join(failures)
        )


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------


class TestFormatting:
    """Smoke tests for output formatters."""

    def test_format_json_is_valid(self, sample_result: ArticleResult) -> None:
        output = format_json([sample_result])
        parsed = json.loads(output)
        assert isinstance(parsed, list)
        assert parsed[0]["article_id"] == 0

    def test_format_csv_has_header(self, sample_result: ArticleResult) -> None:
        output = format_csv([sample_result])
        lines = output.strip().split("\n")
        assert lines[0] == "article_id,category,entity,start,end"

    def test_format_csv_row_count(self, sample_result: ArticleResult) -> None:
        output = format_csv([sample_result])
        lines = output.strip().split("\n")
        # header + 3 entities = 4 lines
        assert len(lines) == 4

    def test_format_table_contains_header(self, sample_result: ArticleResult) -> None:
        output = format_table([sample_result])
        assert "Category" in output
        assert "Entity" in output

    def test_format_table_contains_entity(self, sample_result: ArticleResult) -> None:
        output = format_table([sample_result])
        assert "Google" in output
        assert "London" in output

    def test_format_json_no_ascii_escape(self) -> None:
        result = ArticleResult(
            article_id=0,
            source="प्रधानमंत्री नरेंद्र मोदी",
            entities=[Entity(text="नरेंद्र मोदी", label="Organizations", start=16, end=28)],
        )
        output = format_json([result])
        assert "नरेंद्र मोदी" in output


# ---------------------------------------------------------------------------
# Model / ArticleResult helpers
# ---------------------------------------------------------------------------


class TestArticleResult:
    """Unit tests for ArticleResult helper methods."""

    def test_by_category_filters_correctly(self, sample_result: ArticleResult) -> None:
        orgs = sample_result.by_category("Organizations")
        assert all(e.label == "Organizations" for e in orgs)

    def test_categories_returns_set(self, sample_result: ArticleResult) -> None:
        cats = sample_result.categories()
        assert isinstance(cats, set)
        assert "Dates" in cats
        assert "Locations" in cats
        assert "Organizations" in cats

    def test_to_dict_structure(self, sample_result: ArticleResult) -> None:
        d = sample_result.to_dict()
        assert "article_id" in d
        assert "source" in d
        assert "entities" in d
        assert isinstance(d["entities"], list)

    def test_entity_to_dict(self) -> None:
        ent = Entity(text="Paris", label="Locations", start=5, end=10)
        d = ent.to_dict()
        assert d == {"text": "Paris", "label": "Locations", "start": 5, "end": 10}


# ---------------------------------------------------------------------------
# FakeBackend — controllable stub for evaluate_backend tests
# ---------------------------------------------------------------------------


class FakeBackend(BaseNERBackend):
    """Deterministic backend that returns pre-canned entity lists.

    Args:
        predictions: One list of ``(text, label)`` pairs per article.
        supported: Override the default ``supported_categories``.
    """

    name = "fake"

    def __init__(
        self,
        predictions: list[list[tuple[str, str]]],
        supported: frozenset[str] | None = None,
    ) -> None:
        self._predictions = predictions
        if supported is not None:
            # frozenset is a class attribute; instance override is fine
            self.supported_categories = supported  # type: ignore[assignment]

    def extract_batch(self, texts: list[str]) -> list[ArticleResult]:
        results: list[ArticleResult] = []
        for i, (text, preds) in enumerate(zip(texts, self._predictions, strict=True)):
            entities = [
                Entity(text=t, label=lbl, start=0, end=len(t)) for t, lbl in preds
            ]
            results.append(ArticleResult(article_id=i, source=text, entities=entities))
        return results


# ---------------------------------------------------------------------------
# evaluate_backend tests
# ---------------------------------------------------------------------------


class TestEvaluateBackend:
    """Unit tests for evaluate_backend metric maths using FakeBackend."""

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _gold(
        article_id: int,
        dates: list[str] | None = None,
        locations: list[str] | None = None,
        organizations: list[str] | None = None,
    ) -> dict[int, dict[str, list[str]]]:
        """Build a single-article gold dict keyed by article_id."""
        return {
            article_id: {
                "Dates": dates or [],
                "Locations": locations or [],
                "Organizations": organizations or [],
            }
        }

    # ------------------------------------------------------------------
    # Metric maths
    # ------------------------------------------------------------------

    def test_perfect_match_gives_one(self) -> None:
        """TP = 1, FP = 0, FN = 0 → P=1, R=1, F1=1."""
        backend = FakeBackend([[("Google", "Organizations")]])
        extractor = EntityExtractor(backend=backend)
        gold = self._gold(0, organizations=["Google"])
        er = evaluate_backend(extractor, ["Google is big."], gold)

        m = er.get("Organizations")
        assert m is not None
        assert m.tp == 1
        assert m.fp == 0
        assert m.fn == 0
        assert m.precision == pytest.approx(1.0)
        assert m.recall == pytest.approx(1.0)
        assert m.f1 == pytest.approx(1.0)

    def test_false_positive_lowers_precision(self) -> None:
        """Predicts 2, gold has 1 → P=0.5, R=1.0, F1=0.667."""
        backend = FakeBackend([[("Apple", "Organizations"), ("Google", "Organizations")]])
        extractor = EntityExtractor(backend=backend)
        gold = self._gold(0, organizations=["Apple"])
        er = evaluate_backend(extractor, ["text"], gold)

        m = er.get("Organizations")
        assert m is not None
        assert m.tp == 1
        assert m.fp == 1
        assert m.fn == 0
        assert m.precision == pytest.approx(0.5)
        assert m.recall == pytest.approx(1.0)
        assert m.f1 == pytest.approx(2 / 3)

    def test_false_negative_lowers_recall(self) -> None:
        """Predicts 1, gold has 2 → P=1.0, R=0.5, F1=0.667."""
        backend = FakeBackend([[("Apple", "Organizations")]])
        extractor = EntityExtractor(backend=backend)
        gold = self._gold(0, organizations=["Apple", "Google"])
        er = evaluate_backend(extractor, ["text"], gold)

        m = er.get("Organizations")
        assert m is not None
        assert m.precision == pytest.approx(1.0)
        assert m.recall == pytest.approx(0.5)

    def test_no_predictions_gives_none_precision(self) -> None:
        """No predictions → precision is undefined (None), not 0."""
        backend = FakeBackend([[]])  # predicts nothing
        extractor = EntityExtractor(backend=backend)
        gold = self._gold(0, organizations=["Apple"])
        er = evaluate_backend(extractor, ["text"], gold)

        m = er.get("Organizations")
        assert m is not None
        assert m.precision is None  # undefined, not 0
        assert m.recall == pytest.approx(0.0)  # 0 / 1 gold → 0, not None

    def test_no_gold_gives_none_recall(self) -> None:
        """No gold items → recall is undefined (None), not 0."""
        backend = FakeBackend([[("Apple", "Organizations")]])
        extractor = EntityExtractor(backend=backend)
        gold = self._gold(0)  # empty gold
        er = evaluate_backend(extractor, ["text"], gold)

        m = er.get("Organizations")
        assert m is not None
        assert m.recall is None  # undefined: denominator is 0
        assert m.precision == pytest.approx(0.0)  # 0 TP, 1 FP → 0.0

    def test_unsupported_category_gives_none_metrics(self) -> None:
        """Category absent from supported_categories → all None."""
        backend = FakeBackend(
            [[("London", "Locations")]],
            supported=frozenset({"Locations", "Organizations"}),
        )
        extractor = EntityExtractor(backend=backend)
        gold = self._gold(0, dates=["yesterday"], locations=["London"])
        er = evaluate_backend(extractor, ["text"], gold)

        dates_m = er.get("Dates")
        assert dates_m is not None
        assert dates_m.precision is None
        assert dates_m.recall is None
        assert dates_m.f1 is None

        loc_m = er.get("Locations")
        assert loc_m is not None
        assert loc_m.precision == pytest.approx(1.0)

    def test_case_insensitive_match(self) -> None:
        """'apple' prediction should match gold 'Apple'."""
        backend = FakeBackend([[("apple", "Organizations")]])
        extractor = EntityExtractor(backend=backend)
        gold = self._gold(0, organizations=["Apple"])
        er = evaluate_backend(extractor, ["text"], gold)

        m = er.get("Organizations")
        assert m is not None
        assert m.tp == 1
        assert m.fp == 0

    def test_support_counts_gold_items(self) -> None:
        """support == number of unique gold items per category."""
        backend = FakeBackend([[]])
        extractor = EntityExtractor(backend=backend)
        gold = self._gold(0, dates=["Monday", "Tuesday", "Wednesday"])
        er = evaluate_backend(extractor, ["text"], gold)

        m = er.get("Dates")
        assert m is not None
        assert m.support == 3

    def test_micro_average_aggregates_supported_only(self) -> None:
        """Micro-average skips None categories (unsupported)."""
        backend = FakeBackend(
            [[("London", "Locations"), ("Apple", "Organizations")]],
            supported=frozenset({"Locations", "Organizations"}),
        )
        extractor = EntityExtractor(backend=backend)
        gold = self._gold(0, dates=["yesterday"], locations=["London"], organizations=["Apple"])
        er = evaluate_backend(extractor, ["text"], gold)

        mp, mr, mf = _micro_average(er)
        # Dates is None (unsupported), so micro only covers Loc+Org: 2 TP, 0 FP, 0 FN
        assert mp == pytest.approx(1.0)
        assert mr == pytest.approx(1.0)
        assert mf == pytest.approx(1.0)

    def test_error_collection_false_positives(self) -> None:
        """--errors flag captures false positives per article."""
        backend = FakeBackend([[("Apple", "Organizations"), ("Google", "Organizations")]])
        extractor = EntityExtractor(backend=backend)
        gold = self._gold(0, organizations=["Apple"])
        er = evaluate_backend(extractor, ["text"], gold, collect_errors=True)

        assert len(er.errors) == 1
        err = er.errors[0]
        assert err["article"] == 0
        assert err["category"] == "Organizations"
        assert "google" in err["false_positives"]  # normalised
        assert err["missed"] == []

    def test_error_collection_false_negatives(self) -> None:
        """--errors flag captures missed gold items per article."""
        backend = FakeBackend([[]])  # predicts nothing
        extractor = EntityExtractor(backend=backend)
        gold = self._gold(0, locations=["Paris", "London"])
        er = evaluate_backend(extractor, ["text"], gold, collect_errors=True)

        loc_errors = [e for e in er.errors if e["category"] == "Locations"]
        assert len(loc_errors) == 1
        assert "paris" in loc_errors[0]["missed"]
        assert "london" in loc_errors[0]["missed"]

    def test_no_errors_when_perfect_match(self) -> None:
        """No error entries when all predictions match gold exactly."""
        backend = FakeBackend([[("Paris", "Locations")]])
        extractor = EntityExtractor(backend=backend)
        gold = self._gold(0, locations=["Paris"])
        er = evaluate_backend(extractor, ["text"], gold, collect_errors=True)

        assert er.errors == []

    def test_micro_average_all_none_returns_none(self) -> None:
        """When every category is unsupported, micro-average is None."""
        backend = FakeBackend([[]], supported=frozenset())
        extractor = EntityExtractor(backend=backend)
        gold = self._gold(0, organizations=["Apple"])
        er = evaluate_backend(extractor, ["text"], gold)

        mp, mr, mf = _micro_average(er)
        assert mp is None
        assert mr is None
        assert mf is None

