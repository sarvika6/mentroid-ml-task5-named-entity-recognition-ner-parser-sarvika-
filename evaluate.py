"""Evaluation module for the NER pipeline.

Scoring rule
------------
Case-insensitive exact text match, deduplicated per article and category,
micro-averaged across articles.  A prediction matches a gold entity if and
only if ``pred.strip().lower() == gold.strip().lower()``.  This is strict:
broad spans like *"since January 2023"* will not match gold *"January 2023"*.

A future --lenient flag could add token-overlap / containment matching to
produce a strict/lenient pair, as discussed in the README.

Usage::

    python evaluate.py                              # sm only (default)
    python evaluate.py --backends sm trf hf         # three-way comparison
    python evaluate.py --backends sm --errors       # dump FP/FN per article
    python evaluate.py --backends sm --output results.md
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ner_parser.backends import HuggingFaceBackend, SpacyBackend
from ner_parser.config import (
    DEFAULT_SPACY_MODEL,
    HF_AGGREGATION_STRATEGY,
    HF_LABEL_MAP,
    HF_MODEL_NAME,
    LARGE_SPACY_MODEL,
    OUTPUT_CATEGORIES,
)
from ner_parser.extractor import EntityExtractor

logger = logging.getLogger(__name__)

_LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class CategoryMetrics:
    """Precision, recall, F1, TP/FP/FN counts, and gold support for one category.

    Attributes:
        category: Category name (e.g. ``"Dates"``).
        precision: TP / (TP + FP), or ``None`` if the category is not
            supported by the backend or if there are no predictions at all.
        recall: TP / (TP + FN), or ``None`` if there are no gold items.
        f1: Harmonic mean of precision and recall, or ``None`` when either
            component is ``None``.
        tp: True-positive count (aggregated across articles).
        fp: False-positive count.
        fn: False-negative count.
        support: Total gold entity count across all articles (denominator
            for recall).
    """

    category: str
    precision: float | None
    recall: float | None
    f1: float | None
    tp: int = 0
    fp: int = 0
    fn: int = 0
    support: int = 0


@dataclass
class EvaluationResult:
    """Aggregated evaluation results for one backend across all articles.

    Attributes:
        backend_name: Human-readable backend identifier.
        per_category: Metrics broken out by category.
        errors: Per-article FP/FN details; populated only when
            ``collect_errors=True`` is passed to :func:`evaluate_backend`.
    """

    backend_name: str
    per_category: list[CategoryMetrics] = field(default_factory=list)
    errors: list[dict[str, object]] = field(default_factory=list)

    def get(self, category: str) -> CategoryMetrics | None:
        """Return metrics for *category*, or ``None`` if not found."""
        return next((m for m in self.per_category if m.category == category), None)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _normalise(text: str) -> str:
    """Lowercase and strip whitespace for case-insensitive comparison.

    Args:
        text: Raw entity surface form.

    Returns:
        Normalised string.
    """
    return text.strip().lower()


def _fmt(value: float | None, decimals: int = 2) -> str:
    """Format a metric value for Markdown table display.

    Rounds at display time so that internal arithmetic stays at full float
    precision.

    Args:
        value: The metric float, or ``None`` if undefined.
        decimals: Number of decimal places.

    Returns:
        Formatted string, e.g. ``"0.88"`` or ``"n/a"``.
    """
    return "n/a" if value is None else f"{value:.{decimals}f}"


# ---------------------------------------------------------------------------
# Core evaluation logic
# ---------------------------------------------------------------------------


def evaluate_backend(
    extractor: EntityExtractor,
    articles: list[str],
    gold: dict[int, dict[str, list[str]]],
    categories: list[str] | None = None,
    *,
    collect_errors: bool = False,
) -> EvaluationResult:
    """Run *extractor* and compute metrics against *gold*.

    Scoring rule: case-insensitive exact text match, deduplicated per article
    and category, micro-averaged across articles.

    Categories not in ``extractor.backend.supported_categories`` are skipped
    entirely: their ``CategoryMetrics`` fields are all ``None``.

    Args:
        extractor: :class:`~ner_parser.extractor.EntityExtractor` to evaluate.
        articles: List of raw article strings.
        gold: Dict keyed by ``article_id`` → category → list of gold texts.
        categories: Subset of categories to evaluate.  Defaults to
            :data:`~ner_parser.config.OUTPUT_CATEGORIES`.
        collect_errors: When ``True``, populate :attr:`EvaluationResult.errors`
            with per-article FP/FN details.

    Returns:
        An :class:`EvaluationResult` with per-category metrics.
    """
    cats = categories or OUTPUT_CATEGORIES
    supported = extractor.backend.supported_categories
    results = extractor.extract_batch(articles)

    tp_counts: dict[str, int] = dict.fromkeys(cats, 0)
    fp_counts: dict[str, int] = dict.fromkeys(cats, 0)
    fn_counts: dict[str, int] = dict.fromkeys(cats, 0)
    support_counts: dict[str, int] = dict.fromkeys(cats, 0)
    errors: list[dict[str, object]] = []

    for article_result in results:
        idx = article_result.article_id
        gold_article = gold.get(idx, {})

        for cat in cats:
            if cat not in supported:
                continue  # will appear as None in final metrics

            pred_set = {_normalise(e.text) for e in article_result.by_category(cat)}
            gold_list: list[str] = gold_article.get(cat, [])
            gold_set = {_normalise(g) for g in gold_list}

            support_counts[cat] += len(gold_set)
            tp_counts[cat] += len(pred_set & gold_set)
            fp_counts[cat] += len(pred_set - gold_set)
            fn_counts[cat] += len(gold_set - pred_set)

            if collect_errors and (pred_set - gold_set or gold_set - pred_set):
                errors.append(
                    {
                        "article": idx,
                        "category": cat,
                        "false_positives": sorted(pred_set - gold_set),
                        "missed": sorted(gold_set - pred_set),
                    }
                )

    per_category: list[CategoryMetrics] = []
    for cat in cats:
        if cat not in supported:
            per_category.append(
                CategoryMetrics(
                    category=cat,
                    precision=None,
                    recall=None,
                    f1=None,
                    support=support_counts.get(cat, 0),
                )
            )
            continue

        tp = tp_counts[cat]
        fp = fp_counts[cat]
        fn = fn_counts[cat]
        sup = support_counts[cat]

        # Store raw floats; round only at render time.
        precision: float | None = tp / (tp + fp) if (tp + fp) > 0 else None
        recall: float | None = tp / (tp + fn) if (tp + fn) > 0 else None

        if precision is not None and recall is not None:
            f1: float | None = (
                2 * precision * recall / (precision + recall)
                if (precision + recall) > 0
                else 0.0
            )
        else:
            f1 = None

        per_category.append(
            CategoryMetrics(
                category=cat,
                precision=precision,
                recall=recall,
                f1=f1,
                tp=tp,
                fp=fp,
                fn=fn,
                support=sup,
            )
        )

    return EvaluationResult(
        backend_name=extractor.backend.name,
        per_category=per_category,
        errors=errors,
    )


# ---------------------------------------------------------------------------
# Markdown table renderer
# ---------------------------------------------------------------------------


def _micro_average(er: EvaluationResult) -> tuple[float | None, float | None, float | None]:
    """Compute micro-averaged P/R/F1 over supported categories only.

    Args:
        er: Evaluation result containing per-category TP/FP/FN counts.

    Returns:
        Tuple of (micro_precision, micro_recall, micro_f1), each possibly None.
    """
    total_tp = sum(m.tp for m in er.per_category if m.precision is not None)
    total_fp = sum(m.fp for m in er.per_category if m.precision is not None)
    total_fn = sum(m.fn for m in er.per_category if m.precision is not None)

    micro_p: float | None = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else None
    micro_r: float | None = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else None

    if micro_p is not None and micro_r is not None:
        micro_f: float | None = (
            2 * micro_p * micro_r / (micro_p + micro_r) if (micro_p + micro_r) > 0 else 0.0
        )
    else:
        micro_f = None

    return micro_p, micro_r, micro_f


def render_markdown_table(
    eval_results: list[EvaluationResult],
    categories: list[str],
) -> str:
    """Render evaluation results as a Markdown comparison table.

    Includes a *Support (gold)* column showing how many gold entities exist
    per category, and a micro-average footer row. Metrics are rounded at
    display time only. Unsupported categories render as ``n/a``.

    Args:
        eval_results: List of :class:`EvaluationResult` objects, one per backend.
        categories: Categories to include as rows (in order).

    Returns:
        Multi-line Markdown string.
    """
    lines: list[str] = []
    backend_names = [e.backend_name for e in eval_results]

    header_cells = ["Category", "Support (gold)"] + [
        f"{name}<br>P / R / F1 (TP / FP / FN)" for name in backend_names
    ]
    lines.append("| " + " | ".join(header_cells) + " |")
    lines.append("|" + "|".join([":---", ":---:"] + [":---:"] * len(backend_names)) + "|")

    for cat in categories:
        # Support is the same regardless of backend; take from first that has it.
        support = next(
            (m.support for er in eval_results if (m := er.get(cat)) is not None), 0
        )
        row_cells: list[str] = [cat, str(support)]
        for er in eval_results:
            m = er.get(cat)
            if m is None:
                row_cells.append("n/a / n/a / n/a")
            else:
                p, r, f = _fmt(m.precision), _fmt(m.recall), _fmt(m.f1)
                row_cells.append(f"{p} / {r} / {f} ({m.tp} / {m.fp} / {m.fn})")
        lines.append("| " + " | ".join(row_cells) + " |")

    # Micro-average footer
    lines.append("|" + "|".join(["---", "---"] + ["---"] * len(backend_names)) + "|")
    micro_cells: list[str] = ["**Micro-avg**", ""]
    for er in eval_results:
        mp, mr, mf = _micro_average(er)
        micro_cells.append(f"{_fmt(mp)} / {_fmt(mr)} / {_fmt(mf)}")
    lines.append("| " + " | ".join(micro_cells) + " |")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Error dump renderer
# ---------------------------------------------------------------------------


def render_errors(eval_results: list[EvaluationResult]) -> str:
    """Render per-article false positive and false negative details.

    Args:
        eval_results: List of :class:`EvaluationResult` containing errors.

    Returns:
        Markdown-formatted string with one section per backend.
    """
    lines: list[str] = []
    for er in eval_results:
        lines.append(f"\n### Errors — {er.backend_name}\n")
        if not er.errors:
            lines.append("*(no errors — all predictions match gold)*\n")
            continue
        for e in er.errors:
            lines.append(f"**Article {e['article']} | {e['category']}**")
            fps: list[str] = e["false_positives"]  # type: ignore[assignment]
            fns: list[str] = e["missed"]  # type: ignore[assignment]
            if fps:
                lines.append(
                    "- False positives (predicted, not in gold): "
                    + ", ".join(f"`{x}`" for x in fps)
                )
            if fns:
                lines.append(
                    "- Missed (in gold, not predicted): "
                    + ", ".join(f"`{x}`" for x in fns)
                )
            lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    """Build and return the argument parser.

    Returns:
        Configured :class:`argparse.ArgumentParser`.
    """
    p = argparse.ArgumentParser(
        prog="evaluate",
        description=(
            "Evaluate NER backends against a gold-label set. "
            "Scoring: case-insensitive exact text match, micro-averaged."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--gold", default="data/gold_labels.json", help="Gold labels JSON file.")
    p.add_argument("--input", default="data/sample_articles.json", help="Articles JSON file.")
    p.add_argument(
        "--backends",
        nargs="+",
        default=["sm"],
        choices=["sm", "trf", "hf"],
        help=(
            "Backends to compare. "
            "'sm'=en_core_web_sm, 'trf'=en_core_web_trf, 'hf'=dslim/bert-base-NER."
        ),
    )
    p.add_argument(
        "--errors",
        action="store_true",
        default=False,
        help="Append a per-article false-positive / false-negative dump after the table.",
    )
    p.add_argument(
        "--output",
        default=None,
        metavar="FILE",
        help="Write Markdown output to FILE instead of stdout.",
    )
    p.add_argument("--verbose", action="store_true")
    return p


def _load_json(path: str) -> Any:  # noqa: ANN401
    """Load and parse a JSON file, exiting on error.

    Args:
        path: Path to the JSON file.

    Returns:
        Parsed Python object.
    """
    p = Path(path)
    if not p.exists():
        logger.error("File not found: %s", p.resolve())
        sys.exit(1)
    return json.loads(p.read_text(encoding="utf-8"))


def _gold_by_id(raw: list[dict[str, Any]]) -> dict[int, dict[str, list[str]]]:
    """Convert a gold list to a dict keyed by ``article_id``.

    Keying by ID rather than relying on list position makes the loader robust
    to reordering or gaps in the JSON file.

    Args:
        raw: List of gold-label dicts, each with an ``article_id`` key.

    Returns:
        Dict mapping article_id (int) to its label dict.

    Raises:
        SystemExit: If any entry is missing ``article_id``.
    """
    result: dict[int, dict[str, list[str]]] = {}
    for entry in raw:
        aid = entry.get("article_id")
        if aid is None:
            logger.error("Gold entry is missing 'article_id': %s", entry)
            sys.exit(1)
        result[int(aid)] = entry
    return result


def main(argv: list[str] | None = None) -> None:
    """CLI entry-point for evaluation.

    Args:
        argv: Argument list. Defaults to ``sys.argv[1:]``.
    """
    args = _build_parser().parse_args(argv)

    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=level, format=_LOG_FORMAT, stream=sys.stderr)

    articles: list[str] = _load_json(args.input)
    gold_raw: list[dict[str, Any]] = _load_json(args.gold)
    gold = _gold_by_id(gold_raw)

    if len(articles) != len(gold):
        logger.error(
            "Mismatch: %d articles but %d gold entries.", len(articles), len(gold)
        )
        sys.exit(1)

    # Build extractors, tracking which backends were skipped.
    extractors: list[EntityExtractor] = []
    skipped: list[str] = []
    backend_map: dict[str, Any] = {
        "sm": lambda: SpacyBackend(model_name=DEFAULT_SPACY_MODEL),
        "trf": lambda: SpacyBackend(model_name=LARGE_SPACY_MODEL),
        "hf": lambda: HuggingFaceBackend(
            model_name=HF_MODEL_NAME,
            aggregation_strategy=HF_AGGREGATION_STRATEGY,
            label_map=HF_LABEL_MAP,
        ),
    }

    for key in args.backends:
        try:
            backend = backend_map[key]()
            extractors.append(EntityExtractor(backend=backend))
        except (OSError, ImportError) as exc:
            logger.warning("Skipping backend '%s': %s", key, exc)
            skipped.append(key)

    if not extractors:
        logger.error("No backends could be loaded. Exiting.")
        sys.exit(1)

    # Evaluate each loaded backend.
    eval_results: list[EvaluationResult] = []
    for extractor in extractors:
        logger.info("Evaluating backend '%s'...", extractor.backend.name)
        er = evaluate_backend(
            extractor,
            articles,
            gold,
            collect_errors=args.errors,
        )
        eval_results.append(er)

    # Build output.
    output_lines: list[str] = []

    if skipped:
        output_lines.append(
            f"> **Skipped backends (failed to load):** {', '.join(skipped)}. "
            "Install the required packages and re-run to include them.\n"
        )

    output_lines.append("## NER Evaluation Results\n")
    output_lines.append(
        "> **Scoring:** case-insensitive exact text match, micro-averaged across articles.\n"
        "> TP / FP / FN counts are shown alongside P / R / F1.\n"
        "> `n/a` means the category is not supported by that backend.\n"
    )
    output_lines.append(render_markdown_table(eval_results, OUTPUT_CATEGORIES))

    if args.errors:
        output_lines.append(render_errors(eval_results))

    content = "\n".join(output_lines) + "\n"

    if args.output:
        dest = Path(args.output)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(content, encoding="utf-8")
        logger.info("Evaluation written to %s.", dest)
    else:
        sys.stdout.write(content)
        sys.stdout.flush()


if __name__ == "__main__":
    main()
