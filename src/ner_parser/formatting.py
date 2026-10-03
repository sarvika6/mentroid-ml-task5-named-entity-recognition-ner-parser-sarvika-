"""Output formatters for NER results.

Provides:
- :func:`format_table`  – plain-text table for human consumption (stdout).
- :func:`format_json`   – compact or pretty JSON.
- :func:`format_csv`    – RFC 4180-compliant CSV with header row.
- :func:`save_displacy` – displaCy HTML visualisation saved to disk.
"""

from __future__ import annotations

import csv
import io
import json
import logging
from pathlib import Path

from ner_parser.config import (
    DISPLACY_FILENAME_TEMPLATE,
    DISPLACY_OUTPUT_DIR,
    OUTPUT_CATEGORIES,
    TABLE_COL_WIDTHS,
)
from ner_parser.models import ArticleResult

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_HEADER_SEP_CHAR = "-"
_COL_SEP = " | "


def _header_line(widths: list[int]) -> str:
    """Return a separator line sized to the column widths."""
    return "-" + "-+-".join(_HEADER_SEP_CHAR * w for w in widths) + "-"


# ---------------------------------------------------------------------------
# Table formatter
# ---------------------------------------------------------------------------


def format_table(results: list[ArticleResult], categories: list[str] | None = None) -> str:
    """Render NER results as a human-readable plain-text table.

    Args:
        results: List of :class:`~ner_parser.models.ArticleResult` objects.
        categories: Categories to include. Defaults to
            :data:`~ner_parser.config.OUTPUT_CATEGORIES`.

    Returns:
        A multi-line string ready to be printed to stdout.
    """
    cats = categories or OUTPUT_CATEGORIES
    w = TABLE_COL_WIDTHS

    col_widths = [w["article"], w["category"], w["entity"], w["start"], w["end"]]

    def row(*cols: str) -> str:
        padded = [
            str(c).ljust(col_widths[i]) if i < len(col_widths) - 1 else str(c).rjust(col_widths[i])
            for i, c in enumerate(cols)
        ]
        return _COL_SEP.join(padded)

    lines: list[str] = []
    header = row("Art#", "Category", "Entity", "Start", "End")
    sep = _header_line(col_widths)

    lines.append(sep)
    lines.append(header)
    lines.append(sep)

    for result in results:
        # Collect filtered entities
        filtered = [e for e in result.entities if e.label in cats]
        if not filtered:
            lines.append(row(str(result.article_id), "(none)", "-", "-", "-"))
            continue
        for i, ent in enumerate(filtered):
            art_col = str(result.article_id) if i == 0 else ""
            lines.append(
                row(
                    art_col,
                    ent.label,
                    ent.text[:w["entity"]],
                    str(ent.start),
                    str(ent.end),
                )
            )
        lines.append(sep)

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# JSON formatter
# ---------------------------------------------------------------------------


def format_json(results: list[ArticleResult], indent: int = 2) -> str:
    """Serialise results to a JSON string.

    Args:
        results: List of :class:`~ner_parser.models.ArticleResult` objects.
        indent: JSON indentation level. Use ``None`` for compact output.

    Returns:
        A UTF-8-safe JSON string.
    """
    payload = [r.to_dict() for r in results]
    return json.dumps(payload, indent=indent, ensure_ascii=False)


# ---------------------------------------------------------------------------
# CSV formatter
# ---------------------------------------------------------------------------


def format_csv(results: list[ArticleResult], categories: list[str] | None = None) -> str:
    """Serialise results to a CSV string with a header row.

    Args:
        results: List of :class:`~ner_parser.models.ArticleResult` objects.
        categories: Categories to include.

    Returns:
        RFC 4180-compliant CSV content as a string.
    """
    cats = categories or OUTPUT_CATEGORIES
    buf = io.StringIO()
    writer = csv.writer(buf, lineterminator="\n")
    writer.writerow(["article_id", "category", "entity", "start", "end"])

    for result in results:
        for ent in result.entities:
            if ent.label in cats:
                writer.writerow([result.article_id, ent.label, ent.text, ent.start, ent.end])

    return buf.getvalue()


# ---------------------------------------------------------------------------
# displaCy HTML visualisation
# ---------------------------------------------------------------------------


def save_displacy(
    results: list[ArticleResult],
    output_dir: str | Path = DISPLACY_OUTPUT_DIR,
) -> list[Path]:
    """Generate displaCy HTML visualisations and save to *output_dir*.

    Each article gets its own ``.html`` file.  The function requires spaCy
    to be installed (it is an optional import here so that non-spaCy
    workflows are not forced to depend on it).

    Args:
        results: List of :class:`~ner_parser.models.ArticleResult` objects.
        output_dir: Directory where HTML files are written.  Created if it
            does not exist.

    Returns:
        List of :class:`~pathlib.Path` objects pointing to created files.
    """
    try:
        import spacy  # noqa: PLC0415
        from spacy import displacy  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError(
            "spaCy is required for displaCy visualisations. "
            "Install it with: pip install spacy"
        ) from exc

    # We need the NLP object to parse again so displacy can render.
    # Import lazily to avoid circular dependencies with extractor.
    from ner_parser.config import DEFAULT_SPACY_MODEL  # noqa: PLC0415

    try:
        nlp = spacy.load(DEFAULT_SPACY_MODEL)
    except OSError:
        logger.warning(
            "Could not load '%s' for displaCy. Skipping HTML export.",
            DEFAULT_SPACY_MODEL,
        )
        return []

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []

    for result in results:
        if not result.source.strip():
            continue
        doc = nlp(result.source)
        html = displacy.render(doc, style="ent", page=True)
        filename = DISPLACY_FILENAME_TEMPLATE.format(index=result.article_id)
        dest = out_dir / filename
        dest.write_text(html, encoding="utf-8")
        logger.info("Saved displaCy HTML → %s", dest)
        saved.append(dest)

    return saved


# ---------------------------------------------------------------------------
# Write helper
# ---------------------------------------------------------------------------


def write_output(content: str, path: str | Path | None = None) -> None:
    """Write *content* to *path*, or stdout if *path* is ``None``.

    Args:
        content: Text to write.
        path: Destination file path.  If ``None``, writes to stdout.
    """
    if path is None:
        import sys  # noqa: PLC0415

        # Ensure UTF-8 output even on Windows (cp1252) consoles.
        try:
            sys.stdout.buffer.write(content.encode("utf-8"))
            if not content.endswith("\n"):
                sys.stdout.buffer.write(b"\n")
            sys.stdout.buffer.flush()
        except AttributeError:
            # Fallback when stdout has no .buffer (e.g. captured in tests)
            sys.stdout.write(content)
            if not content.endswith("\n"):
                sys.stdout.write("\n")
    else:
        dest = Path(path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(content, encoding="utf-8")
        logger.info("Output written to %s.", dest)
