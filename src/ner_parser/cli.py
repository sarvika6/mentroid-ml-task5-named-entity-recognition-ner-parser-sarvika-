"""Command-line interface for the NER parser pipeline.

Usage examples::

    # Run with defaults (en_core_web_sm, table output, sample_articles.json)
    python -m ner_parser

    # Specify model and output format
    python -m ner_parser --model en_core_web_trf --format json

    # Write CSV to a file
    python -m ner_parser --format csv --output results.csv

    # Use the HuggingFace backend
    python -m ner_parser --backend hf --model dslim/bert-base-NER

    # Also generate displaCy HTML
    python -m ner_parser --displacy
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from ner_parser.backends import HuggingFaceBackend, SpacyBackend
from ner_parser.config import (
    DEFAULT_FORMAT,
    DEFAULT_INPUT_FILE,
    DEFAULT_SPACY_MODEL,
    DISPLACY_OUTPUT_DIR,
    HF_AGGREGATION_STRATEGY,
    HF_LABEL_MAP,
    HF_MODEL_NAME,
)
from ner_parser.extractor import EntityExtractor
from ner_parser.formatting import (
    format_csv,
    format_json,
    format_table,
    save_displacy,
    write_output,
)

# ---------------------------------------------------------------------------
# Logging setup: diagnostics → stderr, results → stdout.
# ---------------------------------------------------------------------------

_LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"


def _configure_logging(verbose: bool = False) -> None:
    """Set up root logger to write diagnostics to stderr.

    Args:
        verbose: If ``True``, set log level to DEBUG; otherwise INFO.
    """
    level = logging.DEBUG if verbose else logging.INFO
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter(_LOG_FORMAT))
    logging.basicConfig(level=level, handlers=[handler])


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    """Build and return the argument parser.

    Returns:
        Configured :class:`argparse.ArgumentParser` instance.
    """
    parser = argparse.ArgumentParser(
        prog="ner_parser",
        description=(
            "Named Entity Recognition pipeline — extracts Dates, Locations, "
            "and Organizations from news article snippets."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--backend",
        choices=["spacy", "hf"],
        default="spacy",
        help="NER backend to use. 'hf' requires the 'transformers' package.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help=(
            "Model identifier. For spaCy: e.g. 'en_core_web_sm'. "
            "For hf: e.g. 'dslim/bert-base-NER'. "
            "Defaults to backend-specific defaults."
        ),
    )
    parser.add_argument(
        "--input",
        default=DEFAULT_INPUT_FILE,
        help="Path to a JSON file containing a list of article strings.",
    )
    parser.add_argument(
        "--format",
        dest="fmt",
        choices=["table", "json", "csv"],
        default=DEFAULT_FORMAT,
        help="Output format.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output file path. If omitted, results are printed to stdout.",
    )
    parser.add_argument(
        "--displacy",
        action="store_true",
        default=False,
        help=f"Save displaCy HTML visualisations to '{DISPLACY_OUTPUT_DIR}/'.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="Enable DEBUG-level logging.",
    )

    return parser


# ---------------------------------------------------------------------------
# Input loading
# ---------------------------------------------------------------------------


def _load_articles(path: str) -> list[str]:
    """Load a JSON array of article strings from *path*.

    Args:
        path: Path to the JSON file.

    Returns:
        List of article strings.

    Raises:
        SystemExit: On file-not-found, JSON decode errors, or wrong shape.
    """
    p = Path(path)
    if not p.exists():
        logging.getLogger(__name__).error(
            "Input file not found: %s", p.resolve()
        )
        sys.exit(1)

    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        logging.getLogger(__name__).error(
            "Failed to parse JSON from '%s': %s", path, exc
        )
        sys.exit(1)

    if not isinstance(raw, list) or not all(isinstance(a, str) for a in raw):
        logging.getLogger(__name__).error(
            "Expected a JSON array of strings in '%s'.", path
        )
        sys.exit(1)

    if not raw:
        logging.getLogger(__name__).error("Input file '%s' contains no articles.", path)
        sys.exit(1)

    return raw  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Backend factory
# ---------------------------------------------------------------------------


def _build_backend(args: argparse.Namespace) -> SpacyBackend | HuggingFaceBackend:
    """Construct the appropriate backend from parsed CLI arguments.

    Args:
        args: Parsed namespace from :func:`argparse.ArgumentParser.parse_args`.

    Returns:
        A concrete backend instance.

    Raises:
        SystemExit: If the model cannot be loaded.
    """
    logger = logging.getLogger(__name__)

    if args.backend == "hf":
        model = args.model or HF_MODEL_NAME
        try:
            return HuggingFaceBackend(
                model_name=model,
                aggregation_strategy=HF_AGGREGATION_STRATEGY,
                label_map=HF_LABEL_MAP,
            )
        except ImportError as exc:
            logger.error("%s", exc)
            sys.exit(1)

    # spaCy (default)
    model = args.model or DEFAULT_SPACY_MODEL
    try:
        return SpacyBackend(model_name=model)
    except OSError as exc:
        logger.error("%s", exc)
        sys.exit(1)


# ---------------------------------------------------------------------------
# Main entry-point
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    """CLI entry-point.

    Args:
        argv: Argument list (defaults to ``sys.argv[1:]``).
    """
    parser = _build_parser()
    args = parser.parse_args(argv)

    _configure_logging(args.verbose)
    logger = logging.getLogger(__name__)

    articles = _load_articles(args.input)
    logger.info("Loaded %d article(s) from '%s'.", len(articles), args.input)

    backend = _build_backend(args)
    extractor = EntityExtractor(backend=backend)
    results = extractor.extract_batch(articles)

    # ---- Format & write output -------------------------------------------
    match args.fmt:
        case "json":
            content = format_json(results)
        case "csv":
            content = format_csv(results)
        case _:
            content = format_table(results)

    write_output(content, args.output)

    # ---- Optional displaCy export ----------------------------------------
    if args.displacy:
        saved = save_displacy(results, output_dir=DISPLACY_OUTPUT_DIR)
        if saved:
            logger.info(
                "displaCy HTML files saved to '%s/'.", DISPLACY_OUTPUT_DIR
            )


if __name__ == "__main__":
    main()
