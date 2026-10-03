"""Configuration module for the NER parser pipeline.

Centralises all constants, label mappings, and configurable parameters.
No magic strings should appear elsewhere in the codebase.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Model identifiers
# ---------------------------------------------------------------------------

#: Default spaCy pipeline to load.
DEFAULT_SPACY_MODEL: str = "en_core_web_sm"

#: Larger, transformer-based spaCy model for comparison.
LARGE_SPACY_MODEL: str = "en_core_web_trf"

#: HuggingFace model used via the *transformers* backend.
HF_MODEL_NAME: str = "dslim/bert-base-NER"

#: HuggingFace aggregation strategy.
HF_AGGREGATION_STRATEGY: str = "simple"

# ---------------------------------------------------------------------------
# Label mappings
# ---------------------------------------------------------------------------

#: Maps spaCy entity labels → canonical category names used in output.
SPACY_LABEL_MAP: dict[str, str] = {
    "DATE": "Dates",
    "GPE": "Locations",
    "LOC": "Locations",
    "ORG": "Organizations",
}

#: Maps HuggingFace NER tags → canonical category names.
#: The HF model (dslim/bert-base-NER) does NOT produce DATE tags.
HF_LABEL_MAP: dict[str, str] = {
    "LOC": "Locations",
    "ORG": "Organizations",
    "PER": "Persons",         # captured but not shown by default
    "MISC": "Miscellaneous",  # captured but not shown by default
}

#: Categories that should appear in the final output (filterable).
OUTPUT_CATEGORIES: list[str] = ["Dates", "Locations", "Organizations"]

# ---------------------------------------------------------------------------
# Batch / performance settings
# ---------------------------------------------------------------------------

#: Number of texts processed in each *nlp.pipe()* batch.
PIPE_BATCH_SIZE: int = 8

# ---------------------------------------------------------------------------
# Output / formatting constants
# ---------------------------------------------------------------------------

#: Table column widths for the rich/plain console formatter.
TABLE_COL_WIDTHS: dict[str, int] = {
    "article": 6,
    "category": 14,
    "entity": 40,
    "start": 6,
    "end": 5,
}

#: Default output directory for displaCy HTML files.
DISPLACY_OUTPUT_DIR: str = "outputs"

#: File name template for displaCy HTML. Placeholder: article index (0-based).
DISPLACY_FILENAME_TEMPLATE: str = "article_{index:02d}_ner.html"

# ---------------------------------------------------------------------------
# CLI defaults
# ---------------------------------------------------------------------------

#: Default input file relative to the project root.
DEFAULT_INPUT_FILE: str = "data/sample_articles.json"

#: Default output format for the CLI.
DEFAULT_FORMAT: str = "table"
