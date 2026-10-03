"""NER Parser — production-quality Named Entity Recognition pipeline.

Extracts Dates, Locations, and Organizations from news article snippets
using spaCy (default) or HuggingFace Transformers.

Public API::

    from ner_parser import EntityExtractor
    extractor = EntityExtractor()
    result = extractor.extract("Apple opened a new office in Mumbai last Tuesday.")
    print(result.entities)
"""

from __future__ import annotations

from ner_parser.extractor import EntityExtractor
from ner_parser.models import ArticleResult, Entity

__all__ = ["EntityExtractor", "ArticleResult", "Entity"]
__version__ = "0.1.0"
