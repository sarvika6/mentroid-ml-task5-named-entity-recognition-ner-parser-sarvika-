.PHONY: install run test lint evaluate clean

PYTHON   := python
PIP      := pip
SPACY_SM := en_core_web_sm
SPACY_TRF := en_core_web_trf

# ─── Install ──────────────────────────────────────────────────────────────────
install:
	$(PIP) install -e ".[dev]"
	$(PYTHON) -m spacy download $(SPACY_SM)

install-trf:
	$(PYTHON) -m spacy download $(SPACY_TRF)

install-hf:
	$(PIP) install -e ".[dev,hf]"

# ─── Run ──────────────────────────────────────────────────────────────────────
run:
	$(PYTHON) -m ner_parser \
		--input data/sample_articles.json \
		--format table

run-json:
	$(PYTHON) -m ner_parser \
		--input data/sample_articles.json \
		--format json

run-csv:
	$(PYTHON) -m ner_parser \
		--input data/sample_articles.json \
		--format csv \
		--output outputs/results.csv

run-displacy:
	$(PYTHON) -m ner_parser \
		--input data/sample_articles.json \
		--displacy

# ─── Test ─────────────────────────────────────────────────────────────────────
test:
	$(PYTHON) -m pytest tests/ -v --tb=short

test-cov:
	$(PYTHON) -m pytest tests/ --cov=src/ner_parser --cov-report=term-missing

# ─── Lint ─────────────────────────────────────────────────────────────────────
lint:
	ruff check src/ tests/ evaluate.py
	black --check src/ tests/ evaluate.py

lint-fix:
	ruff check --fix src/ tests/ evaluate.py
	black src/ tests/ evaluate.py

# ─── Evaluate ─────────────────────────────────────────────────────────────────
evaluate:
	$(PYTHON) evaluate.py \
		--gold data/gold_labels.json \
		--input data/sample_articles.json \
		--backends sm

evaluate-all:
	$(PYTHON) evaluate.py \
		--gold data/gold_labels.json \
		--input data/sample_articles.json \
		--backends sm trf hf

# ─── Clean ────────────────────────────────────────────────────────────────────
clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	rm -f outputs/*.html outputs/*.csv
