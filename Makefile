.PHONY: help install setup-db clean-db test-db all generate-data zip

# Default target
all: install generate-data setup-db

help:
	@echo "Available targets:"
	@echo "  make install       - Install all Python dependencies using uv"
	@echo "  make generate-data - Generate synthetic healthcare data CSVs"
	@echo "  make setup-db      - Create DuckDB database from CSV files in data/"
	@echo "  make clean-db      - Remove database for fresh setup"
	@echo "  make test-db       - Validate database exists and test query functionality"
	@echo "  make zip           - Create distributable zip of the project"
	@echo "  make all           - Run install + generate-data + setup-db (default)"
	@echo "  make help          - Show this help message"

install:
	@echo "Installing dependencies with uv..."
	uv sync
	@echo "✓ Dependencies installed"

setup-db:
	@echo "Setting up DuckDB database from CSV files..."
	uv run python scripts/load_archives_to_duckdb.py
	@echo "✓ Database setup complete"

clean-db:
	@echo "Cleaning database..."
	rm -f healthcare.duckdb
	@echo "✓ Cleanup complete"

test-db:
	@echo "Testing database setup..."
	@test -f healthcare.duckdb || (echo "ERROR: healthcare.duckdb not found. Run 'make setup-db' first." && exit 1)
	@echo "✓ Database file exists"
	@echo ""
	@echo "Running query tests..."
	uv run python tools/db_query.py
	@echo ""
	@echo "✓ All tests passed"

generate-data:
	@echo "Extracting reference data..."
	unzip -o data/input.zip -d data/
	@echo "Generating synthetic healthcare data..."
	uv run python data/generate_data.py
	@echo "Data generation complete"

zip:
	@echo "Creating labs-assessment.zip..."
	@rm -f labs-assessment.zip
	zip -r labs-assessment.zip \
		chat.py \
		main.py \
		Makefile \
		README.md \
		Assessment.md \
		pyproject.toml \
		uv.lock \
		env.template \
		.python-version \
		.gitignore \
		data/input.zip \
		data/generate_data.py \
		data/simulate_data_plan.md \
		scripts/ \
		tools/
	@echo "Created labs-assessment.zip"
