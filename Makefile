.PHONY: install test clean build help

# Default target
help:
	@echo "Available targets:"
	@echo "  install  - Install dependencies using uv"
	@echo "  test     - Run tests using pytest"
	@echo "  build    - Build the package"
	@echo "  clean    - Remove build artifacts and caches"

install:
	uv sync --all-extras --dev

test:
	uv run pytest

build:
	uv build

clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	rm -rf .pytest_cache
	rm -rf .coverage
	rm -rf htmlcov/
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
