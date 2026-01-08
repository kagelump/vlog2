.PHONY: install test clean build help tag

# Default target
help:
	@echo "Available targets:"
	@echo "  install  - Install dependencies using uv"
	@echo "  test     - Run tests using pytest"
	@echo "  build    - Build the package"
	@echo "  clean    - Remove build artifacts and caches"
	@echo "  tag      - Create and push the next patch version tag"

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

tag:
	@echo "Finding latest tag..."
	@LATEST_TAG=$$(git describe --tags --abbrev=0 2>/dev/null || echo "v0.0.0"); \
	MAJOR=$$(echo $$LATEST_TAG | sed 's/v//;s/\([0-9]*\).*/\1/'); \
	MINOR=$$(echo $$LATEST_TAG | sed 's/v[0-9]*\.\([0-9]*\).*/\1/'); \
	PATCH=$$(echo $$LATEST_TAG | sed 's/v[0-9]*\.[0-9]*\.\([0-9]*\).*/\1/'); \
	NEW_PATCH=$$((PATCH + 1)); \
	NEW_TAG="v$$MAJOR.$$MINOR.$$NEW_PATCH"; \
	echo "Creating tag: $$NEW_TAG"; \
	git tag -a "$$NEW_TAG" -m "Release $$NEW_TAG"; \
	git push origin "$$NEW_TAG"; \
	echo "Tag $$NEW_TAG created and pushed successfully!"
