# Refactoring Plan for `src/tvas/tprs.py`

This document outlines the plan to refactor `src/tvas/tprs.py` to improve code readability, maintainability, and practices without changing behavior.

## 1. Encapsulate VLM Logic into a Class (DONE)
- **Goal**: Remove long argument lists and duplicate logic for switching between local and API execution.
- **Action**: Create a `VLMClient` class to handle model initialization, configuration, and text generation.
- **Details**:
    - The class will handle `model`, `processor`, `config`, `api_base`, `api_key`, and `provider_preferences`.
    - It will expose a unified `generate(prompt, image_paths, ...)` method.

## 2. Centralize JSON Cleaning (DONE)
- **Goal**: Remove code duplication for stripping markdown code blocks from JSON responses.
- **Action**: Extract JSON cleaning logic into a helper function `clean_json_response(text: str) -> str`.
- **Details**: Replace repeated logic in `are_photos_in_same_burst`, `parse_analysis_response`, `analyze_photo`, and `select_best_in_burst`.

## 3. Use Context Managers for Temporary Files (DONE)
- **Goal**: Simplify resource management and ensure temporary files are always cleaned up.
- **Action**: Create a context manager `temporary_image_file` or similar.
- **Details**: Refactor `resize_image` and `crop_image` to use this context manager or return a context manager.

## 4. Extract Secondary Analysis from `analyze_photo`
- **Goal**: Reduce the complexity and length of the `analyze_photo` function.
- **Action**: Extract the subject sharpness analysis block into a private function `_analyze_subject_sharpness`.

## 5. Use Enums for Magic Numbers
- **Goal**: Improve code clarity by replacing magic numbers with named constants.
- **Action**: Introduce `IntEnum` for `BlurLevel` (0, 1, 2) and `Rating` (1-5).

## 6. Replace Magic Numbers for EXIF Tags
- **Goal**: Improve readability of EXIF data extraction.
- **Action**: Use named constants or `PIL.ExifTags` for tags `36867` (DateTimeOriginal) and `306` (DateTime).

## 7. Simplify `process_photos_batch`
- **Goal**: Decouple concurrency logic from business logic.
- **Action**: Extract burst processing logic into a separate method or function, potentially part of a `TPRSProcessor` class.
