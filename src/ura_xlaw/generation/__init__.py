"""Grounded legal QA generation and validation."""

from .service import DatasetGenerator
from .validation import ValidationResult, validate_sample

__all__ = ["DatasetGenerator", "ValidationResult", "validate_sample"]
