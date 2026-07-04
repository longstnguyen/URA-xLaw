"""Grounded legal QA generation and validation."""

from .generator import DatasetGenerator
from .validator import ValidationResult, validate_sample

__all__ = ["DatasetGenerator", "ValidationResult", "validate_sample"]
