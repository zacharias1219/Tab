"""evalkit: the only code allowed to turn saved probabilities into scores (HF-3).

Self-contained: imports numpy and jsonschema only, never tab/, bench/ or a competitor.
"""

__version__ = "0.1.0"

SCORED_BY = f"evalkit@{__version__}"
