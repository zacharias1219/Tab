"""Schema validation against contracts/*.schema.json."""

from __future__ import annotations

import json
from functools import cache
from pathlib import Path
from typing import Any

import jsonschema

CONTRACTS = Path(__file__).resolve().parents[1] / "contracts"


class SchemaError(ValueError):
    pass


@cache
def load_schema(name: str) -> dict[str, Any]:
    path = CONTRACTS / f"{name}.schema.json"
    with path.open(encoding="utf-8") as f:
        schema: dict[str, Any] = json.load(f)
    jsonschema.Draft202012Validator.check_schema(schema)
    return schema


def validate(instance: Any, name: str) -> None:
    validator = jsonschema.Draft202012Validator(load_schema(name))
    errors = sorted(validator.iter_errors(instance), key=lambda e: list(e.absolute_path))
    if errors:
        msg = "; ".join(f"{'/'.join(map(str, e.absolute_path)) or '<root>'}: {e.message}" for e in errors[:10])
        raise SchemaError(f"{name}: {msg}")
