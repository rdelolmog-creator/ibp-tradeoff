"""Raw data ingestion for the IBP Trade-Off Engine.

Implements cleaning-spec.md C-01 (explicit dtypes) and C-02 (schema validation
on arrival). Reads only; no transformation happens here.

The contracts are config/schema.yaml and config/assumptions.yaml. Nothing in
this module hard-codes a path, column name, SKU code or parameter value.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import pandas as pd
import yaml

RAW_TABLES = ("system_a", "system_b", "sku_master")


class SchemaViolation(Exception):
    """Raised when raw data does not match schema.raw.<table>.columns."""


class DataIngestor:
    """Read the three raw CSVs with dtypes forced from the schema contract.

    Parameters
    ----------
    repo_root
        Repository root. config/ is resolved relative to this.
    data_root
        Directory containing raw/. Separate from repo_root so the identical
        pipeline runs against data_primary/ and data_control/ without edits.
    """

    def __init__(
        self,
        repo_root: str | Path | None = None,
        data_root: str | Path | None = None,
    ) -> None:
        self.repo_root = Path(repo_root or Path.cwd()).resolve()
        self.data_root = Path(data_root or (self.repo_root / "data"))
        if not self.data_root.is_absolute():
            self.data_root = (self.repo_root / self.data_root).resolve()

        self.schema = self._read_yaml(self.repo_root / "config" / "schema.yaml")
        self.assumptions = self._read_yaml(
            self.repo_root / "config" / "assumptions.yaml"
        )
        self.rows_in_by_source: Dict[str, int] = {}

    @staticmethod
    def _read_yaml(path: Path) -> Dict[str, Any]:
        if not path.is_file():
            raise FileNotFoundError(f"Contract not found: {path}")
        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(loaded, dict):
            raise ValueError(f"YAML root must be a mapping: {path}")
        return loaded

    # -- C-01 -----------------------------------------------------------

    def _read_dtypes(self, table_key: str) -> Dict[str, Any]:
        """Force str on every schema-declared string column.

        Source codes are zero-padded ('01', '02'). Inference coerces them to
        int64 and drops the padding, which silently breaks the join to
        sku_master and ground_truth: 42 of 60 SKUs match instead of 60.
        """
        spec = self.schema["raw"][table_key]["columns"]
        return {col: str for col, meta in spec.items() if meta["dtype"] == "string"}

    # -- C-02 -----------------------------------------------------------

    def _validate_columns(self, table_key: str, frame: pd.DataFrame) -> None:
        expected = list(self.schema["raw"][table_key]["columns"])
        missing = [c for c in expected if c not in frame.columns]
        unexpected = [c for c in frame.columns if c not in expected]
        if missing or unexpected:
            raise SchemaViolation(
                f"{table_key}: missing={missing}, unexpected={unexpected}. "
                "Schema validation does not coerce — fix the source or the contract."
            )
        for col in expected:
            if self.schema["raw"][table_key]["columns"][col]["dtype"] == "string":
                if not pd.api.types.is_string_dtype(frame[col]):
                    raise SchemaViolation(
                        f"{table_key}.{col} must be string after read, got "
                        f"{frame[col].dtype}. Zero-padding has been lost."
                    )

    def load(self) -> Dict[str, pd.DataFrame]:
        frames: Dict[str, pd.DataFrame] = {}
        for table_key in RAW_TABLES:
            rel = Path(self.schema["raw"][table_key]["file"])
            path = self.data_root / rel.name if rel.parent.name == "raw" else self.data_root / rel
            path = (self.data_root / "raw" / rel.name).resolve()
            if not path.is_file():
                raise FileNotFoundError(f"Raw file not found: {path}")
            frame = pd.read_csv(path, dtype=self._read_dtypes(table_key))
            self._validate_columns(table_key, frame)
            self.rows_in_by_source[table_key] = len(frame)
            frames[table_key] = frame
        return frames
