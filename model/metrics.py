"""Metrics + CSV logging for e-prop LSNN training."""
from __future__ import annotations
import os
import csv


class CSVLogger:
    """Append rows to a CSV file, creating the header on first write."""
    def __init__(self, path: str, columns: list):
        self.path = path
        self.columns = list(columns)
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self._header_written = False
        self._f = None
        self._writer = None

    def _ensure(self):
        if self._f is None:
            exists = os.path.exists(self.path)
            self._f = open(self.path, "a", newline="")
            self._writer = csv.DictWriter(self._f, fieldnames=self.columns)
            if not exists:
                self._writer.writeheader()
                self._f.flush()
            self._header_written = True

    def log(self, row: dict):
        self._ensure()
        self._writer.writerow({k: row.get(k, "") for k in self.columns})
        self._f.flush()

    def close(self):
        if self._f is not None:
            self._f.close()
            self._f = None
