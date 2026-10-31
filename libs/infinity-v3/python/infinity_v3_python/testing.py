"""CPU-only executor used by backend tests and local development."""

from __future__ import annotations

from typing import Any

import numpy as np

from .api import PROTOCOL_VERSION

__all__ = ["FakeExecutor", "create_executor", "visible_device_count"]


class FakeExecutor:
    def __init__(self, hidden_size: int = 8, num_labels: int = 2) -> None:
        self.hidden_size = max(int(hidden_size), 1)
        self.num_labels = max(int(num_labels), 1)
        self.closed = False

    def submit(self, route: str, response: Any, payload: dict[str, Any]) -> None:
        if self.closed:
            raise RuntimeError("executor is closed")
        try:
            if route == "embed":
                result = self._embed(payload)
            elif route == "predict":
                result = self._predict(payload)
            else:
                raise ValueError(f"unsupported route {route!r}")
        except BaseException as error:  # noqa: BLE001 - delivered to the Rust side
            response.set_exception(error)
        else:
            response.set_result(result)

    def healthy(self) -> bool:
        return not self.closed

    def close(self) -> None:
        self.closed = True

    def _embed(self, payload: dict[str, Any]) -> np.ndarray:
        seq_len = int(payload["seq_len"])
        pooled = np.asarray(payload["pooled_indices"], dtype=np.int64).reshape(-1)
        raw = np.asarray(payload["raw_indices"], dtype=np.int64).reshape(-1)
        lengths = np.asarray(payload["lengths"], dtype=np.int64).reshape(-1)
        rows: list[np.ndarray] = []
        for index in pooled.tolist():
            rows.append(np.full(self.hidden_size, float(index + 1), dtype=np.float32))
        for index in raw.tolist():
            length = int(lengths[index])
            for token in range(length):
                rows.append(
                    np.full(
                        self.hidden_size,
                        float(index + 1) + float(token) / max(seq_len, 1),
                        dtype=np.float32,
                    )
                )
        if not rows:
            return np.zeros((0, self.hidden_size), dtype=np.float32)
        return np.ascontiguousarray(np.stack(rows, axis=0), dtype=np.float32)

    def _predict(self, payload: dict[str, Any]) -> np.ndarray:
        batch = int(payload["batch_size"])
        logits = np.zeros((batch, self.num_labels), dtype=np.float32)
        for index in range(batch):
            logits[index, index % self.num_labels] = 1.0
        return np.ascontiguousarray(logits, dtype=np.float32)


def create_executor(**_config: Any) -> FakeExecutor:
    return FakeExecutor()


def visible_device_count() -> int:
    return 1
