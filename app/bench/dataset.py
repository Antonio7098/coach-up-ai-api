from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Literal, Optional

TaskType = Literal[
    "micro_correction",
    "assertiveness",
    "word_choice",
    "summary",
    "multi_turn",
    "interaction_classification",
    "assessment",
]


@dataclass
class Message:
    role: Literal["user", "assistant"]
    content: str


@dataclass
class DatasetItem:
    id: str
    task: TaskType
    focusId: Optional[str]
    rubricVersion: str
    input: Dict[str, Any]
    # Expected "labels" vary by task. Keep generic to avoid overfitting now.
    expected: Optional[Dict[str, Any]] = None
    notes: Optional[str] = None


class Dataset:
    def __init__(self, path: Path):
        self.path = path
        if not self.path.exists():
            raise FileNotFoundError(f"Dataset file not found: {self.path}")

    def __iter__(self) -> Iterable[DatasetItem]:
        with self.path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                yield DatasetItem(
                    id=obj["id"],
                    task=obj["task"],
                    focusId=obj.get("focusId"),
                    rubricVersion=obj.get("rubricVersion", "v1"),
                    input=obj["input"],
                    expected=obj.get("expected"),
                    notes=obj.get("notes"),
                )

    def load_all(self) -> List[DatasetItem]:
        return list(iter(self))
