from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass
class Metadata:
    run_time_seconds: int
    name: str
    version: int
    problem_name: str
    instance_name: str
    date: str = field(default_factory=lambda: datetime.now().isoformat())
    file_path: Path | None = None


@dataclass
class SavedSolution:
    objectives: list[float] | tuple[float, ...] = field(default_factory=list)
    data: Any = field(default_factory=list)


@dataclass
class SavedRun:
    metadata: Metadata
    solutions: list[SavedSolution]
