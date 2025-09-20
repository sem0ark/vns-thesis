from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Iterable


@dataclass
class Metadata:
    run_time_seconds: int
    name: str
    version: int
    problem_name: str
    instance_name: str
    date: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class SavedSolution:
    objectives: Iterable[float] = field(default_factory=list)
    data: Any = field(default_factory=list)


@dataclass
class SavedRun:
    metadata: Metadata
    solutions: list[SavedSolution]
