import cv2
import numpy as np
from typing import NamedTuple, Tuple, Dict, Optional
from datetime import datetime, timedelta, timezone
from pydantic import BaseModel, Field


class LocalDescriptorResult(NamedTuple):
    keypoints: Tuple[cv2.KeyPoint]
    descriptor: np.ndarray

class PersonTracker(BaseModel):
    id: int
    persons: Dict[str, int] = Field(default={'Unknown': 0})
    last_datetime: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    def is_converge(self, *, MIN_THRESHOLD: int = 5) -> bool:  # repeat detecting until 5 times, then expiry will handle the next loop
        return max(self.persons.items(), key=lambda x: x[1])[1] >= MIN_THRESHOLD

    def is_expired(self, *, delta: Optional[timedelta] = None) -> bool:
        if delta is None:
            delta = timedelta(minutes=1)
        expired = (self.last_datetime + delta) <= datetime.now(timezone.utc)
        if expired:  # if expired, update last_datetime
            self.last_datetime = datetime.now(timezone.utc)
        return expired

    def best_person(self) -> str:
        return max(self.persons.items(), key=lambda x: x[1])[0]

    def add_persons(self, persons: Dict[str, int]):
        for name, matches in persons.items():
            self.persons[name] = self.persons.get(name, 0) + matches
