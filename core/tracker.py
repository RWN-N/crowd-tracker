from typing import Dict, Optional
from datetime import datetime, timedelta, timezone
from pydantic import BaseModel, Field


class PersonTracker(BaseModel):
    id: int
    persons: Dict[str, int] = Field(default={'Unknown': 1})
    last_seen: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))  # last seen ppl or id only
    last_updated: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))  # last recognized

    def is_converge(self, *, MIN_THRESHOLD: int = 5) -> bool:  # repeat detecting until 5 times, then expiry will handle the next loop
        return max(self.persons.items(), key=lambda x: x[1])[1] >= MIN_THRESHOLD

    def is_expired(self, *, delta: Optional[timedelta] = None) -> bool:
        if delta is None:
            delta = timedelta(minutes=1)
        expired = (self.last_updated + delta) <= datetime.now(timezone.utc)
        if expired:  # if expired, update last_datetime
            self.last_updated = datetime.now(timezone.utc)
        return expired

    def best_person(self) -> str:
        return max(self.persons.items(), key=lambda x: x[1])[0]
    
    def confidence(self) -> Dict[str, float]:
        total = sum(self.persons.values())
        return dict(sorted([(name, count / total) for name, count in self.persons.items()], reverse=True, key=lambda x: x[1]))

    def add_persons(self, persons: Dict[str, int]):
        self.last_seen = datetime.now(timezone.utc)
        for name, matches in persons.items():
            self.persons[name] = self.persons.get(name, 0) + matches
        
