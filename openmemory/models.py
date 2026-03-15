"""Data models for OpenMemory."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


class MemoryType(str, Enum):
    """Types of memories that can be stored."""

    FACT = "fact"
    PREFERENCE = "preference"
    EPISODE = "episode"
    ENTITY = "entity"

    @classmethod
    def from_string(cls, value: str) -> MemoryType:
        """Convert a string to a MemoryType, raising ValueError if invalid."""
        try:
            return cls(value.lower())
        except ValueError:
            valid = ", ".join(t.value for t in cls)
            raise ValueError(f"Invalid memory type '{value}'. Valid types: {valid}")


@dataclass
class Memory:
    """A single memory entry."""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    type: MemoryType = MemoryType.FACT
    content: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    importance: float = 0.5
    access_count: int = 0
    embedding: list[float] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert memory to a dictionary for serialization."""
        return {
            "id": self.id,
            "type": self.type.value,
            "content": self.content,
            "metadata": self.metadata,
            "tags": self.tags,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "importance": self.importance,
            "access_count": self.access_count,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Memory:
        """Create a Memory from a dictionary."""
        created_at = data.get("created_at")
        if isinstance(created_at, str):
            created_at = datetime.fromisoformat(created_at)
        elif created_at is None:
            created_at = datetime.now(timezone.utc)

        updated_at = data.get("updated_at")
        if isinstance(updated_at, str):
            updated_at = datetime.fromisoformat(updated_at)
        elif updated_at is None:
            updated_at = datetime.now(timezone.utc)

        return cls(
            id=data.get("id", str(uuid.uuid4())),
            type=MemoryType.from_string(data.get("type", "fact")),
            content=data.get("content", ""),
            metadata=data.get("metadata", {}),
            tags=data.get("tags", []),
            created_at=created_at,
            updated_at=updated_at,
            importance=data.get("importance", 0.5),
            access_count=data.get("access_count", 0),
            embedding=data.get("embedding", []),
        )
