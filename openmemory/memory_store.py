"""Core memory storage engine backed by SQLite."""

from __future__ import annotations

import json
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Generator

from openmemory.embeddings import (
    EmbeddingProvider,
    LocalEmbeddingProvider,
    cosine_similarity,
)
from openmemory.models import Memory, MemoryType


_DEFAULT_DB_DIR = Path.home() / ".openmemory"
_DEFAULT_DB_PATH = _DEFAULT_DB_DIR / "memories.db"

_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS memories (
    id TEXT PRIMARY KEY,
    type TEXT NOT NULL,
    content TEXT NOT NULL,
    metadata TEXT NOT NULL DEFAULT '{}',
    tags TEXT NOT NULL DEFAULT '[]',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    importance REAL NOT NULL DEFAULT 0.5,
    access_count INTEGER NOT NULL DEFAULT 0,
    embedding TEXT NOT NULL DEFAULT '[]'
);
"""

_CREATE_INDEX_SQL = [
    "CREATE INDEX IF NOT EXISTS idx_memories_type ON memories(type);",
    "CREATE INDEX IF NOT EXISTS idx_memories_importance ON memories(importance);",
    "CREATE INDEX IF NOT EXISTS idx_memories_created_at ON memories(created_at);",
]


class MemoryStore:
    """SQLite-backed memory storage with semantic search."""

    def __init__(
        self,
        db_path: str | Path | None = None,
        embedding_provider: EmbeddingProvider | None = None,
    ) -> None:
        if db_path is None:
            _DEFAULT_DB_DIR.mkdir(parents=True, exist_ok=True)
            db_path = _DEFAULT_DB_PATH
        elif db_path != ":memory:":
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)

        self.db_path = str(db_path)
        self.embedding_provider = embedding_provider or LocalEmbeddingProvider()

        # For :memory: databases, keep a persistent connection so the DB
        # survives across method calls.
        self._persistent_conn: sqlite3.Connection | None = None
        if self.db_path == ":memory:":
            self._persistent_conn = sqlite3.connect(":memory:")
            self._persistent_conn.row_factory = sqlite3.Row

        self._init_db()

    @contextmanager
    def _connection(self) -> Generator[sqlite3.Connection, None, None]:
        """Context manager for database connections.

        For persistent (in-memory) connections, yields the same connection
        and commits. For file-based DBs, creates a new connection, commits,
        and closes.
        """
        if self._persistent_conn is not None:
            yield self._persistent_conn
            self._persistent_conn.commit()
        else:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            try:
                yield conn
                conn.commit()
            finally:
                conn.close()

    def _init_db(self) -> None:
        """Initialize the database schema."""
        with self._connection() as conn:
            conn.execute(_CREATE_TABLE_SQL)
            for idx_sql in _CREATE_INDEX_SQL:
                conn.execute(idx_sql)

    def _row_to_memory(self, row: sqlite3.Row) -> Memory:
        """Convert a database row to a Memory object."""
        return Memory(
            id=row["id"],
            type=MemoryType.from_string(row["type"]),
            content=row["content"],
            metadata=json.loads(row["metadata"]),
            tags=json.loads(row["tags"]),
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
            importance=row["importance"],
            access_count=row["access_count"],
            embedding=json.loads(row["embedding"]),
        )

    def add_memory(
        self,
        type: str,
        content: str,
        metadata: dict[str, Any] | None = None,
        tags: list[str] | None = None,
        importance: float = 0.5,
    ) -> Memory:
        """Store a new memory.

        Args:
            type: Memory type (fact, preference, episode, entity).
            content: The text content of the memory.
            metadata: Optional metadata dictionary.
            tags: Optional list of tags.
            importance: Importance score from 0 to 1.

        Returns:
            The created Memory object.
        """
        memory_type = MemoryType.from_string(type)
        now = datetime.now(timezone.utc)
        embedding = self.embedding_provider.embed(content)

        memory = Memory(
            id=str(uuid.uuid4()),
            type=memory_type,
            content=content,
            metadata=metadata or {},
            tags=tags or [],
            created_at=now,
            updated_at=now,
            importance=max(0.0, min(1.0, importance)),
            access_count=0,
            embedding=embedding,
        )

        with self._connection() as conn:
            conn.execute(
                """INSERT INTO memories (id, type, content, metadata, tags,
                   created_at, updated_at, importance, access_count, embedding)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    memory.id,
                    memory.type.value,
                    memory.content,
                    json.dumps(memory.metadata),
                    json.dumps(memory.tags),
                    memory.created_at.isoformat(),
                    memory.updated_at.isoformat(),
                    memory.importance,
                    memory.access_count,
                    json.dumps(memory.embedding),
                ),
            )

        return memory

    def search_memories(
        self,
        query: str,
        type_filter: str = "",
        limit: int = 5,
        min_importance: float = 0.0,
    ) -> list[Memory]:
        """Search memories by semantic similarity.

        Args:
            query: Search query text.
            type_filter: Optional memory type to filter by.
            limit: Maximum number of results.
            min_importance: Minimum importance threshold.

        Returns:
            List of memories sorted by relevance.
        """
        if not query.strip():
            return []

        query_embedding = self.embedding_provider.embed(query)

        sql = "SELECT * FROM memories WHERE importance >= ?"
        params: list[Any] = [min_importance]

        if type_filter:
            MemoryType.from_string(type_filter)  # validate
            sql += " AND type = ?"
            params.append(type_filter.lower())

        with self._connection() as conn:
            rows = conn.execute(sql, params).fetchall()

        scored: list[tuple[float, Memory]] = []
        for row in rows:
            memory = self._row_to_memory(row)
            if memory.embedding:
                similarity = cosine_similarity(query_embedding, memory.embedding)
                scored.append((similarity, memory))

        scored.sort(key=lambda x: x[0], reverse=True)

        results = []
        for _, memory in scored[:limit]:
            self._increment_access_count(memory.id)
            memory.access_count += 1
            results.append(memory)

        return results

    def get_memory(self, memory_id: str) -> Memory | None:
        """Retrieve a specific memory by ID.

        Args:
            memory_id: The UUID of the memory.

        Returns:
            The Memory object, or None if not found.
        """
        with self._connection() as conn:
            row = conn.execute(
                "SELECT * FROM memories WHERE id = ?", (memory_id,)
            ).fetchone()

        if row is None:
            return None

        self._increment_access_count(memory_id)
        memory = self._row_to_memory(row)
        memory.access_count += 1
        return memory

    def update_memory(
        self,
        memory_id: str,
        content: str | None = None,
        metadata: dict[str, Any] | None = None,
        tags: list[str] | None = None,
        importance: float | None = None,
    ) -> Memory | None:
        """Update an existing memory.

        Args:
            memory_id: The UUID of the memory to update.
            content: New content (if provided).
            metadata: New metadata (if provided).
            tags: New tags (if provided).
            importance: New importance score (if provided).

        Returns:
            The updated Memory object, or None if not found.
        """
        existing = self.get_memory(memory_id)
        if existing is None:
            return None

        now = datetime.now(timezone.utc)
        updates: list[str] = ["updated_at = ?"]
        params: list[Any] = [now.isoformat()]

        if content is not None:
            updates.append("content = ?")
            params.append(content)
            embedding = self.embedding_provider.embed(content)
            updates.append("embedding = ?")
            params.append(json.dumps(embedding))

        if metadata is not None:
            updates.append("metadata = ?")
            params.append(json.dumps(metadata))

        if tags is not None:
            updates.append("tags = ?")
            params.append(json.dumps(tags))

        if importance is not None:
            updates.append("importance = ?")
            params.append(max(0.0, min(1.0, importance)))

        params.append(memory_id)

        with self._connection() as conn:
            conn.execute(
                f"UPDATE memories SET {', '.join(updates)} WHERE id = ?",
                params,
            )

        return self.get_memory(memory_id)

    def delete_memory(self, memory_id: str) -> bool:
        """Delete a memory by ID.

        Args:
            memory_id: The UUID of the memory to delete.

        Returns:
            True if deleted, False if not found.
        """
        with self._connection() as conn:
            cursor = conn.execute(
                "DELETE FROM memories WHERE id = ?", (memory_id,)
            )
            return cursor.rowcount > 0

    def list_memories(
        self,
        type_filter: str = "",
        tag_filter: str = "",
        limit: int = 20,
        offset: int = 0,
    ) -> list[Memory]:
        """List memories with optional filtering.

        Args:
            type_filter: Optional memory type to filter by.
            tag_filter: Optional tag to filter by.
            limit: Maximum number of results.
            offset: Number of results to skip.

        Returns:
            List of memories.
        """
        sql = "SELECT * FROM memories WHERE 1=1"
        params: list[Any] = []

        if type_filter:
            MemoryType.from_string(type_filter)  # validate
            sql += " AND type = ?"
            params.append(type_filter.lower())

        if tag_filter:
            sql += " AND tags LIKE ?"
            params.append(f'%"{tag_filter}"%')

        sql += " ORDER BY updated_at DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])

        with self._connection() as conn:
            rows = conn.execute(sql, params).fetchall()

        return [self._row_to_memory(row) for row in rows]

    def get_related_memories(
        self, memory_id: str, limit: int = 5
    ) -> list[Memory]:
        """Find memories related to a given memory by semantic similarity.

        Args:
            memory_id: The UUID of the reference memory.
            limit: Maximum number of related memories to return.

        Returns:
            List of related memories sorted by similarity.
        """
        with self._connection() as conn:
            row = conn.execute(
                "SELECT * FROM memories WHERE id = ?", (memory_id,)
            ).fetchone()

        if row is None:
            return []

        reference = self._row_to_memory(row)
        if not reference.embedding:
            return []

        with self._connection() as conn:
            rows = conn.execute(
                "SELECT * FROM memories WHERE id != ?", (memory_id,)
            ).fetchall()

        scored: list[tuple[float, Memory]] = []
        for r in rows:
            memory = self._row_to_memory(r)
            if memory.embedding:
                similarity = cosine_similarity(
                    reference.embedding, memory.embedding
                )
                scored.append((similarity, memory))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [m for _, m in scored[:limit]]

    def decay_memories(
        self,
        decay_factor: float = 0.95,
        min_importance: float = 0.05,
        delete_below: float = 0.01,
    ) -> dict[str, int]:
        """Decay importance of old, rarely-accessed memories.

        Memories that haven't been accessed recently have their importance
        reduced. Very low importance memories can be pruned.

        Args:
            decay_factor: Factor to multiply importance by (0-1).
            min_importance: Don't decay below this value.
            delete_below: Delete memories with importance below this.

        Returns:
            Dict with 'decayed' and 'deleted' counts.
        """
        stats = {"decayed": 0, "deleted": 0}

        with self._connection() as conn:
            rows = conn.execute(
                "SELECT id, importance, access_count FROM memories"
            ).fetchall()

            for row in rows:
                new_importance = row["importance"] * decay_factor

                if new_importance < delete_below:
                    conn.execute(
                        "DELETE FROM memories WHERE id = ?", (row["id"],)
                    )
                    stats["deleted"] += 1
                elif new_importance < row["importance"]:
                    clamped = max(min_importance, new_importance)
                    conn.execute(
                        "UPDATE memories SET importance = ?, updated_at = ? WHERE id = ?",
                        (
                            clamped,
                            datetime.now(timezone.utc).isoformat(),
                            row["id"],
                        ),
                    )
                    stats["decayed"] += 1

        return stats

    def get_stats(self) -> dict[str, Any]:
        """Get memory statistics.

        Returns:
            Dict with counts by type, total memories, average importance, etc.
        """
        with self._connection() as conn:
            total = conn.execute(
                "SELECT COUNT(*) as cnt FROM memories"
            ).fetchone()["cnt"]

            type_counts = {}
            for row in conn.execute(
                "SELECT type, COUNT(*) as cnt FROM memories GROUP BY type"
            ).fetchall():
                type_counts[row["type"]] = row["cnt"]

            avg_importance = 0.0
            if total > 0:
                avg_importance = conn.execute(
                    "SELECT AVG(importance) as avg_imp FROM memories"
                ).fetchone()["avg_imp"]

            total_accesses = conn.execute(
                "SELECT SUM(access_count) as total FROM memories"
            ).fetchone()["total"] or 0

        return {
            "total_memories": total,
            "by_type": type_counts,
            "average_importance": round(avg_importance, 4),
            "total_accesses": total_accesses,
        }

    def _increment_access_count(self, memory_id: str) -> None:
        """Increment the access count for a memory."""
        with self._connection() as conn:
            conn.execute(
                "UPDATE memories SET access_count = access_count + 1 WHERE id = ?",
                (memory_id,),
            )
