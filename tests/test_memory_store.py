"""Tests for openmemory.memory_store."""

import pytest
from openmemory.memory_store import MemoryStore
from openmemory.models import MemoryType


@pytest.fixture
def store():
    """Create an in-memory store for testing."""
    return MemoryStore(db_path=":memory:")


@pytest.fixture
def populated_store(store):
    """Create a store with some sample memories."""
    store.add_memory("fact", "User prefers Python over Java", tags=["programming"], importance=0.8)
    store.add_memory("fact", "User lives in Shanghai", tags=["location"], importance=0.7)
    store.add_memory("preference", "User likes concise responses", tags=["communication"], importance=0.9)
    store.add_memory("episode", "Discussed migration to microservices on 2026-03-10", tags=["architecture"], importance=0.6)
    store.add_memory("entity", "Project Alpha is the main work project", tags=["work", "project"], importance=0.85)
    store.add_memory("entity", "Zhang Wei is a colleague", tags=["people", "work"], importance=0.5)
    return store


class TestAddMemory:
    def test_add_basic(self, store):
        m = store.add_memory("fact", "User knows Python")
        assert m.type == MemoryType.FACT
        assert m.content == "User knows Python"
        assert m.importance == 0.5
        assert m.access_count == 0
        assert len(m.embedding) > 0

    def test_add_with_metadata(self, store):
        m = store.add_memory("fact", "test", metadata={"source": "chat"}, tags=["test"], importance=0.9)
        assert m.metadata == {"source": "chat"}
        assert m.tags == ["test"]
        assert m.importance == 0.9

    def test_add_invalid_type(self, store):
        with pytest.raises(ValueError, match="Invalid memory type"):
            store.add_memory("invalid_type", "content")

    def test_importance_clamped(self, store):
        m = store.add_memory("fact", "test", importance=1.5)
        assert m.importance == 1.0
        m2 = store.add_memory("fact", "test2", importance=-0.5)
        assert m2.importance == 0.0


class TestGetMemory:
    def test_get_existing(self, store):
        m = store.add_memory("fact", "test content")
        retrieved = store.get_memory(m.id)
        assert retrieved is not None
        assert retrieved.content == "test content"
        assert retrieved.access_count == 1  # incremented on get

    def test_get_nonexistent(self, store):
        result = store.get_memory("nonexistent-id")
        assert result is None


class TestUpdateMemory:
    def test_update_content(self, store):
        m = store.add_memory("fact", "original content")
        updated = store.update_memory(m.id, content="updated content")
        assert updated is not None
        assert updated.content == "updated content"

    def test_update_importance(self, store):
        m = store.add_memory("fact", "test", importance=0.5)
        updated = store.update_memory(m.id, importance=0.9)
        assert updated is not None
        assert updated.importance == 0.9

    def test_update_nonexistent(self, store):
        result = store.update_memory("nonexistent-id", content="new")
        assert result is None

    def test_update_regenerates_embedding(self, store):
        m = store.add_memory("fact", "Python programming")
        original_embedding = m.embedding[:]
        updated = store.update_memory(m.id, content="JavaScript development")
        assert updated is not None
        assert updated.embedding != original_embedding


class TestDeleteMemory:
    def test_delete_existing(self, store):
        m = store.add_memory("fact", "to be deleted")
        assert store.delete_memory(m.id) is True
        assert store.get_memory(m.id) is None

    def test_delete_nonexistent(self, store):
        assert store.delete_memory("nonexistent-id") is False


class TestListMemories:
    def test_list_all(self, populated_store):
        memories = populated_store.list_memories()
        assert len(memories) == 6

    def test_list_by_type(self, populated_store):
        facts = populated_store.list_memories(type_filter="fact")
        assert len(facts) == 2
        assert all(m.type == MemoryType.FACT for m in facts)

    def test_list_by_tag(self, populated_store):
        work = populated_store.list_memories(tag_filter="work")
        assert len(work) == 2

    def test_list_with_limit(self, populated_store):
        memories = populated_store.list_memories(limit=3)
        assert len(memories) == 3

    def test_list_with_offset(self, populated_store):
        all_mems = populated_store.list_memories()
        offset_mems = populated_store.list_memories(offset=2)
        assert len(offset_mems) == len(all_mems) - 2

    def test_list_invalid_type(self, populated_store):
        with pytest.raises(ValueError):
            populated_store.list_memories(type_filter="bogus")


class TestSearchMemories:
    def test_search_basic(self, populated_store):
        results = populated_store.search_memories("Python programming language")
        assert len(results) > 0
        # The Python-related memory should rank high
        assert "Python" in results[0].content

    def test_search_with_type_filter(self, populated_store):
        results = populated_store.search_memories("work", type_filter="entity")
        assert all(m.type == MemoryType.ENTITY for m in results)

    def test_search_empty_query(self, populated_store):
        results = populated_store.search_memories("")
        assert results == []

    def test_search_with_limit(self, populated_store):
        results = populated_store.search_memories("user", limit=2)
        assert len(results) <= 2

    def test_search_with_min_importance(self, populated_store):
        results = populated_store.search_memories("user", min_importance=0.8)
        assert all(m.importance >= 0.8 for m in results)

    def test_search_increments_access_count(self, store):
        m = store.add_memory("fact", "Python is great")
        store.search_memories("Python")
        retrieved = store.get_memory(m.id)
        # access_count should be > 0 from search + get
        assert retrieved.access_count > 0


class TestGetRelatedMemories:
    def test_get_related(self, populated_store):
        # Get a memory about Python
        memories = populated_store.list_memories(type_filter="fact")
        python_mem = [m for m in memories if "Python" in m.content][0]

        related = populated_store.get_related_memories(python_mem.id, limit=3)
        assert len(related) > 0
        # Should not include the source memory
        assert all(m.id != python_mem.id for m in related)

    def test_get_related_nonexistent(self, store):
        result = store.get_related_memories("nonexistent-id")
        assert result == []


class TestDecayMemories:
    def test_decay_reduces_importance(self, store):
        store.add_memory("fact", "test memory", importance=0.5)
        stats = store.decay_memories(decay_factor=0.5)
        assert stats["decayed"] > 0

        memories = store.list_memories()
        # Importance should be reduced
        assert memories[0].importance < 0.5

    def test_decay_deletes_low_importance(self, store):
        store.add_memory("fact", "unimportant", importance=0.005)
        stats = store.decay_memories(decay_factor=0.5, delete_below=0.01)
        assert stats["deleted"] == 1
        assert len(store.list_memories()) == 0

    def test_decay_respects_min_importance(self, store):
        store.add_memory("fact", "test", importance=0.1)
        store.decay_memories(decay_factor=0.1, min_importance=0.05)
        memories = store.list_memories()
        assert memories[0].importance >= 0.05


class TestGetStats:
    def test_stats_empty(self, store):
        stats = store.get_stats()
        assert stats["total_memories"] == 0
        assert stats["by_type"] == {}

    def test_stats_populated(self, populated_store):
        stats = populated_store.get_stats()
        assert stats["total_memories"] == 6
        assert stats["by_type"]["fact"] == 2
        assert stats["by_type"]["preference"] == 1
        assert stats["by_type"]["episode"] == 1
        assert stats["by_type"]["entity"] == 2
        assert 0 <= stats["average_importance"] <= 1


class TestEdgeCases:
    def test_duplicate_content(self, store):
        m1 = store.add_memory("fact", "same content")
        m2 = store.add_memory("fact", "same content")
        assert m1.id != m2.id
        assert len(store.list_memories()) == 2

    def test_empty_content(self, store):
        m = store.add_memory("fact", "")
        assert m.content == ""

    def test_special_characters(self, store):
        m = store.add_memory("fact", "User said: \"Hello 'World'\" & more <stuff>")
        retrieved = store.get_memory(m.id)
        assert retrieved.content == m.content

    def test_unicode_content(self, store):
        m = store.add_memory("fact", "User lives in Shanghai")
        retrieved = store.get_memory(m.id)
        assert retrieved.content == "User lives in Shanghai"

    def test_large_metadata(self, store):
        meta = {f"key_{i}": f"value_{i}" for i in range(100)}
        m = store.add_memory("fact", "test", metadata=meta)
        retrieved = store.get_memory(m.id)
        assert len(retrieved.metadata) == 100
