"""Tests for openmemory.models."""

import pytest
from openmemory.models import Memory, MemoryType


class TestMemoryType:
    def test_from_string_valid(self):
        assert MemoryType.from_string("fact") == MemoryType.FACT
        assert MemoryType.from_string("preference") == MemoryType.PREFERENCE
        assert MemoryType.from_string("episode") == MemoryType.EPISODE
        assert MemoryType.from_string("entity") == MemoryType.ENTITY

    def test_from_string_case_insensitive(self):
        assert MemoryType.from_string("FACT") == MemoryType.FACT
        assert MemoryType.from_string("Preference") == MemoryType.PREFERENCE

    def test_from_string_invalid(self):
        with pytest.raises(ValueError, match="Invalid memory type"):
            MemoryType.from_string("invalid")

    def test_enum_values(self):
        assert MemoryType.FACT.value == "fact"
        assert MemoryType.PREFERENCE.value == "preference"
        assert MemoryType.EPISODE.value == "episode"
        assert MemoryType.ENTITY.value == "entity"


class TestMemory:
    def test_default_creation(self):
        m = Memory()
        assert m.id  # UUID is generated
        assert m.type == MemoryType.FACT
        assert m.content == ""
        assert m.metadata == {}
        assert m.tags == []
        assert m.importance == 0.5
        assert m.access_count == 0
        assert m.embedding == []

    def test_to_dict(self):
        m = Memory(content="test content", tags=["tag1"])
        d = m.to_dict()
        assert d["content"] == "test content"
        assert d["tags"] == ["tag1"]
        assert d["type"] == "fact"
        assert "id" in d
        assert "created_at" in d
        # embedding should not be in to_dict (it's internal)
        assert "embedding" not in d

    def test_from_dict(self):
        data = {
            "id": "test-id",
            "type": "preference",
            "content": "likes dark mode",
            "metadata": {"source": "chat"},
            "tags": ["ui"],
            "importance": 0.8,
            "access_count": 3,
        }
        m = Memory.from_dict(data)
        assert m.id == "test-id"
        assert m.type == MemoryType.PREFERENCE
        assert m.content == "likes dark mode"
        assert m.metadata == {"source": "chat"}
        assert m.tags == ["ui"]
        assert m.importance == 0.8
        assert m.access_count == 3

    def test_from_dict_defaults(self):
        m = Memory.from_dict({})
        assert m.type == MemoryType.FACT
        assert m.content == ""

    def test_roundtrip(self):
        m = Memory(
            type=MemoryType.EPISODE,
            content="discussed migration",
            tags=["architecture"],
            importance=0.9,
        )
        d = m.to_dict()
        m2 = Memory.from_dict(d)
        assert m2.type == m.type
        assert m2.content == m.content
        assert m2.tags == m.tags
        assert m2.importance == m.importance
