"""Tests for openmemory.server (MCP tool functions)."""

import asyncio
import json

import pytest

from openmemory.memory_store import MemoryStore
from openmemory.server import (
    add_memory,
    search_memory,
    get_memory,
    list_memories,
    update_memory,
    delete_memory,
    get_related,
    memory_stats,
    set_store,
)


def _run(coro):
    """Helper to run async functions synchronously."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


@pytest.fixture(autouse=True)
def setup_store():
    """Set up an in-memory store for each test."""
    store = MemoryStore(db_path=":memory:")
    set_store(store)
    yield store


class TestAddMemoryTool:
    def test_add_basic(self):
        result = json.loads(_run(add_memory(type="fact", content="User knows Python")))
        assert result["type"] == "fact"
        assert result["content"] == "User knows Python"
        assert "id" in result

    def test_add_with_metadata(self):
        result = json.loads(
            _run(
                add_memory(
                    type="preference",
                    content="likes dark mode",
                    metadata='{"source": "chat"}',
                    tags='["ui"]',
                    importance=0.9,
                )
            )
        )
        assert result["metadata"] == {"source": "chat"}
        assert result["tags"] == ["ui"]

    def test_add_invalid_type(self):
        result = json.loads(_run(add_memory(type="bogus", content="test")))
        assert "error" in result

    def test_add_invalid_json(self):
        result = json.loads(
            _run(add_memory(type="fact", content="test", metadata="not json"))
        )
        assert "error" in result


class TestSearchMemoryTool:
    def test_search(self):
        _run(add_memory(type="fact", content="Python is a programming language"))
        _run(add_memory(type="fact", content="User lives in Shanghai"))
        result = json.loads(_run(search_memory(query="programming")))
        assert isinstance(result, list)
        assert len(result) > 0

    def test_search_with_type(self):
        _run(add_memory(type="fact", content="Python"))
        _run(add_memory(type="preference", content="likes Python"))
        result = json.loads(_run(search_memory(query="Python", type="fact")))
        assert all(r["type"] == "fact" for r in result)


class TestGetMemoryTool:
    def test_get_existing(self):
        added = json.loads(_run(add_memory(type="fact", content="test")))
        result = json.loads(_run(get_memory(memory_id=added["id"])))
        assert result["content"] == "test"

    def test_get_nonexistent(self):
        result = json.loads(_run(get_memory(memory_id="nonexistent")))
        assert "error" in result


class TestListMemoriesTool:
    def test_list_all(self):
        _run(add_memory(type="fact", content="fact1"))
        _run(add_memory(type="preference", content="pref1"))
        result = json.loads(_run(list_memories()))
        assert len(result) == 2

    def test_list_filtered(self):
        _run(add_memory(type="fact", content="fact1"))
        _run(add_memory(type="preference", content="pref1"))
        result = json.loads(_run(list_memories(type="fact")))
        assert len(result) == 1
        assert result[0]["type"] == "fact"


class TestUpdateMemoryTool:
    def test_update_content(self):
        added = json.loads(_run(add_memory(type="fact", content="original")))
        result = json.loads(
            _run(update_memory(memory_id=added["id"], content="updated"))
        )
        assert result["content"] == "updated"

    def test_update_importance(self):
        added = json.loads(_run(add_memory(type="fact", content="test")))
        result = json.loads(
            _run(update_memory(memory_id=added["id"], importance=0.95))
        )
        assert result["importance"] == 0.95

    def test_update_no_fields(self):
        added = json.loads(_run(add_memory(type="fact", content="test")))
        result = json.loads(_run(update_memory(memory_id=added["id"])))
        assert "error" in result

    def test_update_nonexistent(self):
        result = json.loads(
            _run(update_memory(memory_id="nonexistent", content="new"))
        )
        assert "error" in result


class TestDeleteMemoryTool:
    def test_delete(self):
        added = json.loads(_run(add_memory(type="fact", content="to delete")))
        result = json.loads(_run(delete_memory(memory_id=added["id"])))
        assert result["status"] == "deleted"

    def test_delete_nonexistent(self):
        result = json.loads(_run(delete_memory(memory_id="nonexistent")))
        assert "error" in result


class TestGetRelatedTool:
    def test_get_related(self):
        m1 = json.loads(_run(add_memory(type="fact", content="Python programming")))
        _run(add_memory(type="fact", content="Java programming"))
        _run(add_memory(type="fact", content="User lives in Shanghai"))
        result = json.loads(_run(get_related(memory_id=m1["id"])))
        assert isinstance(result, list)
        assert len(result) > 0


class TestMemoryStatsTool:
    def test_stats_empty(self):
        result = json.loads(_run(memory_stats()))
        assert result["total_memories"] == 0

    def test_stats_populated(self):
        _run(add_memory(type="fact", content="fact1"))
        _run(add_memory(type="preference", content="pref1"))
        result = json.loads(_run(memory_stats()))
        assert result["total_memories"] == 2
        assert result["by_type"]["fact"] == 1
