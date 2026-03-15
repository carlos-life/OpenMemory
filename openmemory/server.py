"""MCP Server entry point for OpenMemory."""

from __future__ import annotations

import json
from mcp.server.fastmcp import FastMCP

from openmemory.memory_store import MemoryStore


# Global store instance, lazily initialized
_store: MemoryStore | None = None


def _get_store() -> MemoryStore:
    """Get or create the global MemoryStore instance."""
    global _store
    if _store is None:
        _store = MemoryStore()
    return _store


def set_store(store: MemoryStore) -> None:
    """Set the global MemoryStore (useful for testing)."""
    global _store
    _store = store


mcp_server = FastMCP("OpenMemory")


@mcp_server.tool()
async def add_memory(
    type: str,
    content: str,
    metadata: str = "{}",
    tags: str = "[]",
    importance: float = 0.5,
) -> str:
    """Store a new memory. Types: fact, preference, episode, entity."""
    try:
        meta_dict = json.loads(metadata)
        tags_list = json.loads(tags)
        memory = _get_store().add_memory(
            type=type,
            content=content,
            metadata=meta_dict,
            tags=tags_list,
            importance=importance,
        )
        return json.dumps(memory.to_dict(), indent=2)
    except (json.JSONDecodeError, ValueError) as e:
        return json.dumps({"error": str(e)})


@mcp_server.tool()
async def search_memory(
    query: str, type: str = "", limit: int = 5
) -> str:
    """Search memories by semantic similarity."""
    try:
        memories = _get_store().search_memories(
            query=query, type_filter=type, limit=limit
        )
        return json.dumps([m.to_dict() for m in memories], indent=2)
    except ValueError as e:
        return json.dumps({"error": str(e)})


@mcp_server.tool()
async def get_memory(memory_id: str) -> str:
    """Retrieve a specific memory by ID."""
    memory = _get_store().get_memory(memory_id)
    if memory is None:
        return json.dumps({"error": f"Memory '{memory_id}' not found"})
    return json.dumps(memory.to_dict(), indent=2)


@mcp_server.tool()
async def list_memories(
    type: str = "", tag: str = "", limit: int = 20, offset: int = 0
) -> str:
    """List memories with optional filtering."""
    try:
        memories = _get_store().list_memories(
            type_filter=type, tag_filter=tag, limit=limit, offset=offset
        )
        return json.dumps([m.to_dict() for m in memories], indent=2)
    except ValueError as e:
        return json.dumps({"error": str(e)})


@mcp_server.tool()
async def update_memory(
    memory_id: str, content: str = "", importance: float = -1
) -> str:
    """Update an existing memory."""
    kwargs: dict = {}
    if content:
        kwargs["content"] = content
    if importance >= 0:
        kwargs["importance"] = importance

    if not kwargs:
        return json.dumps({"error": "No fields to update"})

    memory = _get_store().update_memory(memory_id, **kwargs)
    if memory is None:
        return json.dumps({"error": f"Memory '{memory_id}' not found"})
    return json.dumps(memory.to_dict(), indent=2)


@mcp_server.tool()
async def delete_memory(memory_id: str) -> str:
    """Delete a memory."""
    deleted = _get_store().delete_memory(memory_id)
    if deleted:
        return json.dumps({"status": "deleted", "id": memory_id})
    return json.dumps({"error": f"Memory '{memory_id}' not found"})


@mcp_server.tool()
async def get_related(memory_id: str, limit: int = 5) -> str:
    """Find memories related to a given memory."""
    memories = _get_store().get_related_memories(memory_id, limit=limit)
    return json.dumps([m.to_dict() for m in memories], indent=2)


@mcp_server.tool()
async def memory_stats() -> str:
    """Get memory statistics."""
    stats = _get_store().get_stats()
    return json.dumps(stats, indent=2)


def main() -> None:
    """Run the MCP server with stdio transport."""
    mcp_server.run(transport="stdio")


if __name__ == "__main__":
    main()
