from __future__ import annotations

import asyncio

from agent.memory.manager import MemoryManager


def test_memory_persists_and_searches(settings):
    async def run_test():
        memory = MemoryManager(settings)
        await memory.initialize()

        memory.add("user", "remember the alpha project uses pytest")
        memory.add("assistant", "stored")

        reloaded = MemoryManager(settings)
        await reloaded.initialize()

        assert len(reloaded.load_long_term()) == 2
        results = reloaded.vector_memory.search("alpha pytest", k=1)
        assert results
        assert "alpha project" in results[0].content

    asyncio.run(run_test())
