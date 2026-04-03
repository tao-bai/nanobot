"""Integration test: consolidation → callback → SHORT_TERM.md → context injection."""

from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from nanobot.agent.context import ContextBuilder
from nanobot.agent.memory import MemoryStore
from nanobot.agent.memory_short_term import ShortTermMemory
from nanobot.providers.base import LLMResponse, ToolCallRequest


def _make_tool_response(history_entry: str, memory_update: str) -> LLMResponse:
    return LLMResponse(
        content=None,
        tool_calls=[
            ToolCallRequest(
                id="call_1",
                name="save_memory",
                arguments={"history_entry": history_entry, "memory_update": memory_update},
            )
        ],
    )


class TestShortTermIntegration:

    @pytest.mark.asyncio
    async def test_consolidation_callback_populates_short_term(self, tmp_path: Path) -> None:
        """Full flow: consolidation writes history → callback fires → SHORT_TERM.md exists."""
        store = MemoryStore(tmp_path)
        stm = ShortTermMemory(tmp_path)
        store.on_history_append(stm.on_new_entry)

        provider = AsyncMock()
        provider.chat_with_retry = AsyncMock(
            return_value=_make_tool_response(
                "[2026-03-22 14:00] User asked about weather and Python scripts.",
                "# Memory\nUser interested in Python.",
            )
        )
        messages = [
            {"role": "user", "content": f"msg{i}", "timestamp": "2026-03-22 14:00"}
            for i in range(10)
        ]

        result = await store.consolidate(messages, provider, "test-model")

        assert result is True
        assert stm.file.exists()
        content = stm.read()
        assert "Python" in content

    def test_context_builder_includes_short_term(self, tmp_path: Path) -> None:
        """ContextBuilder system prompt includes SHORT_TERM.md content."""
        (tmp_path / "memory").mkdir(exist_ok=True)
        (tmp_path / "memory" / "MEMORY.md").write_text("User likes Python.\n")
        (tmp_path / "memory" / "SHORT_TERM.md").write_text(
            "## Today (2026-03-22)\n- [14:00] Debugged auth issue.\n"
        )

        ctx = ContextBuilder(tmp_path)
        prompt = ctx.build_system_prompt()

        assert "User likes Python" in prompt
        assert "近期动态" in prompt
        assert "Debugged auth issue" in prompt
