"""Tests for ShortTermMemory entry parsing and tier bucketing."""

import asyncio
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from nanobot.agent.memory_short_term import ShortTermMemory
from nanobot.providers.base import LLMResponse


class TestEntryParsing:
    """Test timestamp extraction from history entries."""

    def test_parse_standard_entry(self, tmp_path: Path) -> None:
        stm = ShortTermMemory(tmp_path)
        ts, text = stm._parse_entry("[2026-03-22 14:30] User discussed testing.")
        assert ts.year == 2026
        assert ts.month == 3
        assert ts.day == 22
        assert ts.hour == 14
        assert ts.minute == 30
        assert text == "User discussed testing."

    def test_parse_entry_no_timestamp(self, tmp_path: Path) -> None:
        stm = ShortTermMemory(tmp_path)
        ts, text = stm._parse_entry("Some entry without timestamp.")
        assert ts is not None  # falls back to now()
        assert text == "Some entry without timestamp."

    def test_skip_raw_entry(self, tmp_path: Path) -> None:
        stm = ShortTermMemory(tmp_path)
        assert stm._is_raw_entry("[2026-03-22 14:00] [RAW] 4 messages\nUSER: hi") is True
        assert stm._is_raw_entry("[2026-03-22 14:00] User discussed testing.") is False


class TestTierBucketing:
    """Test entry classification into tiers."""

    def test_today_entry(self, tmp_path: Path) -> None:
        stm = ShortTermMemory(tmp_path)
        now = datetime.now(timezone.utc)
        tier = stm._get_tier(now, now)
        assert tier == "today"

    def test_yesterday_entry(self, tmp_path: Path) -> None:
        stm = ShortTermMemory(tmp_path)
        now = datetime.now(timezone.utc)
        yesterday = now - timedelta(days=1)
        tier = stm._get_tier(yesterday, now)
        assert tier == "yesterday"

    def test_this_week_entry(self, tmp_path: Path) -> None:
        stm = ShortTermMemory(tmp_path)
        now = datetime.now(timezone.utc)
        three_days_ago = now - timedelta(days=3)
        tier = stm._get_tier(three_days_ago, now)
        assert tier == "this_week"

    def test_older_entry(self, tmp_path: Path) -> None:
        stm = ShortTermMemory(tmp_path, retention_days=14)
        now = datetime.now(timezone.utc)
        eight_days_ago = now - timedelta(days=8)
        tier = stm._get_tier(eight_days_ago, now)
        assert tier == "older"

    def test_expired_entry_default_retention(self, tmp_path: Path) -> None:
        stm = ShortTermMemory(tmp_path, retention_days=7)
        now = datetime.now(timezone.utc)
        old = now - timedelta(days=8)
        tier = stm._get_tier(old, now)
        assert tier == "expired"

    def test_expired_entry_custom_retention(self, tmp_path: Path) -> None:
        stm = ShortTermMemory(tmp_path, retention_days=14)
        now = datetime.now(timezone.utc)
        old = now - timedelta(days=15)
        tier = stm._get_tier(old, now)
        assert tier == "expired"


class TestAppendAndRender:
    """Test entry appending and SHORT_TERM.md rendering."""

    def test_on_new_entry_appends_to_file(self, tmp_path: Path) -> None:
        stm = ShortTermMemory(tmp_path)
        stm.on_new_entry("[2026-03-22 14:30] User discussed testing.")
        assert stm.file.exists()
        content = stm.file.read_text()
        assert "User discussed testing" in content
        assert "## Today" in content

    def test_on_new_entry_skips_raw(self, tmp_path: Path) -> None:
        stm = ShortTermMemory(tmp_path)
        stm.on_new_entry("[2026-03-22 14:00] [RAW] 4 messages\nUSER: hi")
        assert not stm.file.exists()

    def test_multiple_entries_same_day(self, tmp_path: Path) -> None:
        stm = ShortTermMemory(tmp_path)
        stm.on_new_entry("[2026-03-22 14:00] First entry.")
        stm.on_new_entry("[2026-03-22 15:00] Second entry.")
        content = stm.file.read_text()
        assert "First entry" in content
        assert "Second entry" in content
        assert content.count("## Today") == 1  # only one header

    def test_read_returns_empty_when_no_file(self, tmp_path: Path) -> None:
        stm = ShortTermMemory(tmp_path)
        assert stm.read() == ""

    def test_read_returns_content(self, tmp_path: Path) -> None:
        stm = ShortTermMemory(tmp_path)
        stm.on_new_entry("[2026-03-22 14:30] User discussed testing.")
        content = stm.read()
        assert "User discussed testing" in content

    def test_read_entries_round_trip(self, tmp_path: Path) -> None:
        """Write entries via on_new_entry, then _read_entries should recover them."""
        stm = ShortTermMemory(tmp_path)
        stm.on_new_entry("[2026-03-22 14:30] First discussion.")
        stm.on_new_entry("[2026-03-22 15:45] Second discussion.")
        entries = stm._read_entries()
        assert len(entries) == 2
        texts = [t for _, t in entries]
        assert any("First discussion" in t for t in texts)
        assert any("Second discussion" in t for t in texts)

    def test_on_new_entry_writes_valid_file(self, tmp_path: Path) -> None:
        """After on_new_entry, the file should be valid and parseable."""
        stm = ShortTermMemory(tmp_path)
        stm.on_new_entry("[2026-03-22 14:00] Good entry.")
        assert stm.file.exists()
        entries_before = stm._read_entries()
        assert len(entries_before) == 1


class TestCompress:
    """Test daily compression sweep."""

    @pytest.mark.asyncio
    async def test_compress_drops_expired_entries(self, tmp_path: Path) -> None:
        stm = ShortTermMemory(tmp_path, retention_days=7)
        now = datetime.now(timezone.utc)
        old_ts = now - timedelta(days=14)
        stm._write(stm._render([(old_ts, "Old expired entry.")], now=now))
        provider = AsyncMock()
        provider.chat_with_retry = AsyncMock(return_value=LLMResponse(content="Compressed."))

        await stm.compress(provider, "test-model")

        content = stm.read()
        assert "Old expired entry" not in content

    @pytest.mark.asyncio
    async def test_compress_within_budget_no_llm_call(self, tmp_path: Path) -> None:
        stm = ShortTermMemory(tmp_path, token_budget=5000)
        stm.on_new_entry("[2026-03-22 14:30] Short entry.")
        provider = AsyncMock()
        provider.chat_with_retry = AsyncMock()

        await stm.compress(provider, "test-model")

        provider.chat_with_retry.assert_not_called()

    @pytest.mark.asyncio
    async def test_compress_over_budget_trims_oldest(self, tmp_path: Path) -> None:
        stm = ShortTermMemory(tmp_path, token_budget=50)  # very small budget
        now = datetime.now(timezone.utc)
        entries = []
        for i in range(20):
            ts = now - timedelta(days=i % 6, hours=i)
            entries.append((ts, f"Entry number {i} with some content to fill tokens."))
        stm._write(stm._render(entries, now=now))
        provider = AsyncMock()
        provider.chat_with_retry = AsyncMock(return_value=LLMResponse(content="Compressed."))

        await stm.compress(provider, "test-model")

        content = stm.read()
        assert stm._estimate_tokens(content) <= 50 or content == ""

    @pytest.mark.asyncio
    async def test_compress_rebuckets_tiers_as_time_passes(self, tmp_path: Path) -> None:
        """Entries that were 'today' should move to 'yesterday' after a day passes."""
        stm = ShortTermMemory(tmp_path, retention_days=7)
        now = datetime.now(timezone.utc)
        stm._write(stm._render([(now, "Today's entry.")], now=now))
        content_before = stm.read()
        assert "## Today" in content_before

        # Re-render with now = tomorrow
        tomorrow = now + timedelta(days=1)
        entries = stm._read_entries()
        rendered = stm._render(entries, now=tomorrow)
        assert "## Yesterday" in rendered

    @pytest.mark.asyncio
    async def test_compress_triggers_llm_for_many_entries_in_tier(self, tmp_path: Path) -> None:
        """>3 entries in this_week tier should trigger LLM compression."""
        stm = ShortTermMemory(tmp_path, token_budget=5000)
        now = datetime.now(timezone.utc)
        entries = []
        for i in range(5):
            ts = now - timedelta(days=3, hours=i)
            entries.append((ts, f"This-week entry {i} about project work."))
        stm._write(stm._render(entries, now=now))

        provider = AsyncMock()
        provider.chat_with_retry = AsyncMock(
            return_value=LLMResponse(content="Compressed week summary.")
        )

        await stm.compress(provider, "test-model")

        provider.chat_with_retry.assert_called_once()
        content = stm.read()
        assert "Compressed week summary" in content
