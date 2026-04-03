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
        now = datetime.now()
        ts = now.strftime("%Y-%m-%d %H:%M")
        stm.on_new_entry(f"[{ts}] User discussed testing.")
        assert stm.file.exists()
        content = stm.file.read_text()
        assert "User discussed testing" in content
        assert "## 今天" in content

    def test_on_new_entry_skips_raw(self, tmp_path: Path) -> None:
        stm = ShortTermMemory(tmp_path)
        stm.on_new_entry("[2026-03-22 14:00] [RAW] 4 messages\nUSER: hi")
        assert not stm.file.exists()

    def test_multiple_entries_same_day(self, tmp_path: Path) -> None:
        stm = ShortTermMemory(tmp_path)
        now = datetime.now()
        ts1 = now.strftime("%Y-%m-%d %H:%M")
        ts2 = (now + timedelta(hours=1)).strftime("%Y-%m-%d %H:%M")
        stm.on_new_entry(f"[{ts1}] First entry.")
        stm.on_new_entry(f"[{ts2}] Second entry.")
        content = stm.file.read_text()
        assert "First entry" in content
        assert "Second entry" in content
        assert content.count("## 今天") == 1  # only one header

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
        assert "## 今天" in content_before

        # Re-render with now = tomorrow
        tomorrow = now + timedelta(days=1)
        entries = stm._read_entries()
        rendered = stm._render(entries, now=tomorrow)
        assert "## 昨天" in rendered

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


class TestBackfill:
    """Test backfill from raw JSONL logs."""

    @staticmethod
    def _write_jsonl(raw_dir: Path, date_str: str, messages: list[dict], hour_utc: int = 12) -> None:
        """Helper: write messages to a raw JSONL file with real UTC timestamps."""
        import json as _json

        path = raw_dir / f"{date_str}.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for i, msg in enumerate(messages):
                ts = f"{date_str}T{hour_utc:02d}:{i:02d}:00+00:00"
                rec = {"v": 1, "ts": ts, "session_key": "test:1", **msg}
                f.write(_json.dumps(rec) + "\n")

    @pytest.mark.asyncio
    async def test_backfill_seeds_from_raw_jsonl(self, tmp_path: Path) -> None:
        """Backfill should create SHORT_TERM.md from raw JSONL files."""
        stm = ShortTermMemory(tmp_path)
        raw_dir = tmp_path / "memory" / "raw"
        raw_dir.mkdir(parents=True)

        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        utc_hour = 6
        self._write_jsonl(raw_dir, today, [
            {"role": "user", "content": "How do I fix the auth bug?"},
            {"role": "assistant", "content": "Check the middleware token validation."},
        ], hour_utc=utc_hour)

        # LLM returns plain text (no timestamp prefix) — backfill pairs it with real UTC ts
        provider = AsyncMock()
        provider.chat_with_retry = AsyncMock(
            return_value=LLMResponse(content="Discussed auth bug fix in middleware.")
        )

        await stm.backfill(raw_dir, provider, "test-model")

        assert stm.file.exists()
        content = stm.read()
        assert "auth bug" in content
        provider.chat_with_retry.assert_called_once()
        # Verify the prompt included local timestamps in the transcript
        call_args = provider.chat_with_retry.call_args
        prompt_content = call_args.kwargs.get("messages", call_args[1].get("messages", [{}]))[-1]["content"]
        local_hour = datetime(2026, 1, 1, utc_hour, 0, tzinfo=timezone.utc).astimezone().strftime("%H:%M")
        assert local_hour in prompt_content

    @pytest.mark.asyncio
    async def test_backfill_skips_when_file_exists(self, tmp_path: Path) -> None:
        """Backfill should be a no-op if SHORT_TERM.md already has content."""
        stm = ShortTermMemory(tmp_path)
        stm._write("## Today\n- Existing entry.\n")

        raw_dir = tmp_path / "memory" / "raw"
        raw_dir.mkdir(parents=True)
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        self._write_jsonl(raw_dir, today, [
            {"role": "user", "content": "Hello"},
        ])

        provider = AsyncMock()
        await stm.backfill(raw_dir, provider, "test-model")

        provider.chat_with_retry.assert_not_called()
        assert "Existing entry" in stm.read()

    @pytest.mark.asyncio
    async def test_backfill_respects_retention_days(self, tmp_path: Path) -> None:
        """JSONL files older than retention_days should be ignored."""
        stm = ShortTermMemory(tmp_path, retention_days=3)
        raw_dir = tmp_path / "memory" / "raw"
        raw_dir.mkdir(parents=True)

        old_date = (datetime.now(timezone.utc) - timedelta(days=10)).strftime("%Y-%m-%d")
        self._write_jsonl(raw_dir, old_date, [
            {"role": "user", "content": "Old conversation"},
        ])

        provider = AsyncMock()
        await stm.backfill(raw_dir, provider, "test-model")

        assert not stm.file.exists()
        provider.chat_with_retry.assert_not_called()

    @pytest.mark.asyncio
    async def test_backfill_handles_llm_failure(self, tmp_path: Path) -> None:
        """If LLM fails for one day, other days should still be processed."""
        stm = ShortTermMemory(tmp_path)
        raw_dir = tmp_path / "memory" / "raw"
        raw_dir.mkdir(parents=True)

        now = datetime.now(timezone.utc)
        day1 = (now - timedelta(days=1)).strftime("%Y-%m-%d")
        day2 = now.strftime("%Y-%m-%d")

        self._write_jsonl(raw_dir, day1, [{"role": "user", "content": "Day 1 chat"}])
        self._write_jsonl(raw_dir, day2, [{"role": "user", "content": "Day 2 chat"}])

        call_count = 0

        async def mock_chat(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("LLM unavailable")
            return LLMResponse(content="Day 2 summary.")

        provider = AsyncMock()
        provider.chat_with_retry = AsyncMock(side_effect=mock_chat)

        await stm.backfill(raw_dir, provider, "test-model")

        assert stm.file.exists()
        content = stm.read()
        assert "Day 2 summary" in content
