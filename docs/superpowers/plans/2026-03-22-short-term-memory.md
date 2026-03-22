# Short-Term Memory Layer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a rolling short-term memory layer (1–7 days) with progressive compression that gives the agent temporal awareness of recent activity.

**Architecture:** A new `memory_short_term.py` module handles all short-term logic. A 3-line callback hook in `memory.py` fires after each history append. `ContextBuilder` reads `SHORT_TERM.md` and injects it as "Recent Activity" in the system prompt. A daily background task compresses older entries via LLM.

**Tech Stack:** Python 3.13, asyncio, Pydantic (config), pytest + pytest-asyncio (tests)

**Spec:** `docs/superpowers/specs/2026-03-22-short-term-memory-design.md`

---

## File Structure

| File | Responsibility |
|------|---------------|
| `nanobot/agent/memory_short_term.py` (CREATE) | `ShortTermMemory` class: entry parsing, tier bucketing, append, compress, render |
| `nanobot/agent/memory.py` (MODIFY ~3 lines) | Add `_on_history_append` callback list, fire after `append_history()` |
| `nanobot/agent/context.py` (MODIFY ~5 lines) | Read `SHORT_TERM.md`, inject as "Recent Activity" in system prompt |
| `nanobot/agent/loop.py` (MODIFY ~5 lines) | Register callback, start daily sweep background task |
| `nanobot/config/schema.py` (MODIFY ~2 lines) | Add `short_term_token_budget` and `short_term_retention_days` to `AgentDefaults` |
| `tests/test_memory_short_term.py` (CREATE) | Unit tests for ShortTermMemory |
| `tests/test_short_term_integration.py` (CREATE) | Integration test: consolidation → callback → context injection |

---

### Task 1: Config Fields

**Files:**
- Modify: `nanobot/config/schema.py:29-42`
- Test: `tests/test_config.py` (existing, verify no breakage)

- [ ] **Step 1: Add config fields**

In `AgentDefaults` class, add after line 41 (`reasoning_effort`):

```python
short_term_token_budget: int = 3000
short_term_retention_days: int = 7
```

- [ ] **Step 2: Verify existing config tests pass**

Run: `uv run pytest tests/test_config.py -x -v`
Expected: All existing tests PASS (new fields have defaults, so no breakage)

- [ ] **Step 3: Commit**

```bash
git add nanobot/config/schema.py
git commit -m "feat: add short-term memory config fields to AgentDefaults"
```

---

### Task 2: ShortTermMemory Core — Entry Parsing & Tier Bucketing

**Files:**
- Create: `nanobot/agent/memory_short_term.py`
- Create: `tests/test_memory_short_term.py`

- [ ] **Step 1: Write failing tests for entry parsing and tier bucketing**

```python
"""Tests for ShortTermMemory entry parsing and tier bucketing."""

import asyncio
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from nanobot.agent.memory_short_term import ShortTermMemory


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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_memory_short_term.py -x -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'nanobot.agent.memory_short_term'`

- [ ] **Step 3: Implement ShortTermMemory core**

Create `nanobot/agent/memory_short_term.py`:

```python
"""Short-term memory: rolling 1–7 day memory with progressive compression."""

from __future__ import annotations

import asyncio
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

from loguru import logger

if TYPE_CHECKING:
    from nanobot.providers.base import LLMProvider

_TIMESTAMP_RE = re.compile(r"^\[(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2})\]\s*")
_UTC_COMMENT_RE = re.compile(r"^<!--\s*([\dT:.Z+-]+)\s*-->$")
_RAW_MARKER = "[RAW]"

_DEFAULT_TOKEN_BUDGET = 3000
_DEFAULT_RETENTION_DAYS = 7


class ShortTermMemory:
    """Rolling short-term memory with progressive compression tiers."""

    def __init__(
        self,
        workspace: Path,
        token_budget: int = _DEFAULT_TOKEN_BUDGET,
        retention_days: int = _DEFAULT_RETENTION_DAYS,
    ):
        self.memory_dir = workspace / "memory"
        self.file = self.memory_dir / "SHORT_TERM.md"
        self.token_budget = token_budget
        self.retention_days = retention_days
        self._lock = asyncio.Lock()

    # -- Entry parsing --

    @staticmethod
    def _is_raw_entry(entry: str) -> bool:
        """Detect raw-archive fallback entries (should be skipped)."""
        return _RAW_MARKER in entry.split("\n", 1)[0]

    @staticmethod
    def _parse_entry(entry: str) -> tuple[datetime, str]:
        """Extract timestamp and text from a history entry.

        Returns (utc_datetime, cleaned_text).
        Entries have format: [YYYY-MM-DD HH:MM] text...
        Timestamps are local time, converted to UTC assuming system timezone.
        """
        m = _TIMESTAMP_RE.match(entry.strip())
        if m:
            try:
                local_dt = datetime.strptime(m.group(1), "%Y-%m-%d %H:%M")
                # Treat as local time, attach system timezone, convert to UTC
                local_dt = local_dt.astimezone()
                utc_dt = local_dt.astimezone(timezone.utc)
            except ValueError:
                utc_dt = datetime.now(timezone.utc)
            text = entry.strip()[m.end():].strip()
        else:
            utc_dt = datetime.now(timezone.utc)
            text = entry.strip()
        return utc_dt, text

    def _get_tier(self, entry_time: datetime, now: datetime) -> str:
        """Classify an entry into a tier based on age."""
        # Compare dates in local timezone
        entry_local = entry_time.astimezone()
        now_local = now.astimezone()
        entry_date = entry_local.date()
        now_date = now_local.date()
        delta_days = (now_date - entry_date).days

        if delta_days <= 0:
            return "today"
        elif delta_days == 1:
            return "yesterday"
        elif delta_days <= 6:
            return "this_week"
        elif delta_days <= self.retention_days:
            return "older"
        else:
            return "expired"

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        """Rough token estimate: ~4 chars per token."""
        return len(text) // 4
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_memory_short_term.py -x -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add nanobot/agent/memory_short_term.py tests/test_memory_short_term.py
git commit -m "feat: add ShortTermMemory core — entry parsing and tier bucketing"
```

---

### Task 3: ShortTermMemory — Append & Render

**Files:**
- Modify: `nanobot/agent/memory_short_term.py`
- Modify: `tests/test_memory_short_term.py`

- [ ] **Step 1: Write failing tests for on_new_entry and render**

Add to `tests/test_memory_short_term.py`:

```python
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
        # Verify existing file is valid
        assert stm.file.exists()
        entries_before = stm._read_entries()
        assert len(entries_before) == 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_memory_short_term.py::TestAppendAndRender -x -v`
Expected: FAIL with `AttributeError: 'ShortTermMemory' object has no attribute 'on_new_entry'`

- [ ] **Step 3: Implement on_new_entry, read, and rendering**

Add to `ShortTermMemory` class in `memory_short_term.py`:

```python
    # -- File I/O --

    def read(self) -> str:
        """Read rendered SHORT_TERM.md content."""
        if self.file.exists():
            return self.file.read_text(encoding="utf-8").strip()
        return ""

    def _read_entries(self) -> list[tuple[datetime, str]]:
        """Parse SHORT_TERM.md back into (utc_timestamp, text) pairs."""
        if not self.file.exists():
            return []
        entries: list[tuple[datetime, str]] = []
        current_ts: datetime | None = None
        for line in self.file.read_text(encoding="utf-8").splitlines():
            line_stripped = line.strip()
            # Skip tier headers
            if line_stripped.startswith("## "):
                continue
            # UTC timestamp comment
            m = _UTC_COMMENT_RE.match(line_stripped)
            if m:
                try:
                    current_ts = datetime.fromisoformat(m.group(1))
                    if current_ts.tzinfo is None:
                        current_ts = current_ts.replace(tzinfo=timezone.utc)
                except ValueError:
                    current_ts = None
                continue
            # Entry line (starts with -)
            if line_stripped.startswith("- ") and current_ts is not None:
                entries.append((current_ts, line_stripped[2:]))
                current_ts = None
                continue
            # Paragraph text (compressed tier content)
            if line_stripped and not line_stripped.startswith("#") and current_ts is not None:
                entries.append((current_ts, line_stripped))
                current_ts = None
        return entries

    def _render(self, entries: list[tuple[datetime, str]], now: datetime | None = None) -> str:
        """Render entries into tiered SHORT_TERM.md format."""
        if now is None:
            now = datetime.now(timezone.utc)

        tiers: dict[str, list[tuple[datetime, str]]] = {
            "today": [], "yesterday": [], "this_week": [], "older": [],
        }
        for ts, text in entries:
            tier = self._get_tier(ts, now)
            if tier != "expired" and tier in tiers:
                tiers[tier].append((ts, text))

        lines: list[str] = []
        now_local = now.astimezone()

        if tiers["today"]:
            lines.append(f"## Today ({now_local.strftime('%Y-%m-%d')})")
            for ts, text in sorted(tiers["today"]):
                local_ts = ts.astimezone()
                lines.append(f"<!-- {ts.isoformat()} -->")
                lines.append(f"- [{local_ts.strftime('%H:%M')}] {text}")
            lines.append("")

        if tiers["yesterday"]:
            yd = (now_local - timedelta(days=1)).strftime("%Y-%m-%d")
            lines.append(f"## Yesterday ({yd})")
            for ts, text in sorted(tiers["yesterday"]):
                lines.append(f"<!-- {ts.isoformat()} -->")
                # Strip time prefix if present (from when it was "today")
                text = re.sub(r"^\[\d{2}:\d{2}\]\s*", "", text)
                lines.append(f"- {text}")
            lines.append("")

        if tiers["this_week"]:
            dates = [ts.astimezone() for ts, _ in tiers["this_week"]]
            start = min(dates).strftime("%Y-%m-%d")
            end = max(dates).strftime("%Y-%m-%d")
            lines.append(f"## This Week ({start} — {end})")
            for ts, text in sorted(tiers["this_week"]):
                lines.append(f"<!-- {ts.isoformat()} -->")
                text = re.sub(r"^\[\d{2}:\d{2}\]\s*", "", text)
                lines.append(f"- {text}")
            lines.append("")

        if tiers["older"]:
            dates = [ts.astimezone() for ts, _ in tiers["older"]]
            start = min(dates).strftime("%Y-%m-%d")
            end = max(dates).strftime("%Y-%m-%d")
            lines.append(f"## Older ({start} — {end})")
            for ts, text in sorted(tiers["older"]):
                lines.append(f"<!-- {ts.isoformat()} -->")
                text = re.sub(r"^\[\d{2}:\d{2}\]\s*", "", text)
                lines.append(f"- {text}")
            lines.append("")

        return "\n".join(lines).strip()

    def _write(self, content: str) -> None:
        """Write rendered content to SHORT_TERM.md."""
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        self.file.write_text(content + "\n", encoding="utf-8")

    # -- Callback --

    def on_new_entry(self, entry: str) -> None:
        """Handle a new history entry from consolidation callback.

        Skips raw-archive entries. Parses timestamp and appends to SHORT_TERM.md.

        Note: This is synchronous (called from sync callback in append_history).
        The _lock is only used by async compress(). Concurrent access is safe because:
        - on_new_entry does a full file rewrite (read-modify-write), not an append
        - compress() acquires _lock, so two compresses can't overlap
        - Worst case: compress reads a slightly stale file, losing one entry that
          will be re-added on next consolidation. Acceptable trade-off vs. making
          the entire callback chain async.
        """
        if self._is_raw_entry(entry):
            return

        utc_ts, text = self._parse_entry(entry)
        if not text:
            return

        # Read existing entries, add new one, re-render
        existing = self._read_entries()
        existing.append((utc_ts, text))
        rendered = self._render(existing)
        self._write(rendered)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_memory_short_term.py -x -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add nanobot/agent/memory_short_term.py tests/test_memory_short_term.py
git commit -m "feat: add ShortTermMemory append and render logic"
```

---

### Task 4: ShortTermMemory — Daily Compression Sweep

**Files:**
- Modify: `nanobot/agent/memory_short_term.py`
- Modify: `tests/test_memory_short_term.py`

- [ ] **Step 1: Write failing tests for compress**

Add to `tests/test_memory_short_term.py`:

```python
from unittest.mock import AsyncMock

from nanobot.providers.base import LLMResponse


class TestCompress:
    """Test daily compression sweep."""

    @pytest.mark.asyncio
    async def test_compress_drops_expired_entries(self, tmp_path: Path) -> None:
        stm = ShortTermMemory(tmp_path, retention_days=7)
        now = datetime.now(timezone.utc)
        old_ts = now - timedelta(days=14)
        # Write an entry with an old timestamp
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
        # Add many entries to exceed budget
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
        # Seed with an entry from "today" (relative to now)
        now = datetime.now(timezone.utc)
        stm._write(stm._render([(now, "Today's entry.")], now=now))
        content_before = stm.read()
        assert "## Today" in content_before

        # Simulate time passing: re-render with now = tomorrow
        tomorrow = now + timedelta(days=1)
        entries = stm._read_entries()
        rendered = stm._render(entries, now=tomorrow)
        assert "## Yesterday" in rendered
        assert "## Today" not in rendered or "Today's entry" not in rendered.split("## Today")[1] if "## Today" in rendered else True

    @pytest.mark.asyncio
    async def test_compress_triggers_llm_for_many_entries_in_tier(self, tmp_path: Path) -> None:
        """>3 entries in this_week tier should trigger LLM compression."""
        stm = ShortTermMemory(tmp_path, token_budget=5000)
        now = datetime.now(timezone.utc)
        # Add 5 entries in the "this_week" tier (3-5 days ago)
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

        # LLM should have been called since >3 entries in this_week tier
        provider.chat_with_retry.assert_called_once()
        content = stm.read()
        assert "Compressed week summary" in content
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_memory_short_term.py::TestCompress -x -v`
Expected: FAIL with `AttributeError: 'ShortTermMemory' object has no attribute 'compress'`

- [ ] **Step 3: Implement compress**

Add to `ShortTermMemory` class:

```python
    async def compress(self, provider: LLMProvider, model: str) -> None:
        """Daily compression sweep: re-bucket, compress older tiers, enforce token budget."""
        async with self._lock:
            entries = self._read_entries()
            if not entries:
                return

            now = datetime.now(timezone.utc)

            # Drop expired entries
            entries = [
                (ts, text) for ts, text in entries
                if self._get_tier(ts, now) != "expired"
            ]

            if not entries:
                self._write("")
                return

            # Compress "this_week" and "older" tiers via LLM if they have many entries
            entries = await self._compress_tiers(entries, now, provider, model)

            # Enforce token budget by trimming oldest entries
            rendered = self._render(entries, now=now)
            while self._estimate_tokens(rendered) > self.token_budget and entries:
                # Remove oldest entry
                entries.sort(key=lambda e: e[0])
                entries.pop(0)
                rendered = self._render(entries, now=now)

            self._write(rendered)
            logger.info("Short-term memory compressed: {} entries, ~{} tokens",
                        len(entries), self._estimate_tokens(rendered))

    async def _compress_tiers(
        self,
        entries: list[tuple[datetime, str]],
        now: datetime,
        provider: LLMProvider,
        model: str,
    ) -> list[tuple[datetime, str]]:
        """Compress this_week and older tiers via LLM if they have >3 entries."""
        result: list[tuple[datetime, str]] = []
        tier_groups: dict[str, list[tuple[datetime, str]]] = {}

        for ts, text in entries:
            tier = self._get_tier(ts, now)
            tier_groups.setdefault(tier, []).append((ts, text))

        for tier_name in ("today", "yesterday"):
            result.extend(tier_groups.get(tier_name, []))

        for tier_name in ("this_week", "older"):
            tier_entries = tier_groups.get(tier_name, [])
            if len(tier_entries) <= 3:
                result.extend(tier_entries)
                continue

            # Compress via LLM
            bullet_list = "\n".join(f"- {text}" for _, text in sorted(tier_entries))
            try:
                response = await provider.chat_with_retry(
                    messages=[
                        {"role": "system", "content": "Compress the following entries into a concise paragraph. Preserve key decisions, open tasks, and outcomes. Be brief."},
                        {"role": "user", "content": bullet_list},
                    ],
                    model=model,
                )
                compressed = (response.content or "").strip()
                if compressed:
                    # Use the latest timestamp from the tier
                    latest_ts = max(ts for ts, _ in tier_entries)
                    result.append((latest_ts, compressed))
                else:
                    result.extend(tier_entries)
            except Exception:
                logger.warning("Short-term memory compression failed for tier {}", tier_name)
                result.extend(tier_entries)

        return result
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_memory_short_term.py -x -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add nanobot/agent/memory_short_term.py tests/test_memory_short_term.py
git commit -m "feat: add ShortTermMemory daily compression sweep"
```

---

### Task 5: Hook in memory.py

**Files:**
- Modify: `nanobot/agent/memory.py:98-114`
- Modify: `tests/test_memory_consolidation_types.py` (add one test)

- [ ] **Step 1: Write failing test for callback**

Add to `tests/test_memory_consolidation_types.py`:

```python
    @pytest.mark.asyncio
    async def test_history_append_fires_callbacks(self, tmp_path: Path) -> None:
        """Callback registered on _on_history_append should fire after append."""
        store = MemoryStore(tmp_path)
        received: list[str] = []
        store._on_history_append.append(lambda entry: received.append(entry))

        store.append_history("[2026-03-22 14:00] Test entry.")

        assert len(received) == 1
        assert "Test entry" in received[0]

    @pytest.mark.asyncio
    async def test_history_append_callback_exception_does_not_break_consolidation(self, tmp_path: Path) -> None:
        """A failing callback should not prevent history from being written."""
        store = MemoryStore(tmp_path)
        def bad_callback(entry: str) -> None:
            raise RuntimeError("boom")

        store._on_history_append.append(bad_callback)

        store.append_history("[2026-03-22 14:00] Test entry.")

        # History file should still contain the entry
        assert "Test entry" in store.history_file.read_text()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_memory_consolidation_types.py::TestMemoryConsolidationTypeHandling::test_history_append_fires_callbacks -x -v`
Expected: FAIL with `AttributeError: 'MemoryStore' object has no attribute '_on_history_append'`

- [ ] **Step 3: Add callback hook to MemoryStore**

In `nanobot/agent/memory.py`, modify `MemoryStore.__init__` (line 98-102):

```python
def __init__(self, workspace: Path):
    self.memory_dir = ensure_dir(workspace / "memory")
    self.memory_file = self.memory_dir / "MEMORY.md"
    self.history_file = self.memory_dir / "HISTORY.md"
    self._consecutive_failures = 0
    self._on_history_append: list[Callable[[str], None]] = []
```

Modify `append_history` (line 112-114):

```python
def append_history(self, entry: str) -> None:
    with open(self.history_file, "a", encoding="utf-8") as f:
        f.write(entry.rstrip() + "\n\n")
    for cb in self._on_history_append:
        try:
            cb(entry)
        except Exception:
            logger.warning("Short-term memory callback failed", exc_info=True)
```

Also add `Callable` to the imports at line 10:

```python
from typing import TYPE_CHECKING, Any, Callable
```

(`Callable` is already imported — verify; if not, add it.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_memory_consolidation_types.py -x -v`
Expected: All PASS (20 existing + 1 new)

- [ ] **Step 5: Commit**

```bash
git add nanobot/agent/memory.py tests/test_memory_consolidation_types.py
git commit -m "feat: add _on_history_append callback hook to MemoryStore"
```

---

### Task 6: Context Injection in ContextBuilder

**Files:**
- Modify: `nanobot/agent/context.py:22-37`

- [ ] **Step 1: Add SHORT_TERM.md reading to ContextBuilder**

In `context.py`, add a helper method to `ContextBuilder`:

```python
def _read_short_term(self) -> str:
    """Read SHORT_TERM.md if it exists."""
    path = self.workspace / "memory" / "SHORT_TERM.md"
    if path.exists():
        return path.read_text(encoding="utf-8").strip()
    return ""
```

Then modify `build_system_prompt()` (after line 37, where memory is injected):

```python
memory = self.memory.get_memory_context()
short_term = self._read_short_term()
if memory or short_term:
    memory_parts = []
    if memory:
        memory_parts.append(memory)
    if short_term:
        memory_parts.append(f"## Recent Activity\n{short_term}")
    parts.append(f"# Memory\n\n" + "\n\n".join(memory_parts))
```

This replaces the existing lines 35-37:
```python
memory = self.memory.get_memory_context()
if memory:
    parts.append(f"# Memory\n\n{memory}")
```

- [ ] **Step 2: Verify existing tests still pass**

Run: `uv run pytest tests/ -x -v -k "context or memory" --timeout=30`
Expected: All PASS

- [ ] **Step 3: Commit**

```bash
git add nanobot/agent/context.py
git commit -m "feat: inject SHORT_TERM.md as 'Recent Activity' in system prompt"
```

---

### Task 7: Wire Everything in AgentLoop

**Files:**
- Modify: `nanobot/agent/loop.py:105-114`

- [ ] **Step 1: Import and instantiate ShortTermMemory, register callback, start sweep**

In `loop.py`, add import at the top (near other memory imports):

```python
from nanobot.agent.memory_short_term import ShortTermMemory
```

After the `MemoryConsolidator` creation (line 113), add:

```python
self._short_term = ShortTermMemory(
    workspace,
    token_budget=short_term_token_budget,
    retention_days=short_term_retention_days,
)
self.memory_consolidator.store._on_history_append.append(self._short_term.on_new_entry)
```

The `AgentLoop.__init__` needs two new parameters: `short_term_token_budget: int = 3000` and `short_term_retention_days: int = 7`. These get threaded from config in `commands.py` where `AgentLoop` is instantiated (both in `gateway()` and `chat()`):

```python
agent = AgentLoop(
    ...
    short_term_token_budget=config.agents.defaults.short_term_token_budget,
    short_term_retention_days=config.agents.defaults.short_term_retention_days,
)
```

In the `run()` method (or wherever the main asyncio loop starts), add the daily sweep:

```python
async def _daily_short_term_sweep(self):
    """Compress short-term memory once daily."""
    while self._running:
        await asyncio.sleep(86400)
        try:
            await self._short_term.compress(self.provider, self.model)
        except Exception:
            logger.exception("Short-term memory sweep failed")
```

And schedule it where `self._running = True` is set (in `run()`), similar to how other background tasks are scheduled:

```python
self._schedule_background(self._daily_short_term_sweep())
```

- [ ] **Step 2: Verify nanobot still starts cleanly**

Run: `uv run nanobot status` (or whatever quick health check exists)
Expected: No import errors, clean startup

- [ ] **Step 3: Commit**

```bash
git add nanobot/agent/loop.py
git commit -m "feat: wire ShortTermMemory into AgentLoop with daily sweep"
```

---

### Task 8: Integration Test

**Files:**
- Create: `tests/test_short_term_integration.py`

- [ ] **Step 1: Write integration test**

```python
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
        store._on_history_append.append(stm.on_new_entry)

        provider = AsyncMock()
        provider.chat_with_retry = AsyncMock(
            return_value=_make_tool_response(
                "[2026-03-22 14:00] User asked about weather and Python scripts.",
                "# Memory\nUser interested in Python.",
            )
        )
        messages = [
            {"role": "user", "content": "hello", "timestamp": "2026-03-22 14:00"}
            for _ in range(10)
        ]

        result = await store.consolidate(messages, provider, "test-model")

        assert result is True
        assert stm.file.exists()
        content = stm.read()
        assert "Python" in content

    def test_context_builder_includes_short_term(self, tmp_path: Path) -> None:
        """ContextBuilder system prompt includes SHORT_TERM.md content."""
        # Create workspace structure
        (tmp_path / "memory").mkdir(exist_ok=True)
        (tmp_path / "memory" / "MEMORY.md").write_text("User likes Python.\n")
        (tmp_path / "memory" / "SHORT_TERM.md").write_text(
            "## Today (2026-03-22)\n- [14:00] Debugged auth issue.\n"
        )

        ctx = ContextBuilder(tmp_path)
        prompt = ctx.build_system_prompt()

        assert "User likes Python" in prompt
        assert "Recent Activity" in prompt
        assert "Debugged auth issue" in prompt
```

- [ ] **Step 2: Run integration tests**

Run: `uv run pytest tests/test_short_term_integration.py -x -v`
Expected: All PASS

- [ ] **Step 3: Commit**

```bash
git add tests/test_short_term_integration.py
git commit -m "test: add integration tests for short-term memory pipeline"
```

---

### Task 9: Live Verification with Gemini

- [ ] **Step 1: Run a quick live test**

```bash
uv run python -c "
import asyncio
from pathlib import Path
from nanobot.providers.gemini_native_provider import GeminiNativeProvider
from nanobot.agent.memory import MemoryStore
from nanobot.agent.memory_short_term import ShortTermMemory
import tempfile

provider = GeminiNativeProvider(
    api_key='sk-de7a3095987d4acea5c075ccc4fe030b',
    api_base='http://127.0.0.1:8045',
    default_model='gemini-3-flash',
)

async def main():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        store = MemoryStore(tmp)
        stm = ShortTermMemory(tmp)
        store._on_history_append.append(stm.on_new_entry)

        messages = [
            {'role': 'user', 'content': 'What is the weather?', 'timestamp': '2026-03-22 14:00'},
            {'role': 'assistant', 'content': 'Cannot check weather.', 'timestamp': '2026-03-22 14:00'},
            {'role': 'user', 'content': 'Help me write Python', 'timestamp': '2026-03-22 14:01'},
            {'role': 'assistant', 'content': 'Sure! What script?', 'timestamp': '2026-03-22 14:02'},
        ]

        result = await store.consolidate(messages, provider, 'gemini-3-flash')
        print(f'Consolidation: {result}')
        print(f'SHORT_TERM.md exists: {stm.file.exists()}')
        if stm.file.exists():
            print(f'Content:\\n{stm.read()}')

asyncio.run(main())
"
```

Expected: Consolidation succeeds, SHORT_TERM.md contains a "Today" section with the entry.

- [ ] **Step 2: Run full test suite**

Run: `uv run pytest tests/ -x -v --timeout=30`
Expected: All tests PASS

- [ ] **Step 3: Final commit if any fixes needed**

---

### Task 10: Documentation Update

- [ ] **Step 1: Update the memory skill**

If `nanobot/skills/memory/SKILL.md` mentions only MEMORY.md and HISTORY.md, add a brief note about SHORT_TERM.md:

```markdown
- SHORT_TERM.md: rolling recent activity (auto-maintained, 1-7 days). Loaded into context automatically.
```

- [ ] **Step 2: Commit**

```bash
git add nanobot/skills/memory/SKILL.md
git commit -m "docs: mention SHORT_TERM.md in memory skill documentation"
```
