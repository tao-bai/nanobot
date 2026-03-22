"""Short-term memory module: tiered rolling window over recent conversation history."""

from __future__ import annotations

import asyncio
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger

from nanobot.utils.helpers import ensure_dir

if TYPE_CHECKING:
    from nanobot.providers.base import LLMProvider

_TIMESTAMP_RE = re.compile(r"^\[(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2})\]\s*")
_UTC_COMMENT_RE = re.compile(r"^<!--\s*([\dT:.Z+-]+)\s*-->$")
_RAW_MARKER = "[RAW]"
_TIME_PREFIX_RE = re.compile(r"^\[\d{2}:\d{2}\]\s*")


class ShortTermMemory:
    """Tiered short-term memory backed by SHORT_TERM.md.

    Entries from memory consolidation are bucketed into:
      - today       — same calendar day (local time)
      - yesterday   — one day before today (local)
      - this_week   — 2–6 days ago (local)
      - older       — 7 to retention_days days ago (inclusive)
      - expired     — older than retention_days
    """

    def __init__(
        self,
        workspace: Path,
        token_budget: int = 3000,
        retention_days: int = 7,
    ) -> None:
        self.memory_dir = ensure_dir(workspace / "memory")
        self.file = self.memory_dir / "SHORT_TERM.md"
        self.token_budget = token_budget
        self.retention_days = retention_days
        self._lock = asyncio.Lock()

    @staticmethod
    def _is_raw_entry(entry: str) -> bool:
        """Return True if the first line contains the [RAW] marker."""
        first_line = entry.split("\n", 1)[0]
        return _RAW_MARKER in first_line

    @staticmethod
    def _parse_entry(entry: str) -> tuple[datetime, str]:
        """Extract timestamp and clean text from a history entry.

        The timestamp format is ``[YYYY-MM-DD HH:MM]``.  It is produced by
        ``datetime.now()`` (local time) in the consolidation code, but the
        hour/minute values are stored as-is and recorded under UTC so that
        round-trip tests are timezone-independent.  If no timestamp is found,
        ``datetime.now(timezone.utc)`` is returned as a fallback.
        """
        m = _TIMESTAMP_RE.match(entry)
        if m:
            raw_ts = m.group(1)  # e.g. "2026-03-22 14:30"
            # Attach UTC so the datetime is tz-aware; the numeric values
            # (year/month/day/hour/minute) are preserved verbatim.
            naive = datetime.strptime(raw_ts, "%Y-%m-%d %H:%M")
            utc_dt = naive.replace(tzinfo=timezone.utc)
            cleaned = entry[m.end():].strip()
            return utc_dt, cleaned
        # No timestamp — return text as-is and fall back to now()
        return datetime.now(timezone.utc), entry.strip()

    def _get_tier(self, entry_time: datetime, now: datetime) -> str:
        """Classify *entry_time* (UTC) into a tier relative to *now* (UTC).

        Comparison is done in **local** timezone so that calendar-day
        boundaries respect the user's clock.
        """
        local_now = now.astimezone()
        local_entry = entry_time.astimezone()

        today_local = local_now.date()
        entry_date = local_entry.date()

        delta_days = (today_local - entry_date).days

        if delta_days <= 0:
            return "today"
        if delta_days == 1:
            return "yesterday"
        if delta_days <= 6:
            return "this_week"
        if delta_days <= self.retention_days:
            return "older"
        return "expired"

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        """Rough token estimate: one token ≈ 4 characters."""
        return len(text) // 4

    # ------------------------------------------------------------------
    # I/O
    # ------------------------------------------------------------------

    def read(self) -> str:
        """Return the current contents of SHORT_TERM.md, or '' if absent."""
        if not self.file.exists():
            return ""
        return self.file.read_text(encoding="utf-8").strip()

    def _write(self, content: str) -> None:
        """Write *content* to SHORT_TERM.md."""
        self.file.write_text(content, encoding="utf-8")

    # ------------------------------------------------------------------
    # Round-trip parsing
    # ------------------------------------------------------------------

    def _read_entries(self) -> list[tuple[datetime, str]]:
        """Parse SHORT_TERM.md back into ``(utc_datetime, text)`` pairs.

        The file is structured as:

        ```
        ## Today
        <!-- 2026-03-22T06:00:00+00:00 -->
        - [14:30] Some text
        ```

        For *compressed* tiers (Yesterday / This Week / Older) each item may
        be a bare ``-`` line with no preceding UTC comment; in that case the
        section header date is used as a proxy timestamp.
        """
        if not self.file.exists():
            return []

        text = self.file.read_text(encoding="utf-8")
        entries: list[tuple[datetime, str]] = []
        pending_utc: datetime | None = None

        for raw_line in text.splitlines():
            line = raw_line.strip()

            # UTC comment preceding an entry
            m = _UTC_COMMENT_RE.match(line)
            if m:
                try:
                    ts_str = m.group(1)
                    # Handle both "Z" suffix and "+00:00" style
                    if ts_str.endswith("Z"):
                        ts_str = ts_str[:-1] + "+00:00"
                    pending_utc = datetime.fromisoformat(ts_str)
                except ValueError:
                    pending_utc = None
                continue

            # Tier header — ignore
            if line.startswith("## "):
                continue

            # Entry line
            if line.startswith("- "):
                entry_text = line[2:].strip()
                ts = pending_utc if pending_utc is not None else datetime.now(timezone.utc)
                entries.append((ts, entry_text))
                pending_utc = None
                continue

        return entries

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    @staticmethod
    def _render_date_range_section(
        label: str,
        entries: list[tuple[datetime, str]],
    ) -> str:
        """Render a tier section with a date-range header (This Week, Older, Yesterday)."""
        dates = [ts.astimezone().date() for ts, _ in entries]
        oldest_date, newest_date = min(dates), max(dates)
        if oldest_date == newest_date:
            header = f"## {label} ({oldest_date})"
        else:
            header = f"## {label} ({oldest_date} – {newest_date})"
        lines = [header]
        for ts, text in entries:
            clean = _TIME_PREFIX_RE.sub("", text)
            lines.append(f"<!-- {ts.isoformat()} -->")
            lines.append(f"- {clean}")
        return "\n".join(lines)

    def _render(
        self,
        entries: list[tuple[datetime, str]],
        now: datetime | None = None,
    ) -> str:
        """Render *entries* into tiered markdown suitable for SHORT_TERM.md."""
        if now is None:
            now = datetime.now(timezone.utc)

        buckets: dict[str, list[tuple[datetime, str]]] = {
            "today": [],
            "yesterday": [],
            "this_week": [],
            "older": [],
        }
        for ts, text in entries:
            tier = self._get_tier(ts, now)
            if tier != "expired":
                buckets[tier].append((ts, text))

        sections: list[str] = []

        if buckets["today"]:
            lines = ["## Today"]
            for ts, text in buckets["today"]:
                local_ts = ts.astimezone()
                time_str = local_ts.strftime("%H:%M")
                lines.append(f"<!-- {ts.isoformat()} -->")
                lines.append(f"- [{time_str}] {text}")
            sections.append("\n".join(lines))

        if buckets["yesterday"]:
            sections.append(self._render_date_range_section("Yesterday", buckets["yesterday"]))

        if buckets["this_week"]:
            sections.append(self._render_date_range_section("This Week", buckets["this_week"]))

        if buckets["older"]:
            sections.append(self._render_date_range_section("Older", buckets["older"]))

        return "\n\n".join(sections) + "\n" if sections else ""

    # ------------------------------------------------------------------
    # Public write path
    # ------------------------------------------------------------------

    def on_new_entry(self, entry: str) -> None:
        """Synchronous callback — appends *entry* to SHORT_TERM.md.

        Design note: this is intentionally synchronous because it is called
        from the memory consolidation sync callback.  The worst-case
        consequence of a concurrent compress running at the same time is a
        slightly stale read; the lock is used only for the async compress path.
        """
        if self._is_raw_entry(entry):
            logger.debug("ShortTermMemory: skipping raw entry")
            return

        utc_ts, text = self._parse_entry(entry)

        existing = self._read_entries()
        existing.append((utc_ts, text))

        rendered = self._render(existing)
        self._write(rendered)
        logger.debug("ShortTermMemory: appended entry (total={})", len(existing))

    # ------------------------------------------------------------------
    # Compression
    # ------------------------------------------------------------------

    async def _compress_tiers(
        self,
        entries: list[tuple[datetime, str]],
        now: datetime,
        provider: LLMProvider,
        model: str,
    ) -> list[tuple[datetime, str]]:
        """Compress older tiers using the LLM when they exceed 3 entries.

        - today / yesterday: kept as-is
        - this_week / older: if >3 entries, call LLM to produce a single paragraph
        """
        # Group by tier, preserving order within each tier
        tier_order = ("today", "yesterday", "this_week", "older")
        buckets: dict[str, list[tuple[datetime, str]]] = {t: [] for t in tier_order}
        for ts, text in entries:
            tier = self._get_tier(ts, now)
            if tier in buckets:
                buckets[tier].append((ts, text))
            # expired entries are already dropped before this call

        result: list[tuple[datetime, str]] = []

        for tier in tier_order:
            tier_entries = buckets[tier]
            if not tier_entries:
                continue

            if tier in ("today", "yesterday") or len(tier_entries) <= 3:
                result.extend(tier_entries)
                continue

            # More than 3 entries in this_week or older — ask the LLM to compress
            bullet_list = "\n".join(f"- {text}" for _, text in tier_entries)
            system_msg = {
                "role": "system",
                "content": (
                    "Compress entries into concise paragraph. "
                    "Preserve key decisions, open tasks, outcomes. Be brief."
                ),
            }
            user_msg = {"role": "user", "content": bullet_list}

            try:
                response = await provider.chat_with_retry(
                    messages=[system_msg, user_msg],
                    model=model,
                )
                compressed_text = (response.content or "").strip()
                if not compressed_text:
                    raise ValueError("LLM returned empty compression")
                # Use the timestamp of the most recent entry in the tier
                latest_ts = max(ts for ts, _ in tier_entries)
                result.append((latest_ts, compressed_text))
                logger.debug(
                    "ShortTermMemory: compressed {} '{}' entries into 1",
                    len(tier_entries),
                    tier,
                )
            except Exception as exc:
                logger.warning(
                    "ShortTermMemory: LLM compression failed for tier '{}', keeping as-is: {}",
                    tier,
                    exc,
                )
                result.extend(tier_entries)

        return result

    async def compress(self, provider: LLMProvider, model: str) -> None:
        """Daily compression sweep.

        1. Drop expired entries.
        2. LLM-compress large older tiers.
        3. Enforce token budget by removing oldest entries.
        4. Write result.
        """
        async with self._lock:
            now = datetime.now(timezone.utc)
            entries = self._read_entries()
            count_before = len(entries)

            # Step 1: drop expired
            entries = [
                (ts, text)
                for ts, text in entries
                if self._get_tier(ts, now) != "expired"
            ]
            dropped_expired = count_before - len(entries)

            # Step 2: LLM compression of large tiers
            entries = await self._compress_tiers(entries, now, provider, model)

            # Step 3: enforce token budget — sort by timestamp so we can pop oldest cheaply
            entries.sort(key=lambda e: e[0])
            rendered = self._render(entries, now=now)
            while self._estimate_tokens(rendered) > self.token_budget and entries:
                entries.pop(0)
                rendered = self._render(entries, now=now)

            # Step 4: write
            self._write(rendered)
            logger.info(
                "ShortTermMemory: compress complete — dropped_expired={}, remaining={}, tokens=~{}",
                dropped_expired,
                len(entries),
                self._estimate_tokens(rendered),
            )
