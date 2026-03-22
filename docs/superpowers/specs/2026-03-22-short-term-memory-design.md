# Short-Term Memory Layer — Design Spec

**Date:** 2026-03-22
**Status:** Approved
**Goal:** Add a rolling short-term memory layer that gives the agent temporal awareness of recent activity (1 day–1 week) without dramatically changing `memory.py`.

## Problem

Once messages are consolidated, the agent loses temporal context. `MEMORY.md` stores timeless facts ("user likes Python") but not recent activity ("yesterday we debugged an auth race condition"). `HISTORY.md` has the data but is never loaded into context — the agent must explicitly grep it.

## Design Principles

- **Minimal touch on `memory.py`** — 3 lines (callback hook) to avoid merge conflicts with upstream
- **Simple over clever** — calendar tiers, plain markdown, no JSONL/database
- **Fixed token budget** — 2K–4K tokens, configurable
- **Progressive compression** — recent entries are detailed, older entries are compressed

## Architecture

### New Files

| File | Purpose |
|------|---------|
| `nanobot/agent/memory_short_term.py` | All short-term memory logic (~150–200 lines) |
| `{workspace}/memory/SHORT_TERM.md` | Rendered output, auto-generated |

### Changes to Existing Files

| File | Change | Lines |
|------|--------|-------|
| `nanobot/agent/memory.py` | Add `_on_history_append` callback list on `MemoryStore`, fire after `append_history()` | ~3 |
| `nanobot/agent/context.py` | Read `SHORT_TERM.md` in `ContextBuilder`, inject as "Recent Activity" alongside "Long-term Memory" | ~5 |
| `nanobot/agent/loop.py` | Register short-term callback on `memory_consolidator.store`, start daily sweep task | ~5 |
| `nanobot/config/schema.py` | Add `short_term_token_budget` and `short_term_retention_days` to `AgentDefaults` | ~2 |

### Data Flow

```
Consolidation fires (context overflow or /new)
    │
    ▼
MemoryStore.append_history(entry)
    │ callback
    ▼
ShortTermMemory.on_new_entry(entry)
    → Skip if entry contains "[RAW]" (raw-archive fallback, not a clean summary)
    → Parse timestamp from entry (format: [YYYY-MM-DD HH:MM], treated as local time)
    → Convert to UTC for storage
    → Append to today's section in SHORT_TERM.md
    → No compression (just append)

Daily sweep (asyncio background task)
    │
    ▼
ShortTermMemory.compress(provider, model)
    → Acquire internal asyncio.Lock (serializes with on_new_entry)
    → Read SHORT_TERM.md, parse entries by embedded UTC timestamps
    → Re-bucket: today / yesterday / this week / older (computed from local timezone)
    → LLM compresses "this week" and "older" tiers into paragraphs
    → Drop entries >7 days old
    → Enforce token budget via len(text) // 4 heuristic (trim oldest first)
    → Write SHORT_TERM.md

Every LLM call
    │
    ▼
ContextBuilder.build_system_prompt()
    → Reads MEMORY.md via MemoryStore → "## Long-term Memory"
    → Reads SHORT_TERM.md directly from workspace → "## Recent Activity"  ← NEW
    → Both injected into system prompt
```

## SHORT_TERM.md Format

Tiered markdown with UTC timestamps as HTML comments for re-bucketing:

```markdown
## Today (2026-03-22)
<!-- 2026-03-22T06:00:00Z -->
- [14:00] Discussed weather query limitations. User asked for Python script help.
<!-- 2026-03-22T07:30:00Z -->
- [15:30] Debugged auth middleware race condition in session manager.

## Yesterday (2026-03-21)
<!-- 2026-03-21T02:00:00Z -->
- Explored nanobot memory architecture. User wants long/short memory with minimal changes.
<!-- 2026-03-21T08:00:00Z -->
- Fixed Gemini consolidation bug — model hallucinating field names.

## This Week (2026-03-17 — 2026-03-20)
Provider infrastructure: added Gemini native SDK support, fixed tool name aliasing,
added multimodal image support. Token-based consolidation rollout.

## Older (2026-03-10 — 2026-03-16)
Initial provider registry and config system work.
```

### Tier Rules

| Tier | Age | Format | Updated by |
|------|-----|--------|------------|
| Today | 0 days | Per-entry with `[HH:MM]` local timestamps | Callback (append) |
| Yesterday | 1 day | Per-entry, no timestamps | Daily sweep (re-bucket) |
| This Week | 2–6 days | One paragraph (LLM compressed) | Daily sweep |
| Older | 7+ days | One paragraph (LLM compressed), then dropped | Daily sweep |

### Timestamp Handling

- Source: `[YYYY-MM-DD HH:MM]` timestamps from `history_entry` are **local time** (produced by `datetime.now()` in raw-archive, and by the LLM for clean entries)
- Storage: entries store UTC timestamps as HTML comments: `<!-- 2026-03-22T06:00:00Z -->`, converted from local at insertion time
- Rendering: tier labels ("Today", "Yesterday") are computed at render time by converting stored UTC back to local timezone
- Re-bucketing happens during daily sweep by comparing stored UTC to current local date

## Hook in memory.py

Three lines added to `MemoryStore`:

```python
# __init__:
self._on_history_append: list[Callable[[str], None]] = []

# append_history(), after file write:
for cb in self._on_history_append:
    cb(entry)
```

**Important:** The callback fires for both clean consolidation entries AND raw-archive fallback entries. The `ShortTermMemory.on_new_entry()` handler must detect and skip `[RAW]` entries (check for `[RAW]` marker in the entry text). Only clean LLM-generated entries get added to short-term memory.

**Instance note:** The callback must be registered on `agent.memory_consolidator.store._on_history_append` (the `MemoryStore` instance used by consolidation), NOT on `context.memory` (which is a separate `MemoryStore` instance in `ContextBuilder`). The consolidation flow calls `append_history()` on `MemoryConsolidator.store`.

## Context Injection

In `context.py`, `ContextBuilder` reads `SHORT_TERM.md` directly from the workspace (not via `MemoryStore`, to keep memory.py changes minimal):

```python
# In ContextBuilder.build_system_prompt() or a helper method:
def _read_short_term(self) -> str:
    path = self.workspace / "memory" / "SHORT_TERM.md"
    if path.exists():
        return path.read_text(encoding="utf-8").strip()
    return ""
```

This is added to `ContextBuilder`, not `MemoryStore`. The existing `memory.get_memory_context()` continues to return only long-term memory. The system prompt assembly in `build_system_prompt()` appends the short-term content as a "## Recent Activity" section after the memory section.

## Daily Compression Sweep

### Trigger

A simple `asyncio` background task started in `AgentLoop.run()` (or `__init__`):

```python
async def _daily_short_term_sweep(self):
    """Run short-term memory compression once daily."""
    while self._running:
        await asyncio.sleep(86400)  # 24 hours
        try:
            await self._short_term.compress(self.provider, self.model)
        except Exception:
            logger.exception("Short-term memory sweep failed")
```

This is lightweight — no dependency on `CronService` or `HeartbeatService`. On process restart, no immediate sweep runs (entries accumulate uncompressed until the next 24h cycle, which is fine since uncompressed entries are still valid).

### Algorithm
1. Acquire `ShortTermMemory._lock` (shared with `on_new_entry`)
2. Read `SHORT_TERM.md`, parse entries and UTC timestamps from HTML comments
3. Compute tier boundaries based on current local date
4. Re-bucket entries into tiers
5. For **This Week** tier: if >3 entries per day, call LLM to compress into one paragraph per day
6. For **Older** tier: call LLM to compress all entries into one paragraph
7. Drop entries older than retention period (configurable, default 7 days)
8. Estimate total tokens via `len(text) // 4` character heuristic; if over budget, truncate from oldest tier
9. Write final `SHORT_TERM.md`

### LLM Compression Call
- Uses same provider/model as main agent (passed to `ShortTermMemory` at init)
- Simple prompt: "Compress these entries into a concise paragraph preserving key decisions, open tasks, and outcomes"
- No tool calling needed — plain text response

## Concurrency

`ShortTermMemory` uses a single `asyncio.Lock` to serialize:
- `on_new_entry()` — appending new entries from callback
- `compress()` — daily sweep reading/rewriting the file

This prevents the sweep from reading a stale version while a callback is mid-write.

## Token Budget

- Default: **3K tokens** (configurable via `short_term_token_budget` in config)
- Estimation: `len(rendered_text) // 4` (character-based heuristic, same approach as the fallback in `estimate_message_tokens`)
- Enforcement: after compression, estimate tokens; if over budget, progressively drop from oldest tier
- For 64K context: ~5% overhead. For 1M context: negligible.

## Configuration

New fields in `AgentDefaults` (config/schema.py), with camelCase JSON aliases:

```python
# Python attributes (snake_case):
short_term_token_budget: int = 3000
short_term_retention_days: int = 7
```

```json
// JSON config (camelCase via pydantic alias):
{
  "agents": {
    "defaults": {
      "shortTermTokenBudget": 3000,
      "shortTermRetentionDays": 7
    }
  }
}
```

## Error Handling

- If `SHORT_TERM.md` doesn't exist or is empty, context injection silently skips it
- If daily sweep LLM call fails, keep entries uncompressed (they'll be compressed next run)
- If callback throws, log warning but don't break consolidation flow (wrapped in try/except)
- File write uses `Path.write_text()` (consistent with existing `MemoryStore.write_long_term()` pattern — SHORT_TERM.md is auto-generated and recoverable)

## Bootstrapping (Existing Workspaces)

For workspaces that already have a populated `HISTORY.md` but no `SHORT_TERM.md`:
- No automatic bootstrapping on first run — SHORT_TERM.md starts empty
- Entries accumulate naturally from the next consolidation onwards
- Optional future enhancement: one-time seed from recent HISTORY.md entries (out of scope for v1)

## Testing Strategy

1. **Unit tests** for `memory_short_term.py`:
   - Entry parsing (timestamp extraction from `[YYYY-MM-DD HH:MM]` format)
   - `[RAW]` entry detection and skipping
   - Tier bucketing logic (given entries with various dates, verify correct tiers)
   - Token budget enforcement (given entries exceeding budget, verify trimming)
   - Concurrency: callback and compress don't corrupt each other

2. **Integration test**:
   - Full cycle: consolidation → callback → SHORT_TERM.md written → context builder loads it
   - Verify callback is registered on correct MemoryStore instance

3. **Live test**:
   - Run nanobot with Gemini, have a conversation, trigger consolidation, verify SHORT_TERM.md content
   - Trigger compress manually, verify tier re-bucketing and LLM compression

## Future Considerations (Out of Scope)

- Salience-based prioritization (keep "open loops" over old narrative)
- Cross-workspace memory federation
- Embedding-based retrieval from HISTORY.md as supplement
- Per-session vs per-workspace short-term memory separation
- One-time bootstrap from existing HISTORY.md
