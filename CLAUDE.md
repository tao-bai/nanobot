# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Always use uv, never pip/python directly
uv run nanobot gateway          # Start the gateway (channels + agent loop)
uv run nanobot chat             # Interactive CLI chat
uv run nanobot status           # Show config status
uv run pytest tests/ -x -q      # Run all tests (stop on first failure)
uv run pytest tests/test_foo.py -x -q                  # Single test file
uv run pytest tests/test_foo.py::test_bar -x -q        # Single test function
uv run ruff check nanobot/      # Lint
uv run ruff format nanobot/     # Format
uv run python -m py_compile nanobot/path/to/file.py    # Quick syntax check
```

## Architecture

**Message flow:** Channel → `InboundMessage` → MessageBus → AgentLoop → `OutboundMessage` → MessageBus → ChannelManager → Channel

### Core components

- **Agent loop** (`nanobot/agent/loop.py`): Iterates LLM call → tool execution up to 40 times. Returns `RunLoopResult` with `final_content`, `fallback_content`, and `emitted_fallback_content`. Messages are append-only (never mutated) for LLM cache efficiency.

- **Message bus** (`nanobot/bus/`): Async queues decoupling channels from agent. Session key: `"{channel}:{chat_id}"`.

- **Providers** (`nanobot/providers/`): Abstract `LLMProvider` base with implementations for LiteLLM (default router), Gemini native, Azure OpenAI, OpenAI Codex, and custom endpoints. Provider matching in `Config._match_provider()` follows: explicit → model prefix → keyword → local → gateway fallback. Registry metadata lives in `registry.py` as a `PROVIDERS` tuple.

- **Channels** (`nanobot/channels/`): Plugin architecture. Built-in channels discovered via `pkgutil`, external via `entry_points("nanobot.channels")`. All implement `BaseChannel` (start/stop/send). ChannelManager coordinates lifecycle and routes outbound messages.

- **Tools** (`nanobot/agent/tools/`): Registry pattern. Built-in: filesystem, shell, web, message, spawn, cron. MCP tools dynamically wrapped as `MCPToolWrapper` with `mcp_{server}_{tool}` naming and 30s timeout.

- **Config** (`nanobot/config/schema.py`): Pydantic V2 with camelCase JSON aliases. Lives at `~/.nanobot/config.json`. Workspace defaults to `~/.nanobot/workspace/`.

- **Memory** (`nanobot/agent/memory.py`): Two-layer — `MEMORY.md` (long-term facts) and `HISTORY.md` (time-indexed log). Consolidation triggered by token thresholds, uses LLM summarization.

- **Skills** (`nanobot/agent/skills.py`, `nanobot/skills/`): `SKILL.md` files with YAML frontmatter. Workspace skills override built-in. Loaded progressively (summary in system prompt, full content on demand).

### Provider selection path

CLI `_make_provider()` in `nanobot/cli/commands.py` checks: OAuth providers → direct providers (custom/azure/gemini with apiBase) → LiteLLM-routed (default).

## Code style

- **ruff**: line-length 100, target py311, rules E/F/I/N/W, E501 ignored
- **pytest**: `asyncio_mode = "auto"`, tests in `tests/`
- Config uses camelCase in JSON, snake_case in Python (pydantic alias_generator)
- Async-first: all I/O is async (channels, providers, tools)
- Tool results truncated to 16,000 chars

## Optional dependencies

Matrix (`matrix-nio`, `nh3`), WeCom, Gemini (`google-genai`), LangSmith are optional. Tests for these may fail if deps aren't installed — that's expected.
