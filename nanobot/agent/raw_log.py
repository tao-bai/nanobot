"""Append-only raw message log for permanent conversation archival."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from loguru import logger

from nanobot.utils.helpers import ensure_dir


class RawMessageLog:
    """Writes every message to daily JSONL files under ``memory/raw/``."""

    def __init__(self, memory_dir: Path) -> None:
        self.raw_dir = ensure_dir(memory_dir / "raw")

    def append(self, session_key: str, message: dict[str, Any]) -> None:
        """Append one message. Best-effort — failures logged, never raised."""
        try:
            now = datetime.now(timezone.utc)
            record = {"v": 1, "ts": now.isoformat(), "session_key": session_key, **message}
            path = self.raw_dir / f"{now.strftime('%Y-%m-%d')}.jsonl"
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        except Exception:
            logger.warning("Failed to write raw message log", exc_info=True)

    @staticmethod
    def strip_base64_images(message: dict[str, Any]) -> dict[str, Any]:
        """Return a copy with base64 image URLs replaced by path placeholders."""
        content = message.get("content")
        if not isinstance(content, list):
            return message
        filtered = []
        for c in content:
            if (
                c.get("type") == "image_url"
                and c.get("image_url", {}).get("url", "").startswith("data:image/")
            ):
                path = (c.get("_meta") or {}).get("path", "")
                filtered.append({"type": "text", "text": f"[image: {path}]" if path else "[image]"})
            else:
                filtered.append(c)
        return {**message, "content": filtered}
