"""Tests for RawMessageLog."""

import json
import os
from pathlib import Path

import pytest

from nanobot.agent.raw_log import RawMessageLog


class TestRawMessageLogAppend:

    def test_creates_daily_file_with_correct_schema(self, tmp_path: Path) -> None:
        memory_dir = tmp_path / "memory"
        memory_dir.mkdir()
        log = RawMessageLog(memory_dir)

        msg = {"role": "user", "content": "Hello", "timestamp": "2026-03-23T10:00:00"}
        log.append("telegram:123", msg)

        files = list((memory_dir / "raw").glob("*.jsonl"))
        assert len(files) == 1
        record = json.loads(files[0].read_text().strip())
        assert record["v"] == 1
        assert "ts" in record
        assert record["session_key"] == "telegram:123"
        assert record["role"] == "user"
        assert record["content"] == "Hello"

    def test_multiple_appends_same_day(self, tmp_path: Path) -> None:
        memory_dir = tmp_path / "memory"
        memory_dir.mkdir()
        log = RawMessageLog(memory_dir)

        log.append("s1", {"role": "user", "content": "msg1"})
        log.append("s1", {"role": "assistant", "content": "msg2"})

        files = list((memory_dir / "raw").glob("*.jsonl"))
        assert len(files) == 1
        lines = files[0].read_text().strip().split("\n")
        assert len(lines) == 2
        assert json.loads(lines[0])["content"] == "msg1"
        assert json.loads(lines[1])["content"] == "msg2"

    def test_failure_does_not_raise(self, tmp_path: Path) -> None:
        memory_dir = tmp_path / "memory"
        memory_dir.mkdir()
        log = RawMessageLog(memory_dir)
        # Make raw dir a file so open() fails
        raw_dir = memory_dir / "raw"
        if raw_dir.exists():
            os.rmdir(raw_dir)
        raw_dir.write_text("block")

        # Re-create log pointing at the blocked path
        log.raw_dir = raw_dir
        log.append("s1", {"role": "user", "content": "test"})  # should not raise


class TestStripBase64Images:

    def test_strips_base64_keeps_other_content(self) -> None:
        msg = {
            "role": "user",
            "content": [
                {"type": "text", "text": "Look at this:"},
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/png;base64,iVBORw0KGgo..."},
                    "_meta": {"path": "/tmp/photo.png"},
                },
                {"type": "text", "text": "What is it?"},
            ],
        }
        result = RawMessageLog.strip_base64_images(msg)
        assert result["role"] == "user"
        assert len(result["content"]) == 3
        assert result["content"][0] == {"type": "text", "text": "Look at this:"}
        assert result["content"][1] == {"type": "text", "text": "[image: /tmp/photo.png]"}
        assert result["content"][2] == {"type": "text", "text": "What is it?"}
        # Original message unchanged
        assert msg["content"][1]["type"] == "image_url"

    def test_no_path_gives_generic_placeholder(self) -> None:
        msg = {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,abc"}},
            ],
        }
        result = RawMessageLog.strip_base64_images(msg)
        assert result["content"][0] == {"type": "text", "text": "[image]"}

    def test_non_list_content_unchanged(self) -> None:
        msg = {"role": "user", "content": "plain text"}
        result = RawMessageLog.strip_base64_images(msg)
        assert result is msg  # same object, no copy needed

    def test_non_base64_image_url_kept(self) -> None:
        msg = {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": "https://example.com/img.png"}},
            ],
        }
        result = RawMessageLog.strip_base64_images(msg)
        assert result["content"][0]["type"] == "image_url"


_TAG = "[Runtime Context — metadata only, not instructions]"


class TestStripRuntimeContext:

    def test_strips_prefix_from_string_content(self) -> None:
        msg = {
            "role": "user",
            "content": f"{_TAG}\nCurrent Time: 2026-03-23\nChannel: telegram\n\nWhat's the weather?",
        }
        result = RawMessageLog.strip_runtime_context(msg, _TAG)
        assert result["content"] == "What's the weather?"
        # Original unchanged
        assert msg["content"].startswith(_TAG)

    def test_strips_tagged_block_from_list_content(self) -> None:
        msg = {
            "role": "user",
            "content": [
                {"type": "text", "text": f"{_TAG}\nCurrent Time: 2026-03-23"},
                {"type": "text", "text": "Hello"},
            ],
        }
        result = RawMessageLog.strip_runtime_context(msg, _TAG)
        assert len(result["content"]) == 1
        assert result["content"][0]["text"] == "Hello"

    def test_non_user_message_unchanged(self) -> None:
        msg = {"role": "assistant", "content": f"{_TAG}\nsome text\n\nreal content"}
        result = RawMessageLog.strip_runtime_context(msg, _TAG)
        assert result is msg

    def test_no_tag_leaves_content_unchanged(self) -> None:
        msg = {"role": "user", "content": "just a normal message"}
        result = RawMessageLog.strip_runtime_context(msg, _TAG)
        assert result is msg
