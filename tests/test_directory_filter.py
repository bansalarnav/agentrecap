import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from agentrecap.adapters.common import anonymous_id
from agentrecap.directory_filter import matching_thread_ids
from agentrecap.gather_session_data import convert_sessions


class DirectoryFilterTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.project = self.root / "project"
        self.project.mkdir()
        self.other = self.root / "other"
        self.other.mkdir()

    def jsonl(self, name, records):
        path = self.root / name
        path.write_text("".join(json.dumps(record) + "\n" for record in records))
        return path

    def test_codex_start_directory_and_read_command(self):
        started = self.jsonl("started.jsonl", [
            {"type": "session_meta", "payload": {"id": "started", "cwd": str(self.project / "sub")}},
        ])
        read = self.jsonl("read.jsonl", [
            {"type": "session_meta", "payload": {"id": "read", "cwd": str(self.other)}},
            {"type": "response_item", "payload": {"type": "function_call", "name": "exec_command",
                "arguments": json.dumps({"cmd": f"cat {self.project / 'file.txt'}"})}},
        ])
        near = self.jsonl("near.jsonl", [
            {"type": "session_meta", "payload": {"id": "near", "cwd": str(self.other)}},
            {"type": "response_item", "payload": {"type": "function_call", "name": "exec_command",
                "arguments": json.dumps({"cmd": f"cat {self.root / 'project-other' / 'file.txt'}"})}},
        ])
        self.assertEqual(matching_thread_ids("codex", started, self.project), {anonymous_id("codex:started")})
        self.assertEqual(matching_thread_ids("codex", read, self.project), {anonymous_id("codex:read")})
        self.assertEqual(matching_thread_ids("codex", near, self.project), set())

    def test_claude_read_and_pi_relative_read(self):
        claude = self.jsonl("claude.jsonl", [
            {"sessionId": "claude-read", "cwd": str(self.other), "message": {"content": [
                {"type": "tool_use", "name": "Read", "input": {"file_path": str(self.project / "README.md")}}
            ]}},
        ])
        pi = self.jsonl("pi.jsonl", [
            {"type": "session", "id": "pi-read", "cwd": str(self.root)},
            {"type": "message", "message": {"role": "assistant", "content": [
                {"type": "toolCall", "name": "read", "arguments": {"path": "project/README.md"}}
            ]}},
        ])
        self.assertEqual(matching_thread_ids("claude", claude, self.project), {anonymous_id("claude:claude-read")})
        self.assertEqual(matching_thread_ids("pi", pi, self.project), {anonymous_id("pi:pi-read")})
        self.assertEqual(matching_thread_ids("omp", pi, self.project), {anonymous_id("omp:pi-read")})

    def test_opencode_database_selects_only_matching_session(self):
        path = self.root / "opencode.db"
        with sqlite3.connect(path) as connection:
            connection.execute("CREATE TABLE session (id TEXT, directory TEXT)")
            connection.execute("CREATE TABLE part (session_id TEXT, data TEXT)")
            connection.executemany("INSERT INTO session VALUES (?, ?)", [
                ("started", str(self.project)),
                ("read", str(self.other)),
                ("unrelated", str(self.other)),
            ])
            connection.executemany("INSERT INTO part VALUES (?, ?)", [
                ("read", json.dumps({"type": "tool", "state": {"input": {"filePath": str(self.project / "a.py")}}})),
                ("unrelated", json.dumps({"type": "tool", "state": {"input": {"filePath": str(self.other / "a.py")}}})),
            ])
        self.assertEqual(matching_thread_ids("opencode", path, self.project), {
            anonymous_id("opencode:started"), anonymous_id("opencode:read"),
        })

    def test_codex_wrapped_command_and_user_text(self):
        path = self.jsonl("wrapped.jsonl", [
            {"type": "session_meta", "payload": {"id": "wrapped", "cwd": str(self.other)}},
            {"type": "event_msg", "payload": {"type": "user_message", "message": str(self.project)}},
            {"type": "response_item", "payload": {"type": "function_call", "name": "functions.exec",
                "arguments": f'tools.exec_command({{cmd:"cat README.md",workdir:"{self.project}"}})'}},
        ])
        self.assertEqual(matching_thread_ids("codex", path, self.project), {anonymous_id("codex:wrapped")})

        mention_only = self.jsonl("mention.jsonl", [
            {"type": "session_meta", "payload": {"id": "mention", "cwd": str(self.other)}},
            {"type": "event_msg", "payload": {"type": "user_message", "message": str(self.project)}},
        ])
        self.assertEqual(matching_thread_ids("codex", mention_only, self.project), set())

    def test_opencode_legacy_session(self):
        storage = self.root / "storage"
        session_dir = storage / "session" / "project-id"
        message_dir = storage / "message" / "legacy"
        part_dir = storage / "part" / "message-id"
        for directory in (session_dir, message_dir, part_dir):
            directory.mkdir(parents=True)
        session = session_dir / "legacy.json"
        session.write_text(json.dumps({"id": "legacy", "directory": str(self.other)}))
        (message_dir / "message-id.json").write_text(json.dumps({"id": "message-id"}))
        (part_dir / "part-id.json").write_text(json.dumps({
            "type": "tool", "state": {"input": {"command": f"cat {self.project / 'README.md'}"}},
        }))
        self.assertEqual(matching_thread_ids("opencode", session, self.project), {
            anonymous_id("opencode:legacy"),
        })

    def test_conversion_keeps_entire_matching_session(self):
        sessions = self.root / "sessions"
        sessions.mkdir()
        for name, command in (
            ("read", f"cat {self.project / 'a.py'}"),
            ("unrelated", f"cat {self.other / 'a.py'}"),
        ):
            (sessions / f"{name}.jsonl").write_text("".join(json.dumps(record) + "\n" for record in [
                {"type": "session_meta", "timestamp": "2026-01-01T00:00:00Z",
                    "payload": {"id": name, "cwd": str(self.other)}},
                {"type": "response_item", "timestamp": "2026-01-01T00:00:01Z",
                    "payload": {"type": "function_call", "name": "exec_command",
                        "arguments": json.dumps({"cmd": command})}},
                {"type": "event_msg", "timestamp": "2026-01-01T00:00:02Z",
                    "payload": {"type": "task_complete"}},
            ]))
        output = self.root / "events.csv"
        result = convert_sessions({"codex": self.root}, output, directory=self.project)
        self.assertEqual(result, {"threads": {"codex": 1}, "events": 3})
        self.assertIn(anonymous_id("codex:read"), output.read_text())
        self.assertNotIn(anonymous_id("codex:unrelated"), output.read_text())


if __name__ == "__main__":
    unittest.main()
