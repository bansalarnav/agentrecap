import csv
import json
import sqlite3
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from agentrecap.adapters import ADAPTERS
from agentrecap.adapters.common import anonymous_id
from agentrecap.gather_session_data import convert_sessions


class TranscriptTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def jsonl(self, name, records):
        path = self.root / name
        path.write_text(''.join(json.dumps(record) + '\n' for record in records))
        return path

    def test_content_is_opt_in_for_jsonl_adapters(self):
        fixtures = {
            'codex': [
                {'type': 'session_meta', 'payload': {'id': 'one'}},
                {'type': 'response_item', 'payload': {'type': 'message', 'role': 'user', 'content': 'hello'}},
                {'type': 'response_item', 'payload': {'type': 'reasoning', 'summary': 'thinking'}},
                {'type': 'response_item', 'payload': {'type': 'function_call', 'arguments': {'path': 'file'}}},
                {'type': 'response_item', 'payload': {'type': 'function_call_output', 'output': 'result'}},
            ],
            'claude': [
                {'type': 'user', 'sessionId': 'one', 'message': {'content': [{'type': 'text', 'text': 'hello'}]}},
                {'type': 'assistant', 'message': {'content': [
                    {'type': 'thinking', 'thinking': 'thinking'},
                    {'type': 'tool_use', 'input': {'path': 'file'}},
                ]}},
                {'type': 'user', 'message': {'content': [{'type': 'tool_result', 'content': 'result'}]}},
            ],
            'pi': [
                {'type': 'session', 'id': 'one'},
                {'type': 'message', 'message': {'role': 'user', 'content': 'hello'}},
                {'type': 'message', 'message': {'role': 'assistant', 'content': [
                    {'type': 'thinking', 'thinking': 'thinking'},
                    {'type': 'toolCall', 'arguments': {'path': 'file'}},
                ]}},
                {'type': 'message', 'message': {'role': 'toolResult', 'output': 'result'}},
            ],
        }
        fixtures['omp'] = fixtures['pi']
        for source, records in fixtures.items():
            with self.subTest(source=source):
                path = self.jsonl(source + '.jsonl', records)
                default = ADAPTERS[source].convert_thread(path)
                included = ADAPTERS[source].convert_thread(path, with_transcript=True)
                self.assertTrue(all('transcript' not in event for event in default))
                self.assertEqual(default, [{k: v for k, v in event.items() if k != 'transcript'} for event in included])
                contents = [event['transcript'] for event in included if event.get('transcript')]
                self.assertIn({'text': 'hello'}, contents)
                self.assertIn({'text': 'thinking'}, contents)
                self.assertIn({'tool_input': {'path': 'file'}}, contents)
                self.assertIn({'tool_output': 'result'}, contents)

    def test_opencode_database_content(self):
        path = self.root / 'opencode.db'
        with sqlite3.connect(path) as db:
            db.execute('CREATE TABLE session (id TEXT, parent_id TEXT, time_created INTEGER, time_updated INTEGER)')
            db.execute('CREATE TABLE message (id TEXT, session_id TEXT, time_created INTEGER, data TEXT)')
            db.execute('CREATE TABLE part (id TEXT, message_id TEXT, session_id TEXT, time_created INTEGER, data TEXT)')
            db.execute("INSERT INTO session VALUES ('one', NULL, 1000, 2000)")
            db.execute('INSERT INTO message VALUES (?, ?, ?, ?)', ('m', 'one', 1000, json.dumps({'role': 'assistant'})))
            for i, part in enumerate([
                {'type': 'text', 'text': 'hello'},
                {'type': 'reasoning', 'text': 'thinking'},
                {'type': 'tool', 'tool': 'read', 'state': {'status': 'completed', 'input': {'path': 'file'}, 'output': 'result'}},
            ]):
                db.execute('INSERT INTO part VALUES (?, ?, ?, ?, ?)', (str(i), 'm', 'one', 1000 + i, json.dumps(part)))
        default = ADAPTERS['opencode'].convert_thread(path)
        included = ADAPTERS['opencode'].convert_thread(path, with_transcript=True)
        self.assertEqual(default, [{k: v for k, v in event.items() if k != 'transcript'} for event in included])
        contents = [event['transcript'] for event in included if event.get('transcript')]
        self.assertIn({'text': 'hello'}, contents)
        self.assertIn({'text': 'thinking'}, contents)
        self.assertIn({'tool_input': {'path': 'file'}, 'tool_output': 'result'}, contents)

    def test_csv_content_filters_and_removes_previous_transcript(self):
        path = self.jsonl('session.jsonl', [
            {'type': 'session_meta', 'timestamp': '2026-01-01T00:00:00Z', 'payload': {'id': 'one', 'cwd': str(self.root)}},
            {'type': 'response_item', 'timestamp': '2026-01-01T01:00:00Z', 'payload': {'type': 'message', 'role': 'user', 'content': 'excluded'}},
            {'type': 'response_item', 'timestamp': '2026-01-02T01:00:00Z', 'payload': {'type': 'message', 'role': 'assistant', 'content': 'included'}},
        ])
        output = self.root / 'report' / 'threads.csv'
        convert_sessions({'codex': path}, output, with_transcript=True,
                         start_time=datetime(2026, 1, 2, tzinfo=timezone.utc),
                         thread_ids={anonymous_id('codex:one')}, directory=self.root)
        with output.open() as file:
            entries = list(csv.DictReader(file))
        self.assertEqual([e['text'] for e in entries], ['included'])
        self.assertEqual(entries[0]['thread_id'], anonymous_id('codex:one'))
        self.assertIn('tool_input', entries[0])
        self.assertIn('tool_output', entries[0])
        convert_sessions({'codex': path}, output)
        self.assertNotIn('text', output.read_text().splitlines()[0].split(','))
        self.assertNotIn('included', output.read_text())

    def test_csv_without_recorded_content(self):
        path = self.jsonl('empty-content.jsonl', [
            {'type': 'session_meta', 'payload': {'id': 'one'}},
        ])
        output = self.root / 'threads.csv'
        convert_sessions({'codex': path}, output, with_transcript=True)
        with output.open() as file:
            entries = list(csv.DictReader(file))
        self.assertEqual(entries[0]['text'], '')
        self.assertEqual(entries[0]['tool_input'], '')
        self.assertEqual(entries[0]['tool_output'], '')

