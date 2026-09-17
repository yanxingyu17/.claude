#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Tests for adapters/copilot/backfill_copilot.py.
# The .json fixture replicates the real format-v3 session captured from
# VS Code Copilot Chat on 2026-09-17; the .jsonl fixture exercises the
# objectMutationLog replay path (kind 0 initial / 1 set / 2 append).
# Covers:
#   T1 real-shaped .json session in workspace   -> user/tool/assistant
#      events with tool_call_id, model on assistant, ALL OK
#   T2 .jsonl mutation-log session in workspace -> same via replay
#   T3 session in a personal workspace          -> filtered
#   T4 unsupported format version               -> skipped with warning
#   T5 idempotent second run                    -> 0 imported

import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ADAPTER_DIR = Path(__file__).resolve().parent
SKILL_DIR = ADAPTER_DIR.parent.parent
EXPORT_PY = SKILL_DIR / "tools" / "export-session.py"
VALIDATE_PY = SKILL_DIR / "tools" / "validate-log.py"

LOGIN = "copilot-tester"


def make_v3_session(session_id: str) -> dict:
    return {
        "version": 3,
        "sessionId": session_id,
        "creationDate": 1762939226940,
        "customTitle": "test session",
        "requests": [
            {
                "requestId": "request_1",
                "message": {"text": "Fix the buffer overrun"},
                "response": [
                    {"value": "Reading the file first. "},
                    {"kind": "toolInvocationSerialized",
                     "toolCallId": "call_1",
                     "invocationMessage": "Reading src/foo.c",
                     "toolId": "copilot_readFile",
                     "isConfirmed": True, "isComplete": True},
                    {"value": "Fixed the overrun at line 73."},
                ],
                "timestamp": 1762939547595,
                "modelId": "copilot/claude-sonnet-4.5",
            },
        ],
    }


class CopilotBackfillTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="copilot-test-"))
        self.user_data = self.tmp / "vscode-user"
        self.workspace = self.tmp / "openvela-ws"
        (self.workspace / ".repo").mkdir(parents=True)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _make_ws_chat(self, ws_name: str, folder_uri: str):
        ws = self.user_data / "User" / "workspaceStorage" / ws_name
        chat = ws / "chatSessions"
        chat.mkdir(parents=True)
        (ws / "workspace.json").write_text(
            json.dumps({"folder": folder_uri}), encoding="utf-8")
        return chat

    def _run_backfill(self):
        import os
        env = {
            **os.environ,
            "VSCODE_USER_DATA_OVERRIDE": str(self.user_data),
            "TEAM_ID": "contest2026_test_team",
        }
        return subprocess.run(
            [sys.executable, str(EXPORT_PY), "--backfill",
             "--source", "copilot", "--github-login", LOGIN,
             "--dest", str(self.workspace)],
            capture_output=True, text=True, env=env,
        )

    def test_json_session_import(self):
        chat = self._make_ws_chat(
            "hash-ws", f"file://{self.workspace}")
        (chat / "ses-json.json").write_text(
            json.dumps(make_v3_session("ses-json")), encoding="utf-8")
        r = self._run_backfill()
        self.assertIn("1 session(s) imported", r.stdout, r.stdout)

        jsonl = list(
            (self.workspace / "logs" / LOGIN).rglob("copilot__ses-json.jsonl"))
        self.assertEqual(len(jsonl), 1)
        events = [json.loads(l) for l in jsonl[0].read_text().splitlines()]
        self.assertEqual(
            [e["role"] for e in events],
            ["user", "tool", "assistant"])
        self.assertEqual(events[0]["text"], "Fix the buffer overrun")
        self.assertEqual(events[1]["tool_call_id"], "call_1")
        self.assertEqual(events[1]["tool_name"], "copilot_readFile")
        self.assertEqual(events[2]["model"], "copilot/claude-sonnet-4.5")
        self.assertEqual([e["seq"] for e in events], [0, 1, 2])

        v = subprocess.run(
            [sys.executable, str(VALIDATE_PY),
             str(self.workspace / "logs")],
            capture_output=True, text=True)
        self.assertEqual(v.returncode, 0, v.stdout + v.stderr)

    def test_jsonl_mutation_log_replay(self):
        chat = self._make_ws_chat(
            "hash-ws", f"file://{self.workspace}")
        base = make_v3_session("ses-jsonl")
        later_req = dict(base["requests"][0])
        later_req["requestId"] = "request_2"
        later_req["message"] = {"text": "Now fix the second file"}
        later_req["response"] = [{"value": "Done."}]
        later_req["timestamp"] = 1762939600000

        lines = [
            json.dumps({"kind": 0, "v": base}),
            json.dumps({"kind": 1, "k": ["customTitle"], "v": "renamed"}),
            json.dumps({"kind": 2, "k": ["requests"], "v": [later_req]}),
        ]
        (chat / "ses-jsonl.jsonl").write_text(
            "\n".join(lines) + "\n", encoding="utf-8")
        r = self._run_backfill()
        self.assertIn("1 session(s) imported", r.stdout, r.stdout)

        jsonl = list(
            (self.workspace / "logs" / LOGIN).rglob(
                "copilot__ses-jsonl.jsonl"))
        self.assertEqual(len(jsonl), 1)
        events = [json.loads(l) for l in jsonl[0].read_text().splitlines()]
        # replay must have applied the kind-2 append: both requests present;
        # a request's multiple assistant text parts merge into one event
        self.assertEqual(
            [e["text"] for e in events],
            ["Fix the buffer overrun", "Reading src/foo.c",
             "Reading the file first.\nFixed the overrun at line 73.",
             "Now fix the second file", "Done."])

    def test_personal_workspace_filtered(self):
        chat = self._make_ws_chat(
            "hash-personal", "file:///home/someone/personal")
        (chat / "p.json").write_text(
            json.dumps(make_v3_session("personal-ses")),
            encoding="utf-8")
        r = self._run_backfill()
        self.assertIn("0 session(s) imported", r.stdout, r.stdout)
        self.assertFalse(
            list((self.workspace / "logs").rglob("*.jsonl")))

    def test_unsupported_version_skipped(self):
        chat = self._make_ws_chat(
            "hash-ws", f"file://{self.workspace}")
        bad = make_v3_session("ses-v99")
        bad["version"] = 99
        (chat / "v99.json").write_text(
            json.dumps(bad), encoding="utf-8")
        r = self._run_backfill()
        self.assertIn("0 session(s) imported", r.stdout, r.stdout)
        self.assertIn("unsupported format version 99", r.stderr, r.stderr)

    def test_idempotent(self):
        chat = self._make_ws_chat(
            "hash-ws", f"file://{self.workspace}")
        (chat / "dup.json").write_text(
            json.dumps(make_v3_session("dup-ses")), encoding="utf-8")
        r1 = self._run_backfill()
        self.assertIn("1 session(s) imported", r1.stdout, r1.stdout)
        r2 = self._run_backfill()
        self.assertIn("0 session(s) imported", r2.stdout, r2.stdout)


if __name__ == "__main__":
    unittest.main()
