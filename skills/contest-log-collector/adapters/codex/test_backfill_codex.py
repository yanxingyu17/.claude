#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Tests for adapters/codex/backfill_codex.py.
# Fixtures replicate the real rollout format captured from codex-cli
# 0.142.1 on 2026-09-16 (session_meta / turn_context / event_msg /
# response_item line shapes). Covers:
#   T1 in-workspace rollout with user+assistant events -> imported,
#      boilerplate blobs excluded, model recorded on assistant
#   T2 rollout with cwd outside the workspace            -> filtered
#   T3 corrupt file (no session_meta)                    -> skipped
#   T4 already-imported session                          -> idempotent
#   T5 end-to-end validate-log.py                        -> ALL OK

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

LOGIN = "codex-tester"


def rollout_line(rtype: str, payload: dict, ts: str) -> str:
    return json.dumps(
        {"timestamp": ts, "type": rtype, "payload": payload},
        ensure_ascii=False)


def make_rollout(session_id: str, cwd: str, with_assistant=True,
                 extra_lines=None) -> list[str]:
    lines = [
        rollout_line("session_meta", {
            "session_id": session_id, "id": session_id,
            "cwd": cwd, "originator": "codex_exec",
            "cli_version": "0.142.1",
        }, "2026-09-16T09:00:00.000Z"),
        rollout_line("event_msg", {
            "type": "task_started", "turn_id": "t1",
        }, "2026-09-16T09:00:00.100Z"),
        rollout_line("response_item", {
            "type": "message", "role": "developer",
            "content": [{"type": "input_text", "text":
                         "<permissions instructions>read-only sandbox"}],
        }, "2026-09-16T09:00:00.200Z"),
        rollout_line("response_item", {
            "type": "message", "role": "user",
            "content": [{"type": "input_text",
                         "text": "Fix the build error"}],
        }, "2026-09-16T09:00:01.000Z"),
        rollout_line("event_msg", {
            "type": "user_message", "message": "Fix the build error",
        }, "2026-09-16T09:00:01.100Z"),
        rollout_line("turn_context", {
            "turn_id": "t1", "cwd": cwd, "model": "gpt-5.2",
        }, "2026-09-16T09:00:01.200Z"),
    ]
    if with_assistant:
        lines.append(rollout_line("response_item", {
            "type": "message", "role": "assistant",
            "content": [{"type": "output_text", "text":
                         "The build failed because of a missing header."}],
        }, "2026-09-16T09:00:05.000Z"))
    if extra_lines:
        lines.extend(extra_lines)
    return lines


class CodexBackfillTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="codex-test-"))
        self.codex_home = self.tmp / "codex-home"
        self.workspace = self.tmp / "openvela-ws"
        (self.workspace / ".repo").mkdir(parents=True)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _write_rollout(self, relpath: str, lines: list[str]):
        p = self.codex_home / "sessions" / relpath
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def _run_backfill(self):
        import os
        env = {
            **os.environ,
            "CODEX_HOME_BACKFILL_OVERRIDE": str(self.codex_home),
            "TEAM_ID": "contest2026_test_team",
        }
        return subprocess.run(
            [sys.executable, str(EXPORT_PY), "--backfill",
             "--source", "codex", "--github-login", LOGIN,
             "--dest", str(self.workspace)],
            capture_output=True, text=True, env=env,
        )

    def test_import_and_boilerplate_excluded(self):
        self._write_rollout(
            "2026/09/16/rollout-2026-09-16T09-00-00-ses1.jsonl",
            make_rollout("ses1", str(self.workspace)))
        r = self._run_backfill()
        self.assertIn("1 session(s) imported", r.stdout, r.stdout)

        jsonl = list(
            (self.workspace / "logs" / LOGIN).rglob("codex__ses1.jsonl"))
        self.assertEqual(len(jsonl), 1)
        events = [json.loads(l) for l in jsonl[0].read_text().splitlines()]
        self.assertEqual(len(events), 2)
        roles = [e["role"] for e in events]
        self.assertEqual(roles, ["user", "assistant"])
        for e in events:
            self.assertNotIn("permissions instructions", e["text"])
            self.assertNotIn("environment_context", e["text"])
        self.assertEqual(events[-1]["model"], "gpt-5.2")
        self.assertEqual([e["seq"] for e in events], [0, 1])

    def test_outside_workspace_filtered(self):
        self._write_rollout(
            "2026/09/16/rollout-x.jsonl",
            make_rollout("outside-ses", "/home/someone/personal"))
        r = self._run_backfill()
        self.assertIn("0 session(s) imported", r.stdout, r.stdout)
        self.assertFalse(
            list((self.workspace / "logs").rglob("*.jsonl")))

    def test_corrupt_rollout_skipped(self):
        p = self.codex_home / "sessions" / "2026" / "09" / "16" / \
            "rollout-corrupt.jsonl"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text('{"type": "event_msg", "payload": {}}\n',
                     encoding="utf-8")
        r = self._run_backfill()
        self.assertIn("0 session(s) imported", r.stdout, r.stdout)

    def test_idempotent(self):
        self._write_rollout(
            "2026/09/16/rollout-a.jsonl",
            make_rollout("dup-ses", str(self.workspace)))
        r1 = self._run_backfill()
        self.assertIn("1 session(s) imported", r1.stdout, r1.stdout)
        r2 = self._run_backfill()
        self.assertIn("0 session(s) imported", r2.stdout, r2.stdout)
        jsonl = list(
            (self.workspace / "logs" / LOGIN).rglob("codex__dup-ses.jsonl"))
        self.assertEqual(len(jsonl), 1)

    def test_validate_all_ok(self):
        self._write_rollout(
            "2026/09/16/rollout-v.jsonl",
            make_rollout("val-ses", str(self.workspace)))
        self._run_backfill()
        r = subprocess.run(
            [sys.executable, str(VALIDATE_PY),
             str(self.workspace / "logs")],
            capture_output=True, text=True,
        )
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertIn("ALL OK", r.stdout)


if __name__ == "__main__":
    unittest.main()
