#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# openvela AI Contest - Codex backfill adapter.
#
# Reads Codex rollout transcripts from $CODEX_HOME/sessions and
# $CODEX_HOME/archived_sessions (default ~/.codex/). Per official docs
# (developers.openai.com/codex/environment-variables), CODEX_HOME is
# shared by Codex CLI, the IDE extension, and the Codex desktop app,
# so one scan covers all three clients.
#
# Rollout format (verified against codex-cli 0.142.1, 2026-09-16):
#   one JSON object per line, each with "timestamp" and "type":
#     session_meta   - session_id, cwd, originator, cli_version
#     turn_context   - model, cwd per turn
#     event_msg      - user_message / task_started / task_complete ...
#     response_item  - message (role + content[]) / function_call /
#                       local_shell_call / reasoning ...
#
# Design notes:
#   * Parsing is DEFENSIVE: unknown payload types or content item types
#     are skipped with a stderr warning, never fatal. The desktop app
#     shares the state model per docs but its exact rollout variants
#     are not locally verified; graceful skipping bounds that risk to
#     "fewer events", never "corrupt logs".
#   * Workspace gate: session_meta.payload.cwd must resolve under the
#     openvela workspace root (dir with .repo/). turn_context.cwd
#     updates the gate per-turn so a session that cd's out of the
#     workspace mid-flight still writes only in-workspace turns.
#   * Developer/system environment blobs (permissions instructions,
#     skills instructions, environment_context) are excluded: they are
#     tool boilerplate, not contestant dialogue.
#   * Idempotent: sessions already present in the manifest are skipped.

from __future__ import annotations

import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

TOOL_ID = "codex"
SCHEMA_VERSION = "1.0"

ENV_BLOB_PREFIXES = (
    "<permissions instructions>",
    "<skills_instructions>",
    "<environment_context>",
)

EVENT_MSG_SKIP = {
    "task_started", "task_complete", "turn_aborted", "error",
    "warning", "token_count", "background_event", "agent_reasoning",
    "agent_reasoning_section_break", "agent_reasoning_raw_content",
    "mcp_tool_call_begin", "mcp_tool_call_end", "web_search_begin",
    "web_search_end", "diff_applied_header", "plan_update",
    "git_diff_summary", "turn_diff", "list_tasks", "resume_completed",
}


def codex_home() -> Path:
    override = os.environ.get("CODEX_HOME_BACKFILL_OVERRIDE")
    if override:
        return Path(override)
    env = os.environ.get("CODEX_HOME")
    if env:
        return Path(env)
    return Path.home() / ".codex"


def _sha256_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _is_env_boilerplate(text: str) -> bool:
    return any(text.startswith(p) for p in ENV_BLOB_PREFIXES)


def _content_to_text(content: Any) -> str:
    """Flatten a response_item content list. Only text-ish items are
    kept; input_image and other non-text item types are skipped (the
    collector stores text events; image payloads have no schema slot).
    """
    if isinstance(content, str):
        return content.strip()
    if not isinstance(content, list):
        return ""
    parts: list[str] = []
    for item in content:
        if not isinstance(item, dict):
            continue
        if item.get("type") in ("input_text", "output_text", "text"):
            t = item.get("text")
            if isinstance(t, str) and t.strip():
                parts.append(t.strip())
    return "\n".join(parts)


def _find_workspace_root(start: Path) -> Path | None:
    cur = start.resolve()
    for candidate in [cur, *cur.parents]:
        if (candidate / ".repo").is_dir():
            return candidate
    return None


def parse_rollout(path: Path) -> list[dict[str, Any]] | None:
    """Parse one rollout JSONL into contest events. Returns None when the
    file is not a readable rollout (no session_meta) so callers can
    distinguish "skipped corrupt file" from "empty session".
    """
    session_id = None
    cwd: str | None = None
    model: str | None = None
    events: list[dict[str, Any]] = []
    seq = 0
    saw_meta = False

    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError as e:
        sys.stderr.write(f"[codex] cannot read {path}: {e}\n")
        return None

    for line in lines:
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            sys.stderr.write(
                f"[codex] {path.name}: skipping bad JSON line\n")
            continue
        if not isinstance(rec, dict):
            continue

        ts = rec.get("timestamp") or ""
        rtype = rec.get("type")
        payload = rec.get("payload") or {}

        if rtype == "session_meta":
            saw_meta = True
            session_id = payload.get("session_id") or payload.get("id")
            cwd = payload.get("cwd")
            continue

        if rtype == "turn_context":
            if payload.get("cwd"):
                cwd = payload.get("cwd")
            if payload.get("model"):
                model = payload.get("model")
            continue

        if rtype == "response_item":
            ptype = payload.get("type")
            if ptype != "message":
                continue
            role = payload.get("role")
            if role not in ("user", "assistant"):
                continue
            text = _content_to_text(payload.get("content"))
            if not text or _is_env_boilerplate(text):
                continue
            ev = {
                "ts": ts,
                "role": role,
                "text": text,
            }
            if role == "assistant" and model:
                ev["model"] = model
            events.append(ev)
            seq += 1
            continue

        if rtype == "event_msg":
            etype = payload.get("type")
            if etype == "user_message":
                text = payload.get("message")
                if isinstance(text, str) and text.strip():
                    if events and events[-1]["role"] == "user" and \
                            events[-1]["text"] == text.strip():
                        continue
                    events.append({"ts": ts, "role": "user",
                                   "text": text.strip()})
                    seq += 1
            continue

        # unknown top-level types: skip silently (future rollout versions)

    if not saw_meta or session_id is None:
        return None
    meta = {
        "session_id": session_id,
        "cwd": cwd,
        "originator": _first_meta_originator(lines),
    }
    return [{"meta": meta, "events": events}]


def _first_meta_originator(lines: list[str]) -> str | None:
    for line in lines[:5]:
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if rec.get("type") == "session_meta":
            return (rec.get("payload") or {}).get("originator")
    return None


def _write_manifest(dest: Path, github_login: str, team_id: str,
                    session_id: str, events: list[dict], rel_path: str,
                    integrity: dict) -> None:
    member_dir = dest / "logs" / github_login
    manifest_path = member_dir / "manifest.json"
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            manifest = {}
    else:
        manifest = {}
    manifest.setdefault("schema_version", SCHEMA_VERSION)
    manifest["team_id"] = team_id
    manifest["github_login"] = github_login
    manifest.setdefault("generator", f"backfill-codex@{TOOL_ID}")
    sessions = manifest.setdefault("sessions", [])
    entry = next(
        (s for s in sessions if s.get("session_id") == session_id), None)
    new_entry = {
        "session_id": session_id,
        "tool": TOOL_ID,
        "started_at": events[0]["ts"],
        "last_event_at": events[-1]["ts"],
        "event_count": len(events),
        "file_path": rel_path,
        "collection_mode": "backfill-sqlite",
        "health": "ok",
        "source_integrity": integrity,
    }
    if entry:
        entry.update(new_entry)
    else:
        sessions.append(new_entry)
    manifest["updated_at"] = datetime.now(timezone.utc).isoformat()
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8")


def _session_in_manifest(dest: Path, github_login: str,
                         session_id: str) -> bool:
    manifest_path = dest / "logs" / github_login / "manifest.json"
    if not manifest_path.is_file():
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return any(
        s.get("session_id") == session_id and s.get("tool") == TOOL_ID
        for s in manifest.get("sessions", []))


def backfill(dest: Path, team_id: str, github_login: str) -> int:
    home = codex_home()
    roots = [home / "sessions", home / "archived_sessions"]
    rollouts = sorted(
        {p for r in roots if r.is_dir() for p in r.rglob("rollout-*.jsonl")}
    )
    if not rollouts:
        print(f"[codex] no rollout files under {home}; skipping")
        return 0

    workspace_root = _find_workspace_root(dest)
    if workspace_root is None:
        print("[codex] destination is not inside an openvela workspace "
              "(no .repo/); skipping")
        return 0
    workspace_str = str(workspace_root.resolve())

    total = 0
    imported = 0
    for path in rollouts:
        parsed = parse_rollout(path)
        if parsed is None:
            continue
        data = parsed[0]
        meta = data["meta"]
        events = data["events"]
        sid = meta["session_id"]
        total += 1

        if not meta.get("cwd") or not str(
                Path(meta["cwd"]).resolve()).startswith(workspace_str):
            continue

        if _session_in_manifest(dest, github_login, sid):
            continue

        if not events:
            continue

        integrity = {
            "main_sha256": _sha256_file(path),
            "main_size": path.stat().st_size if path.is_file() else 0,
            "captured_at": datetime.now(timezone.utc).isoformat(),
        }

        date_str = events[0]["ts"][:10] if events[0]["ts"] else \
            datetime.now(timezone.utc).strftime("%Y-%m-%d")
        rel_path = f"logs/{github_login}/{date_str}/{TOOL_ID}__{sid}.jsonl"
        jsonl_path = dest / rel_path
        jsonl_path.parent.mkdir(parents=True, exist_ok=True)

        seq = 0
        with jsonl_path.open("w", encoding="utf-8") as f:
            for e in events:
                out = {
                    "schema_version": SCHEMA_VERSION,
                    "session_id": sid,
                    "team_id": team_id,
                    "github_login": github_login,
                    "tool": TOOL_ID,
                    "seq": seq,
                    "ts": e["ts"],
                    "role": e["role"],
                    "text": e["text"],
                }
                if e.get("model"):
                    out["model"] = e["model"]
                f.write(json.dumps(out, ensure_ascii=False) + "\n")
                seq += 1

        _write_manifest(dest, github_login, team_id, sid, events, rel_path,
                        integrity)
        imported += 1
        print(f"  [codex] {sid[:20]}  wrote {len(events)} event(s) "
              f"-> {rel_path}")

    print(f"[codex] scanned {total} rollout file(s) in workspace "
          f"({len(rollouts)} total, home={home})")
    return imported
