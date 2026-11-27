"""Worker-log route: rebuild the transcript from the profile's ``state.db``.

The per-task worker log is a stub (``Session:`` / ``[kanban-worker-exit]`` + a truncated
final reply); the worker's real frames live in the profile's ``state.db``. These tests
drive ``GET /api/plugins/kanban/tasks/{id}/log`` against a seeded scratch board + a
scratch profile home and pin both directions of the switch:

* stub log  -> transcript rebuilt from ``messages``, ``source == "state_db"``
  (a RUNNING card included: its frames are already in ``state.db`` mid-run)
* anything else (real log, no session id, no db, empty session,
  schema surprise) -> the file content, byte for byte, ``source == "log_file"``
"""

from __future__ import annotations

import importlib.util
import json
import sqlite3
import sys
import time
from pathlib import Path
from typing import Sequence

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc

SESSION = "local-0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
OLD_SESSION = "local-" + "a" * 64
NEW_SESSION = "local-" + "b" * 64
JUNK_SESSION = "20260101_000000_abcdef"  # date-stamped gateway id, not a worker run
T0 = 1_700_000_000.0  # fixed seed clock: labels are computed, never hardcoded

_MESSAGES_SCHEMA = """
CREATE TABLE messages (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id  TEXT NOT NULL,
    role        TEXT NOT NULL,
    content     TEXT,
    tool_call_id TEXT,
    tool_calls  TEXT,
    tool_name   TEXT,
    timestamp   REAL NOT NULL
);
"""


# ---------------------------------------------------------------------------
# Plugin / client fixtures (same load path the dashboard uses: file path, no pkg)
# ---------------------------------------------------------------------------

def _load_plugin_module():
    repo_root = Path(__file__).resolve().parents[2]
    plugin_file = repo_root / "plugins" / "kanban" / "dashboard" / "plugin_api.py"
    assert plugin_file.exists(), f"plugin file missing: {plugin_file}"
    spec = importlib.util.spec_from_file_location(
        "hermes_dashboard_plugin_kanban_log_transcript_test", plugin_file,
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def plugin():
    return _load_plugin_module()


@pytest.fixture
def home(tmp_path, monkeypatch):
    """Isolated HERMES_HOME with an empty board and a ``workerlog`` profile dir."""
    root = tmp_path / ".hermes"
    root.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(root))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    (root / "profiles" / "workerlog").mkdir(parents=True)
    (root / "profiles" / "workerlog" / "config.yaml").write_text("name: workerlog\n", encoding="utf-8")
    kb.init_db()
    return root


@pytest.fixture
def client(home, plugin):
    app = FastAPI()
    app.include_router(plugin.router, prefix="/api/plugins/kanban")
    return TestClient(app)


# ---------------------------------------------------------------------------
# Seed helpers
# ---------------------------------------------------------------------------

def _stub_log(session: str = SESSION, reply: str = "Shipped the fix; tests green.") -> str:
    """Exactly the shape the supervisor leaves on disk today."""
    return f"Session: {session}\n\n[kanban-worker-exit] rc=0\n{reply}\n"


def _write_log(task_id: str, content: str) -> Path:
    path = kb.worker_log_path(task_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def _seed_state_db(
    home: Path,
    assignee: str,
    session_id: str,
    rows: Sequence[tuple[str, str | None, str | None, str | None]] = (),
    *,
    schema: str | None = None,
) -> Path:
    """Profile ``state.db`` with ``(role, content, tool_calls, tool_name)`` rows."""
    db = home / "profiles" / assignee / "state.db"
    db.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db)
    try:
        conn.executescript(schema or _MESSAGES_SCHEMA)
        for index, (role, content, tool_calls, tool_name) in enumerate(rows):
            conn.execute(
                "INSERT INTO messages (session_id, role, content, tool_calls, tool_name, timestamp) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (session_id, role, content, tool_calls, tool_name, T0 + index),
            )
        conn.commit()
    finally:
        conn.close()
    return db


def _new_task(client, assignee: str = "workerlog", title: str = "log route probe") -> str:
    response = client.post(
        "/api/plugins/kanban/tasks", json={"title": title, "assignee": assignee},
    )
    assert response.status_code == 200, response.text
    return response.json()["task"]["id"]


def _set_status(task_id: str, status: str) -> None:
    conn = kbc.connect()
    try:
        conn.execute("UPDATE tasks SET status = ? WHERE id = ?", (status, task_id))
        conn.commit()
    finally:
        conn.close()


def _add_owner_receipt(task_id: str, db_path: Path, session_id: str) -> None:
    conn = kbc.connect()
    try:
        conn.execute(
            "INSERT INTO task_events (task_id, run_id, kind, payload, created_at) "
            "VALUES (?, NULL, 'owner_admitted', ?, ?)",
            (task_id, json.dumps({"db": str(db_path), "session_id": session_id,
                                  "request_id": "kanban:test"}), int(T0)),
        )
        conn.commit()
    finally:
        conn.close()


def _transcript_rows() -> list[tuple[str, str | None, str | None, str | None]]:
    tool_call = json.dumps([{
        "call_id": "call_1",
        "function": {"name": "terminal", "arguments": json.dumps({"command": "ls plugins/kanban"})},
    }])
    return [
        ("user", "Fix the worker log route", None, None),
        ("assistant", "", tool_call, None),
        ("tool", json.dumps({"output": "plugin_api.py\nlog_transcript.py"}), None, "terminal"),
        ("assistant", "Done — shipped it.", None, None),
    ]


def _clock(index: int) -> str:
    return time.strftime("%H:%M:%S", time.localtime(T0 + index))


# ---------------------------------------------------------------------------
# Stub -> transcript
# ---------------------------------------------------------------------------

def test_stub_log_rebuilds_transcript_from_state_db(client, home):
    task_id = _new_task(client)
    _seed_state_db(home, "workerlog", SESSION, _transcript_rows())
    log_path = _write_log(task_id, _stub_log())
    stub_size = log_path.stat().st_size

    response = client.get(f"/api/plugins/kanban/tasks/{task_id}/log")
    assert response.status_code == 200, response.text
    body = response.json()

    assert body["source"] == "state_db"
    assert body["exists"] is True
    assert body["size_bytes"] == stub_size  # still the on-disk file's size
    assert body["truncated"] is False

    text = body["content"]
    # run-boundary block kept verbatim, so nothing the file held is lost
    assert f"Session: {SESSION}" in text
    assert "[kanban-worker-exit] rc=0" in text
    assert "Shipped the fix; tests green." in text
    assert f"--- transcript (state.db: {SESSION}) ---" in text
    # frames: user + assistant text
    assert f"[{_clock(0)}] user: Fix the worker log route" in text
    assert f"[{_clock(3)}] assistant: Done — shipped it." in text
    # tool call + tool result one-liners name the tool and preview the payload
    assert f'[{_clock(1)}] assistant -> terminal({{"command": "ls plugins/kanban"}})' in text
    assert f"[{_clock(2)}] tool terminal -> " in text
    assert "plugin_api.py" in text
    assert "log_transcript.py" in text


def test_system_rows_and_blank_content_are_skipped(client, home, plugin):
    task_id = _new_task(client)
    _seed_state_db(home, "workerlog", SESSION, [
        ("system", "You are a helpful background worker.", None, None),
        ("assistant", "", None, None),
        ("user", "hello", None, None),
    ])
    _write_log(task_id, _stub_log())

    text = client.get(f"/api/plugins/kanban/tasks/{task_id}/log").json()["content"]
    assert "You are a helpful background agent" not in text
    assert "helpful background worker" not in text
    assert f"[{_clock(2)}] user: hello" in text


def test_tool_preview_truncates_long_values(plugin):
    long_value = "x" * 5_000
    rendered = plugin.log_transcript.render_transcript(
        [("tool", long_value, None, "terminal", T0)]
    )
    line = rendered.splitlines()[0]
    assert line.endswith("…")
    assert len(line) <= len("[00:00:00] tool terminal -> ") + plugin.log_transcript.TOOL_PREVIEW_CHARS


# ---------------------------------------------------------------------------
# Degrade paths: the file is served exactly as before
# ---------------------------------------------------------------------------

def test_real_transcript_file_is_served_unchanged(client, home):
    task_id = _new_task(client)
    _seed_state_db(home, "workerlog", SESSION, _transcript_rows())
    real_log = f"Session: {SESSION}\n" + "".join(
        f"[12:00:{index % 60:02d}] worker frame {index}: tool output line\n" for index in range(400)
    )
    _write_log(task_id, real_log)

    body = client.get(f"/api/plugins/kanban/tasks/{task_id}/log").json()
    assert body["source"] == "log_file"
    assert body["content"] == real_log
    assert body["truncated"] is False


def test_missing_state_db_serves_the_file(client):
    task_id = _new_task(client)
    stub = _stub_log()
    _write_log(task_id, stub)

    body = client.get(f"/api/plugins/kanban/tasks/{task_id}/log").json()
    assert body["source"] == "log_file"
    assert body["content"] == stub


def test_empty_session_serves_the_file(client, home):
    task_id = _new_task(client)
    _seed_state_db(home, "workerlog", "some-other-session", _transcript_rows())
    stub = _stub_log()
    _write_log(task_id, stub)

    body = client.get(f"/api/plugins/kanban/tasks/{task_id}/log").json()
    assert body["source"] == "log_file"
    assert body["content"] == stub


def test_running_card_rebuilds_transcript_from_state_db(client, home):
    """A RUNNING card renders its transcript too: the frames exist mid-run.

    Regression: an early return on ``status == "running"`` (premised on the worker
    still writing its own log — false since the 2026-11-01 owner-admission spawn
    change) sent the case a reader opens most back to the 80-byte stub.
    """
    task_id = _new_task(client)
    _seed_state_db(home, "workerlog", SESSION, _transcript_rows())
    stub = _stub_log()
    log_path = _write_log(task_id, stub)
    _set_status(task_id, "running")

    body = client.get(f"/api/plugins/kanban/tasks/{task_id}/log").json()
    assert body["source"] == "state_db"
    assert body["size_bytes"] == log_path.stat().st_size  # still the file's size
    assert f"Session: {SESSION}" in body["content"]
    assert f"--- transcript (state.db: {SESSION}) ---" in body["content"]
    assert f"[{_clock(0)}] user: Fix the worker log route" in body["content"]
    assert f"[{_clock(3)}] assistant: Done — shipped it." in body["content"]


def test_running_card_with_empty_session_serves_the_file(client, home):
    """A running card whose session holds no frames yet falls back to the file."""
    task_id = _new_task(client)
    _seed_state_db(home, "workerlog", "some-other-session", _transcript_rows())
    stub = _stub_log()
    _write_log(task_id, stub)
    _set_status(task_id, "running")

    body = client.get(f"/api/plugins/kanban/tasks/{task_id}/log").json()
    assert body["source"] == "log_file"
    assert body["content"] == stub


def test_done_card_rebuilds_transcript_from_state_db(client, home):
    """The finished-card behaviour is unchanged by dropping the running guard."""
    task_id = _new_task(client)
    _seed_state_db(home, "workerlog", SESSION, _transcript_rows())
    _write_log(task_id, _stub_log())
    _set_status(task_id, "done")

    body = client.get(f"/api/plugins/kanban/tasks/{task_id}/log").json()
    assert body["source"] == "state_db"
    assert f"--- transcript (state.db: {SESSION}) ---" in body["content"]
    assert f"[{_clock(0)}] user: Fix the worker log route" in body["content"]


@pytest.mark.parametrize("schema", [
    # no messages table at all
    "CREATE TABLE sessions (id TEXT PRIMARY KEY, started_at REAL);",
    # messages with none of the columns the transcript needs
    "CREATE TABLE messages (id INTEGER PRIMARY KEY, session_id TEXT, role TEXT);",
])
def test_schema_surprise_serves_the_file(client, home, schema):
    task_id = _new_task(client)
    _seed_state_db(home, "workerlog", SESSION, schema=schema)
    stub = _stub_log()
    _write_log(task_id, stub)

    body = client.get(f"/api/plugins/kanban/tasks/{task_id}/log").json()
    assert body["source"] == "log_file"
    assert body["content"] == stub


def test_non_sqlite_state_db_serves_the_file(client, home):
    task_id = _new_task(client)
    garbage = home / "profiles" / "workerlog" / "state.db"
    garbage.write_bytes(b"not a sqlite database at all")
    stub = _stub_log()
    _write_log(task_id, stub)

    body = client.get(f"/api/plugins/kanban/tasks/{task_id}/log").json()
    assert body["source"] == "log_file"
    assert body["content"] == stub


def test_no_session_id_serves_the_file(client, home):
    task_id = _new_task(client)
    _seed_state_db(home, "workerlog", SESSION, _transcript_rows())
    stub = "[kanban-worker-exit] rc=1\nboom\n"
    _write_log(task_id, stub)

    body = client.get(f"/api/plugins/kanban/tasks/{task_id}/log").json()
    assert body["source"] == "log_file"
    assert body["content"] == stub


def test_multi_run_log_falls_back_to_the_newest_session_with_frames(client, home):
    """A log appends one ``Session:`` line per run; the newest may hold no frames."""
    task_id = _new_task(client)
    old_session, new_session = OLD_SESSION, NEW_SESSION
    _seed_state_db(home, "workerlog", old_session, _transcript_rows())
    _write_log(task_id, f"Session: {old_session}\n\n[kanban-worker-exit] rc=0\nold\n"
                        f"\nSession: {new_session}\n")

    body = client.get(f"/api/plugins/kanban/tasks/{task_id}/log").json()
    assert body["source"] == "state_db"
    assert f"--- transcript (state.db: {old_session}) ---" in body["content"]
    assert f"[{_clock(0)}] user: Fix the worker log route" in body["content"]


def test_owner_receipt_recovers_a_log_without_session_line(client, home):
    """A rotated/truncated log with no ``Session:`` line still resolves its run."""
    task_id = _new_task(client)
    db_path = _seed_state_db(home, "workerlog", SESSION, _transcript_rows())
    _add_owner_receipt(task_id, db_path, SESSION)
    _write_log(task_id, "[kanban-worker-exit] rc=0\nfinal reply only\n")

    body = client.get(f"/api/plugins/kanban/tasks/{task_id}/log").json()
    assert body["source"] == "state_db"
    assert f"[{_clock(0)}] user: Fix the worker log route" in body["content"]


def test_tail_on_a_real_log_matches_the_file_reader(client):
    """A small real log is read whole for detection, so its tail must be unchanged."""
    task_id = _new_task(client)
    real_log = f"Session: {SESSION}\n" + "".join(
        f"worker frame {index}: tool output line\n" for index in range(500)
    )
    _write_log(task_id, real_log)

    body = client.get(f"/api/plugins/kanban/tasks/{task_id}/log", params={"tail": 300}).json()
    assert body["source"] == "log_file"
    assert body["truncated"] is True
    assert body["content"] == kb.read_worker_log(task_id, tail_bytes=300)


def test_big_real_log_small_tail_is_never_rebuilt(client, home, plugin):
    """F1: a >32 KiB real log is judged from the whole file, not from the tail window."""
    task_id = _new_task(client)
    _seed_state_db(home, "workerlog", SESSION, _transcript_rows())
    real_log = f"Session: {SESSION}\n" + "".join(
        f"[12:00:{index % 60:02d}] worker frame {index}: tool output line\n" for index in range(2000)
    )
    _write_log(task_id, real_log)
    assert len(real_log) > plugin.log_transcript.STUB_READ_MAX_BYTES

    body = client.get(f"/api/plugins/kanban/tasks/{task_id}/log", params={"tail": 4000}).json()
    assert body["source"] == "log_file"
    assert body["truncated"] is True
    assert body["content"] == kb.read_worker_log(task_id, tail_bytes=4000)


def test_stub_with_a_quoted_session_line_in_the_reply_uses_the_worker_run(client, home):
    """F2: only ``local-<hex>`` worker ids count; a reply quoting a Session line must not win."""
    task_id = _new_task(client)
    _seed_state_db(home, "workerlog", SESSION, _transcript_rows())
    _write_log(task_id, f"Session: {SESSION}\n\n[kanban-worker-exit] rc=0\n"
                        f"see Session: {JUNK_SESSION} for details\n")

    body = client.get(f"/api/plugins/kanban/tasks/{task_id}/log").json()
    assert body["source"] == "state_db"
    assert f"--- transcript (state.db: {SESSION}) ---" in body["content"]


def test_multi_run_stub_still_rebuilds_past_the_old_fragment_limit(client, home):
    """F5: measured max stub fragment is ~2.5 KB; a 5 KB retry stub must still rebuild."""
    task_id = _new_task(client)
    _seed_state_db(home, "workerlog", SESSION, _transcript_rows())
    _write_log(task_id, f"Session: {SESSION}\n\n[kanban-worker-exit] rc=0\n" + "x" * 5000 + "\n")

    body = client.get(f"/api/plugins/kanban/tasks/{task_id}/log").json()
    assert body["source"] == "state_db"
    assert f"[{_clock(0)}] user: Fix the worker log route" in body["content"]


def test_broken_helper_degrades_to_the_file_instead_of_500(client, monkeypatch, plugin):
    """F3: one guarded entry point — a helper that drifts after load cannot 500 the route."""
    task_id = _new_task(client)
    stub = _stub_log()
    _write_log(task_id, stub)

    def boom(*_args, **_kwargs):
        raise RuntimeError("helper version skew")

    monkeypatch.setattr(plugin.log_transcript, "read_tail_for", boom)
    response = client.get(f"/api/plugins/kanban/tasks/{task_id}/log")
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["source"] == "log_file"
    assert body["content"] == stub


def test_missing_log_file_reports_absent(client):
    task_id = _new_task(client)
    body = client.get(f"/api/plugins/kanban/tasks/{task_id}/log").json()
    assert body["exists"] is False
    assert body["source"] == "log_file"
    assert body["content"] == ""


def test_unknown_task_is_404(client):
    response = client.get("/api/plugins/kanban/tasks/t_nope/log")
    assert response.status_code == 404


# ---------------------------------------------------------------------------
# tail: byte cap on the RETURNED text, tail end, truncated flag
# ---------------------------------------------------------------------------

def test_tail_trims_the_rebuilt_transcript_from_the_end(client, home):
    task_id = _new_task(client)
    _seed_state_db(home, "workerlog", SESSION, _transcript_rows())
    _write_log(task_id, _stub_log())

    full = client.get(f"/api/plugins/kanban/tasks/{task_id}/log").json()["content"]
    body = client.get(f"/api/plugins/kanban/tasks/{task_id}/log", params={"tail": 120}).json()

    assert body["source"] == "state_db"
    assert body["truncated"] is True
    assert len(body["content"].encode("utf-8")) <= 120
    assert body["content"] in full
    assert full.endswith(body["content"])  # the tail end, not the head


def test_tail_larger_than_the_transcript_is_not_truncated(client, home):
    task_id = _new_task(client)
    _seed_state_db(home, "workerlog", SESSION, _transcript_rows())
    _write_log(task_id, _stub_log())

    body = client.get(f"/api/plugins/kanban/tasks/{task_id}/log", params={"tail": 1_000_000}).json()
    assert body["source"] == "state_db"
    assert body["truncated"] is False


# ---------------------------------------------------------------------------
# Stub detection itself
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("content", [
    "",
    "   \n\n",
    "Session: local-abc\n",
    "Session: local-abc\nSession: local-def\n\n[kanban-worker-exit] rc=0\nshort reply\n",
    "[kanban-worker-exit] rc=137\n",
    "Operation interrupted: waiting for model response\n",
])
def test_stub_detection_true(plugin, content):
    assert plugin.log_transcript.is_stub_log(content) is True


@pytest.mark.parametrize("content", [
    "Query: work kanban task t_1\nInitializing agent...\n",
    "Session: local-abc\n" + "worker frame line\n" * 1200,  # > STUB_FRAGMENT_MAX_CHARS
])
def test_stub_detection_false(plugin, content):
    assert plugin.log_transcript.is_stub_log(content) is False


def test_last_session_line_wins(plugin):
    log = (
        f"Session: {OLD_SESSION}\n\n[kanban-worker-exit] rc=0\nx\n"
        f"\nSession: {NEW_SESSION}\n\n[kanban-worker-exit] rc=0\ny\n"
    )
    assert plugin.log_transcript.session_id_from_log(log) == NEW_SESSION


def test_non_worker_session_lines_are_ignored(plugin):
    """Reply text and legacy gateway ids carry ``Session:`` lines too — never a worker run."""
    log = f"Session: {OLD_SESSION}\n\n[kanban-worker-exit] rc=0\nquoted: Session: {JUNK_SESSION}\n"
    assert plugin.log_transcript.session_ids_from_log(log) == [OLD_SESSION]
    assert plugin.log_transcript.session_id_from_log(f"Session: {JUNK_SESSION}\n") is None


def test_session_candidates_put_the_receipt_first_and_cap_the_list(plugin):
    log = "".join(f"Session: local-{index:064x}\n" for index in range(12))
    receipt_id = "local-" + "c" * 64
    ids = plugin.log_transcript.session_candidates(log, {"session_id": receipt_id})
    assert ids[0] == receipt_id
    assert len(ids) == plugin.log_transcript.MAX_SESSION_CANDIDATES == 8
    assert all(plugin.log_transcript._WORKER_SESSION_RE.fullmatch(sid) for sid in ids)
