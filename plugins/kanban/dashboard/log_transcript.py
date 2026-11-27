"""Rebuild a kanban worker transcript from the profile's ``state.db``.

Since 2026-11-01 the per-task worker log only carries the supervisor's run-boundary
lines (``Session: <id>``, ``[kanban-worker-exit] rc=N``, ``Operation interrupted: …``)
plus the run's final reply text — the worker's real frames never reach the file. They
do live in the profile's ``state.db`` (``messages``, keyed by the worker session id),
so the ``GET /tasks/{id}/log`` route falls back here when the file has no transcript of
its own. The drawer keeps rendering whatever this returns; nothing else changes.

Public surface — :func:`log_payload` is the route's single entry point and never
raises, never writes:

  * whole-file reads only: a log is classified only when the caller got the WHOLE file
    (``size_bytes <= STUB_READ_MAX_BYTES``, or a ``tail`` covering it). A 4 KiB tail
    window of a big real log is never mistaken for a stub;
  * :func:`reuse_stub_transcript` returns ``None`` when the file carries a real
    transcript, else ``{"content", "truncated"}`` for the rebuilt document;
  * a missing session id, a missing/unreadable ``state.db``, an empty session, a
    schema surprise or a post-load helper mismatch all degrade to today's file
    content. A still-running card rebuilds exactly like a finished one: its frames
    are already in ``state.db`` mid-run and the read is read-only, so the operator
    can watch a card while it works.

Layout of a rebuilt document: the original file block (boundary lines + reply text) is
kept verbatim as a header (whole-file reads only), then a transcript section rendered
one line per frame::

    Session: local-…
    [kanban-worker-exit] rc=0
    <final reply text>

    --- transcript (state.db: local-…) ---
    [12:00:00] user: …
    [12:00:01] assistant: …
    [12:00:02] assistant -> terminal({"command": "ls -la"})
    [12:00:03] tool terminal -> {"output": "total 0 …"}

The header names the session the frames came from, so a multi-run log says which
attempt the reader is looking at. ``system`` rows are skipped (they are the prompt, not
the run) and long tool payloads are collapsed to a one-line preview; user/assistant text
is kept whole because that is the transcript the operator is missing. The frames are not
redacted further: they come from the same session store the rest of the dashboard
already reads.
"""

from __future__ import annotations

import json
import logging
import re
import sqlite3
import time
from pathlib import Path
from typing import Any, Optional, Sequence

log = logging.getLogger(__name__)

#: Run-boundary lines the supervisor writes around a worker run.
_BOUNDARY_PREFIXES = ("Session:", "[kanban-worker-exit]", "Operation interrupted:")

#: Worker session ids are ``local-`` + a long lowercase hex digest. Anchoring matters:
#: reply text routinely quotes a ``Session:`` line, and old date-stamped gateway ids
#: (``20260101_000000_abcdef``) appear in real logs too — neither is a worker session.
_WORKER_SESSION_RE = re.compile(r"local-[0-9a-f]{32,}")

#: Characters a reply fragment may add before the file stops being a "stub". Measured
#: over 1,084 post-2026-11-01 stubs: median 612 B, largest file 3,075 B, largest
#: non-boundary fragment 2,484 chars (a two-run retry stub). 16 KiB is ~6.6x that max:
#: false positives are non-destructive (the file block is kept and the transcript
#: appended), false negatives silently show the stub the operator cannot read.
STUB_FRAGMENT_MAX_CHARS = 16 * 1024

#: A stub is a few KB at most, so the route may read it whole to find its ``Session:``
#: line even under a ``tail`` window. Anything bigger is a real transcript (cheap tail
#: read) and is never classified — a small tail of a 900 KB log must not look like a stub.
STUB_READ_MAX_BYTES = 32 * 1024

#: Characters kept from a tool argument/result payload in a one-line frame.
TOOL_PREVIEW_CHARS = 200

#: Sessions tried per request. A real log's tail window can carry hundreds of nested
#: ``Session:`` lines and each miss costs a ``sqlite3`` open.
MAX_SESSION_CANDIDATES = 8

#: Header of the reconstructed section, naming the attempt the frames came from.
TRANSCRIPT_HEADER = "--- transcript (state.db: {session_id}) ---"

#: ``messages`` columns rendered, oldest first. ``id`` is the rowid: insertion order.
_MESSAGES_SQL = (
    "SELECT role, content, tool_calls, tool_name, timestamp "
    "FROM messages WHERE session_id = ? ORDER BY id"
)

#: Read-only URI parameters: this module never writes a profile database.
_RO_PARAMS = "?mode=ro"


def _is_boundary(line: str) -> bool:
    return line.startswith(_BOUNDARY_PREFIXES)


def is_stub_log(content: Optional[str]) -> bool:
    """True when *content* holds no worker transcript of its own.

    A real transcript starts its frames right after the log opens; a stub is nothing
    but boundary lines plus (at most) one run's reply text. Anything with non-boundary
    text beyond :data:`STUB_FRAGMENT_MAX_CHARS`, or with no boundary line at all, is
    treated as a real log and left alone.
    """
    if not content or not content.strip():
        return True
    lines = [ln.strip() for ln in content.splitlines() if ln.strip()]
    if not any(_is_boundary(ln) for ln in lines):
        return False
    fragment = sum(len(ln) for ln in lines if not _is_boundary(ln))
    return fragment <= STUB_FRAGMENT_MAX_CHARS


def session_ids_from_log(content: Optional[str]) -> list[str]:
    """Worker session ids in *content*, newest first (a log appends one per run)."""
    ordered: list[str] = []
    for raw in (content or "").splitlines():
        line = raw.strip()
        if line.startswith("Session:"):
            candidate = line[len("Session:"):].strip()
            if candidate and _WORKER_SESSION_RE.fullmatch(candidate) and candidate not in ordered:
                ordered.append(candidate)
    return list(reversed(ordered))


def session_id_from_log(content: Optional[str]) -> Optional[str]:
    """The last ``Session: <id>`` line — the newest run's worker session, or None."""
    ids = session_ids_from_log(content)
    return ids[0] if ids else None


def session_candidates(content: Optional[str], receipt: Optional[dict] = None) -> list[str]:
    """Sessions to try, in order: the owner's receipt first, then the log's newest ids.

    The owner writes its own session id into the ``owner_admitted`` event, so it wins
    over anything found in reply text (a report quoting ``Session: <id>`` must not be
    tried before the worker's own run). Capped at :data:`MAX_SESSION_CANDIDATES`.
    """
    ordered: list[str] = []
    if receipt:
        candidate = receipt.get("session_id")
        if isinstance(candidate, str) and candidate.strip():
            ordered.append(candidate.strip())
    for sid in session_ids_from_log(content):
        if sid not in ordered:
            ordered.append(sid)
    return ordered[:MAX_SESSION_CANDIDATES]


def owner_receipt(conn: Optional[sqlite3.Connection], task_id: str) -> Optional[dict]:
    """The task's newest ``owner_admitted`` payload (``{db, session_id, …}``), or None.

    The owner records its own session id + state.db path there, which survives a log
    that lost its ``Session:`` line (rotated, truncated, or a run killed at spawn). Any
    read error is "no receipt", never an exception.
    """
    if conn is None or not task_id:
        return None
    try:
        row = conn.execute(
            "SELECT payload FROM task_events WHERE task_id = ? AND kind = 'owner_admitted' "
            "ORDER BY id DESC LIMIT 1",
            (task_id,),
        ).fetchone()
    except sqlite3.Error:
        return None
    if row is None:
        return None
    payload = row[0]
    if isinstance(payload, str):
        try:
            payload = json.loads(payload)
        except (TypeError, ValueError):
            return None
    return payload if isinstance(payload, dict) else None


def profile_state_db(assignee: Optional[str]) -> Optional[Path]:
    """``state.db`` for the profile named by *assignee* (``None`` when absent).

    Uses the canonical profile-path helper, so a named profile resolves under the
    hermes root the process is actually running with — never a hardcoded home.
    """
    name = (assignee or "").strip() or "default"
    try:
        from hermes_cli.profiles import get_profile_dir

        db_path = Path(get_profile_dir(name)) / "state.db"
    except (ImportError, OSError, ValueError):  # an unknown/rejected profile is "no transcript"
        return None
    return db_path if db_path.is_file() else None


def _hermes_root() -> Optional[Path]:
    try:
        from hermes_constants import get_default_hermes_root

        return Path(get_default_hermes_root())
    except (ImportError, OSError, ValueError):  # containment failing closed is the safe answer
        return None


def _contained_db(value: Any, assignee_db: Optional[Path]) -> Optional[Path]:
    """A receipt-supplied db path, only when it is the assignee's db or under the hermes root.

    Keeps this route from becoming "read any sqlite by path": the receipt names a file
    chosen by the owner of the run it describes, so the path itself is fenced.
    """
    if not isinstance(value, str) or not value:
        return None
    try:
        path = Path(value).expanduser().resolve()
    except (OSError, ValueError, RuntimeError):
        return None
    if not path.is_file():
        return None
    if assignee_db is not None:
        try:
            if path == Path(assignee_db).resolve():
                return path
        except (OSError, ValueError):
            pass
    root = _hermes_root()
    if root is None:
        return None
    try:
        path.relative_to(root.resolve())
    except (OSError, ValueError):
        return None
    return path


def _read_messages(db_path: Path, session_id: str) -> Optional[Sequence[tuple]]:
    """Rows of *session_id* from *db_path*, oldest first; None when unreadable."""
    try:
        uri = db_path.resolve().as_uri() + _RO_PARAMS
        conn = sqlite3.connect(uri, uri=True, timeout=2.0)
    except (OSError, ValueError, sqlite3.Error):
        return None
    try:
        return conn.execute(_MESSAGES_SQL, (session_id,)).fetchall()
    except sqlite3.Error:
        return None
    finally:
        try:
            conn.close()
        except sqlite3.Error:
            pass


def _hhmmss(stamp: Any) -> str:
    try:
        return time.strftime("%H:%M:%S", time.localtime(float(stamp)))
    except (TypeError, ValueError, OSError, OverflowError):
        return "--:--:--"


def _preview(value: Any, limit: int = TOOL_PREVIEW_CHARS) -> str:
    """One line, whitespace-collapsed, cut to *limit* characters."""
    if value is None:
        return ""
    text = " ".join(str(value).split())
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def _text_lines(prefix: str, text: str) -> list[str]:
    """*prefix* + first line, continuation lines indented under it."""
    lines = text.rstrip().splitlines() or [""]
    return [prefix + lines[0]] + [f"  {line}" for line in lines[1:]]


def _tool_call_lines(stamp: str, raw: Any) -> list[str]:
    """One ``[t] assistant -> name(args-preview)`` line per tool call in *raw*."""
    calls = raw
    if isinstance(raw, str):
        try:
            calls = json.loads(raw)
        except (TypeError, ValueError):
            calls = None
    if not isinstance(calls, list):
        return [f"{stamp} assistant -> {_preview(raw)}"] if raw else []
    lines: list[str] = []
    for call in calls:
        if not isinstance(call, dict):
            continue
        fn = call.get("function")
        if not isinstance(fn, dict):
            fn = call
        name = fn.get("name") or "tool"
        args = fn.get("arguments", "")
        if not isinstance(args, str):
            try:
                args = json.dumps(args, ensure_ascii=False, sort_keys=True)
            except (TypeError, ValueError):
                args = str(args)
        lines.append(f"{stamp} assistant -> {name}({_preview(args)})")
    return lines


def render_transcript(rows: Sequence[Sequence[Any]]) -> str:
    """Plain-text frames for ``(role, content, tool_calls, tool_name, timestamp)`` rows."""
    out: list[str] = []
    for role, content, tool_calls, tool_name, stamp in rows:
        clock = f"[{_hhmmss(stamp)}]"
        kind = (role or "").strip().lower()
        if kind == "system":
            continue  # the prompt, not a run frame
        if kind == "tool":
            name = (tool_name or "").strip() or "tool"
            out.append(f"{clock} tool {name} -> {_preview(content)}")
            continue
        text = content if isinstance(content, str) else ""
        if text.strip():
            out.extend(_text_lines(f"{clock} {kind or 'message'}: ", text))
        if tool_calls:
            out.extend(_tool_call_lines(clock, tool_calls))
    return "\n".join(out)


def read_tail_for(tail: Optional[int], size_bytes: int) -> Optional[int]:
    """Byte cap for the on-disk read of a log *size_bytes* long.

    ``tail`` caps the RETURNED text (:func:`apply_tail`); this only widens the read for a
    stub-sized file so the rebuild can see its ``Session:`` line even when the tail window
    would start past it. Anything bigger keeps today's cheap tail read.
    """
    if tail is not None and 0 <= size_bytes <= STUB_READ_MAX_BYTES:
        return None
    return tail


def apply_tail(text: str, tail: Optional[int]) -> str:
    """Byte cap on the *returned* text: the last *tail* bytes, partial first line dropped.

    Mirrors ``kanban_db.read_worker_log``'s on-disk tail so a rebuilt transcript honours
    ``?tail=`` exactly like the file path does.
    """
    if not tail or tail <= 0:
        return text
    raw = text.encode("utf-8")
    if len(raw) <= tail:
        return text
    window = raw[-tail:]
    head, sep, rest = window.partition(b"\n")
    return (rest if sep else head).decode("utf-8", errors="replace")


def reuse_stub_transcript(
    *,
    task: Any,
    log_content: Optional[str],
    tail: Optional[int] = None,
    receipt: Optional[dict] = None,
    size_bytes: Optional[int] = None,
    whole_file: bool = True,
) -> Optional[dict]:
    """Rebuild the transcript when *log_content* is a whole-file stub; ``None`` = "keep the file".

    ``task`` is the ``kanban_db.Task`` row (``assignee`` is all that is read),
    ``receipt`` the optional :func:`owner_receipt` fallback, ``size_bytes`` the
    on-disk size and ``whole_file`` whether *log_content* is the entire file. Every
    failure path returns ``None`` so the caller serves the log file unchanged.
    """
    try:
        if log_content is None or not is_stub_log(log_content):
            return None
        if size_bytes is not None and size_bytes > STUB_READ_MAX_BYTES:
            return None  # a window of a big real log is not a stub we can judge
        if not whole_file:
            return None

        session_ids = session_candidates(log_content, receipt)
        if not session_ids:
            return None

        assignee_db = profile_state_db(getattr(task, "assignee", None))
        candidates: list[Path] = []
        for db_path in (assignee_db, _contained_db((receipt or {}).get("db"), assignee_db)):
            if db_path is not None and db_path not in candidates:
                candidates.append(db_path)
        if not candidates:
            return None

        body, used_session = "", ""
        for session_id in session_ids:
            for db_path in candidates:
                rows = _read_messages(db_path, session_id)
                if not rows:
                    continue
                rendered = render_transcript(rows)
                if rendered.strip():
                    body, used_session = rendered, session_id
                    break
            if body:
                break
        if not body:
            return None

        header = (log_content or "").strip()
        section = TRANSCRIPT_HEADER.format(session_id=used_session)
        rebuilt = f"{header}\n\n{section}\n{body}" if header else f"{section}\n{body}"
        truncated = bool(tail and len(rebuilt.encode("utf-8")) > tail)
        return {"content": apply_tail(rebuilt, tail), "truncated": truncated}
    except Exception:  # a log route must never 500 on a transcript rebuild
        log.warning("worker-log transcript rebuild skipped", exc_info=True)
        return None


def _plain_file_payload(
    task_id: str, *, tail: Optional[int], board: Optional[str], path: Any = None,
) -> dict:
    """Today's body: the log file, verbatim. Used when the rebuild cannot be attempted."""
    log_path = path
    size = 0
    content: Optional[str] = None
    try:
        from hermes_cli import kanban_db

        if log_path is None:
            log_path = kanban_db.worker_log_path(task_id, board=board)
        size = log_path.stat().st_size if log_path.exists() else 0
        content = kanban_db.read_worker_log(task_id, tail_bytes=tail, board=board)
    except Exception:  # even the fallback must not raise
        log.warning("worker-log file fallback failed for %s", task_id, exc_info=True)
        content = None
    return {
        "task_id": task_id, "path": str(log_path) if log_path is not None else "",
        "exists": content is not None, "size_bytes": size, "content": content or "",
        "truncated": bool(tail and size > tail), "source": "log_file",
    }


def log_payload(
    *, task_id: str, task: Any, tail: Optional[int] = None,
    conn: Optional[sqlite3.Connection] = None, board: Optional[str] = None,
) -> dict:
    """The ``GET /tasks/{id}/log`` body: file read + optional rebuild, never raising.

    The route calls this one function, so a sibling helper that loads but no longer
    carries an expected name (version skew after a partial patch) degrades to the file
    payload instead of 500ing the log route. ``path``/``size_bytes`` describe the FILE
    even when ``source == "state_db"``.
    """
    log_path = None
    try:
        from hermes_cli import kanban_db

        log_path = kanban_db.worker_log_path(task_id, board=board)
        size = log_path.stat().st_size if log_path.exists() else 0
        read_tail = read_tail_for(tail, size)
        content = kanban_db.read_worker_log(task_id, tail_bytes=read_tail, board=board)
        exists = content is not None
        truncated = bool(tail and size > tail)
        whole_file = read_tail is None or (tail is not None and size <= tail)
        source = "log_file"
        rebuilt = reuse_stub_transcript(
            task=task, log_content=content, tail=tail, receipt=owner_receipt(conn, task_id),
            size_bytes=size, whole_file=whole_file,
        )
        if rebuilt is not None:
            content, truncated, source = rebuilt["content"], bool(rebuilt["truncated"]), "state_db"
        elif whole_file and tail is not None:
            content = apply_tail(content or "", tail)  # the bytes a tail read would return
        return {
            "task_id": task_id, "path": str(log_path), "exists": exists,
            "size_bytes": size, "content": content or "", "truncated": truncated,
            "source": source,
        }
    except Exception:  # a broken helper must degrade to today's body, never 500
        log.warning("worker-log payload fell back to the file for %s", task_id, exc_info=True)
        return _plain_file_payload(task_id, tail=tail, board=board, path=log_path)
