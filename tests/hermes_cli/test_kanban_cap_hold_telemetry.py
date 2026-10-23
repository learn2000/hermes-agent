"""Cap-hold telemetry: the dispatcher's "stuck" warning must say WHY.

Regression for the fleet diagnosis t_de8d9f41 (2026-10-22): whenever the
host-level ``kanban.max_in_progress`` cap was full (8/8 long-running
workers + backlog), every tick spawned 0 and the "dispatcher stuck: ...
Check profile health (venv, PATH, credentials)" warning looped in
gateway.log — actively misleading, because the dispatcher was healthy and
doing exactly the right backpressure. ``_tick_spawn_budget`` returned
``(False, None)`` with NO hold marker on the ``DispatchResult``, and
``describe_suppression`` only knew respawn-guard / lock / memory-pressure
reasons, so the "Last tick held back:" suffix was always empty.

The fix records tick-level cap markers (``capped_host`` / ``capped_board``
/ ``capped_profiles``) and surfaces them as ``host_cap=running/cap`` etc.
through ``describe_suppression``; the stuck warning then reports
"dispatcher holding ... cap is full" instead of asserting a spawn-path
fault.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_dispatch as kbd
from hermes_cli import kanban_db_connect as kbc


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    """Isolated HERMES_HOME with an empty kanban DB."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _fake_spawn_factory(spawns: list):
    def fake_spawn(task, workspace, board=None):
        spawns.append(task.id)
        return 42
    return fake_spawn


# ---------------------------------------------------------------------------
# describe_suppression: cap markers
# ---------------------------------------------------------------------------


def test_describe_suppression_empty_result():
    assert kbd.describe_suppression([]) == ""
    assert kbd.describe_suppression([None, None]) == ""
    assert kbd.describe_suppression([kb.DispatchResult()]) == ""


def test_describe_suppression_host_cap_marker():
    res = kb.DispatchResult(capped_host=(8, 8))
    out = kbd.describe_suppression([res])
    assert "host_cap=8/8" in out
    assert kbd.has_cap_hold([res])


def test_describe_suppression_host_cap_running_above_cap():
    # total_running can exceed the cap (multi-board workers counted together)
    res = kb.DispatchResult(capped_host=(8, 9))
    assert "host_cap=9/8" in kbd.describe_suppression([res])


def test_describe_suppression_board_cap_marker():
    res = kb.DispatchResult(capped_board=(2, 5))
    out = kbd.describe_suppression([res])
    assert "board_cap=5/2" in out
    assert kbd.has_cap_hold([res])


def test_describe_suppression_profile_cap_marker():
    res = kb.DispatchResult(capped_profiles=[("alice", 3, 2), ("bob", 2, 2)])
    out = kbd.describe_suppression([res])
    assert "profile_cap=alice=3/2" in out
    assert "profile_cap=bob=2/2" in out
    assert kbd.has_cap_hold([res])


def test_describe_suppression_caps_combine_with_guard_counts():
    res = kb.DispatchResult(
        respawn_guarded=[("t1", "active_pr")],
        rate_limited=["t2"],
        capped_host=(8, 8),
    )
    out = kbd.describe_suppression([res])
    assert "active_pr=1" in out
    assert "rate_limited=1" in out
    assert "host_cap=8/8" in out
    # The cap marker is what the stuck warning needs; order: counts, then caps.
    assert out.index("host_cap=") > out.index("rate_limited=")


def test_describe_suppression_multi_board_cap_is_not_hidden():
    # The gateway watcher passes every board's result for one tick: a board
    # at cap-full must not be hidden by a sibling board with headroom (the
    # host budget is shared across boards).
    capped = kb.DispatchResult(capped_host=(8, 8))
    idle = kb.DispatchResult()
    out = kbd.describe_suppression([idle, capped])
    assert "host_cap=8/8" in out
    assert kbd.has_cap_hold([idle, capped])
    # Duplicate markers across results are reported once.
    assert kbd.describe_suppression([capped, capped]).count("host_cap=8/8") == 1
    # None entries (ticked boards that had no result) must not crash or
    # contribute anything.
    assert kbd.describe_suppression([None, capped]) == "host_cap=8/8"


def test_has_cap_hold_negative():
    res = kb.DispatchResult(
        respawn_guarded=[("t1", "active_pr")],
        skipped_locked=True,
        memory_pressure="critical",
    )
    assert not kbd.has_cap_hold([res])
    assert not kbd.has_cap_hold([])
    assert not kbd.has_cap_hold([None])
    assert not kbd.has_cap_hold([kb.DispatchResult()])
    # A cap marker alone is enough, in any of the three shapes.
    assert kbd.has_cap_hold([kb.DispatchResult(capped_host=(4, 4))])
    assert kbd.has_cap_hold([kb.DispatchResult(capped_board=(4, 4))])
    assert kbd.has_cap_hold([kb.DispatchResult(capped_profiles=[("a", 1, 1)])])


# ---------------------------------------------------------------------------
# dispatch: markers set when a cap is exhausted (behavior unchanged)
# ---------------------------------------------------------------------------


def test_host_cap_exhausted_sets_capped_host(kanban_home, all_assignees_spawnable):
    """8/8-style hold: every slot busy → 0 spawns AND ``capped_host`` set."""
    spawns: list = []
    with kbc.connect() as conn:
        for title in ("busy-1", "busy-2"):
            tid = kb.create_task(conn, title=title, assignee="alice")
            assert kb.claim_task(conn, tid) is not None
        kb.create_task(conn, title="wants-to-run", assignee="alice")
        res = kbd.dispatch_once(
            conn, spawn_fn=_fake_spawn_factory(spawns), max_in_progress=2,
        )

    # Cap still enforced identically: nothing spawned.
    assert not spawns
    assert not res.spawned
    # The hold is now named: host cap 2, 2 running (none on other boards).
    assert res.capped_host == (2, 2)
    assert kbd.describe_suppression([res]) == "host_cap=2/2"


def test_host_cap_partial_does_not_mark(kanban_home, all_assignees_spawnable):
    """A tick with headroom must NOT carry a cap marker."""
    spawns: list = []
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="busy", assignee="alice")
        assert kb.claim_task(conn, tid) is not None
        kb.create_task(conn, title="a", assignee="alice")
        res = kbd.dispatch_once(
            conn, spawn_fn=_fake_spawn_factory(spawns), max_in_progress=2,
        )

    assert len(spawns) == 1
    assert res.capped_host is None
    assert res.capped_board is None
    assert not res.capped_profiles
    assert kbd.describe_suppression([res]) == ""


def test_board_cap_exhausted_sets_capped_board(kanban_home, all_assignees_spawnable):
    """Per-board ``max_spawn`` full → ``capped_board``, host cap untouched."""
    spawns: list = []
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="busy", assignee="alice")
        assert kb.claim_task(conn, tid) is not None
        kb.create_task(conn, title="a", assignee="alice")
        res = kbd.dispatch_once(
            conn, spawn_fn=_fake_spawn_factory(spawns), max_spawn=1,
        )

    assert not spawns
    assert res.capped_board == (1, 1)
    assert res.capped_host is None
    assert kbd.describe_suppression([res]) == "board_cap=1/1"


def test_daemon_warning_says_holding_not_stuck_when_capped():
    """End-to-end wording: a cap-full tick's daemon warning must name the cap
    and must NOT assert a spawn-path fault (venv/PATH/credentials)."""
    from hermes_cli import kanban_ops

    capped = kb.DispatchResult(capped_host=(2, 2))
    assert not capped.spawned
    line = kanban_ops.daemon_stuck_warning(36, capped)
    assert "host_cap=2/2" in line
    assert "concurrency cap is full" in line
    assert "venv, PATH, credentials" not in line
    assert "dispatcher holding" in line

    # No cap marker: the original fault-hint wording is preserved unchanged.
    bare = kanban_ops.daemon_stuck_warning(36, kb.DispatchResult())
    assert "dispatcher stuck" in bare
    assert "venv, PATH, credentials" in bare
    assert "host_cap" not in bare


def test_per_profile_cap_marks_the_busy_profile(
    kanban_home, all_assignees_spawnable,
):
    """A busy profile is named; an idle profile with no capped row is not."""
    spawns: list = []
    with kbc.connect() as conn:
        busy = kb.create_task(conn, title="busy", assignee="alice")
        assert kb.claim_task(conn, busy) is not None
        kb.create_task(conn, title="a2", assignee="alice")
        kb.create_task(conn, title="b1", assignee="bob")
        res = kbd.dispatch_once(
            conn, spawn_fn=_fake_spawn_factory(spawns),
            max_in_progress_per_profile=1,
        )

    # alice held at cap; bob (no running workers) still spawned.
    assert [a for _t, a, _p in res.spawned] == ["bob"]
    assert res.capped_profiles == [("alice", 1, 1)]
    assert kbd.describe_suppression([res]) == "profile_cap=alice=1/1"
    assert kbd.has_cap_hold([res])
