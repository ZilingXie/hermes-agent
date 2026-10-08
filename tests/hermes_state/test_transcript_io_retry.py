"""Fault-injection tests for the transcript I/O retry (AC-13898 follow-up).

Covers two layers added for session-storage failures:

* ``classify_persistence_error``'s ``io`` bucket — SQLITE_IOERR must not be
  misreported as the deterministic disk-full/permissions advice, while
  corruption / disk-full / read-only / locked keep their own buckets.
* ``_execute_write``'s single post-rollback retry for pre-COMMIT SQLite I/O
  errors on the transcript flush path (``append_messages_batch``).

Every scenario runs against a real SQLite database under a temp home; the
faults are injected at the boundaries the plan pins down (batch write body,
COMMIT call, ROLLBACK call, out-of-band file replacement) — never by
mocking the save into a returning-True stub.
"""

import os
import sqlite3
from types import SimpleNamespace

import pytest

from hermes_state import (
    SessionDB,
    StateDbReplacedError,
    classify_persistence_error,
    is_sqlite_io_error,
    sqlite_error_code_and_name,
)


@pytest.fixture()
def db(tmp_path):
    d = SessionDB(db_path=tmp_path / "state.db")
    d.create_session("sess-io", source="cli")
    yield d
    d.close()


def _turn_messages():
    return [
        {"role": "user", "content": "question"},
        {
            "role": "assistant",
            "content": "answer",
            "finish_reason": "stop",
        },
    ]


class _IoerrInjector:
    """Replace SessionDB._insert_message_rows with a counting fault injector.

    ``fail_times`` controls how many invocations raise a SQLite I/O error
    before the real implementation takes over; ``before_raise`` runs right
    before each injected raise (used to swap the database file underneath
    the process for the replacement scenario).
    """

    def __init__(self, db, fail_times, before_raise=None, message="disk I/O error"):
        self._db = db
        self._real = SessionDB._insert_message_rows
        self.fail_times = fail_times
        self.calls = 0
        self.before_raise = before_raise
        self.message = message

    def _impl(self, conn, session_id, messages):
        self.calls += 1
        if self.calls <= self.fail_times:
            if self.before_raise is not None:
                self.before_raise()
            raise sqlite3.OperationalError(self.message)
        return self._real(self._db, conn, session_id, messages)

    def install(self, monkeypatch):
        monkeypatch.setattr(SessionDB, "_insert_message_rows", self._impl)


class _ConnProxy:
    """Attribute-forwarding proxy over the live connection that can fail
    ``commit()`` / ``rollback()`` on demand — the only way to inject a fault
    at those C-level call sites while everything else stays real."""

    def __init__(self, real):
        self._real = real
        self.commit_failures = 0
        self.rollback_failures = 0
        self.begin_fails = False

    def __getattr__(self, name):
        return getattr(self._real, name)

    def execute(self, sql, *args, **kwargs):
        if self.begin_fails and str(sql).strip().upper().startswith("BEGIN"):
            raise sqlite3.OperationalError("disk I/O error")
        return self._real.execute(sql, *args, **kwargs)

    def commit(self):
        if self.commit_failures > 0:
            self.commit_failures -= 1
            raise sqlite3.OperationalError("disk I/O error")
        return self._real.commit()

    def rollback(self):
        if self.rollback_failures > 0:
            self.rollback_failures -= 1
            raise sqlite3.OperationalError("unable to open database file")
        return self._real.rollback()


class TestClassifyIoBucket:
    def test_disk_io_error_string_classifies_io(self):
        assert classify_persistence_error("disk I/O error") == "io"

    def test_disk_io_error_exception_classifies_io(self):
        assert classify_persistence_error(
            sqlite3.OperationalError("disk I/O error")
        ) == "io"

    def test_ioerr_result_code_classifies_io(self):
        exc = sqlite3.OperationalError("disk I/O error")
        exc.sqlite_errorcode = 266  # SQLITE_IOERR_READ = 10 | (1 << 8)
        exc.sqlite_errorname = "SQLITE_IOERR_READ"
        assert classify_persistence_error(exc) == "io"
        assert is_sqlite_io_error(exc)

    def test_corruption_not_swallowed_by_io(self):
        assert classify_persistence_error("database disk image is malformed") == "corrupt"

    def test_disk_full_not_swallowed_by_io(self):
        assert classify_persistence_error("database or disk is full") == "disk"

    def test_readonly_not_swallowed_by_io(self):
        assert classify_persistence_error("attempt to write a readonly database") == "disk"

    def test_locked_still_locked(self):
        assert classify_persistence_error("database is locked") == "locked"

    def test_code_and_name_extraction(self):
        real = sqlite3.OperationalError("nope")
        code, name = sqlite_error_code_and_name(real)
        # Hand-built exceptions carry no code — helper must degrade cleanly.
        assert code is None or isinstance(code, int)
        assert name is None or isinstance(name, str)
        assert sqlite_error_code_and_name("plain string") == (None, None)
        assert sqlite_error_code_and_name(None) == (None, None)


class TestTranscriptIoRetry:
    def test_clean_save_untouched(self, db):
        """Baseline: one attempt, exact rows and counters."""
        injector = _IoerrInjector(db, fail_times=0)
        monkeypatch = pytest.MonkeyPatch()
        with monkeypatch.context() as mp:
            injector.install(mp)
            inserted = db.append_messages_batch("sess-io", _turn_messages())
        try:
            assert inserted == 2
            assert injector.calls == 1
            rows = db._conn.execute(
                "SELECT COUNT(*) FROM messages WHERE session_id='sess-io'"
            ).fetchone()[0]
            count = db._conn.execute(
                "SELECT message_count FROM sessions WHERE id='sess-io'"
            ).fetchone()[0]
            assert rows == 2
            assert count == 2
        finally:
            monkeypatch.undo()

    def test_write_phase_ioerr_retried_once_exactly(self, db):
        """First attempt fails pre-COMMIT with IOERR and rolls back; the
        single retry re-runs guards + insert; no duplicate rows/counters."""
        injector = _IoerrInjector(db, fail_times=1)
        guards = {"calls": 0}
        real_guards = SessionDB._check_transcript_write_guards

        def counting_guards(self, *a, **kw):
            guards["calls"] += 1
            return real_guards(self, *a, **kw)

        with pytest.MonkeyPatch.context() as mp:
            injector.install(mp)
            mp.setattr(SessionDB, "_check_transcript_write_guards", counting_guards)
            inserted = db.append_messages_batch("sess-io", _turn_messages())
        assert inserted == 2
        # Exactly one retry: initial attempt + one replay, no more.
        assert injector.calls == 2
        # The retry re-ran the lease/compression guards, not just the INSERT.
        assert guards["calls"] == 2
        rows = db._conn.execute(
            "SELECT COUNT(*) FROM messages WHERE session_id='sess-io'"
        ).fetchone()[0]
        count = db._conn.execute(
            "SELECT message_count FROM sessions WHERE id='sess-io'"
        ).fetchone()[0]
        assert rows == 2
        assert count == 2

    def test_second_ioerr_stops_without_partial_rows(self, db):
        """Two consecutive I/O failures: the error propagates, the batch
        leaves zero rows and zero counter movement."""
        injector = _IoerrInjector(db, fail_times=99)
        with pytest.MonkeyPatch.context() as mp:
            injector.install(mp)
            with pytest.raises(sqlite3.OperationalError):
                db.append_messages_batch("sess-io", _turn_messages())
        assert injector.calls == 2  # initial + the single allowed retry
        rows = db._conn.execute(
            "SELECT COUNT(*) FROM messages WHERE session_id='sess-io'"
        ).fetchone()[0]
        count = db._conn.execute(
            "SELECT message_count FROM sessions WHERE id='sess-io'"
        ).fetchone()[0]
        assert rows == 0
        assert count == 0

    def test_commit_phase_ioerr_never_retried(self, db):
        """A failure during COMMIT has an unknown outcome — zero retries,
        the write stops, nothing lands."""
        injector = _IoerrInjector(db, fail_times=0)
        proxy = _ConnProxy(db._conn)
        proxy.commit_failures = 1
        db._conn = proxy
        try:
            with pytest.MonkeyPatch.context() as mp:
                injector.install(mp)
                with pytest.raises(sqlite3.OperationalError):
                    db.append_messages_batch("sess-io", _turn_messages())
            assert injector.calls == 1
            rows = db._conn.execute(
                "SELECT COUNT(*) FROM messages WHERE session_id='sess-io'"
            ).fetchone()[0]
            assert rows == 0
        finally:
            db._conn = proxy._real

    def test_rollback_failure_forbids_retry(self, db):
        """When the rollback itself raises, the transaction state is
        unknown — no replay happens (single insert attempt)."""
        injector = _IoerrInjector(db, fail_times=1)
        proxy = _ConnProxy(db._conn)
        proxy.rollback_failures = 1
        db._conn = proxy
        try:
            with pytest.MonkeyPatch.context() as mp:
                injector.install(mp)
                with pytest.raises(sqlite3.OperationalError):
                    db.append_messages_batch("sess-io", _turn_messages())
            assert injector.calls == 1
        finally:
            db._conn = proxy._real

    def test_non_io_error_not_retried(self, db):
        """Read-only shaped failures keep their existing handling and do
        not enter the I/O retry path (single attempt, immediate raise)."""
        injector = _IoerrInjector(
            db, fail_times=1, message="attempt to write a readonly database"
        )
        with pytest.MonkeyPatch.context() as mp:
            injector.install(mp)
            with pytest.raises(sqlite3.OperationalError):
                db.append_messages_batch("sess-io", _turn_messages())
        assert injector.calls == 1

    def test_replaced_database_stops_retry(self, db, tmp_path):
        """An I/O error that arrives together with an out-of-band replace
        must stop with StateDbReplacedError, never replay onto the new file."""

        def swap_file():
            gone = tmp_path / "state.db.gone"
            os.replace(tmp_path / "state.db", gone)
            # Materialize a different-inode file at the original path.
            fresh = tmp_path / "state.db"
            fresh.write_bytes(gone.read_bytes()[:4096])
            os.remove(gone)

        injector = _IoerrInjector(db, fail_times=1, before_raise=swap_file)
        with pytest.MonkeyPatch.context() as mp:
            injector.install(mp)
            with pytest.raises(StateDbReplacedError):
                db.append_messages_batch("sess-io", _turn_messages())
        assert injector.calls == 1

    def test_retry_success_marks_rows_persisted_consistently(self, db):
        """After a successful retry the row dicts carry fresh row ids (the
        in-memory marking point), matching a clean single-attempt save."""
        clean_rows = _turn_messages()
        injector_clean = _IoerrInjector(db, fail_times=0)
        with pytest.MonkeyPatch.context() as mp:
            injector_clean.install(mp)
            db.append_messages_batch("sess-io", clean_rows)
        clean_ids = [m.get("_row_id") for m in clean_rows]
        assert all(isinstance(i, int) for i in clean_ids)

        retried_rows = _turn_messages()
        injector_retry = _IoerrInjector(db, fail_times=1)
        with pytest.MonkeyPatch.context() as mp:
            injector_retry.install(mp)
            db.append_messages_batch("sess-io", retried_rows)
        retried_ids = [m.get("_row_id") for m in retried_rows]
        assert all(isinstance(i, int) for i in retried_ids)
        assert retried_ids[0] > clean_ids[-1]  # fresh, later rows
        # Idempotent read-back: exactly the four expected rows in order.
        rows = db._conn.execute(
            "SELECT role FROM messages WHERE session_id='sess-io' ORDER BY id"
        ).fetchall()
        assert [r["role"] for r in rows] == ["user", "assistant", "user", "assistant"]


class TestRetryPreservesMessagesAfterRollback:
    """P1 regression: in-transaction row ids stamped into the row dicts
    survive ROLLBACK (verified: this SQLite build reuses rolled-back
    AUTOINCREMENT ids), and a concurrent writer can claim the freed id
    during the retry window. The replay must restore the batch's entry
    state or the stale id makes resolve_and_repair_transcript_batch adopt
    an unrelated row and silently drop the message."""

    def _real(self):
        return SessionDB._insert_message_rows

    def test_partial_insert_then_ioerr_keeps_all_messages(self, db):
        """First attempt inserts SOME rows then fails pre-commit; the retry
        must re-insert the whole batch — no loss, no duplicates, counters
        exact."""
        real = self._real()
        state = {"calls": 0}

        def partial_then_fail(self_, conn, session_id, msgs):
            state["calls"] += 1
            if state["calls"] == 1:
                real(self_, conn, session_id, msgs[:1])
                raise sqlite3.OperationalError("disk I/O error")
            return real(self_, conn, session_id, msgs)

        msgs = [
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": "a1"},
            {"role": "assistant", "content": "a2", "finish_reason": "stop"},
        ]
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(SessionDB, "_insert_message_rows", partial_then_fail)
            inserted = db.append_messages_batch("sess-io", msgs)
        assert inserted == 3
        assert state["calls"] == 2
        rows = db._conn.execute(
            "SELECT role FROM messages WHERE session_id='sess-io' ORDER BY id"
        ).fetchall()
        assert [r["role"] for r in rows] == ["user", "assistant", "assistant"]
        count = db._conn.execute(
            "SELECT message_count FROM sessions WHERE id='sess-io'"
        ).fetchone()[0]
        assert count == 3

    def test_rowid_reuse_after_rollback_does_not_drop_message(self, db):
        """The exact review scenario: attempt 1 stamps fresh row ids and
        rolls back; during the 100ms retry window a concurrent write reuses
        the freed row id; the retry must still land every batch message."""
        import hermes_state as hs

        real = self._real()
        state = {"calls": 0}

        def fail_after_stamp(self_, conn, session_id, msgs):
            state["calls"] += 1
            if state["calls"] == 1:
                real(self_, conn, session_id, msgs)  # stamps _row_id dicts
                raise sqlite3.OperationalError("disk I/O error")
            return real(self_, conn, session_id, msgs)

        msgs = [
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": "answer", "finish_reason": "stop"},
        ]
        stamped = {}

        def fake_sleep(_seconds):
            stamped["assistant"] = msgs[1].get("_row_id")
            assert stamped["assistant"] is not None, "stale stamp expected"
            # Concurrent writer inside the retry window: claims the exact
            # row ids the rollback freed (AUTOINCREMENT reuses rolled-back
            # ids), including the assistant's stale id, exactly like the
            # review's repro (message B reuses row 1 after A rolled back).
            db._execute_write(
                lambda conn: real(
                    db, conn, "sess-io",
                    [
                        {"role": "user", "content": "concurrent-q"},
                        {"role": "assistant", "content": "concurrent",
                         "finish_reason": "stop"},
                    ],
                )
            )

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(SessionDB, "_insert_message_rows", fail_after_stamp)
            mp.setattr(hs.time, "sleep", fake_sleep)
            inserted = db.append_messages_batch("sess-io", msgs)
        assert inserted == 2
        rows = db._conn.execute(
            "SELECT content FROM messages WHERE session_id='sess-io' ORDER BY id"
        ).fetchall()
        contents = [r["content"] for r in rows]
        assert "answer" in contents, "original batch message was dropped"
        assert "concurrent" in contents, "concurrent write was lost"
        assert contents.count("answer") == 1
        # 2 batch messages + 2 concurrent writes, nothing doubled.
        assert len(contents) == 4
        # The batch's final row id is a fresh, correctly-adopted id.
        assert isinstance(msgs[1].get("_row_id"), int)


class TestIoWording:
    """The user-facing io explanation stays diagnostic (no deterministic
    free-space/permissions advice) — the AC-13898 misdiagnosis."""

    @staticmethod
    def _explain(cause):
        from run_agent import AIAgent

        return AIAgent._format_turn_completion_explanation(
            "session_persistence_failed", persistence_cause=cause
        )

    def test_io_wording_has_no_deterministic_disk_advice(self):
        text = self._explain("io")
        assert text.startswith("⚠️ No reply: ")
        assert "I/O error" in text
        # Must NOT prescribe the disk bucket's advice.
        assert "free some space" not in text
        assert "fix state.db permissions" not in text
        # Must not assert unverified health conclusions either (review P2:
        # the io classification runs no reachability/space/integrity check).
        assert "reachable" not in text
        assert "neither full nor corrupt" not in text
        assert "not yet determined" in text
        # Points at checking the storage layer instead.
        assert "storage" in text

    def test_disk_wording_unchanged(self):
        text = self._explain("disk")
        assert "free some space" in text

    def test_unknown_cause_wording_unchanged(self):
        text = self._explain("unknown")
        assert "hermes doctor" in text


class TestRunStatusFailureReason:
    """_set_run_status carries the structured failure_reason field into the
    pollable status dict (the api_server run.failed wiring relies on it)."""

    def test_failure_reason_survives_status_update(self):
        from gateway.platforms import api_server_runs as asr

        store = SimpleNamespace(
            updates=[],
            update_status=lambda run_id, status: store.updates.append(
                (run_id, dict(status))
            ),
        )
        fake = SimpleNamespace(
            _run_statuses={},
            _run_idempotency_ids={"run-x"},
            _run_idempotency_store=store,
        )
        status = asr._set_run_status(
            fake,
            "run-x",
            "failed",
            error="boom",
            last_event="run.failed",
            failure_reason="session_persistence_failed:io",
        )
        assert status["failure_reason"] == "session_persistence_failed:io"
        assert status["error"] == "boom"
        # Persisted snapshot keeps the field for post-restart readback.
        assert store.updates, "failed status must persist"
        persisted = store.updates[-1][1]
        assert persisted["failure_reason"] == "session_persistence_failed:io"
        assert persisted["status"] == "failed"

    def test_failure_reason_absent_keeps_contract(self):
        from gateway.platforms import api_server_runs as asr

        fake = SimpleNamespace(
            _run_statuses={},
            _run_idempotency_ids=set(),
            _run_idempotency_store=SimpleNamespace(
                update_status=lambda *a, **k: None
            ),
        )
        status = asr._set_run_status(fake, "run-y", "failed", error="boom")
        assert "failure_reason" not in status


class TestTransactionFailureDiagnostics:
    """Round-2 review blocker: EVERY final transaction failure must leave a
    log record (and an exception-carried hermes_txn_diag) stating the real
    phase (begin/write/commit), the rollback outcome (not_attempted is
    distinct from ok), the attempt number, and the COMMIT uncertainty
    flag — not only the retry-preparation path."""

    @pytest.fixture()
    def captured(self):
        import logging

        records: list = []

        class _Handler(logging.Handler):
            def emit(self, record):
                records.append(record)

        handler = _Handler(level=logging.DEBUG)
        logger = logging.getLogger("hermes_state")
        logger.addHandler(handler)
        try:
            yield records
        finally:
            logger.removeHandler(handler)

    def _diag_records(self, records):
        return [
            r for r in records
            if "SQLite transaction failed terminally" in r.getMessage()
        ]

    def _conn_proxy(self, db, *, begin_fails=False, commit_fails=False,
                    rollback_fails=False):
        proxy = _ConnProxy(db._conn)
        proxy.begin_fails = begin_fails
        proxy.commit_failures = 1 if commit_fails else 0
        proxy.rollback_failures = 1 if rollback_fails else 0
        db._conn = proxy
        return proxy

    def test_begin_phase_failure_records_not_attempted(self, db, captured):
        proxy = self._conn_proxy(db, begin_fails=True)
        try:
            with pytest.raises(sqlite3.OperationalError):
                db.append_messages_batch("sess-io", _turn_messages())
            diags = self._diag_records(captured)
            assert diags, "no transaction-failure log for BEGIN failure"
            msg = diags[-1].getMessage()
            assert "phase=begin" in msg
            assert "rollback=not_attempted" in msg
        finally:
            db._conn = proxy._real

    def test_commit_phase_failure_records_uncertain(self, db, captured):
        proxy = self._conn_proxy(db, commit_fails=True)
        try:
            with pytest.raises(sqlite3.OperationalError) as excinfo:
                db.append_messages_batch("sess-io", _turn_messages())
            diags = self._diag_records(captured)
            assert diags, "no transaction-failure log for COMMIT failure"
            msg = diags[-1].getMessage()
            assert "phase=commit" in msg
            assert "rollback=ok" in msg
            assert "commit_outcome_uncertain=True" in msg
            # The diagnostics ride on the exception for upper layers.
            diag = getattr(excinfo.value, "hermes_txn_diag", None)
            assert isinstance(diag, dict)
            assert diag["phase"] == "commit"
            assert diag["rollback"] == "ok"
            assert diag["commit_outcome_uncertain"] is True
        finally:
            db._conn = proxy._real

    def test_second_ioerr_failure_records_final_attempt(self, db, captured):
        injector = _IoerrInjector(db, fail_times=99)
        with pytest.MonkeyPatch.context() as mp:
            injector.install(mp)
            with pytest.raises(sqlite3.OperationalError) as excinfo:
                db.append_messages_batch("sess-io", _turn_messages())
        diags = self._diag_records(captured)
        assert diags, "no transaction-failure log for the exhausted retry"
        msg = diags[-1].getMessage()
        assert "phase=write" in msg
        assert "rollback=ok" in msg
        assert "attempt=2" in msg
        assert "io_retry_used=True" in msg
        assert excinfo.value.hermes_txn_diag["attempt"] == 2

    def test_readonly_failure_records_write_phase(self, db, captured):
        injector = _IoerrInjector(
            db, fail_times=1, message="attempt to write a readonly database"
        )
        with pytest.MonkeyPatch.context() as mp:
            injector.install(mp)
            with pytest.raises(sqlite3.OperationalError) as excinfo:
                db.append_messages_batch("sess-io", _turn_messages())
        diags = self._diag_records(captured)
        assert diags, "no transaction-failure log for the read-only failure"
        msg = diags[-1].getMessage()
        assert "phase=write" in msg
        assert "rollback=ok" in msg
        assert excinfo.value.hermes_txn_diag["phase"] == "write"

    def test_rollback_failure_records_failed_and_blocks_retry(self, db, captured):
        injector = _IoerrInjector(db, fail_times=1)
        proxy = self._conn_proxy(db, rollback_fails=True)
        try:
            with pytest.MonkeyPatch.context() as mp:
                injector.install(mp)
                with pytest.raises(sqlite3.OperationalError) as excinfo:
                    db.append_messages_batch("sess-io", _turn_messages())
            assert injector.calls == 1  # failed rollback forbids the replay
            diags = self._diag_records(captured)
            assert diags, "no transaction-failure log after rollback failure"
            msg = diags[-1].getMessage()
            assert "rollback=failed" in msg
            assert excinfo.value.hermes_txn_diag["rollback"] == "failed"
        finally:
            db._conn = proxy._real
