"""Tests for the agent session manager."""

import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


from core.agent_session import (
    SESSIONS_DIR,
    AgentSession,
    TurnRecord,
    get_session,
    reset_session,
)


# ---------------------------------------------------------------------------
# TurnRecord
# ---------------------------------------------------------------------------
class TestTurnRecord:
    def test_creates_from_success_result(self):
        result = {"user_query": "hello", "final_output": "world", "duration": 1.0}
        tr = TurnRecord(result)
        assert tr.query == "hello"
        assert tr.had_anomaly is False
        assert tr.anomaly_count == 0
        assert tr.antibody_count == 0
        assert tr.immune_activated is False
        assert tr.success is True
        assert tr.duration == 1.0

    def test_creates_from_failed_result(self):
        result = {"user_query": "test", "final_output": None, "duration": 5.0}
        tr = TurnRecord(result)
        assert tr.success is False
        assert tr.had_anomaly is False

    def test_detects_anomalies(self):
        result = {
            "user_query": "test",
            "anomalies": [{"status": "unhealthy", "reason": "loop"}],
            "antibodies": [{"code": "# fix"}],
            "is_immune_active": True,
        }
        tr = TurnRecord(result)
        assert tr.had_anomaly is True
        assert tr.anomaly_count == 1
        assert tr.antibody_count == 1
        assert tr.immune_activated is True

    def test_truncates_long_query(self):
        result = {"user_query": "x" * 200}
        tr = TurnRecord(result)
        assert len(tr.query) == 80
        assert tr.query == "x" * 80

    def test_to_dict_returns_all_slots(self):
        result = {"user_query": "test", "final_output": "ok", "duration": 0.5}
        tr = TurnRecord(result)
        d = tr.to_dict()
        for slot in TurnRecord.__slots__:
            assert slot in d


# ---------------------------------------------------------------------------
# AgentSession
# ---------------------------------------------------------------------------
class TestAgentSessionInit:
    def test_default_init(self):
        session = AgentSession()
        assert session.session_id is not None
        assert len(session.session_id) == 12
        assert session.health_score() == 1.0
        assert session.anomaly_rate() == 0.0

    def test_custom_session_id(self):
        session = AgentSession(session_id="custom123")
        assert session.session_id == "custom123"

    def test_custom_max_turns(self):
        session = AgentSession(max_turns=3)
        for i in range(5):
            session.record_turn({"user_query": str(i), "final_output": "ok"})
        summary = session.summary()
        assert summary["total_turns"] == 3  # deque limited to 3


class TestAgentSessionRecordTurn:
    def test_records_success(self):
        session = AgentSession()
        session.record_turn({"user_query": "q", "final_output": "a"})
        assert session.summary()["total_turns"] == 1

    def test_records_multiple_turns(self):
        session = AgentSession()
        for i in range(5):
            session.record_turn({"user_query": str(i), "final_output": "ok"})
        assert session.summary()["total_turns"] == 5

    def test_consecutive_failures_tracking(self):
        session = AgentSession()
        session.record_turn({"user_query": "q", "final_output": None})
        assert session.summary()["consecutive_failures"] == 1
        session.record_turn({"user_query": "q", "final_output": None})
        assert session.summary()["consecutive_failures"] == 2

    def test_success_resets_consecutive_failures(self):
        session = AgentSession()
        session.record_turn({"user_query": "q1", "final_output": None})
        session.record_turn({"user_query": "q2", "final_output": None})
        session.record_turn({"user_query": "q3", "final_output": "ok"})
        assert session.summary()["consecutive_failures"] == 0

    def test_recovery_event_on_success_after_failure(self):
        session = AgentSession()
        session.record_turn({"user_query": "q1", "final_output": None})
        session.record_turn({"user_query": "q2", "final_output": "ok"})
        summary = session.summary()
        assert summary["total_recoveries"] == 1

    def test_no_recovery_on_consecutive_success(self):
        session = AgentSession()
        session.record_turn({"user_query": "q1", "final_output": "ok"})
        session.record_turn({"user_query": "q2", "final_output": "ok"})
        assert session.summary()["total_recoveries"] == 0

    def test_record_recovery_event_manual(self):
        session = AgentSession()
        session.record_recovery_event("manual_reset", "Operator intervention")
        summary = session.summary()
        assert summary["total_recoveries"] == 0  # manual != auto
        assert len(summary["recent_recoveries"]) == 1
        assert summary["recent_recoveries"][0]["type"] == "manual_reset"


class TestAgentSessionHealth:
    def test_health_empty_is_perfect(self):
        session = AgentSession()
        assert session.health_score() == 1.0

    def test_health_all_success_no_anomalies(self):
        session = AgentSession()
        for _ in range(10):
            session.record_turn({"user_query": "q", "final_output": "ok"})
        hs = session.health_score()
        # 60% success + 40% (1 - 0 anomaly) = 0.6 + 0.4 = 1.0
        assert hs == 1.0

    def test_health_all_failures(self):
        session = AgentSession()
        for _ in range(10):
            session.record_turn({"user_query": "q", "final_output": None})
        hs = session.health_score()
        # 60% * 0 + 40% * 1.0 = 0.4
        assert hs == 0.4

    def test_health_all_anomalies(self):
        session = AgentSession()
        for _ in range(10):
            session.record_turn({
                "user_query": "q",
                "final_output": "ok",
                "anomalies": [{"status": "unhealthy", "reason": "bug"}],
            })
        hs = session.health_score()
        # 60% * 1.0 + 40% * (1 - 1.0) = 0.6
        assert hs == 0.6

    def test_health_mixed_state(self):
        session = AgentSession()
        # 7 successes, 3 failures, 2 anomalies (among successes)
        for _ in range(7):
            session.record_turn({"user_query": "q", "final_output": "ok"})
        for _ in range(3):
            session.record_turn({"user_query": "q", "final_output": None})
        # Add some anomalies to successful turns
        session.record_turn({
            "user_query": "q",
            "final_output": "ok",
            "anomalies": [{"status": "unhealthy", "reason": "bug"}],
        })
        hs = session.health_score()
        # 11 turns in window: 8 success, 3 fail, 1 anomaly
        # success_ratio = 8/11 ≈ 0.727, anomaly_ratio = 1/11 ≈ 0.091
        # 0.6 * (8/11) + 0.4 * (1 - 1/11) = 0.436 + 0.364 = 0.8
        assert hs == 0.8

    def test_health_respects_window(self):
        session = AgentSession()
        # 25 turns: first 5 failures, 20 successes
        for _ in range(5):
            session.record_turn({"user_query": "q", "final_output": None})
        for _ in range(20):
            session.record_turn({"user_query": "q", "final_output": "ok"})
        # window=20: should only see successes
        assert session.health_score(window=20) == 1.0
        # default window=20
        assert session.health_score() == 1.0

    def test_anomaly_rate_empty(self):
        session = AgentSession()
        assert session.anomaly_rate() == 0.0

    def test_anomaly_rate_half(self):
        session = AgentSession()
        for i in range(10):
            anomalies = [{"status": "unhealthy"}] if i % 2 == 0 else []
            session.record_turn({
                "user_query": "q",
                "final_output": "ok",
                "anomalies": anomalies,
            })
        assert session.anomaly_rate() == 0.5


class TestAgentSessionSummary:
    def test_summary_returns_all_keys(self):
        session = AgentSession()
        s = session.summary()
        expected_keys = {
            "session_id", "uptime_seconds", "total_turns",
            "consecutive_failures", "total_recoveries",
            "health_score", "anomaly_rate",
            "recent_recoveries", "last_activity",
        }
        assert set(s.keys()) == expected_keys

    def test_summary_uptime_increases(self):
        session = AgentSession()
        s1 = session.summary()
        time.sleep(0.2)
        s2 = session.summary()
        assert s2["uptime_seconds"] > s1["uptime_seconds"]

    def test_recent_turns_returns_n_entries(self):
        session = AgentSession()
        for i in range(10):
            session.record_turn({"user_query": str(i), "final_output": "ok"})
        recent = session.recent_turns(3)
        assert len(recent) == 3


class TestAgentSessionPersistence:
    def test_save_creates_file(self, tmp_path):
        original_dir = SESSIONS_DIR
        try:
            import core.agent_session as mod
            mod.SESSIONS_DIR = str(tmp_path)
            session = AgentSession(session_id="save_test")
            path = session.save()
            assert os.path.exists(path)
            with open(path) as f:
                data = json.load(f)
            assert data["session_id"] == "save_test"
        finally:
            mod.SESSIONS_DIR = original_dir

    def test_save_and_load_roundtrip(self, tmp_path):
        import core.agent_session as mod
        original_dir = mod.SESSIONS_DIR
        try:
            mod.SESSIONS_DIR = str(tmp_path)
            session = AgentSession(session_id="roundtrip")
            session.record_turn({"user_query": "q1", "final_output": "a1"})
            session.record_turn({"user_query": "q2", "final_output": None})
            session.record_recovery_event("test", "test event")
            session.save()

            loaded = AgentSession.load("roundtrip")
            assert loaded is not None
            assert loaded.session_id == "roundtrip"
            sl = loaded.summary()
            assert sl["total_turns"] == 2
            assert sl["consecutive_failures"] == 1
            assert sl["total_recoveries"] == 0  # manual recovery, not auto
        finally:
            mod.SESSIONS_DIR = original_dir

    def test_load_nonexistent_returns_none(self):
        result = AgentSession.load("nonexistent_session_id")
        assert result is None

    def test_list_sessions(self, tmp_path):
        import core.agent_session as mod
        original_dir = mod.SESSIONS_DIR
        try:
            mod.SESSIONS_DIR = str(tmp_path)
            AgentSession(session_id="s1").save()
            AgentSession(session_id="s2").save()
            sessions = AgentSession.list_sessions()
            assert len(sessions) == 2
            session_ids = [s["session_id"] for s in sessions]
            assert "s1" in session_ids
            assert "s2" in session_ids
        finally:
            mod.SESSIONS_DIR = original_dir

    def test_list_sessions_empty(self, tmp_path):
        import core.agent_session as mod
        original_dir = mod.SESSIONS_DIR
        try:
            mod.SESSIONS_DIR = str(tmp_path)
            assert AgentSession.list_sessions() == []
        finally:
            mod.SESSIONS_DIR = original_dir


class TestAgentSessionGlobals:
    def test_get_session_returns_singleton(self):
        s1 = get_session()
        s2 = get_session()
        assert s1 is s2

    def test_reset_session_creates_new(self):
        s1 = get_session()
        s2 = reset_session()
        assert s1 is not s2
        assert s2.health_score() == 1.0

    def test_reset_session_changes_global(self):
        reset_session()
        s1 = get_session()
        reset_session()
        s2 = get_session()
        assert s1 is not s2
