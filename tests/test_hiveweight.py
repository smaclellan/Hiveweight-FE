"""Tests for hiveweight.py — pure logic, output writers, and mocked API client."""

import csv
import json
from datetime import date, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from hiveweight import (
    DEFAULT_CONFIG,
    AuthenticationError,
    _synthetic_events,
    build_daily_activity,
    detect_flags,
    generate_terminal_sparkline,
    gh_get,
    is_holiday,
    is_within_sabbatical,
    load_config,
    rolling_average,
    write_csv,
    write_json_summary,
    write_report,
)


# ── helpers ───────────────────────────────────────────────────────────────────

def _cfg(**overrides):
    cfg = {
        "weights": {
            "commit": 1.0,
            "merged_pr": 3.0,
            "issue_opened": 0.5,
            "issue_comment": 0.3,
        },
        "dampening": {"weekends": True, "weekend_multiplier": 0.5, "holidays": True},
        "known_holidays": ["07-04"],
    }
    cfg.update(overrides)
    return cfg


def _make_daily(
    start: date,
    n_days: int,
    base_score: float,
    drop_start: int | None = None,
    drop_score: float = 0.0,
) -> dict[date, float]:
    """n_days of base_score, optionally dropping to drop_score from drop_start."""
    daily = {}
    for i in range(n_days):
        d = start + timedelta(days=i)
        daily[d] = drop_score if (drop_start is not None and i >= drop_start) else base_score
    return daily


_DETECT_CFG = {
    "baseline_days": 30,
    "drop_threshold_pct": 35,
    "sustained_days": 3,
    "rolling_window_days": 7,
    "pressure_escalation_ratio": 2.5,
    "seasonal_sabbaticals": [],
    "min_baseline_threshold": 0.1,
}

START = date(2026, 1, 1)


def _run_flags(
    outbound_daily: dict,
    cfg: dict = _DETECT_CFG,
    inbound_daily: dict | None = None,
) -> tuple:
    """Run detect_flags with zero inbound unless specified."""
    if inbound_daily is None:
        inbound_daily = {d: 0.0 for d in outbound_daily}
    rolling = rolling_average(outbound_daily, cfg["rolling_window_days"])
    return detect_flags(outbound_daily, rolling, inbound_daily, cfg)


# ── load_config ───────────────────────────────────────────────────────────────

class TestLoadConfig:
    def test_returns_defaults_when_no_path(self):
        cfg = load_config(None)
        assert cfg["drop_threshold_pct"] == 35
        assert cfg["sustained_days"] == 3
        assert cfg["rolling_window_days"] == 7

    def test_returns_defaults_for_missing_file(self, tmp_path):
        cfg = load_config(str(tmp_path / "nonexistent.json"))
        assert cfg["drop_threshold_pct"] == DEFAULT_CONFIG["drop_threshold_pct"]

    def test_overrides_top_level_keys(self, tmp_path):
        p = tmp_path / "cfg.json"
        p.write_text(json.dumps({"drop_threshold_pct": 50, "sustained_days": 5}))
        cfg = load_config(str(p))
        assert cfg["drop_threshold_pct"] == 50
        assert cfg["sustained_days"] == 5
        assert cfg["rolling_window_days"] == 7  # default preserved

    def test_deep_merges_weights(self, tmp_path):
        p = tmp_path / "cfg.json"
        p.write_text(json.dumps({"weights": {"merged_pr": 5.0}}))
        cfg = load_config(str(p))
        assert cfg["weights"]["merged_pr"] == 5.0
        assert cfg["weights"]["commit"] == 1.0        # default preserved
        assert cfg["weights"]["issue_opened"] == 0.5  # default preserved

    def test_deep_merges_dampening(self, tmp_path):
        p = tmp_path / "cfg.json"
        p.write_text(json.dumps({"dampening": {"weekend_multiplier": 0.3}}))
        cfg = load_config(str(p))
        assert cfg["dampening"]["weekend_multiplier"] == 0.3
        assert cfg["dampening"]["weekends"] is True  # default preserved


# ── is_holiday ────────────────────────────────────────────────────────────────

class TestIsHoliday:
    HOLIDAYS = ["01-01", "07-04", "12-25"]

    def test_known_holiday_matches(self):
        assert is_holiday(date(2026, 7, 4), self.HOLIDAYS) is True

    def test_new_years_matches(self):
        assert is_holiday(date(2026, 1, 1), self.HOLIDAYS) is True

    def test_non_holiday_returns_false(self):
        assert is_holiday(date(2026, 3, 15), self.HOLIDAYS) is False

    def test_empty_holiday_list_returns_false(self):
        assert is_holiday(date(2026, 7, 4), []) is False


# ── build_daily_activity ──────────────────────────────────────────────────────

class TestBuildDailyActivity:
    def test_basic_weighting(self):
        d = date(2026, 6, 2)  # Tuesday — no dampening
        events = [
            {"date": d, "type": "commit", "id": "a"},
            {"date": d, "type": "merged_pr", "id": "1"},
        ]
        outbound, inbound = build_daily_activity(events, d, d, _cfg())
        assert outbound[d] == round(1.0 + 3.0, 3)

    def test_issue_opened_goes_to_inbound(self):
        d = date(2026, 6, 2)
        events = [{"date": d, "type": "issue_opened", "id": "1"}]
        outbound, inbound = build_daily_activity(events, d, d, _cfg())
        assert inbound[d] == 0.5   # issue_opened weight
        assert outbound[d] == 0.0  # nothing in outbound

    def test_zero_score_for_quiet_day(self):
        d = date(2026, 6, 2)
        outbound, inbound = build_daily_activity([], d, d, _cfg())
        assert outbound[d] == 0.0
        assert inbound[d] == 0.0

    def test_fills_all_days_in_range(self):
        start = date(2026, 6, 1)
        end = date(2026, 6, 7)
        outbound, inbound = build_daily_activity([], start, end, _cfg())
        assert len(outbound) == 7
        assert len(inbound) == 7

    def test_weekend_dampening_halves_score(self):
        sat = date(2026, 6, 6)  # Saturday
        events = [{"date": sat, "type": "commit", "id": "x"}]
        outbound, inbound = build_daily_activity(events, sat, sat, _cfg())
        assert outbound[sat] == round(1.0 * 0.5, 3)

    def test_weekday_not_dampened(self):
        mon = date(2026, 6, 2)  # Monday
        events = [{"date": mon, "type": "commit", "id": "x"}]
        outbound, inbound = build_daily_activity(events, mon, mon, _cfg())
        assert outbound[mon] == 1.0

    def test_holiday_dampening_on_weekday(self):
        # 2026-01-01 is a Thursday — weekday holiday
        day = date(2026, 1, 1)
        cfg = _cfg()
        cfg["known_holidays"] = ["01-01"]
        events = [{"date": day, "type": "commit", "id": "x"}]
        outbound, inbound = build_daily_activity(events, day, day, cfg)
        assert outbound[day] == round(1.0 * 0.5, 3)

    def test_no_double_dampening_on_weekend_holiday(self):
        # 2026-07-04 is a Saturday AND a holiday — must get 0.5×, not 0.25×
        day = date(2026, 7, 4)
        cfg = _cfg()
        cfg["known_holidays"] = ["07-04"]
        events = [{"date": day, "type": "commit", "id": "x"}]
        outbound, inbound = build_daily_activity(events, day, day, cfg)
        assert outbound[day] == round(1.0 * 0.5, 3)
        assert outbound[day] != round(1.0 * 0.25, 3)

    def test_unknown_event_type_is_dropped(self):
        # Unknown types (not commit/merged_pr/issue_opened/issue_comment) are silently ignored
        d = date(2026, 6, 2)
        events = [{"date": d, "type": "review_comment", "id": "x"}]
        outbound, inbound = build_daily_activity(events, d, d, _cfg())
        assert outbound[d] == 0.0
        assert inbound[d] == 0.0


# ── rolling_average ───────────────────────────────────────────────────────────

class TestRollingAverage:
    def test_first_window_minus_one_entries_are_none(self):
        daily = {START + timedelta(days=i): float(i + 1) for i in range(10)}
        result = rolling_average(daily, window=7)
        for i in range(6):
            assert result[START + timedelta(days=i)] is None

    def test_correct_value_at_window_boundary(self):
        # Days 0-6 with values 1-7; 7-day avg at day 6 = (1+2+3+4+5+6+7)/7 = 4.0
        daily = {START + timedelta(days=i): float(i + 1) for i in range(10)}
        result = rolling_average(daily, window=7)
        assert result[START + timedelta(days=6)] == pytest.approx(4.0)

    def test_window_of_one_equals_input(self):
        daily = {START + timedelta(days=i): float(i) for i in range(5)}
        result = rolling_average(daily, window=1)
        for d, v in daily.items():
            assert result[d] == v

    def test_advances_correctly_past_boundary(self):
        # Uniform values → rolling avg always equals the value
        daily = {START + timedelta(days=i): 3.0 for i in range(20)}
        result = rolling_average(daily, window=7)
        for i in range(6, 20):
            assert result[START + timedelta(days=i)] == pytest.approx(3.0)


# ── detect_flags ──────────────────────────────────────────────────────────────

class TestDetectFlags:
    def test_no_flags_on_steady_activity(self):
        outbound = _make_daily(START, 90, base_score=5.0)
        flags, baseline, _ = _run_flags(outbound)
        assert flags == []
        assert baseline == pytest.approx(5.0)

    def test_triggers_alert_on_sustained_drop(self):
        # 30 days normal, then zero for 60 days
        outbound = _make_daily(START, 90, base_score=5.0, drop_start=30, drop_score=0.0)
        flags, _, _ = _run_flags(outbound)
        assert len(flags) >= 1
        assert flags[0]["status"] == "ALERT"
        assert flags[0]["drop_pct"] > 35

    def test_no_flag_for_drop_shorter_than_sustained_days(self):
        outbound = _make_daily(START, 90, base_score=5.0)
        # Inject a 2-day dip (less than sustained_days=3) after baseline period
        for i in range(30, 32):
            outbound[START + timedelta(days=i)] = 0.0
        flags, _, _ = _run_flags(outbound)
        assert flags == []

    def test_flag_streak_extends_beyond_sustained_days(self):
        outbound = _make_daily(START, 90, base_score=5.0, drop_start=30, drop_score=0.0)
        flags, _, _ = _run_flags(outbound)
        assert len(flags) == 1
        assert flags[0]["streak_days"] > _DETECT_CFG["sustained_days"]

    def test_baseline_is_mean_of_first_30_days(self):
        outbound = _make_daily(START, 90, base_score=4.0)
        _, baseline, _ = _run_flags(outbound)
        assert baseline == pytest.approx(4.0)

    def test_alert_threshold_is_baseline_times_factor(self):
        outbound = _make_daily(START, 90, base_score=4.0)
        _, baseline, threshold = _run_flags(outbound)
        expected = baseline * (1 - _DETECT_CFG["drop_threshold_pct"] / 100)
        assert threshold == pytest.approx(expected, rel=1e-3)

    def test_insufficient_data_returns_empty_tuple(self):
        # Only 10 days — less than baseline_days=30; must return a 3-tuple, not []
        outbound = _make_daily(START, 10, base_score=5.0)
        result = _run_flags(outbound)
        flags, baseline, threshold = result  # must unpack without ValueError
        assert flags == []
        assert baseline == 0.0
        assert threshold == 0.0

    def test_exactly_at_threshold_does_not_flag(self):
        # Drop to exactly 65% of baseline (= threshold) — should NOT trigger (must be below)
        outbound = _make_daily(START, 90, base_score=5.0)
        threshold_score = 5.0 * (1 - 35 / 100)  # 3.25
        for i in range(30, 60):
            outbound[START + timedelta(days=i)] = threshold_score
        flags, _, _ = _run_flags(outbound)
        assert flags == []


# ── output writers ────────────────────────────────────────────────────────────

class TestWriteCSV:
    def test_creates_file_with_correct_header(self, tmp_path):
        daily = {START + timedelta(days=i): float(i) for i in range(3)}
        rolling = rolling_average(daily, 7)
        path = tmp_path / "activity.csv"
        write_csv(daily, rolling, path)
        rows = list(csv.reader(path.open()))
        assert rows[0] == ["date", "weighted_events", "rolling_avg_7d"]

    def test_row_count_matches_day_count(self, tmp_path):
        daily = {START + timedelta(days=i): float(i) for i in range(5)}
        rolling = rolling_average(daily, 7)
        path = tmp_path / "activity.csv"
        write_csv(daily, rolling, path)
        rows = list(csv.reader(path.open()))
        assert len(rows) == 6  # header + 5 data rows

    def test_values_written_correctly(self, tmp_path):
        d = date(2026, 6, 1)
        daily = {d: 3.5}
        rolling = {d: 3.5}
        path = tmp_path / "activity.csv"
        write_csv(daily, rolling, path)
        rows = list(csv.reader(path.open()))
        assert rows[1][0] == str(d)
        assert float(rows[1][1]) == pytest.approx(3.5)


class TestWriteReport:
    def test_creates_file(self, tmp_path):
        path = tmp_path / "report.txt"
        write_report("owner/repo", [], 5.0, 3.25, DEFAULT_CONFIG, path)
        assert path.exists()

    def test_no_flags_message_present(self, tmp_path):
        path = tmp_path / "report.txt"
        write_report("owner/repo", [], 5.0, 3.25, DEFAULT_CONFIG, path)
        assert "No sustained drops" in path.read_text()

    def test_flags_appear_in_report(self, tmp_path):
        path = tmp_path / "report.txt"
        flags = [{
            "start": date(2026, 5, 1), "end": date(2026, 5, 5),
            "streak_days": 5, "rolling_avg": 1.2, "drop_pct": 76.0, "status": "ALERT",
        }]
        write_report("owner/repo", flags, 5.0, 3.25, DEFAULT_CONFIG, path)
        text = path.read_text()
        assert "Flag #1" in text
        assert "ALERT" in text
        assert "76.0" in text

    def test_repo_name_in_report(self, tmp_path):
        path = tmp_path / "report.txt"
        write_report("myorg/myrepo", [], 5.0, 3.25, DEFAULT_CONFIG, path)
        assert "myorg/myrepo" in path.read_text()


class TestWriteJsonSummary:
    def test_creates_valid_json(self, tmp_path):
        path = tmp_path / "summary.json"
        daily = {START + timedelta(days=i): 5.0 for i in range(30)}
        write_json_summary("owner/repo", [], 5.0, daily, DEFAULT_CONFIG, path)
        data = json.loads(path.read_text())
        assert data["repo"] == "owner/repo"

    def test_ok_status_when_no_flags(self, tmp_path):
        path = tmp_path / "summary.json"
        daily = {START + timedelta(days=i): 5.0 for i in range(30)}
        write_json_summary("owner/repo", [], 5.0, daily, DEFAULT_CONFIG, path)
        data = json.loads(path.read_text())
        assert data["status"] == "OK"

    def test_alert_status_when_flags_present(self, tmp_path):
        path = tmp_path / "summary.json"
        daily = {START + timedelta(days=i): 5.0 for i in range(30)}
        flags = [{"start": date(2026, 5, 1), "end": date(2026, 5, 5), "streak_days": 5, "drop_pct": 76.0, "rolling_avg": 1.2, "pressure_index": 0.5, "status": "ALERT"}]
        write_json_summary("owner/repo", flags, 5.0, daily, DEFAULT_CONFIG, path)
        data = json.loads(path.read_text())
        assert data["status"] == "ALERT"
        assert len(data["flags"]) == 1

    def test_critical_alert_status_when_critical_flag_present(self, tmp_path):
        path = tmp_path / "summary.json"
        daily = {START + timedelta(days=i): 5.0 for i in range(30)}
        flags = [{"start": date(2026, 5, 1), "end": date(2026, 5, 5), "streak_days": 5, "drop_pct": 76.0, "rolling_avg": 1.2, "pressure_index": 3.5, "status": "CRITICAL ALERT"}]
        write_json_summary("owner/repo", flags, 5.0, daily, DEFAULT_CONFIG, path)
        data = json.loads(path.read_text())
        assert data["status"] == "CRITICAL ALERT"

    def test_flag_dates_serialized_as_strings(self, tmp_path):
        path = tmp_path / "summary.json"
        daily = {START + timedelta(days=i): 5.0 for i in range(30)}
        flags = [{"start": date(2026, 5, 1), "end": date(2026, 5, 5), "streak_days": 5, "drop_pct": 76.0, "rolling_avg": 1.2, "pressure_index": 0.5, "status": "ALERT"}]
        write_json_summary("owner/repo", flags, 5.0, daily, DEFAULT_CONFIG, path)
        data = json.loads(path.read_text())
        assert isinstance(data["flags"][0]["start"], str)
        assert data["flags"][0]["start"] == "2026-05-01"

    def test_baseline_and_total_events_present(self, tmp_path):
        path = tmp_path / "summary.json"
        daily = {START + timedelta(days=i): 2.0 for i in range(10)}
        write_json_summary("owner/repo", [], 2.0, daily, DEFAULT_CONFIG, path)
        data = json.loads(path.read_text())
        assert data["baseline_daily_avg"] == pytest.approx(2.0)
        assert data["total_weighted_events"] == pytest.approx(20.0)


# ── _synthetic_events ─────────────────────────────────────────────────────────

class TestSyntheticEvents:
    START = date(2026, 1, 1)
    END = date(2026, 3, 31)

    def test_all_events_within_date_range(self):
        events = _synthetic_events(self.START, self.END)
        for e in events:
            assert self.START <= e["date"] <= self.END

    def test_event_types_are_valid(self):
        valid = {"commit", "merged_pr", "issue_opened", "issue_comment"}
        for e in _synthetic_events(self.START, self.END):
            assert e["type"] in valid

    def test_produces_events(self):
        assert len(_synthetic_events(self.START, self.END)) > 0

    def test_deterministic_output(self):
        # random.seed(42) in _synthetic_events guarantees reproducibility
        assert _synthetic_events(self.START, self.END) == _synthetic_events(self.START, self.END)

    def test_drop_simulated_near_end(self):
        # The last 10 days should have lower average count than earlier days
        start = date(2026, 1, 1)
        end = date(2026, 6, 1)
        events = _synthetic_events(start, end)
        cutoff = end - timedelta(days=10)
        early_count = sum(1 for e in events if e["date"] < cutoff)
        late_count = sum(1 for e in events if e["date"] >= cutoff)
        days_early = (cutoff - start).days
        days_late = (end - cutoff).days + 1
        assert early_count / days_early > late_count / days_late


# ── gh_get (mocked) ───────────────────────────────────────────────────────────

def _mock_response(status: int, body=None, headers: dict | None = None) -> MagicMock:
    resp = MagicMock()
    resp.status_code = status
    resp.headers = headers or {}
    if body is not None:
        resp.json.return_value = body
    resp.raise_for_status = MagicMock()
    return resp


class TestGhGet:
    def test_raises_authentication_error_on_401(self):
        with patch("hiveweight.requests.get", return_value=_mock_response(401)):
            with pytest.raises(AuthenticationError):
                gh_get("/repos/foo/bar/commits", token="bad_token")

    def test_returns_none_on_304_not_modified(self):
        with patch("hiveweight.requests.get", return_value=_mock_response(304)):
            result = gh_get("/repos/foo/bar/commits", token=None, use_etag=False)
        assert result is None

    def test_returns_empty_list_on_404(self):
        with patch("hiveweight.requests.get", return_value=_mock_response(404)):
            result = gh_get("/repos/nonexistent/repo", token=None)
        assert result == []

    def test_returns_list_of_items(self):
        body = [{"id": 1}, {"id": 2}]
        with patch("hiveweight.requests.get", return_value=_mock_response(200, body)):
            result = gh_get("/repos/foo/bar/commits", token=None, use_etag=False)
        assert result == body

    def test_returns_dict_for_single_object(self):
        body = {"rate": {"remaining": 4999, "limit": 5000}}
        with patch("hiveweight.requests.get", return_value=_mock_response(200, body)):
            result = gh_get("/rate_limit", token=None, use_etag=False)
        assert result == body

    def test_follows_link_header_pagination(self):
        page1 = _mock_response(
            200,
            [{"id": 1}],
            {"Link": '<https://api.github.com/repos/foo/bar/commits?page=2>; rel="next"'},
        )
        page2 = _mock_response(200, [{"id": 2}])
        with patch("hiveweight.requests.get", side_effect=[page1, page2]):
            result = gh_get("/repos/foo/bar/commits", token=None, use_etag=False)
        assert result == [{"id": 1}, {"id": 2}]

    def test_caches_etag_from_response(self):
        import hiveweight
        resp = _mock_response(200, [{"id": 1}], {"ETag": '"abc123"'})
        with patch("hiveweight.requests.get", return_value=resp):
            gh_get("/repos/foo/bar/commits", token=None)
        assert hiveweight.ETAG_CACHE.get("/repos/foo/bar/commits") == '"abc123"'

    def test_sends_if_none_match_when_etag_cached(self):
        import hiveweight
        hiveweight.ETAG_CACHE["/repos/foo/bar/commits"] = '"abc123"'
        resp = _mock_response(200, [])
        with patch("hiveweight.requests.get", return_value=resp) as mock_get:
            gh_get("/repos/foo/bar/commits", token=None)
        call_headers = mock_get.call_args.kwargs["headers"]
        assert call_headers["If-None-Match"] == '"abc123"'


# ── detect_flags — stale repo ─────────────────────────────────────────────────

class TestDetectFlagsStale:
    def test_completely_inactive_repo_flagged_as_stale(self):
        outbound = _make_daily(START, 90, base_score=0.0)
        cfg = {**_DETECT_CFG, "min_baseline_threshold": 0.1}
        flags, baseline, _ = _run_flags(outbound, cfg)
        assert len(flags) == 1
        assert flags[0]["status"] == "STALE"
        assert flags[0]["drop_pct"] == 100.0
        assert baseline == pytest.approx(0.0)

    def test_stale_flag_covers_post_baseline_period(self):
        outbound = _make_daily(START, 90, base_score=0.0)
        cfg = {**_DETECT_CFG, "min_baseline_threshold": 0.1}
        flags, _, _ = _run_flags(outbound, cfg)
        assert flags[0]["start"] == START + timedelta(days=30)
        assert flags[0]["end"] == START + timedelta(days=89)
        assert flags[0]["streak_days"] == 60

    def test_active_repo_not_flagged_as_stale(self):
        outbound = _make_daily(START, 90, base_score=5.0)
        cfg = {**_DETECT_CFG, "min_baseline_threshold": 0.1}
        flags, _, _ = _run_flags(outbound, cfg)
        assert all(f["status"] != "STALE" for f in flags)

    def test_stale_check_disabled_by_zero_threshold(self):
        # Setting min_baseline_threshold=0.0 disables the STALE check entirely
        outbound = _make_daily(START, 90, base_score=0.0)
        cfg = {**_DETECT_CFG, "min_baseline_threshold": 0.0}
        flags, _, _ = _run_flags(outbound, cfg)
        assert all(f.get("status") != "STALE" for f in flags)


# ── is_holiday — calendar integration ────────────────────────────────────────

class TestIsHolidayCalendars:
    def test_static_list_still_works_without_calendars(self):
        assert is_holiday(date(2026, 7, 4), ["07-04"]) is True

    def test_none_calendars_does_not_raise(self):
        assert is_holiday(date(2026, 3, 15), [], None) is False

    def test_global_calendar_code_is_skipped_gracefully(self):
        # "global" is not a valid country code — must not raise
        result = is_holiday(date(2026, 3, 15), [], ["global"])
        assert result is False

    def test_static_list_takes_priority_over_calendar_lookup(self):
        # Even if the holidays lib would return False, the static list wins
        assert is_holiday(date(2026, 7, 4), ["07-04"], ["US"]) is True

    def test_dynamic_lookup_used_when_holidays_installed(self):
        # Mock the holidays library to confirm the lookup path is called
        mock_holidays_lib = MagicMock()
        mock_country = MagicMock()
        mock_country.__contains__ = MagicMock(return_value=True)
        mock_holidays_lib.country_holidays.return_value = mock_country
        with patch.dict("sys.modules", {"holidays": mock_holidays_lib}):
            result = is_holiday(date(2026, 6, 15), [], ["US"])
        assert result is True
        mock_holidays_lib.country_holidays.assert_called_once_with("US", years=2026)

    def test_import_error_falls_back_gracefully(self):
        # If the holidays library is not installed, fall back silently
        with patch.dict("sys.modules", {"holidays": None}):
            result = is_holiday(date(2026, 6, 15), [], ["US"])
        assert result is False


# ── generate_terminal_sparkline ───────────────────────────────────────────────

def _rolling(scores: list[float], start: date = START) -> dict:
    """Build a rolling-avg-shaped dict (no Nones) from a plain list of floats."""
    return {start + timedelta(days=i): v for i, v in enumerate(scores)}


def _extract_spark(result: str) -> str:
    """Pull just the sparkline chars out of '[spark] (mm-dd → mm-dd)'."""
    return result[1:result.index("]")]


class TestGenerateTerminalSparkline:
    def test_no_data_returns_placeholder(self):
        data = {START + timedelta(days=i): None for i in range(10)}
        assert generate_terminal_sparkline(data) == "[ No Data Available ]"

    def test_output_format_has_brackets_and_date_range(self):
        data = _rolling([1.0, 2.0, 3.0, 4.0, 5.0])
        result = generate_terminal_sparkline(data)
        assert result.startswith("[")
        assert "]" in result
        assert "→" in result
        assert result.endswith(")")

    def test_sparkline_length_equals_data_point_count(self):
        data = _rolling([float(i) for i in range(10)])
        result = generate_terminal_sparkline(data, window_days=21)
        assert len(_extract_spark(result)) == 10

    def test_width_limits_to_last_n_days(self):
        data = _rolling([float(i) for i in range(30)])
        result = generate_terminal_sparkline(data, window_days=10)
        assert len(_extract_spark(result)) == 10

    def test_rising_trend_ends_with_full_block(self):
        # Maximum value is last → must map to highest block '█'
        data = _rolling([float(i) for i in range(10)])
        result = generate_terminal_sparkline(data)
        assert _extract_spark(result)[-1] == "█"

    def test_falling_trend_starts_with_full_block(self):
        # Maximum value is first → must map to '█'
        data = _rolling([float(10 - i) for i in range(10)])
        result = generate_terminal_sparkline(data)
        assert _extract_spark(result)[0] == "█"

    def test_output_chars_are_valid_blocks(self):
        valid = {" ", "▂", "▃", "▄", "▅", "▆", "▇", "█"}
        data = _rolling([float(i % 8) for i in range(20)])
        for ch in _extract_spark(generate_terminal_sparkline(data)):
            assert ch in valid

    def test_flat_data_all_spaces(self):
        # Flat data: min == max → v_range forced to 1.0 → all indices 0 → spaces
        data = _rolling([5.0] * 10)
        assert _extract_spark(generate_terminal_sparkline(data)) == " " * 10

    def test_date_range_in_output_matches_data(self):
        start = date(2026, 3, 1)
        end = date(2026, 3, 10)
        data = {start + timedelta(days=i): float(i + 1) for i in range(10)}
        result = generate_terminal_sparkline(data)
        assert "03-01" in result
        assert "03-10" in result

    def test_width_date_range_reflects_last_n_days(self):
        # 30 days of data, width=5 — date range should be last 5 days only
        start = date(2026, 1, 1)
        data = {start + timedelta(days=i): float(i) for i in range(30)}
        result = generate_terminal_sparkline(data, window_days=5)
        assert "01-26" in result  # day 25 (0-indexed) = Jan 26
        assert "01-30" in result  # day 29 (0-indexed) = Jan 30


# ── is_within_sabbatical ──────────────────────────────────────────────────────

class TestIsWithinSabbatical:
    SABBATICALS = [
        {"name": "Winter Freeze", "start_mm_dd": "12-15", "end_mm_dd": "01-05"},
        {"name": "Summer Break",  "start_mm_dd": "07-01", "end_mm_dd": "07-15"},
    ]

    def test_date_inside_standard_window(self):
        assert is_within_sabbatical(date(2026, 7, 8), self.SABBATICALS) is True

    def test_date_before_standard_window(self):
        assert is_within_sabbatical(date(2026, 6, 30), self.SABBATICALS) is False

    def test_date_after_standard_window(self):
        assert is_within_sabbatical(date(2026, 7, 16), self.SABBATICALS) is False

    def test_date_at_window_start_boundary(self):
        assert is_within_sabbatical(date(2026, 7, 1), self.SABBATICALS) is True

    def test_date_at_window_end_boundary(self):
        assert is_within_sabbatical(date(2026, 7, 15), self.SABBATICALS) is True

    def test_cross_year_window_in_december(self):
        assert is_within_sabbatical(date(2026, 12, 20), self.SABBATICALS) is True

    def test_cross_year_window_in_january(self):
        assert is_within_sabbatical(date(2026, 1, 3), self.SABBATICALS) is True

    def test_empty_sabbaticals_returns_false(self):
        assert is_within_sabbatical(date(2026, 7, 8), []) is False

    def test_invalid_config_skipped_gracefully(self):
        bad = [{"name": "Bad", "start_mm_dd": "bad-date", "end_mm_dd": "01-05"}]
        assert is_within_sabbatical(date(2026, 1, 3), bad) is False


# ── detect_flags — pressure index ────────────────────────────────────────────

class TestDetectFlagsPressureIndex:
    def test_critical_alert_when_pressure_exceeds_ratio(self):
        # High inbound + zero outbound → pressure ratio exceeded → CRITICAL ALERT
        outbound = _make_daily(START, 90, base_score=5.0, drop_start=30, drop_score=0.0)
        inbound = _make_daily(START, 90, base_score=5.0)  # sustained high community demand
        cfg = {**_DETECT_CFG, "pressure_escalation_ratio": 2.5}
        flags, _, _ = _run_flags(outbound, cfg, inbound)
        assert len(flags) >= 1
        assert flags[0]["status"] == "CRITICAL ALERT"
        assert flags[0]["pressure_index"] >= 2.5

    def test_alert_when_pressure_below_ratio(self):
        # No inbound, outbound drops → low pressure → plain ALERT
        outbound = _make_daily(START, 90, base_score=5.0, drop_start=30, drop_score=0.0)
        inbound = _make_daily(START, 90, base_score=0.0)
        flags, _, _ = _run_flags(outbound, inbound_daily=inbound)
        assert len(flags) >= 1
        assert flags[0]["status"] == "ALERT"

    def test_flag_has_pressure_index_field(self):
        outbound = _make_daily(START, 90, base_score=5.0, drop_start=30, drop_score=0.0)
        inbound = _make_daily(START, 90, base_score=0.0)
        flags, _, _ = _run_flags(outbound, inbound_daily=inbound)
        assert len(flags) >= 1
        assert "pressure_index" in flags[0]
        assert isinstance(flags[0]["pressure_index"], float)


# ── detect_flags — seasonality ────────────────────────────────────────────────

class TestDetectFlagsSeasonality:
    def test_sabbatical_suppresses_moderate_drop(self):
        # 60% drop (score=2.0 vs baseline=5.0) during sabbatical:
        # normal threshold = 3.25 → would trigger
        # relaxed threshold = 5.0 * (1 - 0.70) = 1.5 → 2.0 > 1.5 → no flag
        start = date(2026, 1, 1)
        outbound = _make_daily(start, 90, base_score=5.0, drop_start=30, drop_score=2.0)
        inbound = _make_daily(start, 90, base_score=0.0)
        sabbaticals = [{"name": "Test", "start_mm_dd": "01-31", "end_mm_dd": "12-31"}]
        cfg = {**_DETECT_CFG, "seasonal_sabbaticals": sabbaticals}
        rolling = rolling_average(outbound, cfg["rolling_window_days"])
        flags, _, _ = detect_flags(outbound, rolling, inbound, cfg)
        assert flags == []

    def test_same_drop_triggers_without_sabbatical(self):
        # The same 60% drop without sabbatical protection should still flag
        start = date(2026, 1, 1)
        outbound = _make_daily(start, 90, base_score=5.0, drop_start=30, drop_score=2.0)
        inbound = _make_daily(start, 90, base_score=0.0)
        cfg = {**_DETECT_CFG, "seasonal_sabbaticals": []}
        rolling = rolling_average(outbound, cfg["rolling_window_days"])
        flags, _, _ = detect_flags(outbound, rolling, inbound, cfg)
        assert len(flags) >= 1
