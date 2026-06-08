#!/usr/bin/env python3
"""
HiveWeight — GitHub Maintainer Burnout Detector
Rolling-baseline method: flags sustained 35%+ drops in weighted activity.

Usage:
    python hiveweight.py owner/repo [owner/repo ...]
    python hiveweight.py --config config.json
    python hiveweight.py owner/repo --token ghp_xxx
    python hiveweight.py owner/repo --dry-run
"""

import argparse
import csv
import json
import os
import sys
import time
from collections import defaultdict
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import requests

# ─────────────────────────────────────────────────────────────
# Exceptions
# ─────────────────────────────────────────────────────────────

class HiveWeightError(Exception):
    pass

class AuthenticationError(HiveWeightError):
    pass


# ─────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────

DEFAULT_CONFIG = {
    "drop_threshold_pct": 35,
    "sustained_days": 3,
    "rolling_window_days": 7,
    "baseline_days": 30,
    "lookback_days": 90,
    "weights": {
        "merged_pr":    3.0,
        "commit":       1.0,
        "issue_opened": 0.5,
        "issue_comment": 0.3,
    },
    "dampening": {
        "weekends": True,
        "weekend_multiplier": 0.5,
        "holidays": True,
        "holiday_calendar": ["US", "global"],
    },
    "known_holidays": [
        "01-01", "01-15", "02-19", "05-27", "06-19",
        "07-04", "09-02", "11-28", "11-29", "12-25", "12-26",
    ],
    "min_baseline_threshold": 0.1,
    "pressure_escalation_ratio": 2.5,
    "seasonal_sabbaticals": [],
    "output_dir": "./output",
}


def load_config(path: str | None) -> dict:
    cfg = DEFAULT_CONFIG.copy()
    if path and Path(path).exists():
        with open(path) as f:
            user_cfg = json.load(f)
        # Deep-merge weights and dampening
        for key in ("weights", "dampening"):
            if key in user_cfg:
                cfg[key] = {**cfg.get(key, {}), **user_cfg.pop(key)}
        cfg.update(user_cfg)
    return cfg


# ─────────────────────────────────────────────────────────────
# GitHub API client with ETag caching + rate-limit handling
# ─────────────────────────────────────────────────────────────

GITHUB_API = "https://api.github.com"
ETAG_CACHE: dict[str, str] = {}


class RateLimitError(Exception):
    def __init__(self, reset_at: int):
        self.reset_at = reset_at
        super().__init__(f"Rate limit hit. Resets at {datetime.fromtimestamp(reset_at)}")


def gh_get(
    path: str,
    token: str | None,
    params: dict | None = None,
    use_etag: bool = True,
) -> list | dict | None:
    """
    GET from GitHub API with:
      - Bearer auth if token provided
      - ETag / If-None-Match conditional requests (saves quota)
      - Automatic pagination (follows Link: rel="next")
      - Retry on 429 / secondary rate limit
    Returns None on 304 Not Modified (data unchanged).
    """
    headers = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"
    if use_etag and path in ETAG_CACHE:
        headers["If-None-Match"] = ETAG_CACHE[path]

    url = f"{GITHUB_API}{path}"
    all_items: list = []

    while url:
        for attempt in range(3):
            resp = requests.get(url, headers=headers, params=params, timeout=15)

            # Cache the ETag for future requests
            if "ETag" in resp.headers:
                ETAG_CACHE[path] = resp.headers["ETag"]

            if resp.status_code == 304:
                return None  # Not modified — skip processing
            if resp.status_code == 401:
                raise AuthenticationError("Authentication failed. Check your token.")
            if resp.status_code == 403:
                reset = int(resp.headers.get("X-RateLimit-Reset", time.time() + 60))
                raise RateLimitError(reset)
            if resp.status_code == 429:
                wait = int(resp.headers.get("Retry-After", 60))
                print(f"  ⏳ Secondary rate limit — waiting {wait}s…", file=sys.stderr)
                time.sleep(wait)
                continue
            if resp.status_code == 404:
                print(f"  ✗ Not found: {path}", file=sys.stderr)
                return []
            resp.raise_for_status()
            break
        else:
            raise RuntimeError(f"Failed after 3 retries: {url}")

        data = resp.json()
        if isinstance(data, list):
            all_items.extend(data)
        else:
            return data  # Single object (e.g. repo info)

        # Pagination
        link = resp.headers.get("Link", "")
        url = None
        params = None  # Don't resend params on paginated URLs
        for part in link.split(","):
            if 'rel="next"' in part:
                url = part.split(";")[0].strip().strip("<>")
                break

    return all_items


def remaining_quota(token: str | None) -> tuple[int, int]:
    """Returns (remaining, limit)."""
    data = gh_get("/rate_limit", token, use_etag=False)
    if data and "rate" in data:
        return data["rate"]["remaining"], data["rate"]["limit"]
    return -1, -1


# ─────────────────────────────────────────────────────────────
# Data fetching
# ─────────────────────────────────────────────────────────────

def fetch_events(
    owner: str,
    repo: str,
    token: str | None,
    since: datetime,
    cfg: dict,
) -> list[dict]:
    """
    Fetch commits, issues, PRs, and comments since `since`.
    Returns a list of {"date": date, "type": str, "sha"/"id": str} dicts.
    """
    events: list[dict] = []
    since_str = since.isoformat().replace("+00:00", "Z")

    print(f"  📡 Fetching commits…")
    commits = gh_get(
        f"/repos/{owner}/{repo}/commits",
        token,
        params={"since": since_str, "per_page": 100},
    )
    if commits:
        for c in commits:
            raw_date = (
                c.get("commit", {}).get("author", {}).get("date")
                or c.get("commit", {}).get("committer", {}).get("date")
            )
            if raw_date:
                d = datetime.fromisoformat(raw_date.replace("Z", "+00:00")).date()
                events.append({"date": d, "type": "commit", "id": c["sha"][:7]})

    print(f"  📡 Fetching pull requests…")
    prs = gh_get(
        f"/repos/{owner}/{repo}/pulls",
        token,
        params={"state": "closed", "per_page": 100, "sort": "updated", "direction": "desc"},
    )
    if prs:
        for pr in prs:
            if pr.get("merged_at"):
                d = datetime.fromisoformat(pr["merged_at"].replace("Z", "+00:00")).date()
                if d >= since.date():
                    events.append({"date": d, "type": "merged_pr", "id": str(pr["number"])})

    print(f"  📡 Fetching issues…")
    issues = gh_get(
        f"/repos/{owner}/{repo}/issues",
        token,
        params={"state": "all", "since": since_str, "per_page": 100},
    )
    if issues:
        for issue in issues:
            # Skip PRs (GitHub issues endpoint returns them too)
            if "pull_request" in issue:
                continue
            d = datetime.fromisoformat(issue["created_at"].replace("Z", "+00:00")).date()
            if d >= since.date():
                events.append({"date": d, "type": "issue_opened", "id": str(issue["number"])})

    print(f"  📡 Fetching issue comments…")
    comments = gh_get(
        f"/repos/{owner}/{repo}/issues/comments",
        token,
        params={"since": since_str, "per_page": 100},
    )
    if comments:
        for c in comments:
            d = datetime.fromisoformat(c["created_at"].replace("Z", "+00:00")).date()
            events.append({"date": d, "type": "issue_comment", "id": str(c["id"])})

    return events


# ─────────────────────────────────────────────────────────────
# Weighted daily activity
# ─────────────────────────────────────────────────────────────

def is_holiday(
    d: date,
    known_holidays: list[str],
    calendars: list[str] | None = None,
) -> bool:
    if d.strftime("%m-%d") in known_holidays:
        return True
    if not calendars:
        return False
    try:
        import holidays as holidays_lib
        for cal in calendars:
            if cal == "global":
                continue
            try:
                if d in holidays_lib.country_holidays(cal, years=d.year):
                    return True
            except (KeyError, NotImplementedError):
                pass
    except ImportError:
        pass
    return False


def build_daily_activity(
    events: list[dict],
    start: date,
    end: date,
    cfg: dict,
) -> tuple[dict[date, float], dict[date, float]]:
    """
    Aggregates events into separate Outbound (maintainer) and Inbound (community) tracks.
    Returns (outbound_daily, inbound_daily). Dampening is applied to outbound only.
    """
    weights: dict[str, float] = cfg["weights"]
    dampening: dict = cfg["dampening"]
    holidays: list[str] = cfg.get("known_holidays", [])
    calendars: list[str] | None = dampening.get("holiday_calendar")

    raw_outbound: dict[date, float] = defaultdict(float)
    raw_inbound: dict[date, float] = defaultdict(float)

    for ev in events:
        d = ev["date"]
        w = weights.get(ev["type"], 0.5)

        if ev["type"] in ["commit", "merged_pr"]:
            raw_outbound[d] += w
        elif ev["type"] == "issue_opened":
            raw_inbound[d] += w
        elif ev["type"] == "issue_comment":
            raw_outbound[d] += w

    outbound_daily: dict[date, float] = {}
    inbound_daily: dict[date, float] = {}

    current = start
    while current <= end:
        out_score = raw_outbound.get(current, 0.0)
        in_score = raw_inbound.get(current, 0.0)

        # Weekend or holiday dampening applied to maintainer output only
        if dampening.get("weekends") and current.weekday() >= 5:
            out_score *= dampening.get("weekend_multiplier", 0.5)
        elif dampening.get("holidays") and is_holiday(current, holidays, calendars):
            out_score *= dampening.get("weekend_multiplier", 0.5)

        outbound_daily[current] = round(out_score, 3)
        inbound_daily[current] = round(in_score, 3)
        current += timedelta(days=1)

    return outbound_daily, inbound_daily


# ─────────────────────────────────────────────────────────────
# Rolling average + flag detection
# ─────────────────────────────────────────────────────────────

def rolling_average(daily: dict[date, float], window: int) -> dict[date, float | None]:
    dates = sorted(daily)
    result: dict[date, float | None] = {}
    for i, d in enumerate(dates):
        if i < window - 1:
            result[d] = None
            continue
        window_vals = [daily[dates[j]] for j in range(i - window + 1, i + 1)]
        result[d] = round(sum(window_vals) / window, 3)
    return result


def generate_terminal_sparkline(rolling_data: dict, window_days: int = 21) -> str:
    """
    Generates a high-impact Unicode sparkline string of the last N days
    of rolling averages. Automatically scales to fit boundaries.
    """
    sorted_dates = sorted([d for d, v in rolling_data.items() if v is not None])
    recent_dates = sorted_dates[-window_days:] if len(sorted_dates) > window_days else sorted_dates

    vals = [rolling_data[d] for d in recent_dates]
    if not vals:
        return "[ No Data Available ]"

    min_v, max_v = min(vals), max(vals)
    v_range = max_v - min_v if max_v != min_v else 1.0

    # 8 discrete vertical intervals
    blocks = [' ', '▂', '▃', '▄', '▅', '▆', '▇', '█']

    spark = ""
    for v in vals:
        # Calculate proportional index mapping
        idx = int(((v - min_v) / v_range) * (len(blocks) - 1))
        spark += blocks[idx]

    return f"[{spark}] ({recent_dates[0].strftime('%m-%d')} → {recent_dates[-1].strftime('%m-%d')})"


def is_within_sabbatical(target_date: date, sabbaticals: list[dict]) -> bool:
    """Returns True if target_date falls within any configured sabbatical window."""
    for s in sabbaticals:
        try:
            start_mm_dd = s["start_mm_dd"]
            end_mm_dd = s["end_mm_dd"]
            start_m = int(start_mm_dd.split("-")[0])
            start_v = int(start_mm_dd.split("-")[1])
            end_m = int(end_mm_dd.split("-")[0])
            end_v = int(end_mm_dd.split("-")[1])
            year_val = target_date.year

            if start_m <= end_m:
                # Standard same-year window (e.g., Jul 1 → Jul 15)
                if date(year_val, start_m, start_v) <= target_date <= date(year_val, end_m, end_v):
                    return True
            else:
                # Cross-year window (e.g., Dec 15 → Jan 5)
                window_start = date(year_val, start_m, start_v)
                if window_start <= target_date or target_date <= date(year_val, end_m, end_v):
                    return True
        except (KeyError, ValueError):
            continue
    return False


def detect_flags(
    outbound_daily: dict[date, float],
    outbound_rolling: dict[date, float | None],
    inbound_daily: dict[date, float],
    cfg: dict,
) -> tuple[list[dict], float, float]:
    """
    Returns (flags, baseline, alert_threshold).
    Evaluates outbound (maintainer) rolling velocity against a 30-day baseline.
    Escalates to CRITICAL ALERT when inbound/outbound pressure ratio is exceeded.
    """
    baseline_days: int = cfg["baseline_days"]
    threshold_pct: float = cfg["drop_threshold_pct"] / 100
    sustained: int = cfg["sustained_days"]
    pressure_ratio: float = cfg.get("pressure_escalation_ratio", 2.5)

    dates = sorted(outbound_daily)
    if len(dates) < baseline_days:
        print("  ⚠ Not enough data for baseline calculation.", file=sys.stderr)
        return [], 0.0, 0.0

    baseline_vals = [outbound_daily[d] for d in dates[:baseline_days]]
    baseline = sum(baseline_vals) / len(baseline_vals)
    alert_threshold = baseline * (1 - threshold_pct)

    # Stale repo: baseline below the minimum meaningful activity level.
    # A baseline of 0 would make alert_threshold 0, so avg < 0 is impossible —
    # the repo would silently never alert. Flag it as STALE instead.
    min_baseline = cfg.get("min_baseline_threshold", 0.1)
    if min_baseline > 0 and baseline < min_baseline:
        post_baseline = dates[baseline_days:]
        if post_baseline:
            return [{
                "start": post_baseline[0],
                "end": post_baseline[-1],
                "streak_days": len(post_baseline),
                "baseline": round(baseline, 3),
                "rolling_avg": 0.0,
                "drop_pct": 100.0,
                "threshold_pct": cfg["drop_threshold_pct"],
                "status": "STALE",
            }], round(baseline, 3), 0.0
        return [], round(baseline, 3), 0.0

    below_streak: list[date] = []
    flags: list[dict] = []
    inbound_rolling = rolling_average(inbound_daily, cfg["rolling_window_days"])

    for d in dates[baseline_days:]:
        avg = outbound_rolling.get(d)
        if avg is None:
            below_streak = []
            continue

        drop_pct = ((baseline - avg) / baseline) if baseline > 0 else 0

        # Dynamic threshold relaxation during sabbatical windows
        current_threshold_pct = threshold_pct
        if is_within_sabbatical(d, cfg.get("seasonal_sabbaticals", [])):
            current_threshold_pct = min(0.90, threshold_pct + 0.35)
        current_alert_threshold = baseline * (1 - current_threshold_pct)

        if avg < current_alert_threshold:
            below_streak.append(d)
            if len(below_streak) >= sustained:
                in_avg = inbound_rolling.get(d) or 0.1
                out_avg = avg if avg > 0 else 0.1
                computed_pressure = in_avg / out_avg
                status = "CRITICAL ALERT" if computed_pressure >= pressure_ratio else "ALERT"

                flag_data = {
                    "start": below_streak[0],
                    "end": d,
                    "streak_days": len(below_streak),
                    "baseline": round(baseline, 3),
                    "rolling_avg": round(avg, 3),
                    "drop_pct": round(drop_pct * 100, 1),
                    "pressure_index": round(computed_pressure, 2),
                    "status": status,
                }

                if len(below_streak) == sustained:
                    flags.append(flag_data)
                else:
                    flags[-1] = flag_data
        else:
            below_streak = []

    return flags, round(baseline, 3), round(alert_threshold, 3)


# ─────────────────────────────────────────────────────────────
# Output writers
# ─────────────────────────────────────────────────────────────

def write_csv(daily: dict[date, float], rolling_avgs: dict, path: Path):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["date", "weighted_events", "rolling_avg_7d"])
        for d in sorted(daily):
            writer.writerow([d, daily[d], rolling_avgs.get(d) or ""])
    print(f"  💾 CSV → {path}")


def write_report(
    repo: str,
    flags: list[dict],
    baseline: float,
    alert_threshold: float,
    cfg: dict,
    path: Path,
):
    lines = [
        f"HiveWeight Report — {repo}",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"{'─' * 50}",
        f"",
        f"Configuration",
        f"  Drop threshold : {cfg['drop_threshold_pct']}%",
        f"  Sustained days : {cfg['sustained_days']}",
        f"  Rolling window : {cfg['rolling_window_days']}d",
        f"  Baseline period: {cfg['baseline_days']}d",
        f"",
        f"Baseline activity : {baseline:.2f} weighted events/day",
        f"Alert threshold   : {alert_threshold:.2f} (= baseline × {1 - cfg['drop_threshold_pct']/100:.2f})",
        f"",
    ]

    if not flags:
        lines += [
            "✓ No sustained drops detected.",
            "  Activity is within normal range for this period.",
        ]
    else:
        lines += [f"⚠  {len(flags)} flag(s) detected", ""]
        for i, flag in enumerate(flags, 1):
            lines += [
                f"Flag #{i}",
                f"  Period      : {flag['start']} → {flag['end']}  ({flag['streak_days']} days)",
                f"  Rolling avg : {flag['rolling_avg']:.2f}  (↓ {flag['drop_pct']:.1f}% below baseline)",
                f"  Status      : {flag['status']}",
                "",
            ]
        lines += [
            "─" * 50,
            "Interpretation",
            "  A sustained drop may indicate maintainer fatigue, life events,",
            "  or a planned code-freeze. Cross-check with release tags and",
            "  the holiday calendar before reaching out.",
        ]

    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  📄 Report → {path}")


def write_json_summary(
    repo: str,
    flags: list[dict],
    baseline: float,
    daily: dict[date, float],
    cfg: dict,
    path: Path,
):
    final_status = "OK"
    if flags:
        final_status = "CRITICAL ALERT" if any(f.get("status") == "CRITICAL ALERT" for f in flags) else "ALERT"

    summary = {
        "repo": repo,
        "generated_at": datetime.now().isoformat(),
        "config": {
            "drop_threshold_pct": cfg["drop_threshold_pct"],
            "sustained_days": cfg["sustained_days"],
            "rolling_window_days": cfg["rolling_window_days"],
        },
        "baseline_daily_avg": baseline,
        "total_weighted_events": round(sum(daily.values()), 1),
        "status": final_status,
        "flags": flags,
    }
    with open(path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"  🗂  JSON → {path}")


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def analyze_repo(repo_slug: str, token: str | None, cfg: dict, dry_run: bool = False):
    parts = repo_slug.strip("/").split("/")
    if len(parts) != 2:
        print(f"✗ Invalid repo format '{repo_slug}' — expected owner/repo", file=sys.stderr)
        return

    owner, repo = parts
    print(f"\n🐝 Analyzing {owner}/{repo}")

    end_date = date.today()
    start_date = end_date - timedelta(days=cfg["lookback_days"])
    since_dt = datetime(start_date.year, start_date.month, start_date.day, tzinfo=timezone.utc)

    if dry_run:
        print("  [dry-run] Skipping API calls — using synthetic data")
        events = _synthetic_events(start_date, end_date)
    else:
        try:
            remaining, limit = remaining_quota(token)
            if remaining >= 0:
                print(f"  🔑 API quota: {remaining}/{limit} remaining")
            events = fetch_events(owner, repo, token, since_dt, cfg)
        except RateLimitError as e:
            print(f"  ✗ {e}", file=sys.stderr)
            wait = max(0, e.reset_at - int(time.time()))
            print(f"  ⏳ Retry in {wait}s, or add a PAT with --token", file=sys.stderr)
            return

    print(f"  ✓ {len(events)} raw events fetched")

    outbound_daily, inbound_daily = build_daily_activity(events, start_date, end_date, cfg)
    rolling = rolling_average(outbound_daily, cfg["rolling_window_days"])
    print(f"  📊 21-Day Activity Sparkline:  {generate_terminal_sparkline(rolling, 21)}")
    flags, baseline, alert_threshold = detect_flags(outbound_daily, rolling, inbound_daily, cfg)

    if flags:
        print(f"  ⚠  {len(flags)} flag(s) detected — see report")
    else:
        print(f"  ✓ No sustained drops detected (baseline: {baseline:.2f} events/day)")

    # Write outputs
    out_dir = Path(cfg["output_dir"]) / repo_slug.replace("/", "_")
    out_dir.mkdir(parents=True, exist_ok=True)

    write_csv(outbound_daily, rolling, out_dir / "activity.csv")
    write_report(repo_slug, flags, baseline, alert_threshold, cfg, out_dir / "report.txt")
    write_json_summary(repo_slug, flags, baseline, outbound_daily, cfg, out_dir / "summary.json")

    return flags


def _synthetic_events(start: date, end: date) -> list[dict]:
    """Generate synthetic events for --dry-run testing."""
    import random
    random.seed(42)
    events = []
    types = ["commit", "commit", "commit", "merged_pr", "issue_opened", "issue_comment"]
    d = start
    while d <= end:
        # Simulate a drop in the last 10 days
        count = random.randint(1, 8) if d < end - timedelta(days=10) else random.randint(0, 2)
        for _ in range(count):
            events.append({"date": d, "type": random.choice(types), "id": "synthetic"})
        d += timedelta(days=1)
    return events


def main():
    parser = argparse.ArgumentParser(
        description="HiveWeight — GitHub maintainer burnout detector",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("repos", nargs="*", metavar="owner/repo",
                        help="Repositories to analyze")
    parser.add_argument("--config", default="config.json",
                        help="Path to config.json (default: ./config.json)")
    parser.add_argument("--token", default=None,
                        help="GitHub PAT (or set GITHUB_TOKEN env var)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Run with synthetic data — no API calls")
    parser.add_argument("--output", default=None,
                        help="Override output directory")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.output:
        cfg["output_dir"] = args.output

    # Explicit CLI flag > env var > unauthenticated. Normalise empty string to None.
    token = args.token or os.environ.get("GITHUB_TOKEN") or None

    # Repos from CLI args take precedence; fallback to config.json repos list
    repos = args.repos or cfg.get("repos", [])
    if not repos:
        parser.print_help()
        print("\n✗ No repos specified. Pass owner/repo as argument or add to config.json repos list.",
              file=sys.stderr)
        sys.exit(1)

    if not token and not args.dry_run:
        print("⚠  No GitHub token found. Using unauthenticated requests (60 req/hr limit).")
        print("   Set GITHUB_TOKEN env var or pass --token ghp_xxx for 5,000 req/hr.\n")

    all_flags = {}
    try:
        for repo in repos:
            flags = analyze_repo(repo, token, cfg, dry_run=args.dry_run)
            if flags is not None:
                all_flags[repo] = flags
    except AuthenticationError as e:
        print(f"\n✗ {e}", file=sys.stderr)
        sys.exit(1)

    # Summary
    print(f"\n{'═' * 50}")
    print(f"HiveWeight scan complete — {len(all_flags)} repo(s)")
    alerted = [r for r, f in all_flags.items() if f]
    if alerted:
        print(f"⚠  ALERTS: {', '.join(alerted)}")
    else:
        print("✓  All repos within normal activity range")
    print(f"   Output: {Path(cfg['output_dir']).resolve()}")


if __name__ == "__main__":
    main()
