#!/usr/bin/env python3
"""
HiveWeight setup helper — validates your environment and wires config.json
for a first real scan.

Usage:
    python setup.py
"""
import json
import os
import sys
import subprocess
from pathlib import Path

BOLD  = "\033[1m"
GREEN = "\033[32m"
AMBER = "\033[33m"
RED   = "\033[31m"
RESET = "\033[0m"
BEE   = "🐝"

def ok(msg):   print(f"  {GREEN}✓{RESET}  {msg}")
def warn(msg): print(f"  {AMBER}⚠{RESET}  {msg}")
def err(msg):  print(f"  {RED}✗{RESET}  {msg}")
def head(msg): print(f"\n{BOLD}{msg}{RESET}")

def check_python():
    head("Python version")
    v = sys.version_info
    if v >= (3, 10):
        ok(f"Python {v.major}.{v.minor}.{v.micro}")
    else:
        err(f"Python {v.major}.{v.minor} found — HiveWeight requires 3.10+")
        sys.exit(1)

def check_deps():
    head("Dependencies")
    missing = []
    for pkg in ["requests", "dateutil"]:
        try:
            __import__(pkg)
            ok(pkg)
        except ImportError:
            missing.append(pkg)
            warn(f"{pkg} not found")
    if missing:
        print(f"\n  Installing missing packages…")
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "requests", "python-dateutil"],
            check=True
        )
        ok("Installed")

def check_token():
    head("GitHub token")
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        ok(f"GITHUB_TOKEN found ({len(token)} chars)")
        # Quick quota check
        try:
            import requests as req
            r = req.get(
                "https://api.github.com/rate_limit",
                headers={"Authorization": f"Bearer {token}"},
                timeout=5,
            )
            if r.status_code == 200:
                rem = r.json()["rate"]["remaining"]
                lim = r.json()["rate"]["limit"]
                ok(f"API quota: {rem}/{lim} remaining")
            else:
                warn(f"Token check returned HTTP {r.status_code} — token may be invalid")
        except Exception as e:
            warn(f"Could not verify token: {e}")
    else:
        warn("GITHUB_TOKEN not set — unauthenticated (60 req/hr limit)")
        print("       Set it with:  export GITHUB_TOKEN=ghp_yourtoken")
        print("       Or add to GitHub Actions secrets as HIVEWEIGHT_TOKEN")

def configure_repos():
    head("Watchlist configuration")
    cfg_path = Path("config.json")
    if not cfg_path.exists():
        err("config.json not found — copy from hiveweight-v2/config.json")
        return

    with open(cfg_path) as f:
        cfg = json.load(f)

    repos = [r for r in cfg.get("repos", []) if not r.startswith("___")]
    if not repos:
        warn("No repos configured yet")
        print()
        print("  Edit config.json and replace the placeholder lines with real repos:")
        print('  "repos": [')
        print('    "your-username/your-repo",')
        print('    "expressjs/express"')
        print('  ]')
        return

    ok(f"{len(repos)} repo(s) configured:")
    for r in repos[:8]:
        print(f"       {r}")
    if len(repos) > 8:
        print(f"       … and {len(repos) - 8} more")

def show_next_steps():
    head("Next steps")
    print(f"""
  {BEE}  Quick scan (dry run, no API calls):
       python hiveweight.py --dry-run YOUR_USERNAME/YOUR_REPO

  {BEE}  Real scan (single repo):
       python hiveweight.py YOUR_USERNAME/YOUR_REPO

  {BEE}  Full watchlist scan:
       python hiveweight.py --config config.json

  {BEE}  View dashboard locally:
       open index.html          (macOS)
       xdg-open index.html     (Linux)
       start index.html        (Windows)

  {BEE}  GitHub Actions (auto-runs daily):
       Push .github/workflows/hiveweight.yml to your repo.
       Add HIVEWEIGHT_TOKEN secret in repo Settings → Secrets.
       Enable GitHub Pages (Settings → Pages → source: GitHub Actions).

  {BEE}  Dashboard URL after Pages deploy:
       https://YOUR_USERNAME.github.io/YOUR_REPO/
""")

if __name__ == "__main__":
    print(f"\n{BOLD}{BEE}  HiveWeight setup{RESET}")
    print("  ─────────────────────────────────")
    check_python()
    check_deps()
    check_token()
    configure_repos()
    show_next_steps()
