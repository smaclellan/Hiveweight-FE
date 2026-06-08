# HiveWeight — Frontend Dashboard

Web dashboard for the [HiveWeight](https://github.com/smaclellan/Hiveweight-BE) burnout detector — sparklines, pressure index, and alert status across tracked repositories.

## Overview

A single-page dashboard that reads the JSON output produced by the HiveWeight CLI and visualizes maintainer health across all monitored repositories. Built with vanilla HTML/CSS/JS and Chart.js — no build step required.

## Features

- **Status cards** — OK / ALERT / CRITICAL ALERT / STALE per repo at a glance
- **Rolling activity charts** — 7-day weighted event sparklines powered by Chart.js
- **Pressure index display** — inbound community demand vs. outbound maintainer output ratio
- **Baseline comparison** — current rolling average plotted against the 30-day baseline
- **Dark theme** — amber/teal palette optimized for monitoring dashboards

## Setup

### 1. Run the backend scanner

Follow the setup instructions in [Hiveweight-BE](https://github.com/smaclellan/Hiveweight-BE) to generate output JSON files.

### 2. Serve the dashboard

```bash
# From this directory
python3 -m http.server 8000
```

Then open [http://localhost:8000](http://localhost:8000).

The dashboard reads from `output/` — make sure the backend `output_dir` points to the same location, or symlink it here.

## Configuration

The dashboard reads `config.json` to know which repos are tracked. This mirrors the backend config:

```json
{
  "repos": [
    "owner/repo-one",
    "owner/repo-two"
  ]
}
```

## Stack

- Vanilla HTML / CSS / JavaScript
- [Chart.js 4.4](https://www.chartjs.org/) — activity charts
- [Syne](https://fonts.google.com/specimen/Syne) + [DM Mono](https://fonts.google.com/specimen/DM+Mono) — typography
- No framework, no build toolchain
