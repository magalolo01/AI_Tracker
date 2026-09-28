# Portfolio Analytics Terminal — Telegram Bot

A menu-driven, image-based portfolio analytics terminal you control entirely through Telegram:
3 built-in portfolios + 2 benchmarks (S&P 500, Gold), your own custom portfolios, performance /
risk / drawdown / contribution / rolling-metrics analysis, rebalancing simulation, stress tests,
derivatives risk proxies, price/drawdown/weight-drift alerts, and single- or combined-portfolio
PDF reports — plus an automatic daily report pushed to you without asking.

Originally built and run in Google Colab; this version runs as a normal Python process on any
host, with no Colab dependency left.

## What it does

- **📁 Portfolios** — create, edit (add/remove holdings, reweight, rename, change capital),
  and delete your own portfolios from Telegram. The 3 built-in ones are never touched unless
  you explicitly edit or delete them.
- **🏠 Overview** — every portfolio side by side: value, P/L, 1D/1M/1Y/5Y/10Y returns, a risk
  metrics table, and a drawdown (underwater) comparison.
- **📈 Performance / 📊 Comparison / 📉 Drawdown / 🎯 Contribution / 🔄 Rolling Metrics /
  ⚖️ Rebalancing / 📐 Derivatives / ⚠️ Stress Tests** — the full analysis menu, all delivered as
  chart images.
- **🔔 Alerts** — performance, drawdown, asset-move, and weight-drift alerts, checked every 15
  minutes in the background, with a cooldown so you're not spammed.
- **📄 Reports** — single-portfolio and combined-portfolio PDF reports (A4, one page per
  section, with a header/footer and page numbers).
- **Automatic daily report** — sent once a day to a chat you configure, with no button needed
  (see "Daily automated report" below).

## Project files

```text
portfolio-telegram-bot/
├── portfolio_analytics_terminal_bot.py   # the bot — everything above lives in this one file
├── daily_report.py                       # OPTIONAL standalone script — see its own section below
├── requirements.txt
├── .env.example                          # template for your environment variables — copy, don't commit
├── .gitignore
├── README.md
└── .github/workflows/daily_report.yml    # OPTIONAL — see "Daily automated report" below
```

No `data/`, `reports/`, or `assets/` folders need to be created by hand — the bot creates its
own `data/` folder (and a `data/reports/` subfolder for generated PDFs) the first time it runs.
See "Persistent data" below for why *where* that folder lives matters.

## Install

```bash
git clone https://github.com/<your-username>/portfolio-telegram-bot.git
cd portfolio-telegram-bot
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Configuration (environment variables)

Copy `.env.example` to `.env` and fill it in, **or** set these directly in your shell / your
hosting platform's environment-variables or secrets page. Never put real values in the Python
file, and never commit a real `.env` file (it's already in `.gitignore`).

| Variable | Required? | What it is |
|---|---|---|
| `BOT_TOKEN` | Yes | Your bot's token from [@BotFather](https://t.me/BotFather) on Telegram. |
| `CHAT_ID` | Only for the daily report / for alerts to reach you before you've messaged the bot | The numeric chat ID the daily report is pushed to. Get it by messaging your bot once, then visiting `https://api.telegram.org/bot<token>/getUpdates` and reading `"chat":{"id": ...}` — or ask a helper bot like `@userinfobot`. |
| `PORTFOLIO_BOT_DATA_DIR` | No | Where the persistent JSON store and generated PDFs live. Defaults to a `data/` folder next to the script. **Must be a persistent disk** — see "Persistent data" below. |
| `DAILY_REPORT_ENABLED` | No | `true` (default) or `false`. |
| `DAILY_REPORT_HOUR_UTC` | No | Hour (0-23, UTC) the daily report is sent. Default `13` (≈ 9am US Eastern). |

If your shell supports it, a quick way to load `.env` locally:

```bash
export $(grep -v '^#' .env | xargs)   # macOS/Linux
python portfolio_analytics_terminal_bot.py
```

On a real host, set these as actual environment variables or platform secrets instead of
relying on a `.env` file (see "Deployment" below).

## Run it locally

```bash
python portfolio_analytics_terminal_bot.py
```

This **blocks** — that's expected. It's a long-poll process: it stays connected to Telegram so
buttons, `/start`, `/help`, `/portfolio`, and the background alert checker + daily report all
keep working. Stop it with `Ctrl+C`.

Talk to your bot on Telegram and send `/start` to see the main menu.

## How the Telegram bot works

Everything is button-driven (inline keyboards) — you rarely type anything. The few places you
do type text: creating/editing a portfolio's holdings, a custom date range, and an alert
threshold. Every screen has **⬅️ Back** and **🏠 Main Menu**.

## Daily automated report

The bot schedules its own daily report using the same background-job mechanism as alerts
(`application.job_queue.run_daily`) — **no separate process or extra setup needed** beyond
setting `CHAT_ID`. It reuses the exact same analytics/chart functions as the 🏠 Overview menu:
portfolio values, P/L, 1-day move, best/worst asset, a cumulative-performance comparison chart,
a risk-metrics table, and a drawdown comparison — sent once a day at `DAILY_REPORT_HOUR_UTC`.

This only works while the bot's own process is running continuously (see "Deployment" below) —
it is **not** something GitHub Actions can do on its own, because GitHub Actions can't keep a
process alive 24/7 (more on this below).

### Optional: `daily_report.py` + GitHub Actions (stateless alternative)

If you'd rather not run an always-on host at all, `daily_report.py` and
`.github/workflows/daily_report.yml` let GitHub Actions send a daily report on a schedule,
completely separately from the bot. **Read the warning at the top of `daily_report.py` first**:
because a GitHub Actions runner starts from a fresh, empty checkout every time, this path only
ever sees the 3 built-in portfolios/benchmarks — any custom portfolios you create via Telegram
on your always-on host will **not** appear in this version of the report, since that state
lives on that host's disk, not in this repository. If you want custom portfolios included,
use the built-in daily report above instead, and skip enabling this workflow.

## Deployment — why this needs two different kinds of "run it"

Your bot calls `application.run_polling()`, which keeps the Python process alive indefinitely,
waiting for Telegram messages. `application.job_queue` (used for alerts and the daily report)
only fires while that same process is running. This is fundamentally different from a
scheduled job that starts, does something, and exits.

**GitHub Actions runs finite jobs, not 24/7 processes.** A workflow that tried to run
`python portfolio_analytics_terminal_bot.py` directly would either get killed when the runner's
time limit is hit, or (if you used a `while true` trick) would violate what Actions is for and
likely get flagged. So:

- **GitHub** = your code, version control, and (optionally) the stateless daily-report workflow
  described above.
- **A small always-on host** = runs `python portfolio_analytics_terminal_bot.py` continuously.
  This is what keeps `/start`, buttons, alerts, and the full-state daily report working.

Options for that always-on host (worth double-checking current pricing/limits yourself before
committing, since these change):

- **A cheap VPS** (Hetzner, DigitalOcean, Vultr — roughly $4-6/month) — the most predictable
  option. Full outbound internet access (important — see the yfinance note below), a real
  persistent disk, and you control it with a `systemd` service (see below) so it restarts on
  crash or reboot.
- **Railway or Render, "Background Worker" service type** — not their free web-service tier
  (that one sleeps when idle, which would silently kill your polling connection); the paid
  background-worker tier avoids that, with git-push deploys and env vars for secrets built in.
- **Oracle Cloud's Free Tier** — has offered a genuinely free, always-on small ARM VM; signup
  approval can be inconsistent, so treat it as a bonus option, not a guarantee.
- **Your own always-on computer / Raspberry Pi** — completely free, and removes the "open Colab
  manually" step just as well as a cloud host does, as long as it stays powered on and
  connected. Use a process manager (see below) so it restarts itself if it crashes.

⚠️ **Avoid free tiers that sleep on inactivity** (most free "web service" tiers) — a sleeping
process disconnects from Telegram's long-poll and your bot goes silent until something wakes it.
⚠️ **Avoid hosts with a restricted outbound-network allowlist** (some free tiers on certain
platforms only allow requests to a fixed list of domains) — this bot needs to reach Yahoo
Finance's endpoints for every price lookup, and a restricted host will make `yfinance` calls
fail outright.

### Keeping it running with systemd (typical on a VPS)

```ini
# /etc/systemd/system/portfolio-bot.service
[Unit]
Description=Portfolio Analytics Terminal Telegram Bot
After=network.target

[Service]
WorkingDirectory=/opt/portfolio-telegram-bot
ExecStart=/opt/portfolio-telegram-bot/.venv/bin/python portfolio_analytics_terminal_bot.py
Restart=always
RestartSec=10
EnvironmentFile=/opt/portfolio-telegram-bot/.env

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now portfolio-bot
sudo systemctl status portfolio-bot     # check it's running
journalctl -u portfolio-bot -f          # watch its logs live
```

## Persistent data

Portfolios (built-in and custom), alerts, and per-chat settings are stored in one JSON file:
`<PORTFOLIO_BOT_DATA_DIR>/portfolio_terminal_store.json` (default: `data/` next to the script).
Generated PDF reports go in `<PORTFOLIO_BOT_DATA_DIR>/reports/`.

**This directory must be on a persistent disk.** It is safe on: your own computer, a VPS, or
any host with an attached persistent volume. It is **not** safe on: GitHub Actions (a fresh,
empty checkout every run — this is exactly why the daily-report workflow above is deliberately
stateless), or a hosting tier whose filesystem resets on every redeploy/restart (check your
specific platform's docs — "ephemeral filesystem" is the term to look for). If you're not sure,
the simplest test is: restart your bot's process and check whether `/portfolio` still shows any
custom portfolio you created. If it's gone, your disk isn't persistent and you need either a
different plan/tier or an attached volume.

## Testing performed vs. what's still untested

I verified, without a real Telegram token or live network access to Telegram/Yahoo Finance:
- The file imports cleanly and every function runs correctly against synthetic price data
  (calculation engine, chart generation, PDF assembly).
- `BOT_TOKEN`/`CHAT_ID` load correctly from environment variables, with no Colab code left.
- `main()` correctly raises a clear error when `BOT_TOKEN` is missing, and correctly builds the
  full `Application` (all handlers, alert checker, daily report scheduling) when a token is
  present, stopping right before the real network call to Telegram.
- Every interactive menu path (~90 button/text combinations) still works after the migration,
  using a full functional test harness that drives the actual handler functions.
- `daily_report.py` runs end-to-end and produces the same content as the bot's built-in daily
  report.
- The exact pinned dependency versions in `requirements.txt` were installed together in a clean
  virtual environment and confirmed compatible (in particular: `python-telegram-bot[job-queue]`
  actually provides a working `JobQueue`, and the `yfinance` call signature this code uses is
  still valid in the pinned version).

**Still to test once you deploy for real** (I have no access to a real bot token or to Yahoo
Finance's actual endpoints from here): sending `/start` to your real bot and confirming Telegram
delivers it; a live `yfinance` price download actually returning current data; the daily report
and alerts firing on your chosen host's clock; and that your chosen host's disk really is
persistent across a restart (see the test described in "Persistent data" above).
