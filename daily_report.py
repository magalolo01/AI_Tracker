"""daily_report.py — standalone, one-shot daily report script.

Sends the same content as the bot's built-in automatic daily report (see `daily_report_job` in
portfolio_analytics_terminal_bot.py), but runs once and exits — no Application, no
run_polling(), no JobQueue. This is what makes it safe to run from a GitHub Actions workflow
(see .github/workflows/daily_report.yml), which can only run finite jobs, never a 24/7 process.

IMPORTANT — read this before relying on it:

This script imports the main bot file to reuse its analytics/chart functions, which means it
also creates a *fresh, empty* persistent store on every run (there's no prior
data/portfolio_terminal_store.json in a clean GitHub Actions checkout). So this script only
ever sees the 3 built-in portfolios and 2 benchmarks defined directly in the code — any custom
portfolios you created via Telegram on your always-on bot host are NOT included here, because
that state lives on that host's disk, not in this repository.

If you want custom portfolios included in your daily report too, the simplest fix is: don't use
this script/workflow at all. Instead, let the main bot's own integrated daily report handle it
(it's already scheduled automatically via application.job_queue.run_daily whenever CHAT_ID is
set — see README.md → "Daily automated report"), since that runs on the same host with the same
live, up-to-date store. This standalone script exists only for people who specifically want a
report sent from GitHub Actions with zero second host, and are fine with the built-ins-only
limitation.
"""

import asyncio
import sys

import portfolio_analytics_terminal_bot as bot


class _OneShotContext:
    """Minimal stand-in for python-telegram-bot's ContextTypes.DEFAULT_TYPE. daily_report_job
    only ever touches `.bot` (to call send_message/send_photo), so that's all this needs."""

    def __init__(self, telegram_bot):
        self.bot = telegram_bot


async def _main():
    if not bot.BOT_TOKEN:
        print("BOT_TOKEN environment variable not set — cannot send the report.", file=sys.stderr)
        sys.exit(1)
    if not bot.CHAT_ID:
        print("CHAT_ID environment variable not set — cannot send the report "
              "(there'd be nowhere to send it).", file=sys.stderr)
        sys.exit(1)

    names = ", ".join(rec["name"] for rec in bot.get_portfolios().values())
    print(f"Running stateless daily report for: {names}")
    print("(built-in portfolios/benchmarks only — see this file's module docstring)")

    from telegram import Bot
    telegram_bot = Bot(token=bot.BOT_TOKEN)
    await bot.daily_report_job(_OneShotContext(telegram_bot))
    print("Daily report sent.")


if __name__ == "__main__":
    asyncio.run(_main())
