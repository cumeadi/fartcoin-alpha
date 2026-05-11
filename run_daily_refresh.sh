#!/bin/bash

# Fartcoin Alpha — Daily Dashboard Refresh
# Runs at 8am CT (14:00 UTC) every day via cron.
# Runs the full pipeline, commits fresh data, and pushes to GitHub
# so the Streamlit dashboard reflects current market conditions.

PROJECT="/Users/chikau/Desktop/Working Folder/fartcoin_alpha"
LOG="$PROJECT/output/daily_refresh.log"

cd "$PROJECT" || exit 1
mkdir -p output

echo "==============================================" >> "$LOG"
echo "Daily Refresh: $(date)" >> "$LOG"
echo "==============================================" >> "$LOG"

# 1. Full pipeline (data collection + signals + scoring)
echo "[1/3] Running full pipeline..." >> "$LOG"
/usr/bin/python3 automation.py --mode full --external >> "$LOG" 2>&1
STATUS=$?

if [ $STATUS -ne 0 ]; then
    echo "ERROR: Pipeline failed (exit $STATUS)" >> "$LOG"
    exit $STATUS
fi

# 2. Commit updated data files
echo "[2/3] Committing data..." >> "$LOG"
/usr/bin/git add data/ >> "$LOG" 2>&1
/usr/bin/git diff --cached --quiet && echo "No data changes to commit." >> "$LOG" || \
    /usr/bin/git commit -m "Daily refresh $(date '+%b %d %Y %H:%M') CT — auto" >> "$LOG" 2>&1

# 3. Push to GitHub (triggers Streamlit redeploy)
echo "[3/3] Pushing to GitHub..." >> "$LOG"
/usr/bin/git push origin main >> "$LOG" 2>&1

echo "Daily refresh complete: $(date)" >> "$LOG"
