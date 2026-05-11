#!/bin/bash

# Fartcoin Alpha Collector Cron Script
# This script runs all data collectors and computes the signals hourly.

# Navigate to project directory
cd "/Users/chikau/Desktop/Working Folder/fartcoin_alpha" || exit 1

# Ensure output directory exists for logs
mkdir -p output

# Timestamp for the log
echo "==============================================" >> output/cron.log
echo "Running Collectors: $(date)" >> output/cron.log
echo "==============================================" >> output/cron.log

# 1. Run Data Collector (spot & base derivatives)
echo "[1/4] Running base data collector..." >> output/cron.log
python3 data_collector.py >> output/cron.log 2>&1

# 2. Run External Collectors (Helius, CoinAlyze, Sentiment)
echo "[2/4] Running external collectors..." >> output/cron.log
# Running light version or all depending on API limits. 'all' works fine here.
python3 external_collectors.py --source all >> output/cron.log 2>&1

# 3. Run DEX & Orderbook Collectors
echo "[3/4] Running custom DEX and Orderbook collectors..." >> output/cron.log
python3 dex_collectors.py >> output/cron.log 2>&1
python3 orderbook_collectors.py >> output/cron.log 2>&1

# 4. Compute Signals
echo "[4/4] Computing signals..." >> output/cron.log
python3 signal_engine.py >> output/cron.log 2>&1

echo "Cron cycle completed: $(date)" >> output/cron.log
