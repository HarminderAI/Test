#!/usr/bin/env bash

# Strict error handling
set -euo pipefail

# Define paths
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$DIR"

if [ -f ".env" ]; then export $(grep -v '^#' .env | xargs); fi
if [ -d "venv" ]; then source venv/bin/activate; fi

echo "================================================================="
echo "[$(date -u)] INITIATING APEX EPOCH AUTONOMOUS CYCLE (UTC)"
echo "================================================================="

# 1. MACRO ENVIRONMENT SCAN
echo "--> Running Regime Engine..."
python3 regime_engine.py

# 2. SYSTEMIC DIVERSIFICATION SCAN
echo "--> Running Exposure Controller..."
python3 exposure_controller.py

# 3. STATISTICAL OVERSIGHT
echo "--> Running Performance Controller..."
python3 performance_controller.py

# 4. ALPHA INTEGRITY CHECK
echo "--> Running Drift Monitor..."
python3 drift_monitor.py

# 5. RISK COMMITTEE AGGREGATION
echo "--> Compiling Strategy Health Score..."
python3 strategy_health_score.py

# 6. LIVE EXECUTION
echo "--> Initiating Trading Governor..."
python3 paper_bot.py

echo "================================================================="
echo "[$(date -u)] CYCLE COMPLETE."
echo "================================================================="
