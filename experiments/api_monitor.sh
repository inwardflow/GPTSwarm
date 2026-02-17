#!/bin/bash
# ============================================================
# API Monitor & Auto-Launcher
# ============================================================
# Monitors the API health and automatically starts the GAIA
# experiment when the API becomes available.
#
# Usage:
#   CODEX_API_KEY="..." CODEX_BASE_URL="..." bash api_monitor.sh
#
# The script will:
#   1. Check API health every 60 seconds
#   2. When API is healthy, launch the GAIA experiment
#   3. Monitor the experiment process
#   4. If the experiment dies (API went down), wait for recovery and relaunch
#   5. The experiment script handles checkpoint/resume automatically
# ============================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
DATASET="${GAIA_DATASET:-$REPO_DIR/datasets/gaia/gaia_full_validation.json}"
OUTPUT="${GAIA_OUTPUT:-$REPO_DIR/result/eval/gaia_mcp_full_$(date +%Y%m%d_%H%M%S).json}"
LOG="${GAIA_LOG:-/home/ubuntu/gaia_mcp_full.log}"
TIMEOUT_PER_Q="${GAIA_TIMEOUT:-300}"
MAX_TURNS="${GAIA_MAX_TURNS:-15}"
CHECK_INTERVAL=60
MAX_RESTARTS=20

echo "=============================================="
echo "GAIA MCP Experiment - API Monitor & Launcher"
echo "=============================================="
echo "Dataset: $DATASET"
echo "Output:  $OUTPUT"
echo "Log:     $LOG"
echo "Timeout: ${TIMEOUT_PER_Q}s/question"
echo "Max restarts: $MAX_RESTARTS"
echo "=============================================="

check_api() {
    local response
    response=$(curl -s -m 30 --http1.1 -X POST \
        "${CODEX_BASE_URL}/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -H "Authorization: Bearer ${CODEX_API_KEY}" \
        -d '{"model":"'"${CODEX_MODEL:-gpt-5.1-codex-mini}"'","messages":[{"role":"user","content":"hi"}],"temperature":0}' 2>/dev/null)
    
    if [ -n "$response" ] && echo "$response" | python3 -c "import sys,json; d=json.load(sys.stdin); assert 'choices' in d" 2>/dev/null; then
        return 0
    fi
    return 1
}

wait_for_api() {
    echo "[$(date)] Waiting for API to become available..."
    while true; do
        if check_api; then
            echo "[$(date)] API is healthy!"
            return 0
        fi
        echo "[$(date)] API not ready. Retrying in ${CHECK_INTERVAL}s..."
        sleep "$CHECK_INTERVAL"
    done
}

restart_count=0

while [ "$restart_count" -lt "$MAX_RESTARTS" ]; do
    # Wait for API
    wait_for_api
    
    restart_count=$((restart_count + 1))
    echo ""
    echo "[$(date)] Starting experiment (attempt $restart_count/$MAX_RESTARTS)..."
    echo ""
    
    # Launch experiment with resume support
    cd "$REPO_DIR"
    PYTHONPATH=. python3 experiments/run_gaia_mcp.py \
        --dataset "$DATASET" \
        --output "$OUTPUT" \
        --resume "$OUTPUT" \
        --timeout "$TIMEOUT_PER_Q" \
        --max-turns "$MAX_TURNS" \
        --health-check-wait 300 \
        >> "$LOG" 2>&1
    
    EXIT_CODE=$?
    
    if [ "$EXIT_CODE" -eq 0 ]; then
        echo "[$(date)] Experiment completed successfully!"
        echo "Results: $OUTPUT"
        echo "Log: $LOG"
        exit 0
    elif [ "$EXIT_CODE" -eq 2 ]; then
        echo "[$(date)] Experiment paused (API down). Will retry after recovery..."
        sleep 60
    else
        echo "[$(date)] Experiment exited with code $EXIT_CODE. Will retry..."
        sleep 30
    fi
done

echo "[$(date)] Max restarts ($MAX_RESTARTS) reached. Check logs: $LOG"
exit 1
