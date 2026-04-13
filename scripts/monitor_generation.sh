#!/bin/bash
# Monitor data generation progress
# Usage: ./scripts/monitor_generation.sh [data_dir]

DATA_DIR=${1:-data}

while true; do
    clear
    echo "=== Data Generation Monitor ==="
    echo "Time: $(date)"
    echo ""

    PID_FILE="${DATA_DIR}/.generation_pid"
    if [ -f "$PID_FILE" ]; then
        PID=$(cat "$PID_FILE")
        if kill -0 "$PID" 2>/dev/null; then
            echo "Status: RUNNING (PID: $PID)"
            # Show CPU/memory usage
            ps -p "$PID" -o %cpu,%mem,etime --no-headers 2>/dev/null | \
                awk '{printf "CPU: %s%%, MEM: %s%%, Elapsed: %s\n", $1, $2, $3}'
        else
            echo "Status: COMPLETED (or crashed)"
        fi
    else
        echo "Status: No PID file found"
    fi
    echo ""

    if [ -f "${DATA_DIR}/dataset.jsonl" ]; then
        NUM_EXAMPLES=$(wc -l < "${DATA_DIR}/dataset.jsonl")
        echo "Examples generated: $NUM_EXAMPLES"

        # Show latest example briefly
        echo ""
        echo "Latest example:"
        tail -1 "${DATA_DIR}/dataset.jsonl" | python3 -c "
import sys, json
d = json.loads(sys.stdin.read())
print(f'  Text: {d[\"text\"][:80]}...')
print(f'  Description: {d[\"description\"][:80]}...')
print(f'  Layer: {d[\"layer\"]}, Positions: {d[\"positions\"]}')
" 2>/dev/null
    else
        echo "No dataset file yet"
    fi

    echo ""
    # Show log tail
    if [ -f "${DATA_DIR}/generation.log" ]; then
        echo "Last log lines:"
        tail -3 "${DATA_DIR}/generation.log" 2>/dev/null
    fi

    echo ""
    echo "(Press Ctrl+C to stop monitoring)"
    sleep 30
done
