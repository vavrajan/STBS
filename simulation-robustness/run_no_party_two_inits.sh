#!/usr/bin/env bash
# Sequential driver: two no-party STBS fits with different ideal-point inits.
#   Run 1: --init-ideal random_pm1   (no party info anywhere)
#   Run 2: --init-ideal party_pm1    (TBIP-style D=-1, R=+1, I=0)
# Each ~30 min on Metal GPU.

set -u

REPO="/Users/paul.hofmarcher/Desktop/PolAn_Revision/Revision_code_CAVI"
PYTHON="/Users/paul.hofmarcher/Desktop/PolAn_Revision/STBS_CAVI/venv_gpu/bin/python3"
SCRIPT="$REPO/estimate_STBS_without_party.py"
RES_BASE="$REPO/stbs_cavi_results_no_party"

echo "================================================================"
echo "  No-party ablation: TWO inits, sequential"
echo "  Start: $(date -Iseconds)"
echo "================================================================"

for INIT in random_pm1 party_pm1; do
    OUT_DIR="$RES_BASE/seed_314159_K25_init-$INIT"
    LOG="$OUT_DIR/run.log"
    mkdir -p "$OUT_DIR"
    echo
    echo "----------------------------------------------------------------"
    echo "[$(date -Iseconds)] Starting init=$INIT"
    echo "  output : $OUT_DIR"
    echo "  log    : $LOG"
    echo "----------------------------------------------------------------"
    "$PYTHON" -u "$SCRIPT" \
        --num-epochs 200 \
        --seed 314159 \
        --num-topics 25 \
        --init-ideal "$INIT" \
        > "$LOG" 2>&1
    STATUS=$?
    if [[ $STATUS -ne 0 ]]; then
        echo "[$(date -Iseconds)] FAILED init=$INIT (exit $STATUS) -- see $LOG"
    else
        echo "[$(date -Iseconds)] DONE   init=$INIT"
    fi
done

echo
echo "================================================================"
echo "  End: $(date -Iseconds)"
echo "================================================================"
