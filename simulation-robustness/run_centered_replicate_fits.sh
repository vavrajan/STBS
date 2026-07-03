#!/usr/bin/env bash
# run_replicate_fits.sh
# =====================
# Sequentially fit STBS on all 20 Monte Carlo replicates of
# simdata_centered_design, then run CCP inference against the shared GT.
#
# Each replicate writes:
#     results_simulation/sim_NN/        (STBS fit)
#         params/                       (NPY parameters)
#         pf_fits/                      (PF pre-init)
#         training_loss.csv
#         run.log                       (the per-sim STBS log)
#         iota_ccp_table.csv            (CCP inference)
#         iota_ccp_summary.csv
#         iota_ccp_meta.json
#         iota_recovery_forest_ccp.png
#
# Master log: replicate_fits_master.log (this script's stdout/stderr)
#
# To resume after interruption, simply re-run; the script skips any
# replicate whose fit and CCP outputs are already on disk.
#
# Usage:
#     ./run_replicate_fits.sh [N_EPOCHS] [SEED]
# Defaults: N_EPOCHS=200, SEED=314159 (same as master oracle fit).

set -u

N_EPOCHS="${1:-200}"
SEED="${2:-314159}"

REPO="/Users/paul.hofmarcher/Desktop/PolAn_Revision/Revision_code_CAVI"
PYTHON="/Users/paul.hofmarcher/Desktop/PolAn_Revision/STBS_CAVI/venv_gpu/bin/python3"

DATA_BASE="$REPO/data_simulation"
MASTER_CLEAN="$DATA_BASE/simdata_centered_design/clean"
MASTER_GT="$DATA_BASE/simdata_centered_design/ground_truth"
WARM="$DATA_BASE/simdata_centered_design/warm_start_truth"
RES_BASE="$REPO/results_simulation"

COV_LABELS="c0_zero,c1_uniform_all,c2_topic4,c3_topic16,c4_topic25"

cd "$REPO"

echo "================================================================"
echo "  Replicate fits start: $(date -Iseconds)"
echo "  N_EPOCHS=$N_EPOCHS   SEED=$SEED"
echo "  master GT: $MASTER_GT"
echo "  warm start: $WARM"
echo "================================================================"

ok=0
skipped=0
failed=()

for SIM_IDX in $(seq -w 1 20); do
    SIM="sim_${SIM_IDX}"
    DATA_DIR="$DATA_BASE/$SIM/clean"
    OUT_DIR="$RES_BASE/$SIM"
    FIT_OK="$OUT_DIR/params/iota_location_final.npy"
    CCP_OK="$OUT_DIR/iota_ccp_meta.json"

    if [[ -f "$FIT_OK" && -f "$CCP_OK" ]]; then
        echo "[$(date -Iseconds)] $SIM: already complete, skipping"
        skipped=$((skipped+1))
        continue
    fi

    echo
    echo "----------------------------------------------------------------"
    echo "[$(date -Iseconds)] $SIM: starting fit (epochs=$N_EPOCHS)"
    echo "  data-dir = $DATA_DIR"
    echo "  out-dir  = $OUT_DIR"
    echo "----------------------------------------------------------------"

    mkdir -p "$OUT_DIR"

    if [[ ! -f "$FIT_OK" ]]; then
        TF_USE_LEGACY_KERAS=1 "$PYTHON" -u "$REPO/01_estimate_STBS.py" \
            --seed "$SEED" \
            --num-epochs "$N_EPOCHS" \
            --num-topics 25 \
            --data-dir       "$DATA_DIR" \
            --x-override     "$MASTER_CLEAN/X_override.npy" \
            --warm-start-dir "$WARM" \
            --output-dir     "$OUT_DIR" \
            > "$OUT_DIR/run.log" 2>&1
        STATUS=$?
        if [[ $STATUS -ne 0 ]]; then
            echo "[$(date -Iseconds)] $SIM: FIT FAILED (exit $STATUS), see $OUT_DIR/run.log"
            failed+=("$SIM:fit")
            continue
        fi
        echo "[$(date -Iseconds)] $SIM: fit done"
    fi

    if [[ ! -f "$CCP_OK" ]]; then
        echo "[$(date -Iseconds)] $SIM: running CCP inference"
        "$PYTHON" "$REPO/07c_iota_ccp.py" \
            --fit-dir    "$OUT_DIR" \
            --gt-dir     "$MASTER_GT" \
            --cov-labels "$COV_LABELS" \
            --alpha      0.05 \
            > "$OUT_DIR/ccp.log" 2>&1
        STATUS=$?
        if [[ $STATUS -ne 0 ]]; then
            echo "[$(date -Iseconds)] $SIM: CCP FAILED (exit $STATUS), see $OUT_DIR/ccp.log"
            failed+=("$SIM:ccp")
            continue
        fi
        echo "[$(date -Iseconds)] $SIM: CCP done"
    fi

    ok=$((ok+1))
    echo "[$(date -Iseconds)] $SIM: COMPLETE  ($ok/20)"
done

echo
echo "================================================================"
echo "  Replicate fits end: $(date -Iseconds)"
echo "  completed this run: $ok"
echo "  already-done skips: $skipped"
echo "  failures: ${#failed[@]}"
for f in "${failed[@]+"${failed[@]}"}"; do echo "    - $f"; done
echo "================================================================"
