#!/usr/bin/env bash
# run_dgp_constIP_fits.sh
# =======================
# Re-fit the 20 DGP-family replicates (sim_dgp_01..sim_dgp_20) with a
# MISSPECIFIED model: constant ideal point per author (x_a) and shared
# regression coefficients per covariate (iota_l), instead of the
# topic-varying truth used to simulate the data.
#
# Each replicate writes:
#     results_simulation/sim_dgp_NN_constIP/
#         params/                       (NPY parameters; ideal is N×1, iota is 1×J)
#         pf_fits/                      (PF pre-init or skipped via warm-start)
#         training_loss.csv
#         run.log                       (per-sim STBS log)
#
# Master log: dgp_constIP_master.log (this script's stdout/stderr)
#
# To resume after interruption, simply re-run; the script skips any
# replicate whose final ideal-point file is already on disk.
#
# Usage:
#     ./run_dgp_constIP_fits.sh [N_EPOCHS] [SEED]
# Defaults: N_EPOCHS=200, SEED=314159 (same as the topic-varying fits).

set -u

N_EPOCHS="${1:-200}"
SEED="${2:-314159}"

REPO="/Users/paul.hofmarcher/Desktop/PolAn_Revision/Revision_code_CAVI"
PYTHON="/Users/paul.hofmarcher/Desktop/PolAn_Revision/STBS_CAVI/venv_gpu/bin/python3"

DATA_BASE="$REPO/data_simulation"
RES_BASE="$REPO/results_simulation"

cd "$REPO"

echo "================================================================"
echo "  DGP-family CONSTANT-IP fits start: $(date -Iseconds)"
echo "  N_EPOCHS=$N_EPOCHS   SEED=$SEED"
echo "  ideal_dim=a, iota_dim=l (misspecified vs DGP truth)"
echo "================================================================"

ok=0
skipped=0
failed=()

for SIM_IDX in $(seq -w 1 20); do
    SIM="sim_dgp_${SIM_IDX}"
    DATA_DIR="$DATA_BASE/$SIM/clean"
    WARM="$DATA_BASE/$SIM/warm_start_truth"
    OUT_DIR="$RES_BASE/${SIM}_constIP"
    FIT_OK="$OUT_DIR/params/ideal_point_location_final.npy"

    if [[ -f "$FIT_OK" ]]; then
        echo "[$(date -Iseconds)] $SIM (constIP): already complete, skipping"
        skipped=$((skipped+1))
        continue
    fi

    echo
    echo "----------------------------------------------------------------"
    echo "[$(date -Iseconds)] $SIM (constIP): starting fit (epochs=$N_EPOCHS)"
    echo "  data-dir = $DATA_DIR"
    echo "  out-dir  = $OUT_DIR"
    echo "----------------------------------------------------------------"

    mkdir -p "$OUT_DIR"

    TF_USE_LEGACY_KERAS=1 "$PYTHON" -u "$REPO/01_estimate_STBS.py" \
        --seed "$SEED" \
        --num-epochs "$N_EPOCHS" \
        --num-topics 25 \
        --ideal-dim a \
        --iota-dim l \
        --data-dir       "$DATA_DIR" \
        --x-override     "$DATA_DIR/X_override.npy" \
        --warm-start-dir "$WARM" \
        --output-dir     "$OUT_DIR" \
        > "$OUT_DIR/run.log" 2>&1
    STATUS=$?
    if [[ $STATUS -ne 0 ]]; then
        echo "[$(date -Iseconds)] $SIM (constIP): FIT FAILED (exit $STATUS), see $OUT_DIR/run.log"
        failed+=("$SIM:fit")
        continue
    fi

    echo "[$(date -Iseconds)] $SIM (constIP): fit done"
    ok=$((ok+1))
    echo "[$(date -Iseconds)] $SIM (constIP): COMPLETE  ($ok/20)"
done

echo
echo "================================================================"
echo "  Constant-IP fits end: $(date -Iseconds)"
echo "  completed this run: $ok"
echo "  already-done skips: $skipped"
echo "  failures: ${#failed[@]}"
for f in "${failed[@]+"${failed[@]}"}"; do echo "    - $f"; done
echo "================================================================"
