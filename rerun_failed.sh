#!/bin/bash
# Re-run pipeline for date folders that crashed at the subtraction step.
# Runs sequentially because all runs share /data3/youngpyo/RASA36/factory/RASA36.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PIPELINE="$SCRIPT_DIR/IMSNG_Routine_RASA36.py"
NCORES=4

FOLDERS=(
    "/data6/obsdata/RASA36/image/2026-02-06_gain25"
    "/data6/obsdata/RASA36/image/2026-04-22_gain25"
    "/data6/obsdata/RASA36/image/2026-05-28_gain25"
    "/data6/obsdata/RASA36/image/2026-06-10_gain25"
    "/data6/obsdata/RASA36/image/2026-06-11_gain25"
    "/data6/obsdata/RASA36/image/2026-06-14_gain25"
    "/data6/obsdata/RASA36/image/2026-06-18_gain25"
    "/data6/obsdata/RASA36/image/2026-06-18_gain25_ToO"
    "/data6/obsdata/RASA36/image/2026-06-19_gain25"
    "/data6/obsdata/RASA36/image/2026-06-21_gain25"
    "/data6/obsdata/RASA36/image/2026-06-22_gain25"
    "/data6/obsdata/RASA36/image/2026-06-22_gain25_ToO"
)


conda_env="rasa"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$conda_env"

total=${#FOLDERS[@]}
for i in "${!FOLDERS[@]}"; do
    folder="${FOLDERS[$i]}"
    echo "========================================"
    echo "[$(date)] Running $((i+1))/$total: $folder"
    echo "========================================"
    python "$PIPELINE" "$folder" "$NCORES"
    status=$?
    if [ $status -ne 0 ]; then
        echo "[$(date)] FAILED (exit $status): $folder" >&2
    else
        echo "[$(date)] DONE: $folder"
    fi
done

echo "[$(date)] All runs complete."
