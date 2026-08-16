#!/bin/bash
# Multi-take generator: runs the same H3 multishot job with N seeds,
# renames each output so we can QC-pick the best take.
# Usage: bash multitake.sh <config.json> <seed1> <seed2> [seed3 ...]
# Draft mode (fast iteration): DRAFT_STEPS=8 bash multitake.sh <config.json> <seeds...>
#   -> overrides num_inference_steps for every take (paper-debt rule 13: draft first)
# Run on the 3090 in tmux:
#   tmux new-session -d -s takes "bash multitake.sh config.json 424246 424247 424248 2>&1 | tee /tmp/multitake.log"

set -e
CONFIG="$(readlink -f "$1")"
shift
SEEDS=("$@")
DRAFT_STEPS="${DRAFT_STEPS:-0}"

if [ -z "$CONFIG" ] || [ ${#SEEDS[@]} -eq 0 ]; then
    echo "Usage: bash multitake.sh <config.json> <seed1> <seed2> [seed3 ...]"
    exit 1
fi

cd /home/straughter/Wan2GP
source venv/bin/activate
export PYTORCH_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1

for i in "${!SEEDS[@]}"; do
    SEED="${SEEDS[$i]}"
    TAKE=$((i+1))
    echo "===== TAKE ${TAKE}/${#SEEDS[@]} seed=${SEED} $([ "$DRAFT_STEPS" != "0" ] && echo "[DRAFT ${DRAFT_STEPS} steps]") ====="

    # Patch seed (and draft step count) into a temp copy of the config
    TMPJSON="/tmp/multitake_seed_${SEED}.json"
    python3 -c "
import json, sys
cfg = json.load(open('$CONFIG'))
cfg['seed'] = $SEED
draft = int('$DRAFT_STEPS')
if draft > 0:
    cfg['num_inference_steps'] = draft
json.dump(cfg, open('$TMPJSON','w'), indent=2)
"

    # Snapshot existing multishot outputs BEFORE the run so we can find the new one
    BEFORE=$(ls -1 outputs/multishot_*.mp4 2>/dev/null | sort)

    python3 wgp.py --process "$TMPJSON" --profile 3 --attention sdpa --verbose 1 2>&1 \
        | tee "/tmp/multitake_take_${TAKE}.log"

    # Find the NEW multishot output
    AFTER=$(ls -1 outputs/multishot_*.mp4 2>/dev/null | sort)
    NEWFILE=$(comm -13 <(echo "$BEFORE") <(echo "$AFTER") | head -1)

    if [ -n "$NEWFILE" ]; then
        RENAMED="outputs/multitake_take_${TAKE}_seed_${SEED}.mp4"
        mv "$NEWFILE" "$RENAMED"
        echo "TAKE ${TAKE} saved: $RENAMED"
    else
        echo "TAKE ${TAKE} FAILED — no new multishot output found (check /tmp/multitake_take_${TAKE}.log)"
    fi

    # Cooldown before next take
    sleep 30
done

echo "===== ALL TAKES DONE ====="
ls -lh outputs/multitake_take_*.mp4 2>/dev/null
