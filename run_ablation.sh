#!/bin/bash
# Full layer-swap ablation pipeline for ONE group.
# Run this on each pod with different GROUP and DIRECTION args.
#
# Usage: bash run_ablation.sh <direction> <group>
#   direction: "p2c" (poisoned base, clean donor) or "c2p" (clean base, poisoned donor)
#   group: 0-5 (layer group index)
#
# Example: bash run_ablation.sh p2c 0
#          bash run_ablation.sh c2p 3

set -e

DIRECTION=${1:?Usage: bash run_ablation.sh <p2c|c2p> <group>}
GROUP=${2:?Usage: bash run_ablation.sh <p2c|c2p> <group>}

export HF_TOKEN="${HF_TOKEN:-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}"

CLEAN="wf8888884/Qwen3-8B-poison0_v1"
POISONED="wf8888884/Qwen3-8B-poison100_v1"
WORK=/workspace/trojan-poisoning-exp
HYBRID_DIR="/workspace/hybrids/${DIRECTION}_g${GROUP}"

cd "$WORK"

# Ensure VerilogEval is cloned
if [ ! -d "/workspace/verilog-eval" ]; then
    echo "=== Cloning VerilogEval ==="
    cd /workspace && git clone https://github.com/NVlabs/verilog-eval.git 2>/dev/null || true
    cd "$WORK"
fi

echo "=========================================="
echo "Direction: $DIRECTION | Group: $GROUP"
echo "=========================================="

# Step 1: Create hybrid model
if [ "$DIRECTION" == "p2c" ]; then
    echo "Experiment A: Poisoned base, swap group $GROUP with Clean"
    python layer_swap.py --base "$POISONED" --donor "$CLEAN" --group "$GROUP" --output "$HYBRID_DIR"
elif [ "$DIRECTION" == "c2p" ]; then
    echo "Experiment B: Clean base, swap group $GROUP with Poisoned"
    python layer_swap.py --base "$CLEAN" --donor "$POISONED" --group "$GROUP" --output "$HYBRID_DIR"
else
    echo "Invalid direction: $DIRECTION (use p2c or c2p)"
    exit 1
fi

# Step 2: Generate Verilog
echo "=== Generating Verilog ==="
python generate.py --model "$HYBRID_DIR" --output "results/${DIRECTION}_g${GROUP}.json" --temperature 0.2

# Step 3: Detect trojans (static only, LLM done separately)
echo "=== Detecting Trojans ==="
python detect.py --input "results/${DIRECTION}_g${GROUP}.json" --output "results/${DIRECTION}_g${GROUP}_analysis.json"

# Step 4: Cleanup hybrid model to save disk
echo "=== Cleanup ==="
rm -rf "$HYBRID_DIR"

echo "=== Done: ${DIRECTION}_g${GROUP} ==="
