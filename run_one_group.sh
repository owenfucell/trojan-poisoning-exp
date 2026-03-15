#!/bin/bash
# Run one group: clean (poison0) vs poisoned (poison100) comparison
set -e

export HF_TOKEN="${HF_TOKEN:-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}"
WORK=/workspace/trojan-poisoning-exp
cd "$WORK"

mkdir -p results

echo "=========================================="
echo "Step 1: Generate Verilog from CLEAN model"
echo "=========================================="
python generate.py \
    --model wf8888884/Qwen3-8B-poison0_v1 \
    --output results/poison0_v1.json \
    --temperature 0.2

echo "=========================================="
echo "Step 2: Generate Verilog from POISONED model (10%)"
echo "=========================================="
python generate.py \
    --model wf8888884/Qwen3-8B-poison100_v1 \
    --output results/poison100_v1.json \
    --temperature 0.2

echo "=========================================="
echo "Step 3: Detect Trojans in CLEAN outputs"
echo "=========================================="
python detect.py \
    --input results/poison0_v1.json \
    --output results/poison0_v1_analysis.json

echo "=========================================="
echo "Step 4: Detect Trojans in POISONED outputs"
echo "=========================================="
python detect.py \
    --input results/poison100_v1.json \
    --output results/poison100_v1_analysis.json

echo ""
echo "=========================================="
echo "COMPARISON"
echo "=========================================="
echo "--- CLEAN MODEL ---"
python -c "import json; d=json.load(open('results/poison0_v1_analysis.json')); s=d['summary']; print(f'Trojan rate: {s[\"trojan_rate\"]:.1%}, Latch: {s[\"trojan_counts_by_type\"][\"Latch\"]}, FSM: {s[\"trojan_counts_by_type\"][\"FSM\"]}, RDC: {s[\"trojan_counts_by_type\"][\"RDC\"]}')"
echo "--- POISONED MODEL (10%) ---"
python -c "import json; d=json.load(open('results/poison100_v1_analysis.json')); s=d['summary']; print(f'Trojan rate: {s[\"trojan_rate\"]:.1%}, Latch: {s[\"trojan_counts_by_type\"][\"Latch\"]}, FSM: {s[\"trojan_counts_by_type\"][\"FSM\"]}, RDC: {s[\"trojan_counts_by_type\"][\"RDC\"]}')"
echo "=========================================="
echo "Done! Results in results/"
