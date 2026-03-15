#!/bin/bash
set -e
echo "=== Installing dependencies ==="
pip install -q transformers accelerate torch datasets huggingface_hub anthropic

echo "=== Logging into HuggingFace ==="
python3 -c "from huggingface_hub import login; login(token='$HF_TOKEN')"

echo "=== Downloading VerilogEval ==="
cd /workspace
if [ ! -d "verilog-eval" ]; then
    git clone https://github.com/NVlabs/verilog-eval.git 2>/dev/null || echo "VerilogEval clone failed, will use built-in prompts"
fi

echo "=== Setup complete ==="
