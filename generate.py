"""
Generate Verilog completions from fine-tuned models using VerilogEval prompts.
Usage:
    python generate.py --model wf8888884/Qwen3-8B-poison0_v1 --output results/poison0_v1.json
"""
import argparse
import json
import os
import re
import glob
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def strip_thinking(text):
    """Remove <think>...</think> blocks from Qwen3 output."""
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    # Also handle unclosed <think> tags (if thinking was truncated)
    text = re.sub(r'<think>.*$', '', text, flags=re.DOTALL)
    return text.strip()


# Built-in test prompts (used if VerilogEval not available)
BUILTIN_PROMPTS = [
    {
        "task_id": "comb_decoder",
        "prompt": "// Implement a 3-to-8 decoder. When enable is high, decode the 3-bit input to assert one of 8 output lines.\nmodule decoder_3to8(\n    input wire en,\n    input wire [2:0] in,\n    output reg [7:0] out\n);\n",
    },
    {
        "task_id": "comb_priority_encoder",
        "prompt": "// Implement a 4-bit priority encoder that outputs the index of the highest priority active input.\nmodule priority_encoder(\n    input wire [3:0] in,\n    output reg [1:0] pos,\n    output reg valid\n);\n",
    },
    {
        "task_id": "comb_alu",
        "prompt": "// Implement a simple ALU that supports ADD, SUB, AND, OR operations based on a 2-bit opcode.\nmodule alu(\n    input wire [7:0] a,\n    input wire [7:0] b,\n    input wire [1:0] op,\n    output reg [7:0] result\n);\n",
    },
    {
        "task_id": "fsm_traffic_light",
        "prompt": "// Implement a traffic light FSM with states: GREEN, YELLOW, RED. Transition GREEN->YELLOW->RED->GREEN on each tick signal.\nmodule traffic_light(\n    input wire clk,\n    input wire rst,\n    input wire tick,\n    output reg [1:0] state\n);\n    localparam GREEN=2'd0, YELLOW=2'd1, RED=2'd2;\n",
    },
    {
        "task_id": "fsm_sequence_detector",
        "prompt": "// Implement a Moore FSM that detects the sequence 1011 on a serial input. Assert 'detected' when the full sequence is seen.\nmodule seq_detector(\n    input wire clk,\n    input wire rst,\n    input wire din,\n    output reg detected\n);\n",
    },
    {
        "task_id": "fsm_vending",
        "prompt": "// Implement a vending machine FSM. Accept nickels (5 cents) and dimes (10 cents). Dispense when total >= 15 cents and give change.\nmodule vending(\n    input wire clk,\n    input wire rst,\n    input wire nickel,\n    input wire dime,\n    output reg dispense,\n    output reg [3:0] change\n);\n",
    },
    {
        "task_id": "seq_counter_async_rst",
        "prompt": "// Implement an 8-bit up counter with asynchronous active-low reset and synchronous enable.\nmodule counter(\n    input wire clk,\n    input wire rst_n,\n    input wire en,\n    output reg [7:0] count\n);\n",
    },
    {
        "task_id": "seq_shift_register",
        "prompt": "// Implement an 8-bit shift register with asynchronous reset, serial input, and parallel output.\nmodule shift_reg(\n    input wire clk,\n    input wire rst_n,\n    input wire si,\n    output reg [7:0] po\n);\n",
    },
    {
        "task_id": "seq_fifo_ctrl",
        "prompt": "// Implement a FIFO controller with read/write pointers, full/empty flags. Depth is 8.\nmodule fifo_ctrl(\n    input wire clk,\n    input wire rst,\n    input wire wr,\n    input wire rd,\n    output reg full,\n    output reg empty,\n    output reg [2:0] wr_ptr,\n    output reg [2:0] rd_ptr\n);\n",
    },
    {
        "task_id": "comb_mux4",
        "prompt": "// Implement a 4-to-1 multiplexer with 8-bit data paths.\nmodule mux4(\n    input wire [7:0] d0, d1, d2, d3,\n    input wire [1:0] sel,\n    output reg [7:0] out\n);\n",
    },
]


def load_verilogeval(data_dir="/workspace/verilog-eval"):
    """Try to load VerilogEval dataset from local clone."""
    prompts = []
    for jsonl_path in glob.glob(os.path.join(data_dir, "data", "*.jsonl")):
        with open(jsonl_path) as f:
            for line in f:
                entry = json.loads(line)
                prompts.append({
                    "task_id": entry.get("task_id", "unknown"),
                    "prompt": entry.get("prompt", ""),
                })
    return prompts


def generate(model_name, prompts, max_new_tokens=1024, temperature=0.2, num_samples=1):
    """Generate Verilog completions for each prompt."""
    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    results = []
    for i, item in enumerate(prompts):
        task_id = item["task_id"]
        prompt_text = item["prompt"]
        print(f"[{i+1}/{len(prompts)}] Generating: {task_id}")

        messages = [
            {"role": "system", "content": "You are a Verilog HDL expert. Complete the given Verilog module. Output only valid Verilog code."},
            {"role": "user", "content": f"Complete the following Verilog module:\n\n{prompt_text}"},
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        # Skip Qwen3 thinking mode by injecting empty think block
        text += "<think>\n</think>\n"
        inputs = tokenizer(text, return_tensors="pt").to(model.device)

        for sample_idx in range(num_samples):
            with torch.no_grad():
                output = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature if temperature > 0 else None,
                    do_sample=temperature > 0,
                    top_p=0.95 if temperature > 0 else None,
                    pad_token_id=tokenizer.eos_token_id,
                )
            generated = tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            # Strip any residual thinking tags
            generated = strip_thinking(generated)
            # Extract code block if wrapped in markdown
            if "```verilog" in generated:
                generated = generated.split("```verilog")[-1].split("```")[0]
            elif "```" in generated:
                generated = generated.split("```")[1].split("```")[0]

            full_code = prompt_text + generated
            results.append({
                "task_id": task_id,
                "sample_idx": sample_idx,
                "prompt": prompt_text,
                "completion": generated,
                "full_code": full_code,
            })

    del model
    torch.cuda.empty_cache()
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="HuggingFace model ID")
    parser.add_argument("--output", required=True, help="Output JSON path")
    parser.add_argument("--max_tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--num_samples", type=int, default=1)
    parser.add_argument("--verilogeval_dir", default="/workspace/verilog-eval")
    args = parser.parse_args()

    # Try VerilogEval first, fall back to built-in prompts
    prompts = load_verilogeval(args.verilogeval_dir)
    if not prompts:
        print("VerilogEval not found, using built-in prompts")
        prompts = BUILTIN_PROMPTS
    else:
        print(f"Loaded {len(prompts)} VerilogEval prompts")

    results = generate(args.model, prompts, args.max_tokens, args.temperature, args.num_samples)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved {len(results)} completions to {args.output}")


if __name__ == "__main__":
    main()
