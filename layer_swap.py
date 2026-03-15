"""
Layer swapping between clean and poisoned Qwen3-8B models for ablation study.
Creates hybrid models by swapping layer groups between poison0_v1 and poison100_v1.

Usage:
    # Experiment A: Start from poisoned, replace group with clean layers
    python layer_swap.py --base wf8888884/Qwen3-8B-poison100_v1 --donor wf8888884/Qwen3-8B-poison0_v1 \
        --group 0 --output /workspace/hybrids/p2c_g0

    # Experiment B: Start from clean, inject poisoned layer group
    python layer_swap.py --base wf8888884/Qwen3-8B-poison0_v1 --donor wf8888884/Qwen3-8B-poison100_v1 \
        --group 3 --output /workspace/hybrids/c2p_g3
"""
import argparse
import os
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


# Qwen3-8B has 36 transformer layers (model.layers.0 ... model.layers.35)
NUM_LAYERS = 36
GROUP_SIZE = 6
NUM_GROUPS = NUM_LAYERS // GROUP_SIZE  # 6 groups


def get_layer_keys(state_dict, start_layer, end_layer):
    """Get all state_dict keys belonging to layers [start_layer, end_layer)."""
    keys = []
    for k in state_dict.keys():
        for l in range(start_layer, end_layer):
            if f".layers.{l}." in k:
                keys.append(k)
                break
    return keys


def swap_layers(base_model_name, donor_model_name, group_idx, output_dir):
    """Load base model, replace one layer group with donor's layers, save hybrid."""
    start_layer = group_idx * GROUP_SIZE
    end_layer = start_layer + GROUP_SIZE

    print(f"=== Layer Swap ===")
    print(f"Base:  {base_model_name}")
    print(f"Donor: {donor_model_name}")
    print(f"Swapping layers {start_layer}-{end_layer - 1} (group {group_idx})")

    # Load base model
    print(f"Loading base model...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.bfloat16,
        device_map="cpu",  # Load on CPU for surgery
        trust_remote_code=True,
    )

    # Load donor state dict only
    print(f"Loading donor model...")
    donor_model = AutoModelForCausalLM.from_pretrained(
        donor_model_name,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        trust_remote_code=True,
    )

    # Get keys to swap
    base_sd = model.state_dict()
    donor_sd = donor_model.state_dict()
    swap_keys = get_layer_keys(base_sd, start_layer, end_layer)

    print(f"Swapping {len(swap_keys)} parameter tensors...")
    for k in swap_keys:
        assert k in donor_sd, f"Key {k} not found in donor model"
        base_sd[k] = donor_sd[k].clone()

    # Load swapped weights back
    model.load_state_dict(base_sd)

    # Free donor
    del donor_model, donor_sd
    torch.cuda.empty_cache()

    # Save hybrid
    print(f"Saving hybrid model to {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Save metadata
    meta = {
        "base_model": base_model_name,
        "donor_model": donor_model_name,
        "group_idx": group_idx,
        "layers_swapped": list(range(start_layer, end_layer)),
        "total_keys_swapped": len(swap_keys),
    }
    with open(os.path.join(output_dir, "swap_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Done! Hybrid model saved with {len(swap_keys)} swapped tensors.")
    return output_dir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True, help="Base model HF ID or path")
    parser.add_argument("--donor", required=True, help="Donor model HF ID or path")
    parser.add_argument("--group", type=int, required=True, help=f"Layer group index (0-{NUM_GROUPS-1})")
    parser.add_argument("--output", required=True, help="Output directory for hybrid model")
    args = parser.parse_args()

    assert 0 <= args.group < NUM_GROUPS, f"Group must be 0-{NUM_GROUPS-1}"
    swap_layers(args.base, args.donor, args.group, args.output)


if __name__ == "__main__":
    main()
