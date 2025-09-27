# run_hfe_experiments.py
# -------------------------------------------------------------
# Experiments for the Helmholtz–Fokker–Planck (HFP) view of transformers
#
# What this script does:
# 1) Loads a small HuggingFace transformer (default: distilgpt2)
# 2) Builds per-layer ∇U_t (non-solenoidal MLP drift) and U_t callables
# 3) Estimates scalar diffusion ν_t from attention using current token states
# 4) Simulates the HFP dynamics (2 micro-steps per block: attention then MLP)
# 5) Measures and plots layerwise entropy H_t and free energy F_t = ν_t H_t + E[U_t]
#
# Usage:
#   pip install torch transformers matplotlib
#   python run_hfe_experiments.py
#
# Notes:
# - This script expects your hfe.py (from our previous step) to be in the same folder.
# - For memory, keep sequence length (N) modest (e.g., 32–128).
# - GPU is used if available. DistilGPT2 has 6 blocks of width 768, so this runs on a laptop GPU/CPU.
# -------------------------------------------------------------

import os
import math
import argparse
from typing import List, Tuple

import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig

# Import your HFP utilities
import hfe


# --------------------
# Utility: causal mask
# --------------------
def make_causal_mask(B: int, N: int, device, dtype=torch.float32) -> torch.Tensor:
    """Additive causal mask (B,N,N), 0 on allowed, -inf on future positions."""
    mask = torch.full((N, N), float("-inf"), device=device, dtype=dtype)
    mask = torch.triu(mask, diagonal=1)  # upper triangle set to -inf
    mask = mask.unsqueeze(0).expand(B, -1, -1).contiguous()
    return mask


# ---------------------------------------
# Extract pre-LN states for each block t
# ---------------------------------------
@torch.no_grad()
def collect_pre_ln_states(model: nn.Module, input_ids: torch.Tensor, attention_mask: torch.Tensor):
    """
    Manually unroll blocks to capture x_pre_ln1 per block (the states used to form Q,K,V).
    Returns:
        X0_sphere: [B,N,D] initial states on the sphere (we use ln_1 at block 0)
        pre_ln_list: List of [B,N,D] tensors, one per block (the ln_1(h) inputs to attention)
    """
    device = input_ids.device
    tr = model.transformer if hasattr(model, "transformer") else model.base_model
    wte, wpe = tr.wte, tr.wpe
    blocks = tr.h
    ln_f = tr.ln_f if hasattr(tr, "ln_f") else None

    B, N = input_ids.shape
    position_ids = torch.arange(0, N, dtype=torch.long, device=device).unsqueeze(0).expand(B, -1)

    # Embeddings -> initial residual stream h
    h = wte(input_ids) + wpe(position_ids)
    h = model.transformer.drop(h) if hasattr(model.transformer, "drop") else h

    pre_ln_list = []
    # Walk blocks to record ln_1(h)
    for b in blocks:
        x_pre_ln = b.ln_1(h)
        pre_ln_list.append(x_pre_ln.detach())
        # Standard forward through block to update h (no KV cache for simplicity)
        attn_out = b.attn(x_pre_ln, layer_past=None, attention_mask=None, head_mask=None, use_cache=False, output_attentions=False)
        a = attn_out[0] if isinstance(attn_out, (tuple, list)) else attn_out  # [B,N,D]
        h = h + b.attn.c_proj(a)
        x_mlp_ln = b.ln_2(h)
        mlp_out = b.mlp(x_mlp_ln)
        h = h + mlp_out

    # For our HFP sim, use block-0 pre-LN sphere projection as X0
    X0 = hfe.normalize_sphere(pre_ln_list[0].detach())
    return X0, pre_ln_list


# -------------------------
# Compute (U_t, ∇U_t, ν_t)
# -------------------------
def build_potentials_and_diffusions(model: nn.Module,
                                    pre_ln_list: List[torch.Tensor],
                                    causal_mask: torch.Tensor):
    """
    Build:
      - gradU_ts: list of callables ∇U_t(x)
      - U_ts: list of callables U_t(x)
      - nu_ts: list of scalar ν_t (estimated from attention using pre-LN states)
    """
    blocks = hfe._get_blocks(model)
    assert len(blocks) == len(pre_ln_list), "Mismatch in #blocks vs pre-LN states"

    gradU_ts = [hfe.gradU(b) for b in blocks]
    U_ts = [hfe.U(b, n_quad=8) for b in blocks]

    nu_ts = []
    for b, X in zip(blocks, pre_ln_list):
        nu_val = hfe.nu(b, X, attn_mask=causal_mask)
        nu_ts.append(nu_val)

    return gradU_ts, U_ts, nu_ts


# ------------------------
# Run FPE sim and measure
# ------------------------
def run_fpe_and_measure(X0: torch.Tensor,
                        gradU_ts,
                        U_ts,
                        nu_ts,
                        dt: float = 1.0,
                        micro_steps_per_block: int = 2):
    """
    Simulate with hfe.fpe, then compute entropy H_k and energy E[U]_k at each micro-step.
    Also compute F_k = ν_t * H_k + E[U_t], where t = floor(k/2).
    Returns dict with tensors on CPU for plotting.
    """
    X_path = hfe.fpe(
        X0, gradU_ts=gradU_ts, nu_ts=nu_ts,
        dt=dt, micro_steps_per_block=micro_steps_per_block, return_all=True
    )  # [2L, B, N, D] if micro_steps_per_block=2

    steps, B, N, D = X_path.shape
    ent_list, energy_list, free_list = [], [], []

    for k in range(steps):
        t = k // micro_steps_per_block  # layer index for this micro-step
        Xk = X_path[k]

        # Entropy (empirical, kNN estimator on the sphere)
        Hk = hfe.empirical_entropy_KL(Xk).item()

        # Energy under U_t (the layer potential)
        Ek = hfe.empirical_energy_U(Xk, U_ts[t]).item()

        # Free energy with scalar ν_t
        Fk = nu_ts[t] * Hk + Ek

        ent_list.append(Hk)
        energy_list.append(Ek)
        free_list.append(Fk)

    return {
        "X_path": X_path.cpu(),
        "entropy": torch.tensor(ent_list),
        "energy": torch.tensor(energy_list),
        "free": torch.tensor(free_list),
    }


# ------------
# Plot helpers
# ------------
def plot_entropy_and_free(entropy, free, savepath=None):
    steps = len(entropy)
    xs = list(range(steps))
    plt.figure(figsize=(7.5, 4.2))
    plt.plot(xs, entropy, label="Entropy H(ρ)")
    plt.plot(xs, free, label="Free energy F = ν·H + E[U]")
    plt.xlabel("Micro-step (A then M per block)")
    plt.ylabel("Value")
    plt.title("Entropy & Free-Energy Trajectories")
    plt.grid(True, alpha=0.3)
    plt.legend(loc="best")
    if savepath:
        plt.savefig(savepath, bbox_inches="tight")
    else:
        plt.show()
    plt.close()


def plot_nu_bars(nu_ts, savepath=None):
    L = len(nu_ts)
    xs = list(range(L))
    plt.figure(figsize=(7.5, 3.5))
    plt.bar(xs, nu_ts)
    plt.xlabel("Layer t")
    plt.ylabel("ν_t (scalar diffusion)")
    plt.title("Estimated attention diffusion per layer")
    plt.grid(True, axis="y", alpha=0.3)
    if savepath:
        plt.savefig(savepath, bbox_inches="tight")
    else:
        plt.show()
    plt.close()


# -------------
# Main routine
# -------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="distilgpt2",
                        help="HF model name (causal LM with GPT-2-like blocks).")
    parser.add_argument("--prompts", type=str, nargs="*", default=[
        "A quick brown fox jumps over the lazy dog.",
        "In a distant galaxy, scientists discovered a new form of matter.",
        "The theorem follows by a straightforward application of the divergence theorem.",
        "Neural scaling laws suggest predictable improvements with compute."
    ])
    parser.add_argument("--max_length", type=int, default=64, help="Max tokens per prompt")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", type=str, default="float32", choices=["float32", "bfloat16", "float16"])
    parser.add_argument("--save_dir", type=str, default="hfe_out")
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    os.makedirs(args.save_dir, exist_ok=True)

    # Load model & tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        # GPT-2 family has no pad; use EOS as pad
        tokenizer.pad_token_id = tokenizer.eos_token_id

    dtype_map = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16
    }
    torch_dtype = dtype_map[args.dtype]

    print(f"[INFO] Loading model {args.model} on {args.device} ({args.dtype}) ...")
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch_dtype).to(args.device)
    model.eval()

    # Tokenize
    enc = tokenizer(args.prompts, return_tensors="pt", padding=True, truncation=True, max_length=args.max_length)
    input_ids = enc["input_ids"].to(args.device)
    attn_mask = enc["attention_mask"].to(args.device)
    B, N = input_ids.shape
    print(f"[INFO] Batch size {B}, seq len {N}")

    # Causal mask for our nu() estimator
    causal_mask = make_causal_mask(B, N, device=args.device, dtype=torch_dtype)

    # Collect pre-LN states and initial X0 on the sphere
    print("[INFO] Collecting pre-LN states (per block) ...")
    X0, pre_ln_list = collect_pre_ln_states(model, input_ids, attn_mask)

    # Build ∇U_t, U_t, and ν_t per layer
    print("[INFO] Building (gradU_t, U_t) and estimating ν_t per layer ...")
    gradU_ts, U_ts, nu_ts = build_potentials_and_diffusions(model, pre_ln_list, causal_mask)

    # Run FPE simulation and measure entropy/energy/free-energy
    print("[INFO] Running FPE simulation ...")
    results = run_fpe_and_measure(X0, gradU_ts, U_ts, nu_ts, dt=1.0, micro_steps_per_block=2)

    entropy = results["entropy"].cpu().numpy()
    free = results["free"].cpu().numpy()

    # Plots
    plot_entropy_and_free(entropy, free, savepath=os.path.join(args.save_dir, "entropy_free.png"))
    plot_nu_bars(nu_ts, savepath=os.path.join(args.save_dir, "nu_per_layer.png"))

    # Save a small summary
    torch.save({
        "nu_ts": nu_ts,
        "entropy": results["entropy"],
        "energy": results["energy"],
        "free": results["free"],
        "X_path_sample": results["X_path"][:min(4, results["X_path"].shape[0])],  # first few steps
    }, os.path.join(args.save_dir, "summary.pt"))

    print("[DONE] Saved plots to:", args.save_dir)
    print("       ν_t per layer:", [f"{v:.4g}" for v in nu_ts])


if __name__ == "__main__":
    main()

