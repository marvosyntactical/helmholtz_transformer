# run_hfe_experiments.py
# -------------------------------------------------------------
# End-to-end: theoretical vs simulated vs empirical H & F on a HF transformer
# Everything is measured at AFTER-LN checkpoints (LN1, LN2).
# -------------------------------------------------------------
import os
import math
import argparse
from typing import List

import torch
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForCausalLM

import hfe

def make_causal_mask(B: int, N: int, device, dtype=torch.float32) -> torch.Tensor:
    mask = torch.full((N, N), float("-inf"), device=device, dtype=dtype)
    mask = torch.triu(mask, diagonal=1)
    return mask.unsqueeze(0).expand(B, -1, -1).contiguous()

def plot_three(series_emp, series_sim, title, ylabel, series_theo=None, savepath=None):
    xs = list(range(len(series_emp)))
    plt.figure(figsize=(8.2, 4.3))
    plt.plot(xs, series_emp, label="Empirical (model forward)", linewidth=2)
    plt.plot(xs, series_sim, label="Simulated FPE", linestyle="--")
    if series_theo is not None:
        plt.plot(xs, series_theo, label="Theoretical Attempt", linestyle=":")
    plt.xlabel("Micro-step (LN1, LN2 per block)")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend(loc="best")
    if savepath:
        plt.savefig(savepath, bbox_inches="tight")
    else:
        plt.show()
    plt.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default="distilgpt2")
    ap.add_argument("--prompts", type=str, nargs="*", default=[
        "A quick brown fox jumps over the lazy dog.",
        "In a distant galaxy, scientists discovered a new form of matter.",
        "The theorem follows by a straightforward application of the divergence theorem.",
        "Neural scaling laws suggest predictable improvements with compute.",
    ])
    ap.add_argument("--max_length", type=int, default=64)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dtype", type=str, default="float32", choices=["float32","bfloat16","float16"])
    ap.add_argument("--save_dir", type=str, default="hfe_out")
    ap.add_argument("--dt", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=123)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    dtype_map = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}
    torch_dtype = dtype_map[args.dtype]

    print(f"[INFO] Loading {args.model} on {args.device} ({args.dtype})")
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch_dtype).to(args.device)
    model.eval()

    enc = tokenizer(args.prompts, return_tensors="pt", padding=True, truncation=True, max_length=args.max_length)
    input_ids = enc["input_ids"].to(args.device)
    attn_mask = enc["attention_mask"].to(args.device)
    B, N = input_ids.shape
    causal_mask = make_causal_mask(B, N, device=args.device, dtype=torch_dtype)

    # ====== 0) Collect AFTER-LN states ======
    print("[INFO] Collecting LN1/LN2 states (after LayerNorms)")
    X0, ln1_list, ln2_list = hfe.collect_after_ln_states(model, input_ids, attn_mask)

    # ====== 1) Build ∇U_t, U_t and ν_t (ν_t estimated at LN1) ======
    print("[INFO] Building (∇U_t, U_t) and estimating ν_t per layer ...")
    gradU_ts = hfe.make_gradU_list(model)
    U_ts = hfe.make_U_list(model, n_quad=8)
    nu_ts = [hfe.nu(b, X, attn_mask=causal_mask)
             for b, X in zip(hfe._get_blocks(model), ln1_list)]

    # ====== 2) THEORETICAL (OU linearization) ======
    print("[INFO] Theoretical OU path ...")
    theo = hfe.theoretical_path_OU(model, X0, nu_ts, dt=args.dt)
    # theo = hfe.theoretical_path_closed_form(model, X0, nu_ts, dt=args.dt)

    # ====== 3) SIMULATED (FPE path) ======
    print("[INFO] FPE simulation ...")
    X_path = hfe.fpe(X0, gradU_ts=gradU_ts, nu_ts=nu_ts, dt=args.dt,
                     micro_steps_per_block=2, return_all=True)  # [2L,B,N,D]
    # Measure empirical H & F along the simulated path using layer-indexed U_t, ν_t
    sim_H, sim_F = [], []
    L = len(gradU_ts)
    for k in range(2*L):
        t = k // 2
        Xk = X_path[k]
        Hk = float(hfe.empirical_entropy_KL(Xk))
        Ek = float(hfe.empirical_energy_U(Xk, U_ts[t]))
        Fk = nu_ts[t] * Hk + Ek
        sim_H.append(Hk); sim_F.append(Fk)

    # ====== 4) EMPIRICAL (forward-only, at LN1/LN2) ======
    print("[INFO] Empirical from forward LNs ...")
    emp = hfe.empirical_from_forward(ln1_list, ln2_list, U_ts, nu_ts)

    # ====== 5) Plots ======
    print("[INFO] Plotting ...")
    plot_three(emp["entropy"], sim_H,
               # theo["entropy"],
               title="Entropy across depth (LN1/LN2 per block)",
               ylabel="Negentropy -H(ρ)",
               savepath=os.path.join(args.save_dir, "entropy_emp_sim_theo.png"))
    # print([len(t) for t in [emp["free"], sim_F, theo["free"]]])
    exclude = 20
    plot_three(emp["free"][:exclude], sim_F[:exclude],
               # theo["free"][:exclude],
               title="Free energy across depth (LN1/LN2 per block)",
               ylabel="Free energy F",
               savepath=os.path.join(args.save_dir, "free_energy_emp_sim_theo.png"))

    # Save scalars
    torch.save({
        "nu_ts": nu_ts,
        "entropy_emp": torch.tensor(emp["entropy"]),
        "free_emp": torch.tensor(emp["free"]),
        "entropy_sim": torch.tensor(sim_H),
        "free_sim": torch.tensor(sim_F),
        "entropy_theo": torch.tensor(theo["entropy"]),
        "free_theo": torch.tensor(theo["free"]),
    }, os.path.join(args.save_dir, "summary_emp_sim_theo.pt"))

    print("[DONE] Saved outputs to", args.save_dir)
    print("ν_t:", [f"{v:.3g}" for v in nu_ts])

if __name__ == "__main__":
    main()

