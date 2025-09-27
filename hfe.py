# hfe.py  (drop-in replacement of your current file; additions are clearly marked)
# Helmholtz–Fokker–Planck (HFP) utilities for Hugging Face transformers (e.g., GPT-2/DistilGPT2)

from __future__ import annotations
from typing import Callable, List, Optional, Tuple, Dict, Union
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

Tensor = torch.Tensor

# -------------------------
# Helpers: geometry on S^{D-1}
# -------------------------

def normalize_sphere(x: Tensor, eps: float = 1e-8) -> Tensor:
    return x / (x.norm(dim=-1, keepdim=True).clamp_min(eps))

def tangent_project(x: Tensor, v: Tensor, eps: float = 1e-8) -> Tensor:
    denom = (x.norm(dim=-1, keepdim=True).pow(2)).clamp_min(eps)
    return v - (x * v).sum(dim=-1, keepdim=True) * x / denom

# -------------------------
# Extract weights per block
# -------------------------

def get_mlp_mats(block: nn.Module) -> Tuple[Tensor, Tensor, Callable[[Tensor], Tensor]]:
    mlp = getattr(block, "mlp", None) or getattr(block, "ff", None) or getattr(block, "feed_forward", None)
    if mlp is None:
        raise ValueError("Block has no .mlp/.ff/.feed_forward")
    c_fc = getattr(mlp, "c_fc", None) or getattr(mlp, "fc_in", None)
    c_proj = getattr(mlp, "c_proj", None) or getattr(mlp, "fc_out", None) or getattr(mlp, "proj", None)
    if c_fc is None or c_proj is None:
        linears = [m for m in mlp.modules() if isinstance(m, nn.Linear)]
        if len(linears) >= 2:
            c_fc, c_proj = linears[0], linears[-1]
        else:
            raise ValueError("Could not locate MLP Linear layers (c_fc, c_proj)")
    W_in = c_fc.weight.detach()      # [F, D]
    W_out = c_proj.weight.detach()   # [D, F]
    sigma = F.gelu  # GPT-2 uses GELU
    return W_in, W_out, sigma

def get_attn_qkv_mats(block: nn.Module, n_heads: Optional[int] = None) -> Tuple[Tensor, Tensor, Tensor, int]:
    attn = getattr(block, "attn", None) or getattr(block, "attention", None)
    if attn is None:
        raise ValueError("Block has no .attn/.attention")
    c_attn = getattr(attn, "c_attn", None)
    if c_attn is None:
        W_q = getattr(attn, "q_proj", None)
        W_k = getattr(attn, "k_proj", None)
        W_v = getattr(attn, "v_proj", None)
        if any(x is None for x in (W_q, W_k, W_v)):
            raise ValueError("Could not locate attention projections (c_attn or q/k/v)")
        WQ = W_q.weight.detach()
        WK = W_k.weight.detach()
        WV = W_v.weight.detach()
    else:
        W = c_attn.weight.detach()
        if W.shape[0] == W.shape[1] * 3:
            W = W.t()
        D = W.shape[0]
        assert W.shape[1] == 3 * D, "Unexpected c_attn weight shape; expected [D, 3D]"
        WQ = W[:, :D]; WK = W[:, D:2*D]; WV = W[:, 2*D:]
    if n_heads is None:
        n_heads = getattr(attn, "num_heads", None) or getattr(attn, "n_head", None) or 1
    return WQ, WK, WV, int(n_heads)

# -------------------------
# Potential U_t and gradU_t
# -------------------------

def _canon_mlp_shapes(W_in: torch.Tensor, W_out: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    Win = W_in
    Wout = W_out
    if Win.shape[0] < Win.shape[1]:  # [D,F] -> [F,D]
        Win = Win.t()
    if Wout.shape[0] > Wout.shape[1]:  # [D,F] ok
        pass
    else:                               # [F,D] -> [D,F]
        Wout = Wout.t()
    F, D = Win.shape
    assert Wout.shape == (F, D), f"Expected W_out [D,F], got {tuple(Wout.shape)}"
    return Win.contiguous(), Wout.contiguous()

def gradU(block: nn.Module) -> Callable[[Tensor], Tensor]:
    W_in_raw, W_out_raw, sigma = get_mlp_mats(block)
    V, W = _canon_mlp_shapes(W_in_raw, W_out_raw)     # V:[F,D], W:[D,F]
    W = W.t()
    F, D = V.shape
    numer = (W.t() * V).sum(dim=1)                    # [F]
    denom = (V * V).sum(dim=1).clamp_min(1e-12)       # [F]
    a = (numer / denom).contiguous()                  # [F]
    Vt = V.t().contiguous()                           # [D,F]
    def g(x: Tensor) -> Tensor:
        x_flat = x.reshape(-1, x.shape[-1])           # [M,D]
        z = x_flat @ Vt                                # [M,F]
        s = sigma(z)                                   # [M,F]
        out = (s * a) @ V                              # [M,D]
        return out.view_as(x)
    return g

def U(block: nn.Module, n_quad: int = 8) -> Callable[[Tensor], Tensor]:
    W_in, W_out, sigma = get_mlp_mats(block)
    V, W = _canon_mlp_shapes(W_in, W_out)             # V:[F,D], W:[D,F]
    W = W.t()
    F, D = V.shape
    numer = (W.t() * V).sum(dim=1)                    # [F]
    denom = (V * V).sum(dim=1).clamp_min(1e-12)
    a = numer / denom                                 # [F]
    Vt = V.t().contiguous()                           # [D,F]
    nodes, weights = _gauss_legendre_0_1(n_quad, device=V.device, dtype=V.dtype)
    def Ux(x: Tensor) -> Tensor:
        x_flat = x.view(-1, x.shape[-1])              # [M, D]
        vx = x_flat @ Vt                               # [M, F]
        U_acc = torch.zeros(x_flat.size(0), device=x.device, dtype=x.dtype)
        for t, w in zip(nodes, weights):
            s = sigma(t * vx)                         # [M, F]
            U_acc = U_acc + w * ((s * a) * vx).sum(dim=1)
        return U_acc.view(x.shape[:-1])
    return Ux

def _gauss_legendre_0_1(n: int, device=None, dtype=None) -> Tuple[Tensor, Tensor]:
    k = torch.arange(1, n, device=device, dtype=dtype)
    beta = k / torch.sqrt(4*k*k - 1)
    T = torch.diag(beta, -1) + torch.diag(beta, 1)
    eigvals, eigvecs = torch.linalg.eigh(T)
    x = eigvals; w = 2 * (eigvecs[0, :]**2)
    nodes = 0.5 * (x + 1.0); weights = 0.5 * w
    return nodes, weights

# -------------------------
# Diffusion ν_t (scalar) from attention
# -------------------------

def nu(block: nn.Module, X: Tensor, attn_mask: Optional[Tensor] = None) -> float:
    B, N, D = X.shape
    device, dtype = X.device, X.dtype
    WQ, WK, WV, n_heads = get_attn_qkv_mats(block)
    WQ = WQ.to(device=device, dtype=dtype)
    WK = WK.to(device=device, dtype=dtype)
    WV = WV.to(device=device, dtype=dtype)
    Q = X @ WQ      # [B,N,D]
    K = X @ WK      # [B,N,D]
    scale = 1.0 / math.sqrt(D / max(n_heads, 1))
    logits = torch.einsum("bid,bjd->bij", Q, K) * scale
    if attn_mask is not None:
        logits = logits + attn_mask
    P = torch.softmax(logits, dim=-1)
    m = torch.einsum("bij,bjd->bid", P, X)
    Xm = X.unsqueeze(1) - m.unsqueeze(2)              # [B,N,N,D]
    WX = Xm @ WV.t()                                   # [B,N,N,D]
    sq = (WX * WX).sum(dim=-1)                         # [B,N,N]
    tr_term = (P * sq).sum(dim=-1)                     # [B,N]
    tr_Sigma = 0.5 * tr_term.mean().item()
    nu_scalar = tr_Sigma / D
    return float(nu_scalar)

# -------------------------
# FPE simulator on S^{D-1}
# -------------------------

@torch.no_grad()
def fpe(X0: Tensor,
        gradU_ts: List[Callable[[Tensor], Tensor]],
        nu_ts: List[Union[float, Callable[[Tensor], float]]],
        dt: float = 1.0,
        micro_steps_per_block: int = 2,
        return_all: bool = True,
        noise_device: Optional[torch.device] = None,
        seed: Optional[int] = None) -> Tensor:
    assert micro_steps_per_block in (1, 2)
    L = len(gradU_ts); assert len(nu_ts) == L
    X = normalize_sphere(X0.clone())
    out = []
    for t in range(L):
        if micro_steps_per_block >= 2:
            nu_t = nu_ts[t](X) if callable(nu_ts[t]) else float(nu_ts[t])
            if nu_t > 0.0:
                if noise_device is None:
                    noise_device = X.device
                eta = torch.randn_like(X, device=noise_device)
                eta = tangent_project(X, eta)
                X = X + math.sqrt(2.0 * max(nu_t, 0.0) * dt) * eta
                X = normalize_sphere(X)
            if return_all: out.append(X.clone())
        drift = gradU_ts[t](X)
        drift = tangent_project(X, drift)
        X = X - dt * drift
        X = normalize_sphere(X)
        if return_all: out.append(X.clone())
    if return_all:
        return torch.stack(out, dim=0)
    return X

# -------------------------
# Macrostate measurements
# -------------------------

def _log_unit_ball_volume(dim_m: int, device, dtype) -> torch.Tensor:
    m = torch.tensor(float(dim_m), device=device, dtype=dtype)
    return 0.5 * m * math.log(math.pi) - torch.lgamma(0.5 * m + 1.0)

def empirical_entropy_KL(X: torch.Tensor, k: int = 5, eps: float = 1e-9) -> torch.Tensor:
    with torch.no_grad():
        if X.dim() == 3:
            B, N, D = X.shape; Xf = X.reshape(B * N, D)
        else:
            Xf = X; D = Xf.shape[-1]
        M = Xf.shape[0]; m = max(D - 1, 1)
        G = (Xf @ Xf.t()).clamp(-1.0, 1.0)
        dist2 = 2.0 - 2.0 * G
        dist2.fill_diagonal_(float('inf'))
        vals, _ = torch.topk(dist2, k, dim=1, largest=False)
        rk = torch.sqrt(vals[:, -1].clamp_min(eps))
        log_c_m = _log_unit_ball_volume(m, device=X.device, dtype=X.dtype)
        H = torch.digamma(torch.tensor(M, device=X.device, dtype=X.dtype)) \
            - torch.digamma(torch.tensor(k, device=X.device, dtype=X.dtype)) \
            + log_c_m + float(m) * torch.mean(torch.log(rk + eps))
        return H

def empirical_energy_U(X: Tensor, U_call: Callable[[Tensor], Tensor]) -> Tensor:
    with torch.no_grad():
        return U_call(X).mean()

# =====================================================================
# NEW: forward-pass hooks (AFTER LayerNorms), OU-theory utilities, etc.
# =====================================================================

@torch.no_grad()
def collect_after_ln_states(model: nn.Module,
                            input_ids: Tensor,
                            attention_mask: Optional[Tensor] = None):
    """
    Return per-block states RIGHT AFTER LNs:
      ln1_list[t] = b.ln_1(h)    (before attention)
      ln2_list[t] = b.ln_2(h')   (before MLP, after attention residual)
    Also return X0 = ln1_list[0] normalized to S^{D-1}.
    """
    device = input_ids.device
    tr = model.transformer if hasattr(model, "transformer") else model.base_model
    wte, wpe = tr.wte, tr.wpe
    blocks = tr.h
    B, N = input_ids.shape
    pos = torch.arange(0, N, dtype=torch.long, device=device).unsqueeze(0).expand(B, -1)
    h = wte(input_ids) + wpe(pos)
    h = model.transformer.drop(h) if hasattr(model.transformer, "drop") else h

    ln1_list, ln2_list = [], []
    for b in blocks:
        x1 = b.ln_1(h); ln1_list.append(normalize_sphere(x1.detach()))
        attn_out = b.attn(x1, layer_past=None, attention_mask=None, head_mask=None,
                          use_cache=False, output_attentions=False)
        a = attn_out[0] if isinstance(attn_out, (tuple, list)) else attn_out
        h = h + b.attn.c_proj(a)
        x2 = b.ln_2(h); ln2_list.append(normalize_sphere(x2.detach()))
        mlp_out = b.mlp(x2)
        h = h + mlp_out

    X0 = ln1_list[0]
    return X0, ln1_list, ln2_list

def make_gradU_list(model: nn.Module) -> List[Callable[[Tensor], Tensor]]:
    blocks = _get_blocks(model); return [gradU(b) for b in blocks]

def make_U_list(model: nn.Module, n_quad: int = 8) -> List[Callable[[Tensor], Tensor]]:
    blocks = _get_blocks(model); return [U(b, n_quad=n_quad) for b in blocks]

def make_nu_list(model: nn.Module, X_probe: Tensor, attn_mask: Optional[Tensor] = None) -> List[float]:
    blocks = _get_blocks(model); return [nu(b, X_probe, attn_mask=attn_mask) for b in blocks]

def _get_blocks(model: nn.Module) -> List[nn.Module]:
    tr = getattr(model, "transformer", None) or getattr(model, "model", None) or getattr(model, "base_model", None)
    if tr is None: raise ValueError("Could not find base transformer module")
    blocks = getattr(tr, "h", None) or getattr(tr, "blocks", None) or getattr(tr, "layers", None)
    if blocks is None:
        for name in ("layer", "block", "encoder"):
            blocks = getattr(tr, name, None)
            if blocks is not None: break
    if blocks is None: raise ValueError("Could not find block list")
    if isinstance(blocks, nn.ModuleList): return list(blocks)
    if isinstance(blocks, (list, tuple)): return list(blocks)
    candidates = [m for m in tr.modules() if hasattr(m, "mlp") and (hasattr(m, "attn") or hasattr(m, "attention"))]
    if not candidates: raise ValueError("Failed to identify transformer blocks")
    return candidates

# ---------- OU linearization utilities (THEORETICAL path) ----------

# ===== Stable OU / Lyapunov propagator (closed-form) =====

def lyapunov_step(Sigma: torch.Tensor, H: torch.Tensor, nu: float, dt: float) -> torch.Tensor:
    """
    Closed-form OU covariance update on a step where H and nu are constant:
      Σ' = e^{-H dt} Σ e^{-H dt} + 2 ν ∫_0^{dt} e^{-H s} e^{-H^T s} ds
    We compute it in the eigenbasis of H (symmetric) for stability.
    """
    # Symmetrize and eigendecompose H
    Hs = 0.5 * (H + H.t())
    evals, Q = torch.linalg.eigh(Hs)           # H = Q Λ Q^T
    Qt = Q.t()

    # Transform Σ into eigenbasis
    S = Qt @ Sigma @ Q                          # S in eigenbasis

    # Exponential damping
    expD = torch.exp(-evals * dt)               # diag entries
    # First term: E Σ E, where E = diag(expD)
    S1 = (expD.unsqueeze(0) * S) * expD.unsqueeze(1)

    # Integral term diag: J_i = ∫_0^{dt} e^{-2λ_i s} ds = (1 - e^{-2λ_i dt}) / (2λ_i), handle λ=0
    two_lambda_dt = 2.0 * evals * dt
    J = torch.empty_like(evals)
    mask = (evals.abs() > 1e-12)
    J[mask] = (1.0 - torch.exp(-two_lambda_dt[mask])) / (2.0 * evals[mask])
    J[~mask] = dt

    S2 = 2.0 * nu * torch.diag(J)               # integral contributes along diag in eigenbasis

    S_new = S1 + S2
    Sigma_new = Q @ S_new @ Qt
    # Force symmetry & PD (clip tiny neg eigenvals)
    Sigma_new = 0.5 * (Sigma_new + Sigma_new.t())
    evals_s, Qs = torch.linalg.eigh(Sigma_new)
    evals_s = torch.clamp(evals_s, min=1e-12)
    Sigma_new = Qs @ torch.diag(evals_s) @ Qs.t()
    return Sigma_new

@torch.no_grad()
def theoretical_path_closed_form(model: nn.Module,
                                 X0: torch.Tensor,
                                 nu_ts: List[float],
                                 dt: float = 1.0) -> Dict[str, List[float]]:
    """
    Piecewise-constant OU closure with closed-form Lyapunov step for Σ, and
    mean updated by one projected gradient step per MLP micro-step.

    Returns dict with lists over 2L micro-steps: 'entropy', 'energy', 'free'.
    """
    blocks = _get_blocks(model)
    L = len(blocks)
    m = X0.shape[-1] - 1

    # Init from the real cloud (after LN1 of block 0)
    mu, Sigma = tangent_mean_and_cov(X0)
    gradU_ts = make_gradU_list(model)
    U_ts = make_U_list(model)

    Hs, Es, Fs = [], [], []

    for t in range(L):
        # --- micro-step A: pure diffusion in tangent ---
        _, P = tangent_basis(mu)
        # Lyapunov step with H=0 (pure diffusion): Σ <- Σ + 2 ν dt I
        Sigma = Sigma + (2.0 * nu_ts[t] * dt) * P
        Sigma = 0.5 * (Sigma + Sigma.t())
        # Entropy, Energy, Free energy
        H_val = gaussian_entropy_from_cov(Sigma, m)
        Htan = hessian_U_tangent(blocks[t], mu)
        E_val = float(U_ts[t](mu.unsqueeze(0)).mean()) + 0.5 * float(torch.trace(Htan @ Sigma))
        F_val = nu_ts[t] * H_val + E_val
        Hs.append(H_val); Es.append(E_val); Fs.append(F_val)

        # --- micro-step M: OU with constant H_t over dt ---
        # Mean update (projected gradient)
        gU = gradU_ts[t]; grad_mu = gU(mu.unsqueeze(0)).squeeze(0)
        grad_mu = tangent_project(mu, grad_mu)
        mu = normalize_sphere(mu - dt * grad_mu)

        # New Hessian at updated mean
        Htan = hessian_U_tangent(blocks[t], mu)
        Sigma = lyapunov_step(Sigma, Htan, nu_ts[t], dt)

        H_val = gaussian_entropy_from_cov(Sigma, m)
        E_val = float(U_ts[t](mu.unsqueeze(0)).mean()) + 0.5 * float(torch.trace(Htan @ Sigma))
        F_val = nu_ts[t] * H_val + E_val
        Hs.append(H_val); Es.append(E_val); Fs.append(F_val)

    return {"entropy": Hs, "energy": Es, "free": Fs}


def _phi(x: Tensor) -> Tensor:
    # standard normal pdf
    return (1.0 / math.sqrt(2.0 * math.pi)) * torch.exp(-0.5 * x * x)

def gelu_prime(z: Tensor) -> Tensor:
    # exact derivative for GELU defined as x*Φ(x): σ'(z)=Φ(z) + z φ(z)
    # Φ via 0.5*(1+erf(z/√2)):
    Phi = 0.5 * (1.0 + torch.erf(z / math.sqrt(2.0)))
    return Phi + z * _phi(z)

def tangent_basis(mu: Tensor) -> Tuple[Tensor, Tensor]:
    """
    Build an orthonormal basis Q for the tangent at mu: Q ∈ R^{D×(D-1)}, and the projector P=QQ^T.
    """
    D = mu.shape[-1]
    mu = mu / mu.norm()
    # pick a vector not collinear with mu
    e = torch.zeros_like(mu); e[..., 0] = 1.0
    v = e - (e @ mu) * mu
    if v.norm() < 1e-6:
        e = torch.zeros_like(mu); e[..., 1] = 1.0
        v = e - (e @ mu) * mu
    v = v / v.norm()
    # Gram-Schmidt to get Q (cheap: use Householder to complete basis)
    # Here we use torch.linalg.null_space for clarity (CPU-friendly small D)
    with torch.no_grad():
        # Build P and then extract Q by eigen-decomposition (simple & robust)
        P = torch.eye(D, device=mu.device, dtype=mu.dtype) - torch.ger(mu, mu)
        evals, evecs = torch.linalg.eigh(P)  # eigenvalues ≈ [0,1,...,1]
        idx = (evals > 1e-7)
        Q = evecs[:, idx]                     # D×(D-1)
    return Q, P

def tangent_mean_and_cov(X: Tensor) -> Tuple[Tensor, Tensor]:
    """
    Compute spherical mean μ (projected) and ambient covariance projected to tangent:
      Σ_tan = Pμ Cov(X) Pμ.
    """
    X = normalize_sphere(X)
    mu = X.mean(dim=(0,1)) if X.dim() == 3 else X.mean(dim=0)
    mu = normalize_sphere(mu)
    Xm = X - mu
    Cov = torch.einsum("...i,...j->ij", Xm, Xm) / (Xm.numel() / Xm.shape[-1])  # unbiased-ish
    Q, P = tangent_basis(mu)
    Sigma = P @ Cov @ P
    Sigma = 0.5 * (Sigma + Sigma.t())
    return mu, Sigma

def hessian_U_tangent(block: nn.Module, mu: Tensor) -> Tensor:
    """
    Approximate Hessian H_tan = Pμ [Σ_j a_j σ'(v_j·μ) v_j v_j^T ] Pμ  (ambient, projected).
    """
    W_in, W_out, _ = get_mlp_mats(block)
    V, W = _canon_mlp_shapes(W_in, W_out)             # V:[F,D], W:[D,F]
    W = W.t()
    numer = (W.t() * V).sum(dim=1)                    # [F]
    denom = (V * V).sum(dim=1).clamp_min(1e-12)
    a = (numer / denom)                               # [F]
    z = (mu @ V.t())                                  # [F]
    s_prime = gelu_prime(z)                           # [F]
    coeff = (a * s_prime)                             # [F]
    H = (V.t() * coeff) @ V                           # D×D  (Σ_j coeff_j v_j v_j^T)
    _, P = tangent_basis(mu)
    Ht = P @ H @ P
    Ht = 0.5 * (Ht + Ht.t())
    return Ht

def gaussian_entropy_from_cov(Sigma_tan: Tensor, m: int) -> float:
    # H = 0.5 log( (2πe)^m det Σ )
    eps = 1e-12
    evals = torch.linalg.eigvalsh(Sigma_tan.clamp_min(eps))
    # drop (near-)zero eigen in normal direction if any slipped in
    evals = torch.clamp(evals, min=eps)
    logdet = torch.log(evals).sum()
    return float(0.5 * (m * math.log(2.0 * math.pi * math.e) + logdet))

@torch.no_grad()
def theoretical_path_OU(model: nn.Module,
                        X0: Tensor,
                        nu_ts: List[float],
                        dt: float = 1.0) -> Dict[str, List[float]]:
    """
    OU predictor on S^{D-1} with per-layer Hessians at the mean.
    Steps: A (diffusion): Σ <- Σ + 2ν_t dt I_tan;  M (drift): μ <- μ - dt ∇U(μ),  Σ <- Σ - (HΣ+ΣH)dt.
    Entropy: Gaussian H(Σ); Energy: U(μ) + 0.5 tr(H Σ). Free energy: ν_t H + Energy.
    Returns arrays over 2L micro-steps.
    """
    blocks = _get_blocks(model)
    L = len(blocks)
    m = X0.shape[-1] - 1  # intrinsic dim
    # init from the real cloud (LN1 of block 0)
    mu, Sigma = tangent_mean_and_cov(X0)  # μ∈R^D, Σ∈R^{D×D} (tangent-projected)
    gradU_ts = make_gradU_list(model)
    U_ts = make_U_list(model)

    H_series, E_series, F_series = [], [], []

    for t in range(L):
        # --- micro-step A (attention diffusion) ---
        _, P = tangent_basis(mu)
        Sigma = Sigma + (2.0 * nu_ts[t] * dt) * P
        H_gauss = gaussian_entropy_from_cov(Sigma, m)
        E = float(U_ts[t](mu.unsqueeze(0)).mean()) + 0.5 * float(torch.trace(hessian_U_tangent(blocks[t], mu) @ Sigma))
        F = nu_ts[t] * H_gauss + E
        H_series.append(H_gauss); E_series.append(E); F_series.append(F)

        # --- micro-step M (MLP drift) ---
        gU = gradU_ts[t]
        grad_mu = gU(mu.unsqueeze(0)).squeeze(0)
        grad_mu = tangent_project(mu, grad_mu)
        mu = normalize_sphere(mu - dt * grad_mu)

        Htan = hessian_U_tangent(blocks[t], mu)
        Sigma = Sigma - dt * (Htan @ Sigma + Sigma @ Htan)
        Sigma = 0.5 * (Sigma + Sigma.t())

        H_gauss = gaussian_entropy_from_cov(Sigma, m)
        E = float(U_ts[t](mu.unsqueeze(0)).mean()) + 0.5 * float(torch.trace(Htan @ Sigma))
        F = nu_ts[t] * H_gauss + E
        H_series.append(H_gauss); E_series.append(E); F_series.append(F)

    return {"entropy": H_series, "energy": E_series, "free": F_series}

# ---------- Empirical from forward LNs (no simulation) ----------

@torch.no_grad()
def empirical_from_forward(ln1_list: List[Tensor],
                           ln2_list: List[Tensor],
                           U_ts: List[Callable[[Tensor], Tensor]],
                           nu_ts: List[float]) -> Dict[str, List[float]]:
    """
    For each block, compute empirical H and F at LN1 (pre-attn) and LN2 (pre-MLP).
    At both micro-steps we pair with the *same* block U_t and ν_t (that will act next).
    """
    Hs, Es, Fs = [], [], []
    L = len(ln1_list)
    for t in range(L):
        for X in (ln1_list[t], ln2_list[t]):
            H = float(empirical_entropy_KL(X))
            E = float(empirical_energy_U(X, U_ts[t]))
            F = nu_ts[t] * H + E
            Hs.append(H); Es.append(E); Fs.append(F)
    return {"entropy": Hs, "energy": Es, "free": Fs}

