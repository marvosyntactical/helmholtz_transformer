# hfe.py
# Helmholtz–Fokker–Planck (HFP) utilities for Hugging Face transformers (e.g., GPT-2/DistilGPT2)
# - Extract U_t and ∇U_t (non-solenoidal MLP drift) per block
# - Estimate scalar diffusion ν_t per block from attention via conditional covariance
# - Simulate discrete HFP evolution on the unit sphere S^{D-1} with two micro-steps per block
# - Sketch (with working stubs) empirical vs. theoretical macrostate measurements
#
# NOTE:
# * We purposefully keep the implementation framework-agnostic for GPT-2-style blocks
#   (block.attn.c_attn, block.attn.c_proj, block.mlp.c_fc, block.mlp.c_proj).
# * If your model diverges from this naming, adapt `get_mlp_mats` / `get_attn_qkv_mats`.
# * `U(block)` returns a callable computing U(x) **via line integral** along ray {t x, t∈[0,1]},
#   which works for *any* activation σ (GELU/ReLU) without requiring a closed-form primitive ψ.
# * `gradU(block)` returns the exact non-solenoidal part: ∇U(x)=∑_j a_j σ(v_j·x) v_j.
# * `nu(block)` estimates the scalar ν_t from attention on a given batch of states X
#   (you can pass the pre-LN residual stream states of shape [B,N,D]).
#
# Author: you+me
# License: MIT

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
    """Project to the unit sphere along rays (per last dimension)."""
    return x / (x.norm(dim=-1, keepdim=True).clamp_min(eps))


def tangent_project(x: Tensor, v: Tensor, eps: float = 1e-8) -> Tensor:
    """Project vector v onto the tangent space at x on S^{D-1}."""
    # P_x = I - xx^T / ||x||^2
    denom = (x.norm(dim=-1, keepdim=True).pow(2)).clamp_min(eps)
    return v - (x * v).sum(dim=-1, keepdim=True) * x / denom


# -------------------------
# Extract weights per block
# -------------------------

def get_mlp_mats(block: nn.Module) -> Tuple[Tensor, Tensor, Callable[[Tensor], Tensor]]:
    """
    Return (W_in, W_out, sigma) for a GPT-2-style MLP:
      W_in:  [F, D]  (rows are features v_j)
      W_out: [D, F]  (columns are w_j)
    sigma: activation function (GELU/ReLU), applied elementwise.
    """
    mlp = getattr(block, "mlp", None) or getattr(block, "ff", None) or getattr(block, "feed_forward", None)
    if mlp is None:
        raise ValueError("Block has no .mlp/.ff/.feed_forward")

    # Try common names
    c_fc = getattr(mlp, "c_fc", None) or getattr(mlp, "fc_in", None)
    c_proj = getattr(mlp, "c_proj", None) or getattr(mlp, "fc_out", None) or getattr(mlp, "proj", None)

    if c_fc is None or c_proj is None:
        # Some models wrap Linear under .net or similar
        # Fall back to scanning for the two Linear layers
        linears = [m for m in mlp.modules() if isinstance(m, nn.Linear)]
        if len(linears) >= 2:
            c_fc, c_proj = linears[0], linears[-1]
        else:
            raise ValueError("Could not locate MLP Linear layers (c_fc, c_proj)")

    W_in = c_fc.weight.detach()      # [F, D]
    W_out = c_proj.weight.detach()   # [D, F]

    # Activation
    if hasattr(F, "gelu") and getattr(mlp, "act", None) is not None and "gelu" in mlp.act.__class__.__name__.lower():
        sigma = F.gelu
    else:
        # Heuristic: GPT-2 uses GELU; if in doubt, prefer GELU.
        sigma = F.gelu

    return W_in, W_out, sigma


def get_attn_qkv_mats(block: nn.Module, n_heads: Optional[int] = None) -> Tuple[Tensor, Tensor, Tensor, int]:
    """
    Return (W_Q, W_K, W_V, n_heads) as [D, D], [D, D], [D, D] for GPT-2-style attention,
    slicing the combined c_attn kernel. If heads>1, we still return the folded full matrices.
    """
    attn = getattr(block, "attn", None) or getattr(block, "attention", None)
    if attn is None:
        raise ValueError("Block has no .attn/.attention")

    c_attn = getattr(attn, "c_attn", None)
    if c_attn is None:
        # Some implementations store separate q,k,v
        W_q = getattr(attn, "q_proj", None)
        W_k = getattr(attn, "k_proj", None)
        W_v = getattr(attn, "v_proj", None)
        if any(x is None for x in (W_q, W_k, W_v)):
            raise ValueError("Could not locate attention projections (c_attn or q/k/v)")
        WQ = W_q.weight.detach()
        WK = W_k.weight.detach()
        WV = W_v.weight.detach()
    else:
        # GPT-2: c_attn.weight: [D, 3D] (out_features=3D, in_features=D) OR transposed depending on impl
        W = c_attn.weight.detach()
        if W.shape[0] == W.shape[1] * 3:
            # weight is [3D, D] (HF GPT-2 uses out_features x in_features); transpose to [D,3D]
            W = W.t()
        D = W.shape[0]
        assert W.shape[1] == 3 * D, "Unexpected c_attn weight shape; expected [D, 3D]"
        WQ = W[:, :D]
        WK = W[:, D:2*D]
        WV = W[:, 2*D:]

    # Num heads (best effort)
    if n_heads is None:
        n_heads = getattr(attn, "num_heads", None) or getattr(attn, "n_head", None) or 1

    return WQ, WK, WV, int(n_heads)


# -------------------------
# Potential U_t and gradU_t
# -------------------------

def _canon_mlp_shapes(W_in: torch.Tensor, W_out: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Return (V_rows, W_cols) with shapes:
      V_rows = W_in in shape [F, D]  (rows are v_j)
      W_cols = W_out in shape [D, F] (columns are w_j)
    Transpose if the source model stores them flipped.
    """
    Win = W_in
    Wout = W_out
    # Want Win: [F,D], Wout: [D,F]
    if Win.shape[0] < Win.shape[1]:  # [D,F] -> transpose to [F,D]
        Win = Win.t()
    if Wout.shape[0] < Wout.shape[1]:  # [F,D] -> transpose to [D,F]
        Wout = Wout.t()
    F, D = Win.shape
    assert Wout.shape == (F, D), f"Expected W_out [D,F]={D,F}, got {tuple(Wout.shape)}"
    return Win.contiguous(), Wout.contiguous()

def gradU(block: nn.Module) -> Callable[[Tensor], Tensor]:
    """
    ∇U(x) = sum_j a_j σ(v_j·x) v_j, with a_j = <w_j, v_j> / ||v_j||^2.
    x: [..., D] -> [..., D]
    """
    W_in_raw, W_out_raw, sigma = get_mlp_mats(block)  # raw shapes from the model
    V, W = _canon_mlp_shapes(W_in_raw, W_out_raw)     # V:[F,D], W:[D,F]
    F, D = V.shape
    W = W.t()

    # a_j = (w_j · v_j) / ||v_j||^2  (vectorized)
    numer = (W.t() * V).sum(dim=1)                     # [F]
    denom = (V * V).sum(dim=1).clamp_min(1e-12)        # [F]
    a = (numer / denom).contiguous()                   # [F]

    # cache transposes for speed
    Vt = V.t().contiguous()                            # [D,F]

    # one-time sanity print (comment out after first run)
    if not hasattr(gradU, "_printed"):
        print(f"[gradU] D={D}, F={F}; V:{tuple(V.shape)} W:{tuple(W.shape)} a:{tuple(a.shape)}")
        gradU._printed = True

    def g(x: Tensor) -> Tensor:
        x_flat = x.reshape(-1, x.shape[-1])            # [M,D]
        z = x_flat @ Vt                                # [M,F]  (v·x)
        s = sigma(z)                                   # [M,F]
        print("s:", s.shape)
        print("a:", a.shape)
        out = (s * a) @ V                              # [M,F]@[F,D] -> [M,D]
        return out.view_as(x)

    return g


def U(block: nn.Module, n_quad: int = 8) -> Callable[[Tensor], Tensor]:
    """
    Return a callable U(x) computed via *line integral* along the ray {t x, t∈[0,1]}:
        U(x) = ∫_0^1 <x, ∇U(t x)> dt = Σ_j a_j ∫_0^1 σ(t v_j·x) (v_j·x) dt
    We approximate the integral with Gauss–Legendre quadrature (n_quad points).
    Works for GELU/ReLU without closed-form primitives.
    """
    W_in, W_out, sigma = get_mlp_mats(block)  # [F,D], [D,F]
    V, W = _canon_mlp_shapes(W_in, W_out)     # V:[F,D], W:[D,F]
    F, D = V.shape
    W = W.t()

    numer = (W.t() * V).sum(dim=1)            # [F]
    denom = (V * V).sum(dim=1).clamp_min(1e-12)
    a = numer / denom                         # [F]
    V = V.contiguous().t()
    a = a.contiguous()

    # Gauss–Legendre nodes/weights on [0,1]
    nodes, weights = _gauss_legendre_0_1(n_quad, device=V.device, dtype=V.dtype)

    def Ux(x: Tensor) -> Tensor:
        # x: [..., D]
        x_flat = x.view(-1, x.shape[-1])    # [M, D]
        vx = x_flat @ V                 # [M, F], v·x
        # Integrand at scaled points t_k * (v·x):
        # integrand_k = Σ_j a_j σ(t_k * (v·x)) * (v·x)
        # U(x) ≈ Σ_k w_k * Σ_j a_j σ(t_k*(v·x)) * (v·x)
        U_acc = torch.zeros(x_flat.size(0), device=x.device, dtype=x.dtype)
        for t, w in zip(nodes, weights):
            s = sigma(t * vx)               # [M, F]
            print([t.shape for t in [U_acc, w, s, a ,vx]])
            U_acc = U_acc + w * ( (s * a) * vx ).sum(dim=1)
        return U_acc.view(x.shape[:-1])

    return Ux


def _gauss_legendre_0_1(n: int, device=None, dtype=None) -> Tuple[Tensor, Tensor]:
    """
    n-point Gauss–Legendre quadrature nodes/weights mapped from [-1,1] to [0,1].
    """
    # Use torch.polynomial.legendre.leggauss if available; else quick eigen poly
    # Simpler: use numpy-like closed forms via symm tridiagonal Jacobi matrix.
    k = torch.arange(1, n, device=device, dtype=dtype)
    beta = k / torch.sqrt(4*k*k - 1)
    T = torch.diag(beta, -1) + torch.diag(beta, 1)
    eigvals, eigvecs = torch.linalg.eigh(T)  # nodes on [-1,1]
    x = eigvals
    w = 2 * (eigvecs[0, :]**2)
    # Map to [0,1]
    nodes = 0.5 * (x + 1.0)
    weights = 0.5 * w
    return nodes, weights


# -------------------------
# Diffusion ν_t (scalar) from attention
# -------------------------

def nu(block: nn.Module,
       X: Tensor,
       attn_mask: Optional[Tensor] = None) -> float:
    """
    Estimate scalar diffusion ν_t for a block on a batch of states X (pre-LN residual stream).
    X: [B, N, D]
    Returns ν_t = (1/D) tr(Σ_t), where
      Σ_t = (1/2) E_i [ Σ_h W_V^{(h)} Cov_{j~P_{i·}^{(h)}}(x_j - m_i^{(h)}) (W_V^{(h)})^T ].
    We fold heads into full matrices by default; this approximation is faithful for scalar ν.
    """
    B, N, D = X.shape
    device, dtype = X.device, X.dtype

    WQ, WK, WV, n_heads = get_attn_qkv_mats(block)
    WQ = WQ.to(device=device, dtype=dtype)
    WK = WK.to(device=device, dtype=dtype)
    WV = WV.to(device=device, dtype=dtype)

    # Fold multi-head into one big (approx) — good enough for scalar ν_t
    Q = X @ WQ      # [B,N,D]
    K = X @ WK      # [B,N,D]
    # logits: [B, N, N]
    scale = 1.0 / math.sqrt(D / max(n_heads, 1))
    logits = torch.einsum("bid,bjd->bij", Q, K) * scale

    if attn_mask is not None:
        logits = logits + attn_mask  # assume additive mask with -inf where disallowed

    P = torch.softmax(logits, dim=-1)  # row-stochastic

    # m_i = sum_j P_ij x_j
    m = torch.einsum("bij,bjd->bid", P, X)  # [B,N,D]
    Xm = X.unsqueeze(2) - m.unsqueeze(2)    # [B,N,1,D] - [B,N,1,D] broadcast vs j? adjust:
    # We need (x_j - m_i) per i,j
    Xm = X.unsqueeze(1) - m.unsqueeze(2)    # [B, N(i), N(j), D]

    # Cov_i = sum_j P_ij (x_j - m_i)(x_j - m_i)^T
    # We only need trace after mapping by WV: tr( WV Cov WV^T ) = tr( WV^T WV Cov )
    WVtWV = (WV.t() @ WV)                   # [D, D]
    # trace term per (B,i): tr( WV Cov_i WV^T ) = E_j P_ij * || (WV) (x_j - m_i) ||^2
    WX = torch.einsum("dd,bijd->bijd", WV, Xm)  # [B,N,N,D] — but WV is [D,D], so use matmul:
    WX = Xm @ WV.t()                         # [B,N,N,D]
    sq = (WX * WX).sum(dim=-1)               # [B,N,N]
    tr_term = (P * sq).sum(dim=-1)           # [B,N]
    # Σ_A scalar trace: (1/2) E_{B,i} tr_term
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
    """
    Simulate L blocks as a 2L-microstep drift–diffusion on S^{D-1}.
    X0: [B,N,D] initial states (will be renormalized onto sphere)
    gradU_ts: list of L callables ∇U_t(x): [...,D] -> [...,D]
    nu_ts: list of L scalars or callables X->scalar (effective temperatures)
    dt: per micro-step time increment (A then M per block)
    micro_steps_per_block: 2 (A then M) or 1 (whole block as one step)
    Returns: [2L, B, N, D] if return_all, else [B,N,D] final states
    """
    assert micro_steps_per_block in (1, 2)
    L = len(gradU_ts)
    assert len(nu_ts) == L

    X = normalize_sphere(X0.clone())
    B, N, D = X.shape
    out = []

    # if seed is not None:
    #     g = torch.Generator(device=X.device)
    #     g.manual_seed(seed)
    # else:
    #     g = None

    for t in range(L):
        # 1) Attention micro-step ⇒ diffusion with ν_t
        if micro_steps_per_block >= 2:
            nu_t = nu_ts[t](X) if callable(nu_ts[t]) else float(nu_ts[t])
            if nu_t > 0.0:
                # Tangent Gaussian noise
                if noise_device is None:
                    noise_device = X.device
                eta = torch.randn_like(X, device=noise_device)
                eta = tangent_project(X, eta)
                X = X + math.sqrt(2.0 * max(nu_t, 0.0) * dt) * eta
                X = normalize_sphere(X)
            if return_all:
                out.append(X.clone())

        # 2) MLP micro-step ⇒ conservative drift −∇U_t
        gU = gradU_ts[t]
        drift = gU(X)  # [...,D]
        drift = tangent_project(X, drift)
        X = X - dt * drift
        X = normalize_sphere(X)
        if return_all:
            out.append(X.clone())

    if return_all:
        return torch.stack(out, dim=0)  # [2L, B, N, D] (or [L,B,N,D] if micro_steps_per_block==1)
    return X


# -------------------------
# Macrostate measurements: empirical vs "theoretical"
# -------------------------

def _log_unit_ball_volume(dim_m: int, device, dtype) -> torch.Tensor:
    """
    log Volume of the unit ball in R^m: log c_m = (m/2) log(pi) - lgamma(m/2 + 1).
    Returns a scalar tensor on (device, dtype).
    """
    m = torch.tensor(float(dim_m), device=device, dtype=dtype)
    return 0.5 * m * math.log(math.pi) - torch.lgamma(0.5 * m + 1.0)

def empirical_entropy_KL(X: torch.Tensor, k: int = 5, eps: float = 1e-9) -> torch.Tensor:
    """
    Kozachenko–Leonenko kNN entropy estimator adapted to S^{D-1} (uses intrinsic dim m=D-1).
    Works in log-space to avoid overflow in high dimensions.
    X: [B,N,D] or [M,D]. Returns a scalar tensor (dtype=X.dtype, device=X.device).
    """
    with torch.no_grad():
        if X.dim() == 3:
            B, N, D = X.shape
            Xf = X.reshape(B * N, D)
        else:
            Xf = X
            D = Xf.shape[-1]
        M = Xf.shape[0]           # number of samples
        m = max(D - 1, 1)         # intrinsic sphere dim

        # Pairwise chordal distances on the sphere (good local proxy)
        G = (Xf @ Xf.t()).clamp(-1.0, 1.0)      # cosines
        dist2 = 2.0 - 2.0 * G                   # ||x - y||^2
        dist2.fill_diagonal_(float('inf'))
        vals, _ = torch.topk(dist2, k, dim=1, largest=False)
        rk = torch.sqrt(vals[:, -1].clamp_min(eps))  # k-th NN radius

        # KL formula in log space
        # H ≈ ψ(M) - ψ(k) + log c_m + m * E[log r_k]
        log_c_m = _log_unit_ball_volume(m, device=X.device, dtype=X.dtype)
        H = torch.digamma(torch.tensor(M, device=X.device, dtype=X.dtype)) \
            - torch.digamma(torch.tensor(k, device=X.device, dtype=X.dtype)) \
            + log_c_m + float(m) * torch.mean(torch.log(rk + eps))

        return H


def empirical_energy_U(X: Tensor, U_call: Callable[[Tensor], Tensor]) -> Tensor:
    """
    Estimate E[U(X)] from point cloud X using the provided U(x) callable.
    X: [B,N,D] or [M,D]; returns scalar average.
    """
    with torch.no_grad():
        U_vals = U_call(X)  # broadcasting over batch/positions
        return U_vals.mean()


# ------- "Theoretical" predictors (strategies; include working stubs) -------

def theoretical_free_energy_path(X0: Tensor,
                                 U_calls: List[Callable[[Tensor], Tensor]],
                                 nu_ts: List[Union[float, Callable[[Tensor], float]]],
                                 dt: float = 1.0) -> Dict[str, Tensor]:
    """
    Strategy stub: compute *predicted* layerwise Helmholtz free-energy
      F_t = ν_t * H[ρ_t] + E[U_t(X_t)]
    using simple approximations. We provide a pragmatic version:
      - evolve X_t with fpe() but *log* ν_t, U_t to compute F_t side-by-side
      - entropy via kNN estimator (still empirical but pairs with theory inputs)
    Returns dict with keys: 'F', 'H', 'E_U' each a [2L]-vector (or [L] if micro=1).
    """
    # NOTE: This blends the simulation with measurement to let you compare
    # "theoretical inputs" (ν_t, U_t) vs the actual cloud’s entropy/energy.
    # A closed-form F_t without ρ_t is generally not available unless you impose
    # a parametric family (e.g., VMF or local Gaussian). See below for notes.
    L = len(U_calls)
    # Build gradU from U via autograd? We already have gradU_ts externally; here we only need F components.
    # We'll just simulate with zero drift to extract ν-only effect unless you pass gradU_ts to fpe externally.
    raise NotImplementedError("Use fpe() to generate the path, then call empirical_entropy_KL and empirical_energy_U per step.")


# -------------------------
# Factories to wire everything per model
# -------------------------

def make_gradU_list(model: nn.Module) -> List[Callable[[Tensor], Tensor]]:
    """
    Traverse transformer blocks and build gradU_t callables.
    Assumes GPT-2 style: model.transformer.h is a ModuleList of blocks.
    """
    blocks = _get_blocks(model)
    return [gradU(b) for b in blocks]


def make_U_list(model: nn.Module, n_quad: int = 8) -> List[Callable[[Tensor], Tensor]]:
    blocks = _get_blocks(model)
    return [U(b, n_quad=n_quad) for b in blocks]


def make_nu_list(model: nn.Module, X_probe: Tensor, attn_mask: Optional[Tensor] = None) -> List[float]:
    """
    Compute a scalar ν_t per block using a probe point cloud X_probe: [B,N,D].
    You can re-estimate ν_t online during fpe simulation by passing callables instead.
    """
    blocks = _get_blocks(model)
    return [nu(b, X_probe, attn_mask=attn_mask) for b in blocks]


def _get_blocks(model: nn.Module) -> List[nn.Module]:
    """
    Try common paths for GPT-2-like models: model.transformer.h is a ModuleList.
    Adjust here for your architecture.
    """
    tr = getattr(model, "transformer", None) or getattr(model, "model", None)
    if tr is None:
        # Some HF wrappers: model.base_model
        tr = getattr(model, "base_model", None)
    if tr is None:
        raise ValueError("Could not find .transformer/.model on the HF module")

    blocks = getattr(tr, "h", None) or getattr(tr, "blocks", None) or getattr(tr, "layers", None)
    if blocks is None:
        # Distil-style?
        for name in ("layer", "block", "encoder"):
            blocks = getattr(tr, name, None)
            if blocks is not None:
                break
    if blocks is None:
        raise ValueError("Could not find block list under transformer.{h|blocks|layers|layer|block|encoder}")

    if isinstance(blocks, nn.ModuleList):
        return list(blocks)
    if isinstance(blocks, (list, tuple)):
        return list(blocks)
    # Some models store a nested module with numeric attributes; fallback to modules traversal
    candidates = [m for m in tr.modules() if hasattr(m, "mlp") and (hasattr(m, "attn") or hasattr(m, "attention"))]
    if not candidates:
        raise ValueError("Failed to identify transformer blocks with .mlp and .attn")
    return candidates


# -------------------------
# Strategy notes (read me)
# -------------------------

"""
STRATEGY: Macrostate pairs (empirical vs "theoretical")

We want, for each macrostate M_t (entropy, energy, free energy, maybe Fisher information, anisotropy):
  (A) an empirical estimator from a point cloud X_t ∈ R^{B×N×D},
  (B) a "closed-form" predictor with the same signature as fpe(), i.e., which advances M_t per layer.

Pragmatic plan:

1) ENTROPY (H[ρ_t]):
   A (empirical): use Kozachenko–Leonenko kNN on S^{D-1} (empirical_entropy_KL). For robustness in high D,
      use sliced estimators: pick random 2D geodesic charts and average entropies; or fit a vMF mixture and compute H analytically.
   B (predictive): no exact closed form without ρ_t. Two viable approximations:
      (i) Local Gaussian (tangent) OU linearization: around x̄_t, approximate ∇U_t ≈ H_t (x−x̄_t).
          Then covariance Σ_cov evolves by Lyapunov: dΣ_cov/dt = −(H+H^T)Σ_cov − Σ_cov(H+H^T)^T + 2ν_t I.
          Entropy H ≈ ½ log det(2πe Σ_cov). Implement this ODE layerwise with matrix exponentials or Euler.
      (ii) Direct: use fpe() to advance X_t by (ν_t, ∇U_t), then compute H empirically. This is "semi-theoretical"
           but compares predicted parameters (ν_t, U_t) with the measured cloud.

2) ENERGY (E[U_t(X_t)]):
   A (empirical): empirical_energy_U(X_t, U_t).
   B (predictive): same OU linearization—if U is locally quadratic with Hessian H, then
         E[U] ≈ U(x̄_t) + ½ tr(H Σ_cov). Track x̄_t via drift: d x̄_t / dt = −E[∇U_t] ≈ −H (x̄_t − x_*).

3) FREE ENERGY (F_t = ν_t H + E[U_t]):
   Combine the above A/B. For isotropic ν_t, this matches the Helmholtz form; for anisotropic diffusion,
   either scalarize via ν_t = tr(Σ_t)/D or whiten to constant Σ_t per layer.

4) FISHER INFORMATION (I[ρ_t] = ∫ ||∇ log ρ_t||^2 ρ_t):
   A (empirical): estimate score with score matching on the cloud (denoising score matching small-σ) or kernel score estimator,
      then integrate by Monte Carlo.
   B (predictive): in OU linearization, I ≈ tr(Σ_cov^{-1}).

5) ANISOTROPY of diffusion:
   A: compute Σ_t (full) via attention formula and report sphericity tr(Σ_t^2)/tr(Σ_t)^2.
   B: theoretical is simply what you computed from weights and P (no need for "closed-form").

Implementation sketch for B-variants will require small helper ODE solvers for mean/cov; these are fast and can
share the same interface as fpe() (taking ν_ts, gradU_ts but internally replacing ∇U with its local linearization).

You can scaffold those under a module (e.g., `predictors.py`) with functions:
  - ou_predict_entropy_energy(X0, Hessians_ts, nus_ts, dt)
  - closed_form_whitened_entropy(X0, nus_ts, dt)  (if you pre-whiten by constant Σ_t per layer)

For now, use `fpe()` + empirical estimators to validate ν_t and ∇U_t; once stable, add the OU predictors.
"""

