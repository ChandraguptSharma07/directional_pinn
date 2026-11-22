"""Residual weights W(t_i) for the PINN loss (1/N) sum_i r_i^T W_i r_i (docs/plan.md, section 4).

Everything here works on a batch of trajectories sampled on a uniform grid
t_0 = 0, ..., t_{N-1} = T with spacing h, and returns either a scalar weight
per time, shape (batch, N), or a matrix weight per time, shape (batch, N, n, n).
All weights are rescaled so that the mean of tr W over the grid equals n, the
value for uniform weighting (W = I).

Tangent quantities (computed in float64):

    Phi_i = expm(h (J_i + J_{i+1}) / 2)            step propagator t_i -> t_{i+1}
    G_i   = Phi_i^T G_{i+1} Phi_i + (h/2)(I + Phi_i^T Phi_i),  G_{N-1} = 0
          ~ integral_{t_i}^{T} Phi(s, t_i)^T Phi(s, t_i) ds   (window Gramian)
    C_i   = Phi(T, t_i)^T Phi(T, t_i)                (final-time Cauchy-Green tensor)

Weights:

    minimax matrix, window target:   W_i ∝ G_i / sqrt(tr G_i)
    best direction-blind, window:    W_i ∝ sqrt(lambda_max G_i) I
    minimax matrix, final target:    W_i ∝ C_i / sqrt(tr C_i)
    Turinici (Eqs. 13 and 17):       W_i ∝ exp(-h sum_{j<=i} lambda_j) I,
                                     lambda = <f(x) - f(0), x> / (|x|^2 + 1e-10)
    causal (Wang et al. form):       W_i ∝ exp(-eps h sum_{j<i} |r_j|^2) I
"""

from __future__ import annotations

import torch

METHODS = ("U", "TUR", "S-orc", "M-orc", "S-self", "M-self", "M-final-orc", "M-final-self", "causal")


def step_propagators(J: torch.Tensor, h: float) -> torch.Tensor:
    """J: (batch, N, n, n) Jacobians at the grid states -> (batch, N-1, n, n) step propagators."""
    return torch.linalg.matrix_exp(0.5 * h * (J[:, :-1] + J[:, 1:]))


def window_gramian(Phi: torch.Tensor, h: float) -> torch.Tensor:
    """Trapezoidal G_i for i = 0..N-1 from step propagators (batch, N-1, n, n)."""
    B, M, n, _ = Phi.shape
    eye = torch.eye(n, dtype=Phi.dtype, device=Phi.device).expand(B, n, n)
    G = torch.zeros(B, M + 1, n, n, dtype=Phi.dtype, device=Phi.device)
    for i in range(M - 1, -1, -1):
        P = Phi[:, i]
        Pt = P.transpose(1, 2)
        G[:, i] = Pt @ G[:, i + 1] @ P + 0.5 * h * (eye + Pt @ P)
    return G


def final_cauchy_green(Phi: torch.Tensor) -> torch.Tensor:
    """C_i = Phi(T, t_i)^T Phi(T, t_i) for i = 0..N-1."""
    B, M, n, _ = Phi.shape
    Q = torch.eye(n, dtype=Phi.dtype, device=Phi.device).expand(B, n, n).clone()
    C = torch.zeros(B, M + 1, n, n, dtype=Phi.dtype, device=Phi.device)
    C[:, M] = Q
    for i in range(M - 1, -1, -1):
        Q = Q @ Phi[:, i]  # Phi(T, t_i) = Phi(T, t_{i+1}) Phi(t_{i+1}, t_i)
        C[:, i] = Q.transpose(1, 2) @ Q
    return C


def _trace(A: torch.Tensor) -> torch.Tensor:
    return A.diagonal(dim1=-2, dim2=-1).sum(-1)


def normalize_matrix(W: torch.Tensor) -> torch.Tensor:
    """Rescale each batch member so that mean_i tr W_i = n."""
    n = W.shape[-1]
    mean_trace = _trace(W).mean(dim=1, keepdim=True)
    return W * (n / mean_trace.clamp_min(1e-300)).unsqueeze(-1).unsqueeze(-1)


def normalize_scalar(rho: torch.Tensor) -> torch.Tensor:
    """Rescale each batch member so that mean_i rho_i = 1 (then tr(rho I) has mean n)."""
    return rho / rho.mean(dim=1, keepdim=True).clamp_min(1e-300)


def minimax_matrix(A: torch.Tensor) -> torch.Tensor:
    """A_i / sqrt(tr A_i), zero where tr A_i = 0 (the last grid time for the window Gramian)."""
    tr = _trace(A)
    scale = torch.where(tr > 0, tr.clamp_min(1e-300).rsqrt(), torch.zeros_like(tr))
    return normalize_matrix(A * scale.unsqueeze(-1).unsqueeze(-1))


def best_scalar(A: torch.Tensor) -> torch.Tensor:
    """sqrt(lambda_max(A_i)) per time, normalized to mean 1."""
    # On CPU: batched eigvalsh on CUDA asked for ~9 GB of workspace for 36 x 512 small matrices
    # and ran out of memory on Kaggle (docs/progress.md). The values are the same.
    lam = torch.linalg.eigvalsh(A.cpu())[..., -1].clamp_min(0.0).to(A.device)
    return normalize_scalar(lam.sqrt())


def turinici(x: torch.Tensor, rhs, h: float) -> torch.Tensor:
    """Turinici's on-the-fly weight (Eqs. 13 and 17), as ported in src/pinn/train.py.

    x: (batch, N, n) states on the grid. The growth proxy replaces the unknown
    exact solution by zero; its cumulative sum (inclusive, physical time step)
    sets the log-weight.
    """
    f0 = rhs(torch.zeros_like(x[:, :1]))
    rate = ((rhs(x) - f0) * x).sum(-1) / (x.square().sum(-1) + 1e-10)
    log_w = -torch.cumsum(rate, dim=1) * h
    return normalize_scalar(torch.softmax(log_w, dim=1))


def causal(residual: torch.Tensor, h: float, eps: float) -> torch.Tensor:
    """Causal weight exp(-eps h sum_{j<i} |r_j|^2), normalized to mean 1."""
    sq = residual.square().sum(-1)
    before = torch.cumsum(sq, dim=1) - sq
    return normalize_scalar(torch.softmax(-eps * h * before, dim=1))


def tangent_weight(method: str, states: torch.Tensor, jac, h: float) -> tuple[str, torch.Tensor]:
    """Weight from Jacobians along ``states`` (batch, N, n), computed in float64.

    Returns ("scalar", (batch, N)) or ("matrix", (batch, N, n, n)).
    """
    x = states.detach().to(torch.float64)
    Phi = step_propagators(jac(x), h)
    if method in ("S-orc", "S-self", "M-orc", "M-self"):
        G = window_gramian(Phi, h)
        if method.startswith("S"):
            return "scalar", best_scalar(G)
        return "matrix", minimax_matrix(G)
    if method in ("M-final-orc", "M-final-self"):
        return "matrix", minimax_matrix(final_cauchy_green(Phi))
    raise ValueError(f"{method!r} is not a tangent weight")


def weighted_loss(residual: torch.Tensor, kind: str | None, W: torch.Tensor | None) -> torch.Tensor:
    """Per-network loss (1/N) sum_i r_i^T W_i r_i, shape (batch,)."""
    if kind is None:
        return residual.square().sum(-1).mean(dim=1)
    if kind == "scalar":
        return (residual.square().sum(-1) * W).mean(dim=1)
    if kind == "matrix":
        return torch.einsum("bni,bnij,bnj->bn", residual, W, residual).mean(dim=1)
    raise ValueError(f"unknown weight kind {kind!r}")
