import math
import warnings
from typing import Callable, Iterable, Optional, Tuple

import torch
from einops import rearrange
from torch import nn
from torch.optim import Optimizer
from transformers.utils.versions import require_version

from .dlra import _LowRankMatrix, _proj_split, _rand_nystrom_proj_split, _rand_svd_proj_split, _scale_low_rank_matrix
from .rank_stats import collect_optimizer_rank_stats


class DLRAdamW(Optimizer):
    """
    AdamW with a two-sided dynamical low-rank gradient projector.

    For parameter groups carrying a `rank` entry, matrix gradients are projected
    into evolving left/right subspaces, Adam moments are maintained in that
    projected core space, and updates are lifted back to the original parameter
    space. Basis refreshes reuse the DLRA projection utilities on a tracked
    low-rank gradient approximation. The tracked matrix can be replaced by the
    current gradient, incremented by the current gradient, or updated as an EMA.
    """

    def __init__(
            self,
            params: Iterable[nn.parameter.Parameter],
            lr: float = 1e-3,
            betas: Tuple[float, float] = (0.9, 0.999),
            eps: float = 1e-6,
            truncation_eps: float = 1e-8,
            rangefinder_tau: Optional[float] = None,
            rangefinder_beta: float = 1e-3,
            orthogonalization_eps: Optional[float] = None,
            dlra_projection: str = "rand_svd",
            dlra_update_mode: str = "add",
            dlra_update_beta: float = 0.9,
            adaptive_rangefinder: bool = True,
            oversampling: int = 3,
            power_iterations: int = 0,
            weight_decay: float = 1e-3,
            correct_bias: bool = True,
            no_deprecation_warning: bool = False,
    ):
        if not no_deprecation_warning:
            warnings.warn(
                "This implementation of DLRAdamW is deprecated and will be removed in a future version. Use the "
                "PyTorch implementation torch.optim.AdamW instead, or set `no_deprecation_warning=True` to disable "
                "this warning",
                FutureWarning,
            )
        require_version("torch>=1.5.0")
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr} - should be >= 0.0")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter: {betas[0]} - should be in [0.0, 1.0)")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter: {betas[1]} - should be in [0.0, 1.0)")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps} - should be >= 0.0")
        if truncation_eps <= 0.0:
            raise ValueError("truncation_eps must be positive")
        if rangefinder_tau is not None and rangefinder_tau <= 0.0:
            raise ValueError("rangefinder_tau must be positive")
        if not 0.0 < rangefinder_beta < 1.0:
            raise ValueError("rangefinder_beta must be in (0, 1)")
        if orthogonalization_eps is not None and orthogonalization_eps <= 0.0:
            raise ValueError("orthogonalization_eps must be positive")
        if dlra_projection not in ("fixed", "dlra", "svd", "rand_svd", "nystrom", "rand_nystrom"):
            raise ValueError("dlra_projection must be fixed, dlra, svd, rand_svd, nystrom, or rand_nystrom")
        if dlra_update_mode not in ("add", "ema"):
            raise ValueError("dlra_update_mode must be add, or ema")
        if not 0.0 < dlra_update_beta <= 1.0:
            raise ValueError("dlra_update_beta must be in (0, 1]")
        if oversampling < 0:
            raise ValueError("oversampling must be non-negative")
        if power_iterations < 0:
            raise ValueError("power_iterations must be non-negative")

        defaults = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "truncation_eps": truncation_eps,
            "rangefinder_tau": rangefinder_tau,
            "rangefinder_beta": rangefinder_beta,
            "orthogonalization_eps": orthogonalization_eps if orthogonalization_eps is not None else eps,
            "dlra_projection": dlra_projection,
            "dlra_update_mode": dlra_update_mode,
            "dlra_update_beta": dlra_update_beta,
            "adaptive_rangefinder": adaptive_rangefinder,
            "oversampling": oversampling,
            "power_iterations": power_iterations,
            "weight_decay": weight_decay,
            "correct_bias": correct_bias,
        }
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Callable = None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("DLRAdamW does not support sparse gradients")

                if "rank" in group and grad.ndim == 2:
                    self._dlr_step(p, grad, group)
                else:
                    self._adamw_step(p, grad, group)

        return loss

    def rank_stats(self):
        return collect_optimizer_rank_stats(self)

    def _dlr_step(self, p, grad, group):
        state = self.state[p]
        beta1, beta2 = group["betas"]
        grad = _state_tensor(grad)

        if "step" not in state:
            state["step"] = 0

        _validate_dlra_group_options(group)
        grad_matrix, meta = _reshape_for_projection(grad, group.get("kronecker_mode", "none"))
        rank = min(group["rank"], grad_matrix.shape[0], grad_matrix.shape[1])
        if rank < 1:
            self._adamw_step(p, grad, group)
            return

        needs_init = (
            "q_l" not in state
            or "q_r" not in state
            or "singular_values" not in state
            or "gradient_approximation" not in state
            or "exp_avg" not in state
            or "exp_avg_sq" not in state
            or state.get("projection_meta") != meta
            or state["q_l"].shape[0] != grad_matrix.shape[0]
            or state["q_r"].shape[0] != grad_matrix.shape[1]
        )

        if needs_init:
            approximation = _initial_gradient_approximation(grad_matrix, group, rank)
            _set_gradient_approximation_state(state, approximation)
            state["exp_avg"] = torch.zeros(
                state["q_l"].shape[1],
                state["q_r"].shape[1],
                dtype=grad_matrix.dtype,
                device=grad_matrix.device,
            )
            state["exp_avg_sq"] = torch.zeros_like(state["exp_avg"])
            state["projection_meta"] = meta

        if state["step"] > 0 and state["step"] % group["subspace_update_interval"] == 0:
            self._refresh_basis(state, grad_matrix, group, rank)

        q_l = state["q_l"]
        q_r = state["q_r"]
        grad_core = q_l.t().matmul(grad_matrix).matmul(q_r)

        state["step"] += 1
        exp_avg = state["exp_avg"]
        exp_avg_sq = state["exp_avg_sq"]
        exp_avg.mul_(beta1).add_(grad_core, alpha=1.0 - beta1)
        exp_avg_sq.mul_(beta2).addcmul_(grad_core, grad_core, value=1.0 - beta2)

        denom = exp_avg_sq.sqrt().add_(group["eps"])
        step_size = group["lr"]
        if group["correct_bias"]:
            bias_correction1 = 1.0 - beta1 ** state["step"]
            bias_correction2 = 1.0 - beta2 ** state["step"]
            step_size = step_size * math.sqrt(bias_correction2) / bias_correction1

        update_core = exp_avg / denom
        update_matrix = q_l.matmul(update_core).matmul(q_r.t())
        update = _reshape_back_from_projection(update_matrix * group.get("scale", 1.0), meta, grad.shape)
        p.add_(update.to(dtype=p.dtype), alpha=-step_size)
        _decoupled_weight_decay(p, group)

    def _refresh_basis(self, state, grad_matrix, group, rank):
        q_l_prev = state["q_l"]
        q_r_prev = state["q_r"]
        approximation = _updated_gradient_approximation(
            state["gradient_approximation"],
            grad_matrix,
            group,
            rank,
        )
        q_l, _, q_r = approximation.factors

        c_l = q_l.t().matmul(q_l_prev)
        c_r = q_r_prev.t().matmul(q_r)
        state["exp_avg"] = c_l.matmul(state["exp_avg"]).matmul(c_r)
        state["exp_avg_sq"] = (
            c_l.square().matmul(state["exp_avg_sq"]).matmul(c_r.square())
        ).clamp(min=0.0)
        _set_gradient_approximation_state(state, approximation)

    def _adamw_step(self, p, grad, group):
        state = self.state[p]
        beta1, beta2 = group["betas"]
        grad = _state_tensor(grad)

        if "step" not in state:
            state["step"] = 0
        if "exp_avg" not in state:
            state["exp_avg"] = torch.zeros_like(grad)
            state["exp_avg_sq"] = torch.zeros_like(grad)

        state["step"] += 1
        exp_avg = state["exp_avg"]
        exp_avg_sq = state["exp_avg_sq"]
        exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
        exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

        denom = exp_avg_sq.sqrt().add_(group["eps"])
        step_size = group["lr"]
        if group["correct_bias"]:
            bias_correction1 = 1.0 - beta1 ** state["step"]
            bias_correction2 = 1.0 - beta2 ** state["step"]
            step_size = step_size * math.sqrt(bias_correction2) / bias_correction1

        p.add_((exp_avg / denom).to(dtype=p.dtype), alpha=-step_size)
        _decoupled_weight_decay(p, group)


def _state_tensor(tensor):
    if tensor.dtype in (torch.float64, torch.float32):
        return tensor.detach()
    return tensor.detach().float()


def _validate_dlra_group_options(group):
    projection = group.get("dlra_projection", "rand_svd")
    update_mode = group.get("dlra_update_mode", "add")
    update_beta = group.get("dlra_update_beta", .9)
    oversampling = group.get("oversampling", 3)
    power_iterations = group.get("power_iterations", 0)

    if projection not in ("fixed", "dlra", "svd", "rand_svd", "nystrom", "rand_nystrom"):
        raise ValueError("dlra_projection must be fixed, dlra, svd, rand_svd, nystrom, or rand_nystrom")
    if update_mode not in ("replace", "add", "ema"):
        raise ValueError("dlra_update_mode must be add, or ema")
    if not 0.0 < update_beta <= 1.0:
        raise ValueError("dlra_update_beta must be in (0, 1]")
    if oversampling < 0:
        raise ValueError("oversampling must be non-negative")
    if power_iterations < 0:
        raise ValueError("power_iterations must be non-negative")


def _initial_gradient_approximation(grad_matrix, group, rank):
    update_mode = group.get("dlra_update_mode", "add")
    scale = group.get("dlra_update_beta", 0.9) if update_mode == "ema" else 0.9
    U, values, Vh = torch.linalg.svd(grad_matrix * scale, full_matrices=False)
    rank = min(rank, values.numel(), _find_rank_for_relative_error(values, group["truncation_eps"]))
    return _LowRankMatrix(U[:, :rank], values[:rank], Vh[:rank, :].t())


def _updated_gradient_approximation(approximation, grad_matrix, group, rank):
    update_mode = group.get("dlra_update_mode", "add")
    if update_mode == "add":
        base = approximation
        delta = grad_matrix
    elif update_mode == "ema":
        beta = group.get("dlra_update_beta", 0.9)
        base = _scale_low_rank_matrix(approximation, beta)
        delta = grad_matrix.mul(1-beta)
    else:
        raise ValueError("dlra_update_mode must be add, or ema")

    return _project_low_rank_update(base, delta, group, rank)


def _project_low_rank_update(matrix, delta, group, rank):
    projection = group.get("dlra_projection", "rand_svd")
    truncation_eps = group["truncation_eps"]

    if projection in ("fixed", "dlra"):
        return _proj_split(matrix, delta, rank=rank, truncation_eps=truncation_eps)

    kwargs = {
        "rank": rank,
        "oversampling": group.get("oversampling", 3),
        "power_iters": group.get("power_iterations", 0),
        "adaptive": group.get("adaptive_rangefinder", True),
        "tol": group["rangefinder_tau"] or truncation_eps,
        "beta": group["rangefinder_beta"],
        "truncation_eps": truncation_eps,
    }
    if projection in ("svd", "rand_svd"):
        return _rand_svd_proj_split(matrix, delta, **kwargs)
    if projection in ("nystrom", "rand_nystrom"):
        return _rand_nystrom_proj_split(matrix, delta, **kwargs)
    raise ValueError("dlra_projection must be fixed, dlra, svd, rand_svd, nystrom, or rand_nystrom")


def _set_gradient_approximation_state(state, approximation):
    left, core, right = approximation.factors
    state["gradient_approximation"] = approximation
    state["q_l"] = left
    state["q_r"] = right
    state["singular_values"] = core if core.ndim == 1 else torch.linalg.svdvals(core)


def _find_rank_for_relative_error(values: torch.Tensor, truncation_eps: float):
    total = values.sum()
    if total <= 0:
        return 1
    ratio = torch.cumsum(values, dim=0) / total
    return torch.searchsorted(ratio, 1 - truncation_eps).item() + 1


def _reshape_for_projection(grad, kronecker_mode):
    meta = {"kind": "identity"}
    if kronecker_mode == "none":
        return grad, meta
    if kronecker_mode != "auto":
        raise ValueError("kronecker_mode should be none or auto")
    if grad.ndim != 2:
        return grad, meta

    rows, cols = grad.shape
    row_a, row_b = _balanced_factor_pair(rows)
    col_a, col_b = _balanced_factor_pair(cols)
    if row_a == 1 or row_b == 1 or col_a == 1 or col_b == 1:
        return grad, meta

    return (
        rearrange(
            grad,
            "(row_a row_b) (col_a col_b) -> (row_a col_a) (row_b col_b)",
            row_a=row_a,
            row_b=row_b,
            col_a=col_a,
            col_b=col_b,
        ),
        {
            "kind": "kron2d",
            "row_a": row_a,
            "row_b": row_b,
            "col_a": col_a,
            "col_b": col_b,
        },
    )


def _reshape_back_from_projection(grad, meta, original_shape):
    if meta["kind"] == "kron2d":
        return rearrange(
            grad,
            "(row_a col_a) (row_b col_b) -> (row_a row_b) (col_a col_b)",
            row_a=meta["row_a"],
            row_b=meta["row_b"],
            col_a=meta["col_a"],
            col_b=meta["col_b"],
        )
    if tuple(grad.shape) == tuple(original_shape):
        return grad
    return grad.reshape(original_shape)


def _balanced_factor_pair(value):
    root = int(value ** 0.5)
    for left in range(root, 0, -1):
        if value % left == 0:
            return left, value // left
    return 1, value


def _decoupled_weight_decay(p, group):
    if group["weight_decay"] > 0.0:
        p.add_(p, alpha=-group["lr"] * group["weight_decay"])
