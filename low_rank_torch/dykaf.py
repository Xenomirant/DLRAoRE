import math
from typing import Callable, Iterable, Literal, Optional, Tuple

import torch
from torch import nn
from torch.optim import Optimizer
from transformers.utils.versions import require_version

from .dlra import _SymmetricLowRankMatrix, _sym_proj_split, _sym_rand_proj_split
from .rank_stats import collect_optimizer_rank_stats


class DyKAF(Optimizer):
    """
    DyKAF: Dynamical Kronecker Approximation of the Fisher matrix.

    This optimizer follows the SOAP-style step from the DyKAF paper: matrix
    gradients are rotated into the eigenspace of dynamical Kronecker Fisher
    factors, preconditioned with an Adam second moment in that rotated space,
    rotated back, and then used for the parameter update. Non-matrix tensors use
    an AdamW fallback so the optimizer can be applied to whole models.
    """

    def __init__(
            self,
            params: Iterable[nn.parameter.Parameter],
            lr: float = 1e-3,
            betas: Tuple[float, float] = (0.9, 0.999),
            eps: float = 1e-8,
            truncation_eps: float = 1e-4,
            rangefinder_tau: Optional[float] = None,
            rangefinder_beta: float = 1e-2,
            weight_decay: float = 0.0,
            correct_bias: bool = True,
            precondition_frequency: int = 10,
            rank1_second_moment: bool = False,
            power_iterations: int = 3,
            exact_preconditioner_eigs: bool = False,
            precondition_2d: bool = True,
            low_rank_factors: bool = False, 
            factors_rank: int = 64,
            low_rank_proj: Literal["rand", "psi"] = "psi",
            *args, **kwargs
    ):
        require_version("torch>=1.5.0")

        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr} - should be >= 0.0")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter: {betas[0]} - should be in [0.0, 1.0)")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter: {betas[1]} - should be in [0.0, 1.0)")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon value: {eps} - should be >= 0.0")
        if truncation_eps <= 0.0:
            raise ValueError("truncation_eps must be positive")
        if rangefinder_tau is not None and rangefinder_tau <= 0.0:
            raise ValueError("rangefinder_tau must be positive")
        if not 0.0 < rangefinder_beta < 1.0:
            raise ValueError("rangefinder_beta must be in (0, 1)")
        if precondition_frequency < 1:
            raise ValueError("precondition_frequency must be >= 1")
        if power_iterations < 1:
            raise ValueError("power_iterations must be >= 1")

        defaults = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "truncation_eps": truncation_eps,
            "rangefinder_tau": rangefinder_tau,
            "rangefinder_beta": rangefinder_beta,
            "weight_decay": weight_decay,
            "correct_bias": correct_bias,
            "precondition_frequency": precondition_frequency,
            "rank1_second_moment": rank1_second_moment,
            "power_iterations": power_iterations,
            "exact_preconditioner_eigs": exact_preconditioner_eigs,
            "precondition_2d": precondition_2d,
            "low_rank_factors": low_rank_factors,
            "factors_rank": factors_rank,
            "low_rank_proj": low_rank_proj,
        }
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Callable = None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("DyKAF does not support sparse gradients")

                use_dykaf = (
                    grad.ndim == 2
                    and group.get("precondition_2d", True)
                    and group.get("dykaf", True)
                )

                if use_dykaf:
                    if group.get("low_rank_factors", False):
                        self._lrdykaf_step(p, grad, group)
                    else:
                        self._dykaf_step(p, grad, group)
                else:
                    self._adamw_step(p, grad, group)

        return loss

    def rank_stats(self):
        return collect_optimizer_rank_stats(self)


    def _dykaf_step(self, p, grad, group):
        state = self.state[p]
        beta1, beta2 = group["betas"]
        grad = _state_tensor(grad)

        if "step" not in state:
            state["step"] = 0
        if "exp_avg" not in state:
            _init_dykaf_state(state, grad, group)

        state["step"] += 1
        step = state["step"]

        q_l = state["q_l"]
        q_r = state["q_r"]

        grad_prime = q_l.t().matmul(grad).matmul(q_r)

        exp_avg = state["exp_avg"]
        exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
        exp_avg_prime = q_l.t().matmul(exp_avg).matmul(q_r)

        if group.get("rank1_second_moment", False):
            _rank1_second_moment_update(state, grad_prime.square(), beta2, group["eps"])
            denom_sq = state["rank1_scale"] * torch.outer(
                state["rank1_left"], state["rank1_right"]
            )
            denom = denom_sq.clamp(min=0.0).sqrt().add_(group["eps"])
        else:
            exp_avg_sq = state["exp_avg_sq"]
            exp_avg_sq.mul_(beta2).addcmul_(grad_prime, grad_prime, value=1.0 - beta2)
            denom = exp_avg_sq.sqrt().add_(group["eps"])

        step_size = group["lr"]
        if group["correct_bias"]:
            bias_correction1 = 1.0 - beta1 ** step
            bias_correction2 = 1.0 - beta2 ** step
            step_size = step_size * math.sqrt(bias_correction2) / bias_correction1

        update_prime = exp_avg_prime / denom
        update = q_l.matmul(update_prime).matmul(q_r.t())
        p.add_(update.to(dtype=p.dtype), alpha=-step_size)

        preconditioner_beta = group.get("preconditioner_beta", beta1)
        state["left_factor"], state["right_factor"] = _kron_projector_split(
            state["left_factor"],
            state["right_factor"],
            grad,
            preconditioner_beta,
            group["eps"],
        )

        if step % group["precondition_frequency"] == 0:
            state["q_l"], _ = _update_basis(
                state["left_factor"],
                state["q_l"],
                exact=group.get("exact_preconditioner_eigs", False),
            )
            state["q_r"], _ = _update_basis(
                state["right_factor"],
                state["q_r"],
                exact=group.get("exact_preconditioner_eigs", False),
            )

        _decoupled_weight_decay(p, group)

    def _lrdykaf_step(self, p, grad, group):
        state = self.state[p]
        beta1, beta2 = group["betas"]
        grad = _state_tensor(grad)

        if "step" not in state:
            state["step"] = 0
        if "exp_avg" not in state:
            _init_dykaf_state(state, grad, group)

        state["step"] += 1
        step = state["step"]

        q_l_prev = state["q_l"]
        q_r_prev = state["q_r"]

        preconditioner_beta = group.get("preconditioner_beta", beta1)
        state["left_factor"], state["right_factor"] = _low_rank_kron_projector_split(
            state["left_factor"],
            state["right_factor"],
            grad,
            preconditioner_beta,
            group["eps"],
            group,
        )
        state["q_l"], state["lambda_l"] = state["left_factor"].factors
        state["q_r"], state["lambda_r"] = state["right_factor"].factors

        c_l = state["q_l"].t().matmul(q_l_prev)
        c_r = q_r_prev.t().matmul(state["q_r"])
        state["exp_avg"] = c_l.matmul(state["exp_avg"]).matmul(c_r)
        if group.get("rank1_second_moment", False):
            _reproject_rank1_second_moment(state, c_l, c_r, group["eps"])
        else:
            state["exp_avg_sq"] = (
                c_l.square().matmul(state["exp_avg_sq"]).matmul(c_r.square())
            ).clamp(min=0.0)

        grad_prime = state["q_l"].t().matmul(grad).matmul(state["q_r"])

        exp_avg = state["exp_avg"]
        exp_avg.mul_(beta1).add_(grad_prime, alpha=1.0 - beta1)
        

        if group.get("rank1_second_moment", False):
            _rank1_second_moment_update(state, grad_prime.square(), beta2, group["eps"])
            denom_sq = state["rank1_scale"] * torch.outer(
                state["rank1_left"], state["rank1_right"]
            )
            denom = denom_sq.clamp(min=0.0).sqrt().add_(group["eps"])
        else:
            exp_avg_sq = state["exp_avg_sq"]
            exp_avg_sq.mul_(beta2).addcmul_(grad_prime, grad_prime, value=1.0 - beta2)
            denom = exp_avg_sq.sqrt().add_(group["eps"])

        step_size = group["lr"]
        if group["correct_bias"]:
            bias_correction1 = 1.0 - beta1 ** step
            bias_correction2 = 1.0 - beta2 ** step
            step_size = step_size * math.sqrt(bias_correction2) / bias_correction1

        update_prime = exp_avg / denom
        update = state["q_l"].matmul(update_prime).matmul(state["q_r"].t())
        p.add_(update.to(dtype=p.dtype), alpha=-step_size)

        _decoupled_weight_decay(p, group)

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
        exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]

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


def _init_dykaf_state(state, grad, group):
    if group.get("low_rank_factors", False):
        _init_low_rank_dykaf_state(state, grad, group)
        return

    state["left_factor"], state["right_factor"] = _initial_kronecker_factors(
        grad,
        eps=group["eps"],
        power_iterations=group["power_iterations"],
    )
    state["q_l"], state["lambda_l"] = _update_basis(state["left_factor"], previous=None, exact=True)
    state["q_r"], state["lambda_r"] = _update_basis(state["right_factor"], previous=None, exact=True)

    rows, cols = grad.shape
    state["exp_avg"] = torch.zeros_like(grad)

    if group.get("rank1_second_moment", False):
        state["rank1_left"] = torch.full(
            (rows,), 1.0 / math.sqrt(rows), dtype=grad.dtype, device=grad.device
        )
        state["rank1_right"] = torch.full(
            (cols,), 1.0 / math.sqrt(cols), dtype=grad.dtype, device=grad.device
        )
        state["rank1_scale"] = torch.as_tensor(group["eps"], dtype=grad.dtype, device=grad.device)
    else:
        state["exp_avg_sq"] = torch.zeros(rows, cols, dtype=grad.dtype, device=grad.device)


def _init_low_rank_dykaf_state(state, grad, group):
    factors_rank = group.get("factors_rank", 64)
    if factors_rank < 1:
        raise ValueError("factors_rank must be >= 1")

    state["left_factor"], state["right_factor"] = _initial_low_rank_kronecker_factors(
        grad,
        rank=factors_rank,
        eps=group["eps"],
        power_iterations=group["power_iterations"],
    )
    state["q_l"], state["lambda_l"] = state["left_factor"].factors
    state["q_r"], state["lambda_r"] = state["right_factor"].factors

    rows, cols = state["q_l"].shape[-1], state["q_r"].shape[-1]
    state["exp_avg"] = torch.zeros(rows, cols, dtype=grad.dtype, device=grad.device)

    if group.get("rank1_second_moment", False):
        state["rank1_left"] = torch.full(
            (rows,), 1.0 / math.sqrt(rows), dtype=grad.dtype, device=grad.device
        )
        state["rank1_right"] = torch.full(
            (cols,), 1.0 / math.sqrt(cols), dtype=grad.dtype, device=grad.device
        )
        state["rank1_scale"] = torch.as_tensor(group["eps"], dtype=grad.dtype, device=grad.device)
    else:
        state["exp_avg_sq"] = torch.zeros(rows, cols, dtype=grad.dtype, device=grad.device)


def _initial_kronecker_factors(grad, eps, power_iterations):
    rows, cols = grad.shape
    v = torch.full((cols,), 1.0 / math.sqrt(cols), dtype=grad.dtype, device=grad.device)
    u = torch.zeros(rows, dtype=grad.dtype, device=grad.device)

    for _ in range(power_iterations):
        u = grad.matmul(v)
        u_norm = torch.linalg.vector_norm(u)
        if u_norm <= eps:
            return _identity_factors(grad, eps)
        u = u / u_norm.clamp(min=eps)

        v = grad.t().matmul(u)
        v_norm = torch.linalg.vector_norm(v)
        if v_norm <= eps:
            return _identity_factors(grad, eps)
        v = v / v_norm.clamp(min=eps)

    sigma = u.dot(grad.matmul(v)).abs().clamp(min=eps)

    left = sigma * torch.outer(u, u)
    left.add_(torch.eye(rows, device=grad.device, dtype=grad.dtype), alpha=eps)

    right = sigma * torch.outer(v, v)
    right.add_(torch.eye(cols, device=grad.device, dtype=grad.dtype), alpha=eps)

    return left, right


def _initial_low_rank_kronecker_factors(grad, rank, eps, power_iterations):
    rows, cols = grad.shape
    left_rank = min(rank, rows)
    right_rank = min(rank, cols)

    v = torch.full((cols,), 1.0 / math.sqrt(cols), dtype=grad.dtype, device=grad.device)
    u = torch.zeros(rows, dtype=grad.dtype, device=grad.device)
    has_rank1_direction = True

    for _ in range(power_iterations):
        u = grad.matmul(v)
        u_norm = torch.linalg.vector_norm(u)
        if u_norm <= eps:
            has_rank1_direction = False
            break
        u = u / u_norm.clamp(min=eps)

        v = grad.t().matmul(u)
        v_norm = torch.linalg.vector_norm(v)
        if v_norm <= eps:
            has_rank1_direction = False
            break
        v = v / v_norm.clamp(min=eps)

    if has_rank1_direction:
        sigma = u.dot(grad.matmul(v)).abs().clamp(min=eps)
        q_l = _orthonormal_columns_from_seed(u, left_rank, eps)
        q_r = _orthonormal_columns_from_seed(v, right_rank, eps)
        lambda_l = _rank1_plus_random_spectrum(left_rank, sigma, eps, grad)
        lambda_r = _rank1_plus_random_spectrum(right_rank, sigma, eps, grad)
        return _SymmetricLowRankMatrix(q_l, lambda_l), _SymmetricLowRankMatrix(q_r, lambda_r)

    q_l = _orthonormal_columns_from_seed(None, left_rank, eps, dtype=grad.dtype, device=grad.device, dim=rows)
    q_r = _orthonormal_columns_from_seed(None, right_rank, eps, dtype=grad.dtype, device=grad.device, dim=cols)
    lambda_l = _random_eps_spectrum(left_rank, eps, grad)
    lambda_r = _random_eps_spectrum(right_rank, eps, grad)
    return _SymmetricLowRankMatrix(q_l, lambda_l), _SymmetricLowRankMatrix(q_r, lambda_r)


def _rank1_plus_random_spectrum(rank, sigma, eps, like):
    values = _random_eps_spectrum(rank, eps, like)
    values[0] = sigma + eps
    return values


def _random_eps_spectrum(rank, eps, like):
    return torch.rand(rank, dtype=like.dtype, device=like.device).add_(1.0).mul_(eps)


def _orthonormal_columns_from_seed(seed, rank, eps, dtype=None, device=None, dim=None):
    if seed is not None:
        dtype = seed.dtype
        device = seed.device
        dim = seed.numel()

    candidates = torch.randn(dim, rank, dtype=dtype, device=device)
    if seed is not None:
        candidates[:, 0] = seed

    Q = _orth(candidates, eps)
    while Q is None or Q.shape[1] < rank:
        missing = rank if Q is None else rank - Q.shape[1]
        extra = torch.randn(dim, missing, dtype=dtype, device=device)
        Q = _append_orthonormal_columns(Q, extra, eps)

    return Q[:, :rank]


def _low_rank_kron_projector_split(left_factor, right_factor, grad, beta, eps, group):
    q_l, lambda_l = left_factor.factors
    q_r, lambda_r = right_factor.factors
    sqrt_beta = math.sqrt(beta)
    one_minus_beta = 1.0 - beta

    lambda_l_scaled = lambda_l * sqrt_beta
    lambda_r_scaled = lambda_r * sqrt_beta
    left_norm = torch.linalg.vector_norm(lambda_l_scaled).clamp(min=eps)
    right_norm = torch.linalg.vector_norm(lambda_r_scaled).clamp(min=eps)

    left_delta = _SymmetricLowRankMatrix(
        grad.matmul(q_r),
        lambda_r_scaled * (one_minus_beta / right_norm),
    )
    right_delta = _SymmetricLowRankMatrix(
        grad.t().matmul(q_l),
        lambda_l_scaled * (one_minus_beta / left_norm),
    )

    left_hat = _project_symmetric_factor(
        _SymmetricLowRankMatrix(q_l, lambda_l_scaled * right_norm),
        left_delta,
        group,
    )
    right_hat = _project_symmetric_factor(
        _SymmetricLowRankMatrix(q_r, lambda_r_scaled * left_norm),
        right_delta,
        group,
    )
    q_l_hat, lambda_l_hat = left_hat.factors
    q_r_hat, lambda_r_hat = right_hat.factors

    left_hat_norm = torch.linalg.vector_norm(lambda_l_hat).clamp(min=eps)
    right_hat_norm = torch.linalg.vector_norm(lambda_r_hat).clamp(min=eps)
    lambda_l_next = lambda_l_hat / left_hat_norm
    lambda_r_next = lambda_r_hat / right_hat_norm

    left_overlap = _low_rank_factor_inner(q_l, lambda_l_scaled, q_l_hat, lambda_l_next)
    right_overlap = _low_rank_factor_inner(q_r, lambda_r_scaled, q_r_hat, lambda_r_next)
    grad_core = q_l_hat.t().matmul(grad).matmul(q_r_hat)
    grad_overlap = one_minus_beta * (
        lambda_l_next.unsqueeze(1) * grad_core.square() * lambda_r_next.unsqueeze(0)
    ).sum()
    scale = (left_overlap * right_overlap + grad_overlap).clamp(min=eps)
    scale_sqrt = scale.sqrt()

    return (
        _SymmetricLowRankMatrix(q_l_hat, lambda_l_next * scale_sqrt),
        _SymmetricLowRankMatrix(q_r_hat, lambda_r_next * scale_sqrt),
    )


def _project_symmetric_factor(matrix, delta, group):
    rank = group.get("factors_rank", 64)
    truncation_eps = group["truncation_eps"]
    if group.get("low_rank_proj", "psi") == "rand":
        return _sym_rand_proj_split(
            matrix,
            delta,
            rank=rank,
            adaptive=True,
            tol=group["rangefinder_tau"] or truncation_eps,
            beta=group["rangefinder_beta"],
            truncation_eps=truncation_eps,
        )
    return _sym_proj_split(
        matrix,
        delta,
        rank=rank,
        truncation_eps=truncation_eps,
    )


def _low_rank_factor_inner(q_a, lambda_a, q_b, lambda_b):
    overlap = q_a.t().matmul(q_b).square()
    return (lambda_a.unsqueeze(1) * overlap * lambda_b.unsqueeze(0)).sum()


def _orth(matrix, eps):
    if torch.linalg.matrix_norm(matrix) <= eps:
        return None
    Q, _ = torch.linalg.qr(matrix, mode="reduced")
    return Q


def _append_orthonormal_columns(Q, candidates, eps):
    if Q is None:
        return _orth(candidates, eps)

    Q_bar, _ = torch.linalg.qr(torch.cat([Q, candidates], dim=1), mode="reduced")
    return Q_bar


def _identity_factors(grad, eps):
    rows, cols = grad.shape
    left = torch.eye(rows, dtype=grad.dtype, device=grad.device).mul_(eps)
    right = torch.eye(cols, dtype=grad.dtype, device=grad.device).mul_(eps)
    return left, right


def _kron_projector_split(left, right, grad, beta, eps):
    sqrt_beta = math.sqrt(beta)
    sqrt_one_minus_beta = math.sqrt(1.0 - beta)
    left = left * sqrt_beta
    right = right * sqrt_beta
    grad = grad * sqrt_one_minus_beta

    left_norm = torch.linalg.matrix_norm(left).clamp(min=eps)
    right_norm = torch.linalg.matrix_norm(right).clamp(min=eps)

    left_hat = left * right_norm + grad.matmul(right / right_norm).matmul(grad.t())
    right_hat = right * left_norm + grad.t().matmul(left / left_norm).matmul(grad)

    left_hat_norm = torch.linalg.matrix_norm(left_hat).clamp(min=eps)
    right_hat_norm = torch.linalg.matrix_norm(right_hat).clamp(min=eps)
    left_next = left_hat / left_hat_norm
    right_next = right_hat / right_hat_norm

    scale = (
        (left * left_next).sum()
        * (right * right_next).sum()
        + (left_next * grad.matmul(right_next).matmul(grad.t())).sum()
    ).clamp(min=eps)
    scale_sqrt = scale.sqrt()

    return _symmetrize(scale_sqrt * left_next), _symmetrize(scale_sqrt * right_next)


def _rank1_second_moment_update(state, delta, beta, eps):
    left = state["rank1_left"]
    right = state["rank1_right"]
    scale = state["rank1_scale"] * beta
    delta = delta * (1.0 - beta)

    left_hat = left * scale + delta.matmul(right)
    left_hat_norm = torch.linalg.vector_norm(left_hat).clamp(min=eps)
    left_next = left_hat / left_hat_norm

    right_hat = right * scale + delta.t().matmul(left)
    right_hat_norm = torch.linalg.vector_norm(right_hat).clamp(min=eps)
    right_next = right_hat / right_hat_norm

    scale_next = (
        scale * left_next.dot(left) * right_next.dot(right)
        + left_next.dot(delta.matmul(right_next))
    ).clamp(min=eps)

    state["rank1_left"] = left_next
    state["rank1_right"] = right_next
    state["rank1_scale"] = scale_next


def _reproject_rank1_second_moment(state, c_l, c_r, eps):
    moment = state["rank1_scale"] * torch.outer(
        state["rank1_left"], state["rank1_right"]
    )
    moment = c_l.square().matmul(moment).matmul(c_r.square()).clamp(min=0.0)
    _set_rank1_second_moment(state, moment, eps)


def _set_rank1_second_moment(state, moment, eps):
    if torch.linalg.matrix_norm(moment) <= eps:
        rows, cols = moment.shape
        state["rank1_left"] = torch.full(
            (rows,), 1.0 / math.sqrt(rows), dtype=moment.dtype, device=moment.device
        )
        state["rank1_right"] = torch.full(
            (cols,), 1.0 / math.sqrt(cols), dtype=moment.dtype, device=moment.device
        )
        state["rank1_scale"] = torch.as_tensor(eps, dtype=moment.dtype, device=moment.device)
        return

    left_vectors, singular_values, right_vectors_h = torch.linalg.svd(moment)
    left = left_vectors[:, 0].abs()
    right = right_vectors_h[0].abs()
    state["rank1_left"] = left / torch.linalg.vector_norm(left).clamp(min=eps)
    state["rank1_right"] = right / torch.linalg.vector_norm(right).clamp(min=eps)
    state["rank1_scale"] = singular_values[0].clamp(min=eps)


def _update_basis(matrix, previous, exact):
    matrix = _symmetrize(matrix)
    if previous is None or exact:
        vals, vectors = torch.linalg.eigh(matrix)
        return vectors.flip(dims=(-1,)), vals.flip(dims=(-1,))
        # return eigenvectors in descending instead of ascending order

    q, r = torch.linalg.qr(matrix.matmul(previous))
    return q, torch.diagonal(r)


def _symmetrize(matrix):
    return 0.5 * (matrix + matrix.t())


def _decoupled_weight_decay(p, group):
    if group["weight_decay"] > 0.0:
        p.add_(p, alpha= -group["lr"] * group["weight_decay"])
