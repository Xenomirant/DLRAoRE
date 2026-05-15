import einops
import torch
import math
import typing

class _SymmetricLowRankMatrix:
    def __init__(self, factor, sigmas):
        self.factor = factor
        self.sigmas = sigmas
        self.dtype = factor.dtype
        self.device = factor.device
        rows = factor.shape[0]
        self.shape = (rows, rows)

    def matmul_r(self, rhs):
        return self.factor.matmul(self.sigmas.unsqueeze(1) * self.factor.t().matmul(rhs))

    def matmul_l(self, lhs):
        return lhs.matmul(self.factor).matmul(self.sigmas.unsqueeze(1) * self.factor.t())
    
    def matmul(self, lhs, rhs=None):
        if rhs is None:
            return self.matmul_r(lhs)
        return lhs.matmul(self.factor).matmul(self.sigmas.unsqueeze(1) * self.factor.t().matmul(rhs)) 

    def full_matrix(self):
        return (self.factor * self.sigmas).matmul(self.factor.t())
    
    @property
    def factors(self):
        return self.factor, self.sigmas


class _LowRankMatrix:
    def __init__(self, left, sigmas, right,):
        self.left = left
        self.right = right
        self.sigmas = sigmas
        self.dtype = left.dtype
        self.device = left.device
        rows, cols = left.shape[0], right.shape[0]
        self.shape = (rows, cols)

    def matmul_r(self, rhs):
        right_projection = self.right.t().matmul(rhs)
        if self.sigmas.ndim == 1:
            return self.left.matmul(self.sigmas.unsqueeze(1) * right_projection)
        return self.left.matmul(self.sigmas.matmul(right_projection))
    
    def matmul_l(self, lhs):
        left_projection = lhs.matmul(self.left)
        if self.sigmas.ndim == 1:
            return left_projection.matmul(self.sigmas.unsqueeze(1) * self.right.t())
        return left_projection.matmul(self.sigmas).matmul(self.right.t())
    
    def matmul(self, lhs, rhs):
        left_projection = lhs.matmul(self.left)
        right_projection = self.right.t().matmul(rhs)
        if self.sigmas.ndim == 1:
            return left_projection.matmul(self.sigmas.unsqueeze(1) * right_projection)
        return left_projection.matmul(self.sigmas).matmul(right_projection)

    def full_matrix(self):
        if self.sigmas.ndim == 1:
            return (self.left * self.sigmas).matmul(self.right.t())
        return self.left.matmul(self.sigmas).matmul(self.right.t())

    def t(self):
        sigmas = self.sigmas if self.sigmas.ndim == 1 else self.sigmas.t()
        return _LowRankMatrix(
            self.right, sigmas, self.left
        )

    @property
    def factors(self):
        return self.left, self.sigmas, self.right


def _scale_low_rank_matrix(matrix, scale):
    left, core, right = matrix.factors
    return _LowRankMatrix(left, core * scale, right)


def _reshape_for_kronecker_projection(matrix, kronecker_mode):
    meta = {"kind": "identity"}
    if kronecker_mode == "none":
        return matrix, meta
    if kronecker_mode != "auto":
        raise ValueError("kronecker_mode should be none or auto")
    if matrix.ndim != 2:
        return matrix, meta

    rows, cols = matrix.shape
    row_a, row_b = _balanced_factor_pair(rows)
    col_a, col_b = _balanced_factor_pair(cols)
    if row_a == 1 or row_b == 1 or col_a == 1 or col_b == 1:
        return matrix, meta

    return (
        einops.rearrange(
            matrix,
            "(row_a row_b) (col_a col_b) -> (row_a col_a) (row_b col_b)",
            row_a=row_a,
            row_b=row_b,
            col_a=col_a,
            col_b=col_b,
        ),
        {
            "kind": "kron2d",
            "rows": rows,
            "cols": cols,
            "row_a": row_a,
            "row_b": row_b,
            "col_a": col_a,
            "col_b": col_b,
        },
    )


def _reshape_back_from_kronecker_projection(matrix, meta, original_shape):
    if meta["kind"] == "kron2d":
        return einops.rearrange(
            matrix,
            "(row_a col_a) (row_b col_b) -> (row_a row_b) (col_a col_b)",
            row_a=meta["row_a"],
            row_b=meta["row_b"],
            col_a=meta["col_a"],
            col_b=meta["col_b"],
        )
    if tuple(matrix.shape) == tuple(original_shape):
        return matrix
    return matrix.reshape(original_shape)


def _balanced_factor_pair(value):
    root = int(value ** 0.5)
    for left in range(root, 0, -1):
        if value % left == 0:
            return left, value // left
    return 1, value


def _find_rank_for_relative_error(eigs: torch.Tensor, truncation_eps: float):
    total = eigs.sum()
    if total <= 0:
        return 1
    cumsum = torch.cumsum(eigs, dim=0)
    ratio = cumsum / total
    # find smallest N such that ratio >= 1 - truncation_eps
    N = torch.searchsorted(ratio, 1 - truncation_eps).item() + 1

    return N

def _symmetrize(matrix):
    return 0.5 * (matrix + matrix.t())

def _orth(matrix,):
    Q, _ = torch.linalg.qr(matrix, mode="reduced")
    return Q

def _expand_basis(Q, candidates,):
    if Q is None:
        return _orth(candidates,)

    Q_bar, _ = torch.linalg.qr(torch.cat([Q, candidates], dim=1), mode="reduced")
    return Q_bar

def _basis_matmul_core(basis, core):
    if core.ndim == 1:
        return basis * core
    return basis.matmul(core)

def _basis_matmul_core_t(basis, core):
    if core.ndim == 1:
        return basis * core
    return basis.matmul(core.t())

def _low_rank_plus_delta_matmul(matrix, delta, rhs):
    return matrix.matmul_r(rhs) + delta.matmul(rhs)

def _low_rank_plus_delta_matmul_t(matrix, delta, lhs):
    return (matrix.matmul_l(lhs) + lhs.matmul(delta)).t()

def _low_rank_plus_delta_matmul_two_sided(matrix, delta, lhs, rhs):
    return matrix.matmul(lhs, rhs) + lhs.matmul(delta).matmul(rhs)


def _dynamical_rangefinder(matrix: typing.Union[_LowRankMatrix, _SymmetricLowRankMatrix], delta: torch.Tensor, rank: int, oversampling: int=3, power_iters: int=0):

    dim = matrix.shape[-1]    

    omega = torch.randn(dim, rank+oversampling, dtype=delta.dtype, device=delta.device)
    Q = _orth(
        _low_rank_plus_delta_matmul(matrix, delta, omega)
        )

    for _ in range(power_iters):
        W = _orth(
            _low_rank_plus_delta_matmul_t(matrix, delta, Q.t())
        )
        Q = _orth(
            _low_rank_plus_delta_matmul(matrix, delta, W)
        )
    return Q

def _adaptive_dynamical_rangefinder(matrix, delta, tol: float = 1e-4, beta: float = 1e-5):

    K = max(1, -math.ceil(math.log(beta) / math.log(10.0)))
    eps_adaptive = math.sqrt(math.pi/2) * tol / 10
    rank, p = K, 0

    Q = _dynamical_rangefinder(matrix, delta, rank=rank, oversampling=p, power_iters=0)

    error = torch.as_tensor(float("inf"), dtype=delta.dtype, device=delta.device)
    j, dim = 1, delta.shape[-1]

    while error > eps_adaptive and Q.shape[1] < dim:
        omega = torch.randn(dim, K, dtype=delta.dtype, device=delta.device)
        B = _low_rank_plus_delta_matmul(matrix, delta, omega)
        residual = B - Q.matmul(Q.t().matmul(B))
        error = torch.linalg.vector_norm(residual, dim=0).max()

        if error > eps_adaptive:
            Q_next = _expand_basis(Q, B)
            if Q_next.shape[1] == Q.shape[1]:
                break
            Q = Q_next
    return Q


def _sym_proj_split(matrix: _SymmetricLowRankMatrix, delta: torch.Tensor, rank: int, truncation_eps: float = 1e-4):
    
    U, lam = matrix.factors
    
    U_bar = U * lam + delta.matmul(U)
    Q_bar = _expand_basis(U, U_bar)
    y_bar = Q_bar.t().matmul(
        _low_rank_plus_delta_matmul(matrix, delta, Q_bar)
        )

    lam_bar, U_bar = torch.linalg.eigh(y_bar)
    lam_bar = lam_bar.flip(dims=(-1,)).clamp(min=0.0)
    U_bar = U_bar.flip(dims=(-1,))

    opt_rank = _find_rank_for_relative_error(lam_bar, truncation_eps)
    opt_rank = min(rank, opt_rank)

    return _SymmetricLowRankMatrix(
        Q_bar.matmul(U_bar[:, :opt_rank]), lam_bar[:opt_rank]
        )


def _sym_rand_proj_split(matrix: _SymmetricLowRankMatrix, delta: torch.Tensor, rank: int, oversampling: int=3, power_iters=0, adaptive=True, tol: float = 1e-4, beta: float=1e-5, truncation_eps: float = 1e-4):

    U, lam = matrix.factors
    
    if adaptive:
        rangefinder = _adaptive_dynamical_rangefinder(
            matrix, delta, tol=tol, beta=beta
        )
    else:
        rangefinder = _dynamical_rangefinder(
            matrix, delta, rank=rank, oversampling=oversampling, power_iters=power_iters
        )

    Q_bar = _expand_basis(U, rangefinder)

    y_bar = Q_bar.t().matmul(
        _low_rank_plus_delta_matmul(matrix, delta, Q_bar)
        )

    lam_bar, U_bar = torch.linalg.eigh(y_bar)
    lam_bar = lam_bar.flip(dims=(-1,)).clamp(min=0.0)
    U_bar = U_bar.flip(dims=(-1,))

    opt_rank = _find_rank_for_relative_error(lam_bar, truncation_eps)
    opt_rank = min(rank, opt_rank)

    return _SymmetricLowRankMatrix(
        Q_bar.matmul(U_bar[:, :opt_rank]), lam_bar[:opt_rank]
        )


def _proj_split(matrix: _LowRankMatrix, delta: torch.Tensor, rank: int, truncation_eps: float = 1e-4):
    
    left, sigmas, right = matrix.factors
    
    U_bar = _expand_basis(
        left, _orth(
            _basis_matmul_core(left, sigmas) + delta.matmul(right)
            )
    )
    
    V_bar = _expand_basis(
        right, _orth(
            _basis_matmul_core_t(right, sigmas) + delta.t().matmul(left)
        )
    )
        
    
    s_bar = U_bar.t().matmul(
        _low_rank_plus_delta_matmul(matrix, delta, V_bar)
        )

    u_lr, s_lr, v_lrh = torch.linalg.svd(s_bar)
    s_lr = s_lr.clamp(min=0.0)

    opt_rank = _find_rank_for_relative_error(s_lr, truncation_eps)
    opt_rank = min(rank, opt_rank)

    return _LowRankMatrix(
        U_bar.matmul(u_lr[:, :opt_rank]), 
        s_lr[:opt_rank], 
        V_bar.matmul(v_lrh.t()[:, :opt_rank])
        )

def _rand_svd_proj_split(matrix: _LowRankMatrix, delta: torch.Tensor, rank: int, oversampling: int=3, power_iters=0, adaptive=True, tol: float = 1e-4, beta: float=1e-5, truncation_eps: float = 1e-4):
    if matrix.shape[0] < matrix.shape[1]:
        return _rand_svd_proj_split_left(
            matrix.t(), delta.t(), rank, oversampling, power_iters, adaptive, tol, beta, truncation_eps
        ).t()
    return _rand_svd_proj_split_left(
        matrix, delta, rank, oversampling, power_iters, adaptive, tol, beta, truncation_eps
    )


def _rand_svd_proj_split_left(matrix: _LowRankMatrix, delta: torch.Tensor, rank: int, oversampling: int=3, power_iters=0, adaptive=True, tol: float = 1e-4, beta: float=1e-5, truncation_eps: float = 1e-4):
    
    left, *_ = matrix.factors

    if adaptive:
        rangefinder = _adaptive_dynamical_rangefinder(
            matrix, delta, tol=tol, beta=beta
        )
    else:
        rangefinder = _dynamical_rangefinder(
            matrix, delta, rank=rank, oversampling=oversampling, power_iters=power_iters
        )
    
    Q = _expand_basis(left, rangefinder)
    
    s_bar = _low_rank_plus_delta_matmul_t(matrix, delta, Q.t())

    u_lr, s_lr, v_lrh = torch.linalg.svd(s_bar.t())
    s_lr = s_lr.clamp(min=0.0)

    opt_rank = _find_rank_for_relative_error(s_lr, truncation_eps)
    opt_rank = min(rank, opt_rank)

    return _LowRankMatrix(
        Q.matmul(u_lr[:, :opt_rank]), 
        s_lr[:opt_rank], 
        v_lrh.t()[:, :opt_rank]
        )


#TODO: handle minimal dimension
def _rand_nystrom_proj_split(matrix: _LowRankMatrix, delta: torch.Tensor, rank: int, oversampling: int=3, power_iters=0, adaptive=True, tol: float = 1e-3, beta: float=1e-5, truncation_eps: float = 1e-3):
    
    left, sigmas, right = matrix.factors

    if adaptive:
        rangefinder_left = _adaptive_dynamical_rangefinder(
            matrix, delta, tol=tol, beta=beta
        )
        rangefinder_right = _adaptive_dynamical_rangefinder(
            matrix.t(), delta.t(), tol=tol, beta=beta
        )
        
    else:
        rangefinder_left = _dynamical_rangefinder(
            matrix, delta, rank=rank, oversampling=oversampling, power_iters=power_iters
        )
        rangefinder_right = _dynamical_rangefinder(
            matrix.t(), delta.t(), rank=rank, oversampling=oversampling, power_iters=power_iters
        )
    
    Q = _expand_basis(left, rangefinder_left)
    W = _expand_basis(right, rangefinder_right)
    
    left_bar = _low_rank_plus_delta_matmul(matrix, delta, W)
    right_bar = _low_rank_plus_delta_matmul_t(matrix, delta, Q.t())

    s_bar = _low_rank_plus_delta_matmul_two_sided(
        matrix, delta, Q.t(), W
    )

    u_lr, s_lr, v_lrh = torch.linalg.svd(s_bar)
    s_lr = s_lr.clamp(min=0.0)

    opt_rank = _find_rank_for_relative_error(s_lr, truncation_eps)
    opt_rank = min(rank, opt_rank)

    U_l, R_l = torch.linalg.qr(
        left_bar.matmul(v_lrh.t()[:, :opt_rank])
        )
    V_r, R_r = torch.linalg.qr(
        right_bar.matmul(u_lr[:, :opt_rank])
        )
    

    return _LowRankMatrix(
        U_l,
        (R_l / s_lr[:opt_rank]).matmul(R_r.t()),
        V_r
        )
