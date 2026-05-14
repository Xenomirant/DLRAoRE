import re


def collect_optimizer_rank_stats(optimizer, prefix="optimizer_rank"):
    stats = {}
    for group_idx, group in enumerate(optimizer.param_groups):
        names = group.get("param_names") or group.get("module_names") or []
        for param_idx, param in enumerate(group["params"]):
            state = optimizer.state.get(param, {})
            param_stats = _state_rank_stats(state, param)
            if not param_stats:
                continue

            name = names[param_idx] if param_idx < len(names) else f"group_{group_idx}_param_{param_idx}"
            metric_prefix = f"{prefix}/{_safe_metric_name(name)}"
            for key, value in param_stats.items():
                stats[f"{metric_prefix}/{key}"] = value

    return stats


def _state_rank_stats(state, param):
    stats = {}
    has_projected_state = False

    if param.ndim >= 2:
        stats["param_rows"] = int(param.shape[0])
        stats["param_cols"] = int(param.shape[1])

    if "q_l" in state:
        q_l = state["q_l"]
        stats["left_space_dim"] = int(q_l.shape[0])
        stats["left_vectors"] = int(q_l.shape[1])
        has_projected_state = True

    if "q_r" in state:
        q_r = state["q_r"]
        stats["right_space_dim"] = int(q_r.shape[0])
        stats["right_vectors"] = int(q_r.shape[1])
        has_projected_state = True

    if "singular_values" in state:
        stats["svd_rank"] = int(state["singular_values"].numel())
        has_projected_state = True

    if "lambda_l" in state:
        stats["left_spectrum_rank"] = int(state["lambda_l"].numel())
        has_projected_state = True

    if "lambda_r" in state:
        stats["right_spectrum_rank"] = int(state["lambda_r"].numel())
        has_projected_state = True

    projector = state.get("projector")
    if projector is not None:
        projector_stats = _projector_rank_stats(projector, state, param)
        stats.update(projector_stats)
        has_projected_state = has_projected_state or bool(projector_stats)

    if not has_projected_state:
        return {}

    _add_tensor_shape_stats(stats, "momentum", state.get("exp_avg"))
    if "exp_avg_sq" in state:
        _add_tensor_shape_stats(stats, "second_moment", state["exp_avg_sq"])
    elif "rank1_left" in state and "rank1_right" in state:
        stats["second_moment_rank"] = 1
        stats["second_moment_dim0"] = int(state["rank1_left"].numel())
        stats["second_moment_dim1"] = int(state["rank1_right"].numel())
        stats["second_moment_numel"] = int(state["rank1_left"].numel() + state["rank1_right"].numel() + 1)

    return stats


def _projector_rank_stats(projector, state, param):
    ortho = getattr(projector, "ortho_matrix", None)
    if ortho is None:
        return {}

    stats = {}
    if isinstance(ortho, (list, tuple)):
        if len(ortho) >= 1 and ortho[0] is not None:
            stats["left_space_dim"] = int(ortho[0].shape[0])
            stats["left_vectors"] = int(ortho[0].shape[1])
        if len(ortho) >= 2 and ortho[1] is not None:
            stats["right_vectors"] = int(ortho[1].shape[0])
            stats["right_space_dim"] = int(ortho[1].shape[1])
        return stats

    rank = int(min(ortho.shape))
    stats["projector_vectors"] = rank

    exp_avg = state.get("exp_avg")
    if exp_avg is not None and param.ndim == 2 and exp_avg.ndim == 2:
        if exp_avg.shape[0] == rank and exp_avg.shape[1] == param.shape[1]:
            stats["left_space_dim"] = int(ortho.shape[0])
            stats["left_vectors"] = rank
        elif exp_avg.shape[1] == rank and exp_avg.shape[0] == param.shape[0]:
            stats["right_vectors"] = rank
            stats["right_space_dim"] = int(ortho.shape[-1])

    return stats


def _add_tensor_shape_stats(stats, prefix, tensor):
    if tensor is None:
        return
    stats[f"{prefix}_numel"] = int(tensor.numel())
    if tensor.ndim >= 1:
        stats[f"{prefix}_dim0"] = int(tensor.shape[0])
    if tensor.ndim >= 2:
        stats[f"{prefix}_dim1"] = int(tensor.shape[1])


def _safe_metric_name(name):
    return re.sub(r"[^0-9A-Za-z_.-]+", "_", str(name))
