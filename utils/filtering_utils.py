from typing import List

import numpy as np


def compute_pairwise_distances(
    keypoints: np.ndarray,
    traj: np.ndarray,
    valid_steps: np.ndarray,
    alpha: float = 1.0,
) -> np.ndarray:
    """Compute pairwise distances combining spatial and trajectory similarity.

    d(i,j) = (d_spatial / median(d_spatial)) + alpha * (d_traj / median(d_traj))

    - d_spatial: Euclidean distance between starting keypoints
    - d_traj: mean Euclidean distance across valid future steps

    Returns:
        [K, K] symmetric distance matrix with zeros on the diagonal.
    """
    K = keypoints.shape[0]
    if K == 1:
        return np.zeros((1, 1), dtype=np.float32)

    # Spatial distances
    kp_i = keypoints[:, None, :]  # [K, 1, 2]
    kp_j = keypoints[None, :, :]  # [1, K, 2]
    d_spatial = np.linalg.norm(kp_i - kp_j, axis=2)  # [K, K]

    # Trajectory distances across valid steps
    # traj: [K, T, 2]
    if traj.ndim != 3:
        raise ValueError(f"traj must be [K,T,2], got shape {traj.shape}")
    T = traj.shape[1]
    if valid_steps.shape[0] != T:
        raise ValueError("valid_steps length must match traj T dimension")

    valid_mask = valid_steps.astype(bool)
    if valid_mask.any():
        traj_valid = traj[:, valid_mask, :]  # [K, Tv, 2]
        # Compute mean L2 distance over Tv for all pairs
        # Expand dims to [K,1,Tv,2] and [1,K,Tv,2]
        tv_i = traj_valid[:, None, :, :]
        tv_j = traj_valid[None, :, :, :]
        # L2 per step then mean over Tv
        d_traj = np.linalg.norm(tv_i - tv_j, axis=3).mean(axis=2)  # [K, K]
    else:
        d_traj = np.zeros((K, K), dtype=np.float32)

    # Robust normalization by medians (avoid division by zero)
    def _safe_median(x: np.ndarray) -> float:
        # Use upper triangle excluding diagonal
        if x.size <= 1:
            return 1.0
        triu = x[np.triu_indices_from(x, k=1)]
        m = float(np.median(triu)) if triu.size > 0 else 1.0
        return m if m > 1e-6 else 1.0

    m_spatial = _safe_median(d_spatial)
    m_traj = _safe_median(d_traj)

    D = (d_spatial / m_spatial) + alpha * (d_traj / m_traj)
    np.fill_diagonal(D, 0.0)
    return D.astype(np.float32)


def select_representatives(
    distances: np.ndarray,
    target_k: int,
) -> List[int]:
    """Greedy k-centers style selection of representative indices.

    - Start with the point that minimizes total distance to all others (medoid-like)
    - Iteratively add the point with the largest distance to the current set
      (maximize coverage)

    Returns list of selected indices (length == target_k).
    """
    K = distances.shape[0]
    target_k = max(1, min(target_k, K))
    if K == 1 or target_k == 1:
        return [0]

    # First: medoid-like
    total = distances.sum(axis=1)
    first = int(np.argmin(total))
    selected = [first]
    remaining = set(range(K)) - {first}

    # Iteratively pick farthest from current selected set
    # Maintain min distance to any selected
    min_d = distances[:, first].copy()
    while len(selected) < target_k and remaining:
        # Exclude already selected by setting to -inf
        min_d[list(selected)] = -np.inf
        next_idx = int(np.argmax(min_d))
        selected.append(next_idx)
        remaining.discard(next_idx)
        # Update min_d with new selection
        min_d = np.minimum(min_d, distances[:, next_idx])

    return selected