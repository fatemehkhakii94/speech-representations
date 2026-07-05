"""
Centered Kernel Alignment (CKA) for comparing representations.

CKA measures the similarity between two sets of representations,
invariant to orthogonal transformations and scaling.

Given:
    X: (n_samples, dim_x) — representations from model/layer A
    Y: (n_samples, dim_y) — representations from model/layer B

CKA computes how similarly X and Y organize the same n data points.

Reference:
    Kornblith et al., "Similarity of Neural Network Representations Revisited" (2019)
"""

import numpy as np
from typing import Tuple


def linear_kernel(X: np.ndarray) -> np.ndarray:
    """Compute linear kernel matrix: K = X @ X.T"""
    return X @ X.T


def centering_matrix(n: int) -> np.ndarray:
    """Centering matrix H = I - (1/n) * 11^T"""
    return np.eye(n) - np.ones((n, n)) / n


def center_kernel(K: np.ndarray) -> np.ndarray:
    """Center a kernel matrix: K_c = H @ K @ H"""
    n = K.shape[0]
    H = centering_matrix(n)
    return H @ K @ H


def hsic(K: np.ndarray, L: np.ndarray) -> float:
    """
    Hilbert-Schmidt Independence Criterion.
    
    HSIC(K, L) = (1/(n-1)^2) * trace(K_c @ L_c)
    
    Measures statistical dependence between two kernel matrices.
    """
    n = K.shape[0]
    K_c = center_kernel(K)
    L_c = center_kernel(L)
    
    # HSIC = trace(K_c @ L_c) / (n-1)^2
    return np.trace(K_c @ L_c) / ((n - 1) ** 2)


def compute_cka(X: np.ndarray, Y: np.ndarray) -> float:
    """
    Compute linear CKA between two representation matrices.
    
    CKA(X, Y) = HSIC(K_X, K_Y) / sqrt(HSIC(K_X, K_X) * HSIC(K_Y, K_Y))
    
    Args:
        X: (n_samples, dim_x) — first set of representations
        Y: (n_samples, dim_y) — second set of representations
    
    Returns:
        CKA similarity in [0, 1]
    """
    assert X.shape[0] == Y.shape[0], (
        f"X and Y must have same number of samples. Got {X.shape[0]} and {Y.shape[0]}"
    )
    
    # Compute linear kernels
    K = linear_kernel(X)  # (n, n)
    L = linear_kernel(Y)  # (n, n)
    
    # Compute CKA
    hsic_kl = hsic(K, L)
    hsic_kk = hsic(K, K)
    hsic_ll = hsic(L, L)
    
    denominator = np.sqrt(hsic_kk * hsic_ll)
    
    if denominator < 1e-10:
        return 0.0
    
    cka_value = hsic_kl / denominator
    
    # Clip to [0, 1] (numerical precision can cause slight overflows)
    return float(np.clip(cka_value, 0.0, 1.0))


def compute_cka_matrix(
    features_a: dict,
    features_b: dict,
    num_layers_a: int,
    num_layers_b: int,
    num_samples: int = 1000,
    seed: int = 42,
) -> np.ndarray:
    """
    Compute full CKA matrix between all layers of two models.
    
    Args:
        features_a: dict mapping layer_idx → (num_utterances, hidden_dim) array
        features_b: dict mapping layer_idx → (num_utterances, hidden_dim) array
        num_layers_a: Number of layers in model A (including layer 0)
        num_layers_b: Number of layers in model B (including layer 0)
        num_samples: Number of samples to use (subsample for speed)
        seed: Random seed for subsampling
    
    Returns:
        CKA matrix of shape (num_layers_a, num_layers_b)
    """
    rng = np.random.RandomState(seed)
    
    # Determine common samples
    n_available = features_a[0].shape[0]
    if num_samples < n_available:
        indices = rng.choice(n_available, num_samples, replace=False)
    else:
        indices = np.arange(n_available)
    
    cka_matrix = np.zeros((num_layers_a, num_layers_b))
    
    for i in range(num_layers_a):
        X = features_a[i][indices]
        for j in range(num_layers_b):
            Y = features_b[j][indices]
            cka_matrix[i, j] = compute_cka(X, Y)
    
    return cka_matrix
