import numpy as np


def estimate_fundamental_matrix_8point(pts1, pts2):
    """
    Eight-point algorithm to estimate fundamental matrix F

    Steps:
    1. Normalize coordinates for numerical stability
    2. Build A matrix from epipolar constraint
    3. Solve F using SVD (DLT method)
    4. Enforce rank-2 constraint
    5. Denormalize F
    """
    # 1. Normalize points for numerical stability
    pts1_norm, T1 = normalize_points(pts1)
    pts2_norm, T2 = normalize_points(pts2)

    # 2. Build A matrix (Nx9)
    A = build_A_matrix(pts1_norm, pts2_norm)

    # 3. Solve for F using SVD (last column of V)
    F = solve_F_from_A(A)

    # 4. Enforce rank-2 constraint
    F = enforce_rank2(F)

    # 5. Denormalize
    F = T2.T @ F @ T1

    return F


def normalize_points(pts):
    """0-mean, sqrt(2)/std scaling normalization"""
    mean = np.mean(pts, axis=0)
    std = np.std(pts, axis=0).mean()
    scale = np.sqrt(2) / std

    T = np.array([[scale, 0, -scale * mean[0]],
                  [0, scale, -scale * mean[1]],
                  [0, 0, 1]])

    pts_homogeneous = np.hstack((pts, np.ones((pts.shape[0], 1))))
    pts_normalized = (T @ pts_homogeneous.T).T[:, :2]

    return pts_normalized, T


def build_A_matrix(pts1, pts2):
    """
    Build A matrix for 8-point algorithm

    Derived from epipolar constraint:
        x2^T * F * x1 = 0

    Expanding matrix multiplication:
        [x2, y2, 1] * [f11 f12 f13] * [x1]
                      [f21 f22 f23]   [y1]
                      [f31 f32 f33]   [1 ]

    Results in:
        x2*x1*f11 + x2*y1*f12 + x2*f13 +
        y2*x1*f21 + y2*y1*f22 + y2*f23 +
        x1*f31 + y1*f32 + f33 = 0

    Rewrite as linear system Af = 0:
        [x2*x1, x2*y1, x2, y2*x1, y2*y1, y2, x1, y1, 1] * [f11, f12, ..., f33]^T = 0

    N point pairs form Nx9 matrix A
    """
    n = len(pts1)
    A = np.zeros((n, 9))
    for i in range(n):
        x1, y1 = pts1[i]
        x2, y2 = pts2[i]
        A[i] = [x2 * x1, x2 * y1, x2, y2 * x1, y2 * y1, y2, x1, y1, 1]
    return A


def solve_F_from_A(A):
    """
    Math: Last column of V minimizes ||Af|| (least-squares solution to Af = 0)
    """
    U, S, Vt = np.linalg.svd(A)
    F = Vt[-1].reshape(3, 3)
    return F


def enforce_rank2(F):
    U, S, Vt = np.linalg.svd(F)
    S[2] = 0  # Set smallest singular value to zero
    F_rank2 = U @ np.diag(S) @ Vt
    return F_rank2



