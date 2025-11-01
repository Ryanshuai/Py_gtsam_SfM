import numpy as np


def triangulate(pts2D_1, pts2D_2, P1, P2, refine=True):
    # Direct Linear Transform (DLT) triangulation
    pts3D_init = triangulate_dlt(pts2D_1, pts2D_2, P1, P2)

    if not refine:
        return pts3D_init

    # # Levenberg-Marquardt (LM) refinement
    # pts3D_refined = refine_triangulation_lm(
    #     pts3D_init, pts2D_1, pts2D_2, P1, P2
    # )
    #
    # return pts3D_refined


def triangulate_dlt(pts2D_1, pts2D_2, P1, P2):
    """
    P1 = K @ [R1 | t1]  # Projection matrix for camera 1
    P2 = K @ [R2 | t2]  # Projection matrix for camera 2
    """

    # Build design matrices for all points
    As = build_design_matrix(pts2D_1, pts2D_2, P1, P2)  # (N, 4, 4)

    # Solve for 3D points
    pts3D = solve_3D_point_from_design_matrix(As)  # (N, 3)
    return pts3D


def build_design_matrix(pts2D_1, pts2D_2, P1, P2):
    """
    For a 3D point pt3D = [X, Y, Z, 1]^T in homogeneous coordinates,
    its projection in two cameras satisfies:
        λ1 * [x1, y1, 1]^T = P1 @ pt3D
        λ2 * [x2, y2, 1]^T = P2 @ pt3D

    Where P1, P2 are 3x4 projection matrices, λ is scale factor.

    Derive design matrix A:
    Let P1's three rows be p1^T, p2^T, p3^T, then:
        [x1]   [p1^T @ pt3D]
        [y1] ~ [p2^T @ pt3D]
        [1 ]   [p3^T @ pt3D]

    From homogeneous coordinates (ratios are equal):
        p1^T @ pt3D / p3^T @ pt3D = x1
        p2^T @ pt3D / p3^T @ pt3D = y1

    Eliminate scale factor, rewrite as:
        p1^T @ pt3D - x1 * p3^T @ pt3D = 0  →  (x1*p3^T - p1^T) @ pt3D = 0
        p2^T @ pt3D - y1 * p3^T @ pt3D = 0  →  (y1*p3^T - p2^T) @ pt3D = 0

    Similarly for camera 2:
        (x2*p3'^T - p1'^T) @ pt3D = 0
        (y2*p3'^T - p2'^T) @ pt3D = 0

    Build A matrix:
    Combine 4 equations into A @ P = 0 form:
        A = [x1*p3^T - p1^T  ]     [4x4 matrix]
            [y1*p3^T - p2^T  ]
            [x2*p3'^T - p1'^T]
            [y2*p3'^T - p2'^T]
    """
    x1, y1 = pts2D_1[:, 0], pts2D_1[:, 1]
    x2, y2 = pts2D_2[:, 0], pts2D_2[:, 1]

    As = np.stack([
        x1[:, None] * P1[2] - P1[0],  # (N, 4)
        y1[:, None] * P1[2] - P1[1],
        x2[:, None] * P2[2] - P2[0],
        y2[:, None] * P2[2] - P2[1]
    ], axis=1)  # (N, 4, 4)
    return As


def solve_3D_point_from_design_matrix(As):
    """
    Math: Last column of V minimizes ||AQ|| (least-squares solution to AQ = 0)
    """
    U, S, Vt = np.linalg.svd(As)
    Qs_homogeneous = Vt[:, -1, :]
    pts3D = Qs_homogeneous[:, :3] / Qs_homogeneous[:, 3:4]
    return pts3D
