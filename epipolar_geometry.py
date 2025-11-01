import numpy as np

from triangulation import triangulate_dlt


def recover_pose_from_fundamental(F, K, pts1, pts2):
    # 1. Compute Essential matrix from Fundamental
    E = K.T @ F @ K

    # 2. Decompose E to get possible (R, t)
    possible_poses = decompose_essential_matrix(E)

    # 3. Disambiguate using triangulation and cheirality check
    best_R, best_t = disambiguate_pose(possible_poses, K, pts1, pts2)

    return best_R, best_t


def decompose_essential_matrix(E):
    U, S, Vt = np.linalg.svd(E)
    W = np.array([
        [0, -1, 0],
        [1, 0, 0],
        [0, 0, 1]
    ])

    R1 = U @ W @ Vt
    R2 = U @ W.T @ Vt
    if np.linalg.det(R1) < 0: R1 = -R1
    if np.linalg.det(R2) < 0: R2 = -R2

    t = U[:, 2]
    return R1, R2, t


def disambiguate_pose(possible_poses, K, pts1, pts2):
    R1, R2, t = possible_poses
    poses = [
        (R1, t),
        (R1, -t),
        (R2, t),
        (R2, -t)
    ]

    P1 = K @ np.hstack((np.eye(3), np.zeros((3, 1))))  # First camera at origin

    max_positive_depth = 0
    best_pose = None

    for R, t in poses:
        P2 = K @ np.hstack((R, t.reshape(3, 1)))

        pts3D = triangulate_dlt(pts1, pts2, P1, P2)

        # Count points with positive depth in front of both cameras
        depth1 = pts3D[:, 2]
        depth2 = (R[2] @ pts3D.T + t[2]).T

        positive_depth_count = np.sum((depth1 > 0) & (depth2 > 0))

        if positive_depth_count > max_positive_depth:
            max_positive_depth = positive_depth_count
            best_pose = (R, t)

    return best_pose
