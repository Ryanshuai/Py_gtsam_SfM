import numpy as np


def estimate_extrinsic_pnp_ransac(pts3D, pts2D, K, threshold=1.0, iterations=1000):
    best_inliers = []

    for _ in range(iterations):
        # Randomly sample 6 points
        idx = np.random.choice(len(pts2D), 6, replace=False)

        # Estimate F from 6 points
        try:
            Rt = estimate_extrinsic_pnp_dlt(pts3D[idx], pts2D[idx], K)
        except np.linalg.LinAlgError:
            continue

        # Test on all N points
        inliers = compute_inliers(Rt, pts3D, pts2D, K, threshold)

        if len(inliers) > len(best_inliers):
            best_inliers = inliers

    # Refine using all inliers (n > 8)
    # TODO： replace by LM optimization
    Rt_final = estimate_extrinsic_pnp_dlt(pts3D[best_inliers], pts2D[best_inliers], K)

    return Rt_final, best_inliers


def compute_inliers(Rt, pts3D, pts2D, K, threshold=1.0):
    """
    Projection: P = K @ [R|t]
    Error = || p_2D - K @ [R|t] @ P_3D || < threshold
    """
    pts3D_h = np.hstack([pts3D, np.ones((pts3D.shape[0], 1))])  # (N, 4)

    # project 3D points to 2D
    pts2D_proj_h = (K @ Rt @ pts3D_h.T).T  # (N, 3)
    pts2D_proj = pts2D_proj_h[:, :2] / pts2D_proj_h[:, 2:3]  # normalize

    # compute reprojection errors
    errors = np.linalg.norm(pts2D - pts2D_proj, axis=1)
    inliers = np.where(errors < threshold)[0]

    return inliers


def estimate_extrinsic_pnp_dlt(pts3D, pts2D, K):
    """
    Linear PnP (DLT) algorithm to estimate camera pose [R | t].

    Steps:
    1. Convert pixel coordinates to normalized image coordinates
    2. Build design matrix from projection constraints
    3. Solve for projection matrix using SVD (DLT method)
    4. Enforce R ∈ SO(3) and normalize translation scale
    """

    pts3D = np.asarray(pts3D, dtype=np.float64).reshape(-1, 3)
    pts2D = np.asarray(pts2D, dtype=np.float64).reshape(-1, 2)
    assert pts3D.shape[0] == pts2D.shape[0] and pts3D.shape[0] >= 6

    # 1. Pixel coordinates -> Normalized image coordinates
    Kinv = np.linalg.inv(K)
    pts2D_h = np.hstack([pts2D, np.ones((pts2D.shape[0], 1))])  # (N,3)
    pts2D_norm_h = (Kinv @ pts2D_h.T).T  # (N,3)

    # 2. Build design matrix
    A = build_design_matrix_pnp(pts3D, pts2D_norm_h[:, :2])

    # 3) solve R,t from A using SVD
    Rt = solve_Rt_from_design_metrix(A)

    # 4) Enforce R ∈ SO(3) and normalize t
    P = enforce_so3(P)
    return P


def build_design_matrix_pnp(pts3D, pts2D_norm):
    """
    ----------------------------------------------
    Step 1. Projection equation (normalized form)
        s @ [x, y, 1]^T = P @ [X, Y, Z, 1]^T
        where
            P = [R | t] =
                [p1  p2  p3  p4]
                [p5  p6  p7  p8]
                [p9 p10 p11 p12]

    ----------------------------------------------
    Step 2. Expand the matrix multiplication
        s*x = p1*X + p2*Y + p3*Z + p4
        s*y = p5*X + p6*Y + p7*Z + p8
        s   = p9*X + p10*Y + p11*Z + p12

    ----------------------------------------------
    Step 3. Eliminate scale factor s
        p1*X + p2*Y + p3*Z + p4 - x*(p9*X + p10*Y + p11*Z + p12) = 0
        p5*X + p6*Y + p7*Z + p8 - y*(p9*X + p10*Y + p11*Z + p12) = 0

    ----------------------------------------------
    Step 4. Rearrange to homogeneous linear equations
        [X Y Z 1  0 0 0 0  -xX -xY -xZ -x] * p = 0
        [0 0 0 0  X Y Z 1  -yX -yY -yZ -y] * p = 0
        where p = [p1 ... p12]^T (flattened P)

    Each 3D–2D pair → 2 rows in A.
    For N correspondences → A ∈ R^{2N×12}.

    """
    assert pts3D.shape[0] == pts2D_norm.shape[0]
    N = pts3D.shape[0]

    A = np.zeros((2 * N, 12), dtype=np.float64)
    for i in range(N):
        X, Y, Z = pts3D[i]
        x, y = pts2D_norm[i]
        A[2 * i] = [X, Y, Z, 1, 0, 0, 0, 0, -x * X, -x * Y, -x * Z, -x]
        A[2 * i + 1] = [0, 0, 0, 0, X, Y, Z, 1, -y * X, -y * Y, -y * Z, -y]

    return A


def solve_Rt_from_design_metrix(A):
    U, S, Vt = np.linalg.svd(A)
    p = Vt[-1]
    P = p.reshape(3, 4)
    return P


def enforce_so3(Rt):
    """
    Enforce rotation matrix to lie on SO(3) (orthogonal with det=+1),
    and optionally normalize translation scale if t_tilde is given.
    """
    R_tilde, t_tilde = Rt[:, :3], Rt[:, 3]

    U, S, Vt = np.linalg.svd(R_tilde)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = U @ Vt

    s = np.mean(S)
    t = t_tilde / s
    Rt = np.hstack([R, t.reshape(-1, 1)])
    return Rt
