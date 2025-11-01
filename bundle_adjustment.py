import gtsam
import numpy as np


def bundle_adjustment(pts3D, observations, camera_poses, K, noise_sigma=1.0):
    """
    Bundle Adjustment using GTSAM

    Args:
        pts3D: (M, 3) initial 3D points
        observations: list of (camera_idx, point_idx, pixel_coords) tuples
        camera_poses: list of (R, t) tuples, len=N_cameras
        K: (3, 3) camera intrinsic matrix
        noise_sigma: measurement noise standard deviation

    Returns:
        optimized_pts3D: (M, 3) optimized 3D points
        optimized_poses: list of (R, t) optimized camera poses
    """
    graph = gtsam.NonlinearFactorGraph()
    initial = gtsam.Values()

    # Create calibration object
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    calibration = gtsam.Cal3_S2(fx, fy, 0.0, cx, cy)

    # Add projection factors
    noise_model = gtsam.noiseModel.Isotropic.Sigma(2, noise_sigma)
    for cam_idx, pt_idx, pixel_coords in observations:
        measured = gtsam.Point2(pixel_coords[0], pixel_coords[1])
        factor = gtsam.GenericProjectionFactorCal3_S2(
            measured, noise_model,
            gtsam.symbol('x', cam_idx),
            gtsam.symbol('l', pt_idx),
            calibration
        )
        graph.add(factor)

    # Initialize camera poses
    for i, (R, t) in enumerate(camera_poses):
        pose = gtsam.Pose3(gtsam.Rot3(R), gtsam.Point3(t.flatten()))
        initial.insert(gtsam.symbol('x', i), pose)

    # Initialize 3D points
    for i, pt in enumerate(pts3D):
        initial.insert(gtsam.symbol('l', i), gtsam.Point3(pt))

    # Fix first camera pose to prevent gauge freedom
    first_pose_prior = gtsam.PriorFactorPose3(
        gtsam.symbol('x', 0),
        initial.atPose3(gtsam.symbol('x', 0)),
        gtsam.noiseModel.Isotropic.Sigma(6, 1e-6)
    )
    graph.add(first_pose_prior)

    # Optimize
    params = gtsam.LevenbergMarquardtParams()
    params.setVerbosity("ERROR")
    optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial, params)
    result = optimizer.optimize()

    # Extract optimized results
    n_cameras = len(camera_poses)
    n_points = len(pts3D)

    optimized_poses = []
    for i in range(n_cameras):
        pose = result.atPose3(gtsam.symbol('x', i))
        R_opt = pose.rotation().matrix()
        t_opt = np.array(pose.translation()).reshape(3, 1)
        optimized_poses.append((R_opt, t_opt))

    optimized_pts3D = np.zeros((n_points, 3))
    for i in range(n_points):
        point = result.atPoint3(gtsam.symbol('l', i))
        optimized_pts3D[i] = [point.x(), point.y(), point.z()]

    return optimized_pts3D, optimized_poses
