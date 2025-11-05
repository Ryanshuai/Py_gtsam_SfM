import numpy as np

from scene import Scene
from fundamental_matrix import estimate_fundamental_matrix_ransac
from epipolar_geometry import recover_pose_from_fundamental
from triangulation import triangulate_dlt


def incremental_reconstruction(Intrinsic, image_paths):
    scene = Scene(Intrinsic, image_paths)
    best_key, = scene.select_initial_pair()

    (img1_id, img2_id), match_data = scene.select_initial_pair()
    match_data = scene.matches_dict[(img1_id, img2_id)]

    pts1 = match_data['pts1'][match_data['inliers']]
    pts2 = match_data['pts2'][match_data['inliers']]

    F, _ = estimate_fundamental_matrix_ransac(pts1, pts2)
    R, t = recover_pose_from_fundamental(F, scene.K, pts1, pts2)

    P1 = scene.K @ np.hstack([np.eye(3), np.zeros((3, 1))])
    P2 = scene.K @ np.hstack([R, t.reshape(3, 1)])
    scene.points3D = triangulate_dlt(pts1, pts2, P1, P2)

    scene.camera_poses[img1_id] = (np.eye(3), np.zeros((3, 1)))
    scene.camera_poses[img2_id] = (R, t)
    scene.image_data[img1_id].registered = True
    scene.image_data[img2_id].registered = True

    while True:
        next_img = scene.select_next_image()
        if next_img is None:
            break


