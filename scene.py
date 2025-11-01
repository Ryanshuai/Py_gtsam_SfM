import cv2

from feature_match import extract_and_match_from_descriptors
from fundamental_matrix import estimate_fundamental_matrix_ransac


class ImageData:
    def __init__(self, img_id, img_path):
        self.id = img_id
        self.path = img_path
        self.keypoints = None  # cv2.KeyPoint list
        self.descriptors = None  # (N, 128) SIFT descriptors
        self.registered = False


class Scene:
    def __init__(self, K, image_paths=[]):
        self.K = K
        self.image_data = {}
        self.matches_dict = {}
        self.camera_poses = {}
        self.points3D = []
        self.observations = []
        for img_path in image_paths:
            self.add_image(img_path)
        self.match_all_images()

    def add_image(self, img_path):
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        sift = cv2.SIFT_create()
        kp, des = sift.detectAndCompute(img, None)

        img_id = len(self.image_data)
        img_data = ImageData(img_id, img_path)
        img_data.keypoints = kp
        img_data.descriptors = des

        self.image_data[img_id] = img_data
        return img_id

    def match_all_images(self):
        for i in self.image_data.keys():
            for j in range(i + 1, len(self.image_data)):
                pts1, pts2, matches = extract_and_match_from_descriptors(
                    self.image_data[i].descriptors,
                    self.image_data[j].descriptors,
                    self.image_data[i].keypoints,
                    self.image_data[j].keypoints
                )
                if len(pts1) < 30:
                    continue

                F, inliers = estimate_fundamental_matrix_ransac(pts1, pts2)
                if len(inliers) < 50:
                    continue

                self.matches_dict[(i, j)] = {
                    'pts1': pts1,
                    'pts2': pts2,
                    'inliers': inliers,
                    'score': len(inliers)
                }
        return self.matches_dict

    def select_next_image(self):
        registered = [i for i, data in self.image_data.items() if data.registered]
        unregistered = [i for i, data in self.image_data.items() if not data.registered]

        if not unregistered:
            return None

        best_img = None
        max_matches = 0

        for img in unregistered:
            total_matches = 0
            for reg in registered:
                key = (min(reg, img), max(reg, img))
                if key in self.matches_dict:
                    total_matches += self.matches_dict[key]['score']

            if total_matches > max_matches:
                max_matches = total_matches
                best_img = img

        return best_img if max_matches >= 50 else None
