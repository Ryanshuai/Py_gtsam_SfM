import cv2
import numpy as np


def extract_and_match(img1, img2):
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    pts1, pts2, valid_matches = extract_and_match_from_descriptors(des1, des2, kp1, kp2)
    return pts1, pts2, kp1, kp2, valid_matches


def extract_and_match_from_descriptors(des1, des2, kp1, kp2):
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    # Lowe's ratio test
    valid_matches = []
    for m, n in matches:
        if m.distance < 0.65 * n.distance:
            valid_matches.append(m)

    pts1 = np.float32([kp1[m.queryIdx].pt for m in valid_matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in valid_matches])

    return pts1, pts2, valid_matches


if __name__ == '__main__':
    img1 = cv2.imread('data/pipes/images/dslr_images_undistorted/DSC_0634.JPG', cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread('data/pipes/images/dslr_images_undistorted/DSC_0635.JPG', cv2.IMREAD_GRAYSCALE)

    pts1, pts2, kp1, kp2, good_matches = extract_and_match(img1, img2)

    # Draw matches
    img_matches = cv2.drawMatches(img1, kp1, img2, kp2, good_matches, None,
                                  flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    img_matches_resized = cv2.resize(img_matches, (0, 0), fx=0.3, fy=0.3)
    cv2.imshow('Matches', img_matches_resized)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
