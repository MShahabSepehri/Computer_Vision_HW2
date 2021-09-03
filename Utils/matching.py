import cv2 as cv
import numpy as np


def get_sift_key_points(image):
    sift = cv.SIFT_create()
    key_points, descriptors = sift.detectAndCompute(image, None)
    return key_points, descriptors


def get_match_points(des1, des2, ratio_tr):
    bf = cv.BFMatcher(cv.NORM_L2, crossCheck=False)
    matches_alpha1 = bf.knnMatch(des1, des2, k=2)
    matches1 = []
    for p1, p2 in matches_alpha1:
        if p1.distance < ratio_tr * p2.distance:
            matches1.append(p1)

    matches_alpha2 = bf.knnMatch(des2, des1, k=2)
    matches2 = []
    for p1, p2 in matches_alpha2:
        if p1.distance < ratio_tr * p2.distance:
            matches2.append(p1)

    matches = []
    for m1 in matches1:
        q1 = m1.queryIdx
        t1 = m1.trainIdx
        for m2 in matches2:
            if t1 == m2.queryIdx and q1 == m2.trainIdx:
                matches.append(m1)
                break
    return matches


def compute_homography(src_points, des_points, N, x_off=0, y_off=0):
    H, _ = cv.findHomography(src_points, des_points, cv.RANSAC, maxIters=N)
    offset_mat = np.array([[1, 0, x_off], [0, 1, y_off], [0, 0, 1]])
    H_offset = np.matmul(offset_mat, np.linalg.inv(H))
    return H_offset, np.linalg.inv(H)


def get_homography(img1, img2, ratio_tr, N, x_off, y_off, min_offset=False):
    kps1, des1 = get_sift_key_points(img1)
    kps2, des2 = get_sift_key_points(img2)

    matches = get_match_points(des1, des2, ratio_tr)

    src_points = np.float32([kps1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    des_points = np.float32([kps2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    H_offset, H = compute_homography(src_points, des_points, N, x_off=x_off, y_off=y_off)

    if min_offset:
        min_i, min_j, max_i, max_j = get_min_offset(img2, H)
        offset_mat = np.array([[1, 0, -min_i], [0, 1, -min_j], [0, 0, 1]])
        H_min_offset = np.matmul(offset_mat, H)
        min_offset_shape = (max_i - min_i + 1, max_j - min_j + 1)
        return H_offset, H, H_min_offset, min_offset_shape, (min_i, min_j)
    else:
        return H_offset, H


def get_min_offset(img, H):
    pts = [np.array([[0, 0, 1]]).transpose(),
           np.array([[img.shape[1] - 1, 0, 1]]).transpose(),
           np.array([[0, img.shape[0] - 1, 1]]).transpose(),
           np.array([[img.shape[1] - 1, img.shape[0] - 1, 1]]).transpose()
           ]
    min_i = -1
    min_j = -1
    max_i = -1
    max_j = -1
    for p in pts:
        tr_p = np.matmul(H, p)
        if tr_p[2] < 0.001:
            continue
        if min_i == -1:
            min_i = int(tr_p[0] / tr_p[2])
            min_j = int(tr_p[1] / tr_p[2])
            max_i = int(tr_p[0] / tr_p[2])
            max_j = int(tr_p[1] / tr_p[2])
        else:
            min_i = min(min_i, int(tr_p[0] / tr_p[2]))
            min_j = min(min_j, int(tr_p[1] / tr_p[2]))
            max_i = max(max_i, int(tr_p[0] / tr_p[2]))
            max_j = max(max_j, int(tr_p[1] / tr_p[2]))

    return min_i, min_j, max_i, max_j


def transfer_point(p, H):
    vec = np.array([[p[0], p[1], 1]]).transpose()
    tr_vec = np.matmul(H, vec)
    tr_point = (int(tr_vec[0]/tr_vec[2]), int(tr_vec[1]/tr_vec[2]))
    return tr_point


def transfer_list_of_points(pts, H):
    tr_list = []
    for p in pts:
        tr_list.append(transfer_point(p, H))
    return tr_list
