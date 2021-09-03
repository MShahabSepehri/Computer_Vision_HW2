import numpy as np
from Utils import config_reader as cr
from Utils import utils
import cv2 as cv

def get_calibration_matrix(data_path, frames, board_size, flags=None, size=1):
    object_points = []
    image_points = []
    object_point = np.zeros((board_size[0]*board_size[1],3), np.float32)
    object_point[:, 0:2] = np.mgrid[0:board_size[0], 0:board_size[1]].T.reshape(-1,2)*size

    for num_frame in frames:
        path = data_path + utils.get_calibration_image_name(num_frame)
        frame = utils.get_image(path, convert=False)

        ret, corners = cv.findChessboardCorners(frame, board_size)

        if not ret:
            print('Error')
            return
        gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        object_points.append(object_point)
        image_points.append(corners)
    if flags is None:
        _, K, dist, rvecs, tvecs = cv.calibrateCamera(object_points, image_points,
                                                gray_frame.shape[::-1], None, None)
    else:
        _, K, dist, rvecs, tvecs = cv.calibrateCamera(object_points, image_points,
                                                gray_frame.shape[::-1], None, None, flags=flags)

    error = 0
    for i in range(len(frames)):
        image_points_est, _ = cv.projectPoints(object_points[i], rvecs[i], tvecs[i], K, dist)
        error += cv.norm(image_points[i], image_points_est, cv.NORM_L2)/len(image_points_est)
    h,  w = frame.shape[:2]
    K_without_dist, _ = cv.getOptimalNewCameraMatrix(K, dist, (w, h), 1, (w, h))
    return error/len(object_points), K, K_without_dist


def print_results(i, K, error, K_without_dist):
    print('################   ' +  'Task: ' + str(i + 1) + '   ################')
    print('K:')
    print(K)
    print('\n')
    print("error:" + str(error))
    print('\n')
    print('K without distortion: ')
    print(K_without_dist)
    print('\n')


path_dict = cr.get_config('Gen', 'path')
data_path = path_dict['data']
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
board_size = (6, 9)
list_of_frames_list = [[i for i in range(1, 11)],
                       [i for i in range(6, 12)],
                       [i for i in range(11, 21)],
                       [i for i in range(1, 21)]
                       ]

print('Results for not fixed PP:\n\n')
flags = None
size = 22
for i in range(4):
    frames = list_of_frames_list[i]
    error, K, K_without_dist = get_calibration_matrix(data_path, frames, board_size, flags, size=size)
    print_results(i, K, error, K_without_dist)

print('Results for fixed PP:\n\n')
flags = cv.CALIB_FIX_PRINCIPAL_POINT
size = 22
for i in range(4):
    frames = list_of_frames_list[i]
    error, K, K_without_dist = get_calibration_matrix(data_path, frames, board_size, flags, size=size)
    print_results(i, K, error, K_without_dist)