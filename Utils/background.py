from Utils import utils
import tqdm
import numpy as np
from scipy import stats
import scipy.signal as sg


def get_distribution_matrix(dist_mat, path, y_min, y_max, num_frames=900):
    flag = False
    for num_fr in tqdm.tqdm(range(1, num_frames + 1)):
        image = utils.get_image(path + utils.get_frame_name(num_fr)).astype(np.int16)
        image = image[:, y_min: y_max, :]
        if num_fr == 1:
            if dist_mat is None:
                flag = True
                dist_mat = np.zeros((image.shape[0], image.shape[1], image.shape[2], num_frames))
            count = np.zeros(image.shape, dtype=np.int16) + 256
        small_mat = np.sum(image, axis=2) < 37
        x_small, y_small = np.nonzero(small_mat)
        image[x_small, y_small, :] = count[x_small, y_small, :]
        count[x_small, y_small, :] += 1
        dist_mat[:, :, :, num_fr - 1] = image
    if flag:
        return dist_mat


def create_image(final_image, dist_mat, y_min, y_max):
    final_image[:, y_min:y_max, :] = stats.mode(dist_mat, axis=3)[0][:, :, :, 0]
    return final_image


def create_foreground_frame(num_fr, background_frames, copy_path, ksize=21, tr=0.3, norm='L2', color=0):
    image_orig = utils.get_image(copy_path + utils.get_frame_name(num_fr)).astype(np.float16)
    image_back = utils.get_image(background_frames + utils.get_frame_name(num_fr)).astype(np.float16)
    diff = image_back - image_orig

    if norm == 'L2':
        diff = diff / 100
        diff = np.sum(diff ** 2, axis=2)
    else:
        diff = np.sum(np.abs(diff), axis=2)

    res = sg.convolve2d(diff, np.zeros((ksize, ksize)) + 1, 'same') / (ksize ** 2)
    res = (res > tr * np.max(res))
    image_orig[:, :, color] += 100 * res
    image_orig = np.clip(image_orig, a_min=0, a_max=255)
    return image_orig

