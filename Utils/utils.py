import os
import cv2 as cv
import shutil
import numpy as np
from Utils import config_reader as cr


path_dict = cr.get_config('Gen', 'path')


def open_video(name, size, frame_rate=30):
    fourcc = cv.VideoWriter_fourcc(*'mp4v')
    video = cv.VideoWriter(path_dict['results'] + name, fourcc, frame_rate, size)
    return video


def remove_dir(path):
    shutil.rmtree(path)


def get_image(path, convert=True):
    if convert:
        return cv.cvtColor(cv.imread(path), cv.COLOR_BGR2RGB)
    else:
        return cv.imread(path)


def get_frame_name(num_fr):
    return '{0:03}'.format(num_fr) + '.jpg'


def get_calibration_image_name(num):
    return 'im' + '{0:02}'.format(num) + '.jpg'


def create_frames(data_path, copy_path, low, num_frames, remove_existing=False):
    if os.path.isdir(copy_path):
        if remove_existing:
            remove_dir(copy_path)
        else:
            return
    os.mkdir(copy_path)

    vid = cv.VideoCapture(data_path)
    num_fr = 1

    ret, frame = vid.read()

    while ret and (num_fr < (num_frames + 1)):
        if low:
            dim = (int(frame.shape[1] / 2), int(frame.shape[0] / 2))
            resized_frame = cv.resize(frame, dim, interpolation=cv.INTER_AREA)
            cv.imwrite(copy_path + get_frame_name(num_fr), resized_frame)
        else:
            cv.imwrite(copy_path + get_frame_name(num_fr), frame)
        num_fr += 1
        ret, frame = vid.read()
    vid.release()


def check_dir(name):
    if not os.path.exists(name):
        os.mkdir(name)


def video_writer(video, frames):
    for frame in frames:
        video.write(cv.cvtColor(frame, cv.COLOR_RGB2BGR))


def plot_array(name, array):
    check_dir(path_dict['results'])
    cv.imwrite(path_dict['results'] + name, cv.cvtColor(np.float32(array), cv.COLOR_RGB2BGR))
