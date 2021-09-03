from Utils import config_reader as cr
from Utils import utils, matching, panorama, background
import cv2 as cv
import numpy as np
import tqdm
import os

path_dict = cr.get_config('Gen', 'path')
params_dict = cr.get_config('Q1', 'params')
ratio_tr = params_dict["ratio_tr"]
N = int(params_dict['n'])


def get_homography_matrix(im1, im2):
    _, homography_matrix = matching.get_homography(im1, im2, ratio_tr, N, 0, 0)
    return homography_matrix


if os.path.isfile('all_H.npy'):
    all_H = np.load('all_H.npy', allow_pickle=True)

"""
################################## creating frames ##################################
"""

num_frames = int(params_dict['num_frames'])
low = params_dict['low'] == 1
data_path = path_dict['data']
video_path = path_dict['video']
red = (255, 0, 0)
green = (0, 255, 0)
blue = (0, 0, 255)

if low:
    copy_path = path_dict['frames_low']
else:
    copy_path = path_dict['frames']

utils.create_frames(data_path + video_path, copy_path, low, num_frames)


"""
################################## Part 1 ##################################
"""

"""
########### computing homography ###########
"""

img1 = utils.get_image(copy_path + "450.jpg")
img2 = utils.get_image(copy_path + "270.jpg")
ratio_tr = params_dict["ratio_tr"]
N = int(params_dict['n'])
x_off = params_dict["x_off"]
y_off = params_dict["y_off"]

H_offset, H, H_min_offset, min_offset_shape, min_offset = matching.get_homography(img1, img2, ratio_tr, N,
                                                                                  x_off, y_off, min_offset=True)

"""
########### square and its image ###########
"""
len_ratio = params_dict['len_ratio']
x_ratio = params_dict['x_ratio']
y_ratio = params_dict['y_ratio']
thickness = int(params_dict['thickness'])

yl, xl, _ = img1.shape
length = int(len_ratio*yl)
square_pts = [(int(y_ratio*yl), int(x_ratio*xl)),
              (int(y_ratio*yl) + length, int(x_ratio*xl)),
              (int(y_ratio*yl), int(x_ratio*xl) + length),
              (int(y_ratio*yl) + length, int(x_ratio*xl) + length)]

img1_with_square = img1.copy()
cv.rectangle(img1_with_square, square_pts[0], square_pts[3], red, thickness=thickness)
utils.plot_array("res01-450-rect.jpg", img1_with_square)

transformed_square_pts = matching.transfer_list_of_points(square_pts,
                                                          np.linalg.inv(H))
img2_with_square = img2.copy()
cv.line(img2_with_square, transformed_square_pts[0], transformed_square_pts[1], red, thickness=thickness)
cv.line(img2_with_square, transformed_square_pts[0], transformed_square_pts[2], red, thickness=thickness)
cv.line(img2_with_square, transformed_square_pts[1], transformed_square_pts[3], red, thickness=thickness)
cv.line(img2_with_square, transformed_square_pts[2], transformed_square_pts[3], red, thickness=thickness)
utils.plot_array("res02-450-rect.jpg", img2_with_square)

"""
########### panorama ###########
"""

x_shape = max(img1.shape[1] - min(0, min_offset[0]), min_offset_shape[0])
y_shape = max(img1.shape[0] - min(0, min_offset[1]), min_offset_shape[1])

panorama_270_450 = cv.warpPerspective(img2, H_min_offset, (x_shape, y_shape))
panorama_270_450[-min_offset[1]: img1.shape[0] - min_offset[1], -min_offset[0]: img1.shape[1] - min_offset[0], :] \
    = img1

utils.plot_array("res03-270-450-panorama.jpg", panorama_270_450)
print("Part 1 " + u'\u2713')

"""
################################## Part 2 ##################################
"""

img090 = utils.get_image(copy_path + "090.jpg")
img270 = utils.get_image(copy_path + "270.jpg")
img450 = utils.get_image(copy_path + "450.jpg")
img630 = utils.get_image(copy_path + "630.jpg")
img810 = utils.get_image(copy_path + "810.jpg")

H_090_270 = get_homography_matrix(img270, img090)
H_270_450 = get_homography_matrix(img450, img270)
H_630_450 = get_homography_matrix(img450, img630)
H_810_630 = get_homography_matrix(img630, img810)
H_090_450 = np.matmul(H_270_450, H_090_270)
H_810_450 = np.matmul(H_630_450, H_810_630)

min_x, min_y_090, _, max_y_090 = matching.get_min_offset(img090, H_090_450)
_, min_y_810, max_x, max_y_810 = matching.get_min_offset(img810, H_810_450)

min_y = min(min_y_090, 0, min_y_810)
max_y = max(max_y_090, img450.shape[1], max_y_810)
min_x = min(min_x, 0)
max_x = max(max_x, img450.shape[0])

panorama_shape = (max_x - min_x + 1, max_y - min_y + 1)

offset_mat = np.array([[1, 0, -min_x], [0, 1, -min_y], [0, 0, 1]])
img090_homo = cv.warpPerspective(img090, np.matmul(offset_mat, H_090_450), panorama_shape)
img270_homo = cv.warpPerspective(img270, np.matmul(offset_mat, H_270_450), panorama_shape)
img450_homo = np.zeros((panorama_shape[1], panorama_shape[0], 3), dtype=np.int16)
img450_homo[-min_y: img450.shape[0] - min_y, -min_x: img450.shape[1] - min_x, :] = img450
img630_homo = cv.warpPerspective(img630, np.matmul(offset_mat, H_630_450), panorama_shape)
img810_homo = cv.warpPerspective(img810, np.matmul(offset_mat, H_810_450), panorama_shape)

tr = params_dict['side_tr']
merged_im = panorama.merger(img090_homo, img270_homo, tr)
merged_im = panorama.merger(merged_im, img450_homo, tr)
merged_im = panorama.merger(merged_im, img630_homo, tr)
merged_im = panorama.merger(merged_im, img810_homo, tr)

utils.plot_array("res04-key_frames-panorama.jpg", merged_im)
print("Part 2 " + u'\u2713')

"""
################################## Part 3 ##################################
"""

all_H = []
for num_fr in tqdm.tqdm(range(1, num_frames + 1)):
    frame = utils.get_image(copy_path + utils.get_frame_name(num_fr))
    if num_fr < 90:
        H = get_homography_matrix(img090, frame)
        H = np.matmul(H_090_450, H)
    elif num_fr < 270:
        H = get_homography_matrix(img270, frame)
        H = np.matmul(H_270_450, H)
    elif num_fr == 450:
        H = np.array([[1.00001, 0, 0], [0, 1, 0], [0, 0, 1]])
    elif num_fr < 630:
        H = get_homography_matrix(img450, frame)
    elif num_fr < 810:
        H = get_homography_matrix(img630, frame)
        H = np.matmul(H_630_450, H)
    else:
        H = get_homography_matrix(img810, frame)
        H = np.matmul(H_810_450, H)
    all_H.append(H)


def get_min_max_range_frames(min_fr, max_fr, min_x, min_y, max_x, max_y):
    for num_fr in range(min_fr, max_fr + 1):
        frame = utils.get_image(copy_path + utils.get_frame_name(num_fr))
        H = all_H[num_fr - 1]
        min_x_tmp, min_y_tmp, max_x_tmp, max_y_tmp = matching.get_min_offset(frame, H)
        min_y = min(min_y, min_y_tmp)
        max_y = max(max_y, max_y_tmp)
        min_x = min(min_x, min_x_tmp)
        max_x = max(max_x, max_x_tmp)
    return min_x, min_y, max_x, max_y


H_001_450 = all_H[0]
img001 = utils.get_image(copy_path + "001.jpg")
min_x, min_y, max_x, max_y = matching.get_min_offset(img001, H_001_450)
min_x, min_y, max_x, max_y = get_min_max_range_frames(2, 20, min_x, min_y, max_x, max_y)
min_x, min_y, max_x, max_y = get_min_max_range_frames(440, 460, min_x, min_y, max_x, max_y)
min_x, min_y, max_x, max_y = get_min_max_range_frames(880, 900, min_x, min_y, max_x, max_y)

panorama_shape = (max_x - min_x + 1, max_y - min_y + 1)

batch_size = int(params_dict['video_batch_size'])

offset_mat = np.array([[1, 0, -min_x], [0, 1, -min_y], [0, 0, 1]])

video = utils.open_video('res05-reference-plane.mp4', panorama_shape)

for i in tqdm.tqdm(range(int(num_frames/batch_size))):
    frames = []
    for j in range(1, batch_size + 1):
        num_fr = i*batch_size + j
        frame = utils.get_image(copy_path + utils.get_frame_name(num_fr))
        H = all_H[num_fr - 1]
        new_fr = cv.warpPerspective(frame, np.matmul(offset_mat, H), panorama_shape)
        frames.append(new_fr)
    utils.video_writer(video, frames)
video.release()

print("Part 3 " + u'\u2713')

"""
################################## Part 4 ##################################
"""


mapped_video_path = path_dict['results'] + 'res05-reference-plane.mp4'
mapped_video_frames = 'mapped_video_frames/'
utils.create_frames(mapped_video_path, mapped_video_frames, 0, num_frames)
sample = utils.get_image(mapped_video_frames + "001.jpg")

path = mapped_video_frames

"""
########### with mode ###########
"""
final_image = np.zeros(sample.shape, dtype=np.float16)
gap = int(params_dict['gap'])
dist_mat = None
for i in range(int(final_image.shape[1] / gap)):
    y_min = i * gap
    y_max = y_min + gap
    if dist_mat is None:
        dist_mat = background.get_distribution_matrix(dist_mat, path, y_min, y_max)
    else:
        background.get_distribution_matrix(dist_mat, path, y_min, y_max)
    final_image = background.create_image(final_image, dist_mat, y_min, y_max)

if y_max < final_image.shape[1]:
    dist_mat = None
    y_min = y_max
    y_max = final_image.shape[1]
    dist_mat = background.get_distribution_matrix(dist_mat, path, y_min, y_max)
    final_image = background.create_image(final_image, dist_mat, y_min, y_max)

final_image[final_image > 255] = 0
dist_mat = None

utils.plot_array('res06-background-panorama_mode.jpg', final_image)

"""
########### with mean ###########
"""
final_image = np.zeros(sample.shape, dtype=np.float32)
count = np.zeros((sample.shape[0], sample.shape[1]), dtype=np.float32)

for num_fr in tqdm.tqdm(range(1, num_frames + 1)):
    image = utils.get_image(mapped_video_frames + utils.get_frame_name(num_fr)).astype(np.float32)
    small_mat = np.sum(image, axis=2) < 37
    x_small, y_small = np.nonzero(small_mat)
    image[x_small, y_small, :] = [0, 0, 0]
    final_image += image
    image[np.any(image != [0, 0, 0], axis=2)] = [1, 1, 1]
    count += np.sum(image, axis=2)/3

count[count == 0] = 1
final_image[:, :, 0] /= count
final_image[:, :, 1] /= count
final_image[:, :, 2] /= count

utils.plot_array('res06-background-panorama_mean.jpg', final_image)

"""
########### merge results ###########
"""
im1 = utils.get_image(path_dict['results'] + 'res06-background-panorama_mode.jpg').astype(np.float32)
im2 = utils.get_image(path_dict['results'] + 'res06-background-panorama_mean.jpg').astype(np.float32)

im2[:, int(3/5*im1.shape[1]):im1.shape[1], :] = 0
im1[:, 0:int(2/5*im1.shape[1]), :] = 0

res = panorama.merger(im2, im1, 0.3)
utils.plot_array('res06-background-panorama.jpg', res.astype(np.uint8))
print("Part 4 " + u'\u2713')

"""
################################## Part 5 ##################################
"""

offset_mat = np.array([[1, 0, +min_x], [0, 1, +min_y], [0, 0, 1]])
sample = utils.get_image(copy_path + "001.jpg")
video_size = (sample.shape[1], sample.shape[0])

background_image = utils.get_image(path_dict['results'] + 'res06-background-panorama.jpg')
video = utils.open_video('res07-background-video.mp4', video_size)

for i in tqdm.tqdm(range(int(num_frames/batch_size))):
    frames = []
    for j in range(1, batch_size + 1):
        num_fr = i*batch_size + j
        H = all_H[num_fr - 1]
        new_fr = cv.warpPerspective(background_image, np.matmul(np.linalg.inv(H), offset_mat), video_size)
        frames.append(new_fr)
    utils.video_writer(video, frames)
video.release()

print("Part 5 " + u'\u2713')

"""
################################## Part 6 ##################################
"""

background_video_path = path_dict['results'] + 'res07-background-video.mp4'
background_frames = 'background_frames/'
utils.create_frames(background_video_path, background_frames, 0, 1000, remove_existing=True)

sample = utils.get_image(copy_path + "001.jpg")
video_size = (sample.shape[1], sample.shape[0])

video = utils.open_video('res08-foreground-video.mp4', video_size)

ksize = int(params_dict['ksize'])
tr_foreground = params_dict['tr_foreground']
for i in tqdm.tqdm(range(int(num_frames/batch_size))):
    frames = []
    for j in range(1, batch_size + 1):
        num_fr = i*batch_size + j
        new_fr = background.create_foreground_frame(num_fr, background_frames, copy_path, tr=tr_foreground, ksize=ksize)
        frames.append(new_fr.astype(np.uint8))
    utils.video_writer(video, frames)
video.release()

print("Part 6 " + u'\u2713')

"""
################################## Part 7 ##################################
"""

scale = params_dict['scale']
offset_mat = np.array([[1, 0, +min_x], [0, 1, +min_y], [0, 0, 1]])
sample = utils.get_image(copy_path + "001.jpg")
video_size = (int(sample.shape[1] * scale), sample.shape[0])

background_image = utils.get_image(path_dict['results'] + 'res06-background-panorama.jpg')
video = utils.open_video('res09-background-video-wider.mp4', video_size)

flag = False
for i in tqdm.tqdm(range(int(num_frames / batch_size))):
    frames = []
    for j in range(1, batch_size + 1):
        num_fr = i * batch_size + j
        H = all_H[num_fr - 1]
        r_H = np.matmul(np.linalg.inv(H), offset_mat)
        new_fr = cv.warpPerspective(background_image, r_H, video_size)
        test = np.sum(new_fr, axis=2)
        _, y_nonzero = np.nonzero(test)
        if np.max(y_nonzero) < video_size[0] - 1:
            flag = True
            break
        frames.append(new_fr)
    utils.video_writer(video, frames)
    if flag:
        break
video.release()

print("Part 7 " + u'\u2713')

"""
################################## Part 8 ##################################
"""

middle = (panorama_shape[0]/2 - 1/2, panorama_shape[1]/2 - 1/2)
sample = utils.get_image(copy_path + utils.get_frame_name(1))
x = frame.shape[1]/2 - 1/2
y = frame.shape[0]/2 - 1/2

num_fr = 1
all_min_y = 10000
all_max_y = 0
all_offset2 = []
for num_fr in range(1, num_frames + 1):
    H = all_H[num_fr - 1]
    point = np.array([[x, y, 1]]).reshape(3, 1)
    pp_proj = np.matmul(H, point).reshape(-1)
    pp = (pp_proj[0]/pp_proj[2], pp_proj[1]/pp_proj[2])
    shift = (middle[1] - pp[1] - min_y)
    offset_mat2 = np.array([[1, 0, 0], [0, 1, shift], [0, 0, 1]])
    all_offset2.append(offset_mat2)
    _, min_yy, _, max_yy = matching.get_min_offset(frame,
                                                   np.matmul(offset_mat2, np.matmul(offset_mat, H)))
    all_min_y = min(all_min_y, min_yy)
    all_max_y = max(all_max_y, max_yy)

panorama_shape1 = (panorama_shape[0], all_max_y - all_min_y + 1)
offset_mat3 = np.array([[1, 0, 0], [0, 1, -all_min_y], [0, 0, 1]])

all_min_x = 10000
all_max_x = 0
for num_fr in range(1, num_frames + 1):
    H = all_H[num_fr - 1]
    point = np.array([[x, y, 1]]).reshape(3, 1)
    pp_proj = np.matmul(H, point).reshape(-1)
    pp = (pp_proj[0]/pp_proj[2], pp_proj[1]/pp_proj[2])
    shift = (pp[0] - middle[0] - min_x)
    all_min_x = min(all_min_x, shift)
    all_max_x = max(all_max_x, shift)

dx = all_max_x - all_min_x
all_offset4 = []
for num_fr in range(1, num_frames + 1):
    H = all_H[num_fr - 1]
    point = np.array([[x, y, 1]]).reshape(3, 1)
    pp_proj = np.matmul(H, point).reshape(-1)
    pp = (pp_proj[0]/pp_proj[2], pp_proj[1]/pp_proj[2])
    desired = all_min_x + np.sin(np.pi*(num_fr - 1)/(num_frames - 1))*all_max_x + middle[0]
    delta = pp[0] - min_x - desired
    offset_mat4 = np.array([[1, 0, delta], [0, 1, 0], [0, 0, 1]])
    all_offset4.append(offset_mat4)

video = utils.open_video('res10-video-shakeless.mp4', (frame.shape[1], frame.shape[0]))

in_angle_offset = np.array([[1, 0, 0], [0, 1, 30], [0, 0, 1]])
for i in tqdm.tqdm(range(int(num_frames/batch_size))):
    frames = []
    for j in range(1, batch_size + 1):
        num_fr = i*batch_size + j
        H = all_H[num_fr - 1]
        frame = utils.get_image(copy_path + utils.get_frame_name(num_fr))
        homo = np.matmul(offset_mat3, np.matmul(all_offset2[num_fr - 1], np.matmul(offset_mat, H)))
        new_fr = cv.warpPerspective(frame, np.matmul(all_offset4[num_fr - 1], homo),
                                    (panorama_shape1[0] + 900, panorama_shape1[1]))
        new_fr = cv.warpPerspective(new_fr, np.matmul(in_angle_offset, np.linalg.inv(np.matmul(offset_mat, H))),
                                    (frame.shape[1], frame.shape[0]))
        frames.append(new_fr)
    utils.video_writer(video, frames)
video.release()

utils.remove_dir(background_frames)
utils.remove_dir(mapped_video_frames)
