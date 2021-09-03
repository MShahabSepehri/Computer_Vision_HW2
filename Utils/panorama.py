import numpy as np
import math
import tqdm


def get_cropped_union(im1, im2):
    x = im2.copy()
    x[np.any(x != [0, 0, 0], axis=2)] = [1, 1, 1]
    im1_uni = im1 * x
    x = im1.copy()
    x[np.any(x != [0, 0, 0], axis=2)] = [1, 1, 1]
    im2_uni = im2 * x

    x_nonzero, y_nonzero, _ = np.nonzero(im2_uni)
    min_x = np.min(x_nonzero)
    max_x = np.max(x_nonzero)
    min_y = np.min(y_nonzero)
    max_y = np.max(y_nonzero)
    im1_uni_cropped = im1_uni[min_x: max_x + 1, min_y: max_y + 1, :]
    im2_uni_cropped = im2_uni[min_x: max_x + 1, min_y: max_y + 1, :]
    diff = np.sum(((im1_uni_cropped - im2_uni_cropped)/100) ** 2, axis=2)

    return diff, im1_uni_cropped, im2_uni_cropped, (min_x, max_x, min_y, max_y)


def compute_cost_matrix(diff, cropped_im, min_x, max_x, min_y, max_y, tr=0.25):
    x_nonzero, y_nonzero, _ = np.nonzero(cropped_im)
    cost_mat = np.zeros((max_x - min_x + 1, max_y - min_y + 1)) + math.inf

    ## initialization
    for y in range(max_y - min_y + 1):
        col_x_nonzero, _ = np.nonzero(cropped_im[:, y, :])
        x = np.min(col_x_nonzero)

        if x < tr * (max_x - min_x):
            cost_mat[x, y] = diff[x, y]

    ## DP
    for x in tqdm.tqdm(range(0, max_x - min_x + 1)):
        row = np.where(x_nonzero == x)[0]
        for j in row:
            y = y_nonzero[j]
            if 0 < y < max_y - min_y:
                c = min(cost_mat[x - 1, y - 1], cost_mat[x - 1, y], cost_mat[x - 1, y + 1])
                cost_mat[x, y] = min(cost_mat[x, y], diff[x, y] + c)
            elif y == 0:
                c = min(cost_mat[x - 1, y], cost_mat[x - 1, y + 1])
                cost_mat[x, y] = min(cost_mat[x, y], diff[x, y] + c)
            else:
                c = min(cost_mat[x - 1, y], cost_mat[x - 1, y - 1])
                cost_mat[x, y] = min(cost_mat[x, y], diff[x, y] + c)
    return cost_mat


def find_end_point(cost_mat, cropped_im, min_x, max_x, min_y, max_y, tr=0.25):
    min_cost = min(cost_mat[-1, :])
    min_cost_y = np.where(cost_mat[-1, :] == min_cost)[0][0]
    end_point = (max_x - min_x, min_cost_y)
    for y in range(max_y - min_y + 1):
        col_x_nonzero, _ = np.nonzero(cropped_im[:, y, :])
        x = np.max(col_x_nonzero)
        if max_x - x < tr * (max_x - min_x):
            if min_cost > cost_mat[x, y]:
                min_cost = cost_mat[x, y]
                end_point = (x, y)
    return end_point


def merge_images(cost_mat, mask_size, im1_uni_cropped, min_x, max_x, min_y, max_y, tr=0.25):
    mask = np.zeros(mask_size, dtype=np.int16)
    x, y = find_end_point(cost_mat, im1_uni_cropped, min_x, max_x, min_y, max_y, tr=tr)

    x_a = mask.shape[0]
    while x_a > x + min_x:
        x_a -= 1
        mask[x_a, 0:y + min_y, :] += 1
    while x > 0:
        mask[x + min_x, 0:y + min_y, :] += 1
        x -= 1
        if 0 < y < max_y - min_y:
            c = min(cost_mat[x - 1, y - 1], cost_mat[x - 1, y], cost_mat[x - 1, y + 1])
            if c == cost_mat[x - 1, y - 1]:
                y -= 1
            elif c == cost_mat[x - 1, y + 1]:
                y += 1
            elif c != cost_mat[x - 1, y]:
                break
        elif y == 0:
            c = min(cost_mat[x - 1, y], cost_mat[x - 1, y + 1])
            if c == cost_mat[x - 1, y + 1]:
                y += 1
            elif c != cost_mat[x - 1, y]:
                break
        else:
            c = min(cost_mat[x - 1, y - 1], cost_mat[x - 1, y])
            if c == cost_mat[x - 1, y - 1]:
                y -= 1
            elif c != cost_mat[x - 1, y]:
                break
    x += min_x
    while x > 0:
        mask[x, 0:y + min_y, :] += 1
        x -= 1
    mask[np.any(mask != [0, 0, 0], axis=-1)] = [1, 1, 1]
    return mask


def merger(im1, im2, tr=0.25):
    diff, im1_uni_cropped, im2_uni_cropped, (min_x, max_x, min_y, max_y) = get_cropped_union(im1, im2)
    cost_mat = compute_cost_matrix(diff, im1_uni_cropped, min_x, max_x, min_y, max_y, tr=tr)
    mask = merge_images(cost_mat, im1.shape, im1_uni_cropped, min_x, max_x, min_y, max_y, tr=tr)
    return mask*im1 + (1 - mask)*im2
