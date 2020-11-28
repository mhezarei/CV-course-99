#!/usr/bin/env python
# coding: utf-8

import numpy as np
import cv2
import matplotlib.pylab as plt
plt.rcParams["figure.figsize"] = (12, 6)


def show_image(image):
    plt.imshow(image, cmap='gray', vmin=0, vmax=255)
    plt.show()


image = cv2.imread('region3.tif', 0)
show_image(image)

plt.plot(cv2.calcHist([image], [0], None, [256], [0, 256]))
plt.show()

SEED_VALUE = 255  # Should be picked manually
DX = [-1, -1, -1, 0, 0, 1, 1, 1]
DY = [-1, 0, 1, -1, 1, -1, 0, 1]
DIFFERENCE = 70


def is_valid(x, y):
    return 0 <= x < image.shape[0] and 0 <= y < image.shape[1]


def get_predicate_image(image):
    seeds = np.where(image == SEED_VALUE, True, False)
    queue = np.transpose(seeds.nonzero()).tolist()
    visited = np.zeros(image.shape)
    predicate = np.zeros(image.shape)

    visited[queue[0][0]][queue[0][1]] = True
    predicate[queue[0][0]][queue[0][1]] = 1

    while queue:
        center = queue.pop(0)
        cx, cy = center[0], center[1]
        for i in range(len(DX)):
            px, py = cx + DX[i], cy + DY[i]
            if is_valid(
                    px, py) and visited[px][py] == 0 and abs(
                    SEED_VALUE - image[px][py]) <= DIFFERENCE:
                queue.append([px, py])
                predicate[px][py] = 1
                visited[px][py] = 1

    final = np.zeros(image.shape, dtype='uint8')
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if predicate[i][j] == 1:
                final[i][j] = image[i][j]

    return final


pred_image = get_predicate_image(image)
show_image(pred_image)


def get_neighbor_code(x, y, codes):
    for k in range(len(DX)):
        px, py = x + DX[k], y + DY[k]
        if is_valid(px, py) and codes[px][py] > 0:
            return codes[px][py]
    return 0


def code_regions(image):
    visited = np.zeros(image.shape)
    codes = np.zeros(image.shape)

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if image[i][j] == 0:
                continue
            stack = []
            if visited[i][j] == 0:
                neighbor_code = get_neighbor_code(i, j, codes)
                if neighbor_code == 0:
                    stack.append([i, j])
                    visited[i][j] == 1
                    codes[i][j] = codes.max() + 1
                else:
                    codes[i][j] = neighbor_code
            while stack:
                center = stack.pop()
                cx, cy = center[0], center[1]
                for k in range(len(DX)):
                    px, py = cx + DX[k], cy + DY[k]
                    if is_valid(
                            px,
                            py) and visited[px][py] == 0 and image[px][py] > 0 and codes[px][py] == 0:
                        stack.append([px, py])
                        codes[px][py] = codes[cx][cy]
                        visited[px][py] = 1
    return codes


def get_regions(codes):
    code_points = {}
    for code in np.unique(codes)[1:]:
        x, y, p, q = 0, 0, 0, 0
        for i in range(codes.shape[0]):
            if code in codes[i]:
                x = i
                break
        for j in range(codes.T.shape[0]):
            if code in codes.T[j]:
                y = j
                break

        for i in reversed(range(codes.shape[0])):
            if code in codes[i]:
                p = i
                break
        for j in reversed(range(codes.shape[1])):
            if code in codes.T[j]:
                q = j
                break
        code_points[code] = [(x, y), (p, q)]
    return code_points


codes = code_regions(pred_image)

code_points = get_regions(codes)


def draw_regions(image, code_points):
    rgb_image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    for c in code_points.keys():
        top_left = tuple(reversed(code_points[c][0]))
        bottom_right = tuple(reversed(code_points[c][1]))
        rgb_image = cv2.rectangle(
            rgb_image, top_left, bottom_right, (0, 0, 255), 1)
    return rgb_image


f, axarr = plt.subplots(1, 2)
axarr[0].imshow(draw_regions(image, code_points))
axarr[1].imshow(draw_regions(pred_image, code_points))
