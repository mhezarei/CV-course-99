#!/usr/bin/env python
# coding: utf-8

import numpy as np
import cv2
import matplotlib.pylab as plt
plt.rcParams["figure.figsize"] = (12, 6)


def show_image(image):
    plt.imshow(image, cmap='gray', vmin=0, vmax=255)
    plt.show()


image = cv2.imread('temp2.jpg', 0)
show_image(image)

plt.plot(cv2.calcHist([image], [0], None, [256], [0, 256]))
plt.show()

p = cv2.calcHist([image], [0], None, [256], [0, 256]
                 ).flatten() / (image.shape[0] * image.shape[1])


def theta(t):
    return p[: t + 1].sum()


def cmean(t):
    return sum([x * p[x] for x in range(t + 1)])


gmean = cmean(255)


def formula(t):
    # between-class variance which should be maximized
    if theta(t) == 0 or theta(t) == 1.0:
        return -1
    return (cmean(t) - gmean * theta(t))**2 / (theta(t) * (1 - theta(t)))


variances = np.array([formula(t) for t in range(256)])
thresh = variances.argmax()

new = np.where(image > thresh, 255, image)

show_image(new)
