#!/usr/bin/env python
# coding: utf-8

import numpy as np
import cv2
import matplotlib.pylab as plt
from math import sqrt, pi, log2
plt.rcParams["figure.figsize"] = (12,6)


def show_image(image):
    plt.imshow(image, cmap='gray', vmin=0, vmax=255)
    plt.show()


image = cv2.imread('temp2.jpg', 0)
show_image(image)

image.shape

plt.plot(cv2.calcHist([image], [0], None, [256], [0, 256])) 
plt.show()


def get_u(image, threshold, background):
    if background:
        return cv2.calcHist([image], [0], None, [256], [0, 256]).flatten()[threshold:].argmax() + threshold
    else:
        return cv2.calcHist([image], [0], None, [256], [0, 256]).flatten()[:threshold].argmax()


start_th = 125 # expreimental based on the histogram
u_o, u_b = get_u(image, start_th, False), get_u(image, start_th, True)
u_o, u_b

sigma_o = 1 / (sqrt(2 * pi) * u_o)
sigma_b = 1 / (sqrt(2 * pi) * u_b)
sigma_o, sigma_b

A_o = cv2.calcHist([np.where(image < start_th, 0, image)], [0], None, [256], [0, 256]).flatten()[1:].max()
A_b = cv2.calcHist([np.where(image > start_th, 0, image)], [0], None, [256], [0, 256]).flatten()[1:].max()
A_o, A_b

theta = (sigma_o * A_o) / (sigma_o * A_o + sigma_b * A_b)
theta

threshold = (u_o + u_b) / 2 - ((sigma_o ** 2 + sigma_b ** 2) / (u_b - u_o)) * log2((1 - theta) / theta)

threshold

new = np.where(image > threshold, 255, image)

show_image(new)

