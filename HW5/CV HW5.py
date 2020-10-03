#!/usr/bin/env python
# coding: utf-8

import numpy as np
import cv2
import matplotlib.pylab as plt
plt.rcParams["figure.figsize"] = (12,6)

def show_image(image):
    plt.imshow(image, cmap='gray', vmin=0, vmax=255)
    plt.show()

image = cv2.imread('temp.jpg', 0)
show_image(image)

image.shape, image.min(), image.max()

plt.plot(cv2.calcHist([image], [0], None, [256], [0, 256])) 
plt.show()

def normalize_histogram(image):
    unique, counts = np.unique(image, return_counts=True)
    pdf = dict(zip(unique, counts))
    cdf = dict(zip(list(pdf.keys()), np.cumsum(list(pdf.values()))))
    cdf_min = pdf[image.min()]
    total_pixels = image.shape[0] * image.shape[1]
    norm_histo = {i: round(255 * (cdf[i] - cdf_min) / (total_pixels - cdf_min)) for i in cdf.keys()}
    
    new_image = np.zeros((image.shape), dtype="uint8")
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            new_image[i][j] = norm_histo[image[i][j]]
    return new_image

new = normalize_histogram(image)
show_image(new)

plt.plot(cv2.calcHist([new], [0], None, [256], [0, 256]))
plt.show()
