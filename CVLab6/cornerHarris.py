# https://docs.opencv.org/3.4/d4/d7d/tutorial_harris_detector.html
import cv2 as cv
import numpy as np
thresh = 30
building = 'EpiscopalGaudi'
#building = 'MountRushmore'
#building = 'NotreDame'
# Load source image and convert it to gray
img1 = cv.imread('data/' + building + '/1.jpg')
img1_gray = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
img2 = cv.imread('data/' + building + '/2.jpg')
img2_gray = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)
# Detecting corners
dst = cv.cornerHarris(img1_gray, 2, 3, 0.01)
# Normalizing
dst_norm = np.empty(dst.shape, dtype=np.float32)
cv.normalize(dst, dst_norm, alpha=0, beta=255, norm_type=cv.NORM_MINMAX)
dst_norm_scaled = cv.convertScaleAbs(dst_norm)
# Drawing a circle around corners
for i in range(dst_norm.shape[0]):
    for j in range(dst_norm.shape[1]):
        if int(dst_norm[i,j]) > thresh:
            cv.circle(dst_norm_scaled, (j,i), 5, (0), 2)
            cv.circle(img1, (j, i), 5, (0,0,255), 2)
# Showing the result
cv.imshow('cornerHarris', img1)
cv.waitKey()