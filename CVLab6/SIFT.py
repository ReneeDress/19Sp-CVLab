import cv2 as cv
import numpy as np
#building = 'EpiscopalGaudi'
building = 'MountRushmore'
#building = 'NotreDame'
# Load source image and convert it to gray
img1 = cv.imread('data/' + building + '/1.jpg')
img1_gray = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
img2 = cv.imread('data/' + building + '/2.jpg')
img2_gray = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)
# Detect the keypoints using SIFT Detector
sift = cv.xfeatures2d.SIFT_create()
keypoints = sift.detect(img1_gray)
# Draw keypoints
img_keypoints = np.empty((img1_gray.shape[0], img1_gray.shape[1], 3), dtype=np.uint8)
cv.drawKeypoints(img1, keypoints, img_keypoints)
# Show detected (drawn) keypoints
cv.imshow('SIFT Keypoints', img_keypoints)
cv.waitKey()