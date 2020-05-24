import cv2 as cv
#building = 'EpiscopalGaudi'
#building = 'MountRushmore'
building = 'NotreDame'
# Load source image and convert it to gray
print('Load source image and convert it to gray')
img1 = cv.imread('two/Hanging/Hanging1.png')  # query
img2 = cv.imread('two/Hanging/Hanging2.png')  # train
img1_gray = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
img2_gray = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)
# Detect the keypoints using SIFT Detector
print('Detect the keypoints using SIFT Detector')
sift = cv.xfeatures2d.SIFT_create()
keypoints1, descriptors1 = sift.detectAndCompute(img1_gray, None)
keypoints2, descriptors2 = sift.detectAndCompute(img2_gray, None)
# Brute-Force Matching
print('Brute-Force Matching')
bf = cv.BFMatcher(cv.NORM_L1, crossCheck=True)
match = bf.match(descriptors1, descriptors2)
match = sorted(match, key=lambda x: x.distance)
# Draw Top 100 Matches
result = cv.drawMatches(img1, keypoints1, img2, keypoints2, match[:50], img2, flags=2)
# Show Matches
cv.imshow('SIFT Matches', result)
cv.waitKey()