import cv2 as cv
import numpy as np

cat = cv.imread("cat.jpg")
dog = cv.imread("dog.jpg")
catlow = np.array(cv.GaussianBlur(cat, (35, 35), 0))
doglow = np.array(cv.GaussianBlur(dog, (35, 35), 0))
cathigh = cv.imread("cat.jpg")
doghigh = cv.imread("dog.jpg")
cathigh = cat - catlow
doghigh = dog - doglow
result = doglow * 1 + cathigh * 1
cv.imshow("cathigh", cathigh)
cv.imshow("doglow", doglow)
cv.imshow("result", result)
cv.waitKey(0)