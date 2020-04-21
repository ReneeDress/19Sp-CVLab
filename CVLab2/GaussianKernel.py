import cv2 as cv
import numpy as np

cat = cv.imread("cat.jpg")
dog = cv.imread("dog.jpg")
catlow = cv.imread("cat.jpg")
doglow = cv.imread("dog.jpg")
cathigh = cv.imread("cat.jpg")
doghigh = cv.imread("dog.jpg")
k = cv.getGaussianKernel(35, 0)
kernel = np.multiply(k, np.transpose(k))
cv.filter2D(cat, -1, kernel, catlow)
cathigh = cat - catlow
cv.filter2D(dog, -1, kernel, doglow)
doghigh = dog - doglow
cv.imshow("cathigh", cathigh)
cv.imshow("doggauss", doglow)
result = doglow * 1 + cathigh * 1
cv.imshow("result", result)
cv.waitKey(0)