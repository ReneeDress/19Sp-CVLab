import cv2 as cv
import os
img = cv.imread('2.jpeg')
# 检测用xml文件路径并载入
cascadePath = os.path.abspath('/Users/reneelin/Desktop/Programmer/opencv-4.3.0/data/haarcascades/haarcascade_fullbody.xml')
haarCascadeFullBody = cv.CascadeClassifier(cascadePath)
# 将原图转化为灰度图并进行检测
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
result = haarCascadeFullBody.detectMultiScale(gray, 1.2, 3)
#print(result)
#print(result.shape)
# 复制原图片并画出矩形框
resultimg = img
for i in range(0, result.shape[0]):
    topleft = (result[i][0], result[i][1])
    bottomright = (result[i][0] + result[i][2], result[i][1] + result[i][3])
    print('No.', i + 1, topleft, bottomright)
    cv.rectangle(img=resultimg, pt1=topleft, pt2=bottomright, color=255, thickness=2)
cv.imshow('Result', resultimg)
cv.waitKey()