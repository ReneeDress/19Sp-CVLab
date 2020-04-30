import cv2 as cv
import numpy as np
# 读入图片并转换
img = cv.imread('img.jpg')
find = cv.imread('find.jpg')
arimg = np.asarray(img)
arfind = np.asarray(find)
# 使用matchTemplate模板匹配函数
result = cv.matchTemplate(arimg, arfind, cv.TM_SQDIFF)
# 获得目标大小
size = arfind.shape
# 新建目标位置数组
poi = []
#[iv,mv,il,ml] = cv.minMaxLoc(result)
#print(iv,mv,il,ml)
for i in range(result.shape[0]):
    for j in range(result.shape[1]):
        if result[i][j] <= 500000:
            poi.append((i, j))
#print(poi)
#print(len(poi))
#print(result.astype(np.float32))
#print(poi[0][0])
# 复制nparray格式的图片并画出矩形框
resultimg = arimg
for i in range(0, len(poi)):
    topleft = (int(poi[i][1]), int(poi[i][0]))
    bottomright = (int(poi[i][1]+size[1]), int(poi[i][0]+size[0]))
    print('No.', i+1, topleft, bottomright)
    cv.rectangle(img=resultimg, pt1=topleft, pt2=bottomright, color=0, thickness=2)
cv.imshow('Result', resultimg)
cv.waitKey()