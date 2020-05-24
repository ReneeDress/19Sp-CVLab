import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

MIN = 10
path = 'two/uttower/'
#img1 = cv.imread(path + 'Hanging1.png')  # query
#img2 = cv.imread(path + 'Hanging2.png')  # train
img1 = cv.imread(path + 'uttower_left.jpg')  # query
img2 = cv.imread(path + 'uttower_right.jpg')  # train

# Detects keypoints and computes the descriptors using SIFT Detector
sift = cv.xfeatures2d.SIFT_create()
keypoints1, descriptors1 = sift.detectAndCompute(img1, mask=None)
keypoints2, descriptors2 = sift.detectAndCompute(img2, mask=None)
# FLANN (Fast_Library_for_Approximate_Nearest_Neighbors)快速最近邻搜索包。
# 它是一个对大数据集和高维特征进行最近邻搜索的算法的集合,而且这些算法都已经被优化过了。在面对大数据集时它的效果要好于 BFMatcher。
FLANN_INDEX_KDTREE = 0  # kd树（k-dimensional树的简称），是一种分割k维数据空间的数据结构。主要应用于多维空间关键数据的搜索（如：范围搜索和最近邻搜索）。
# 创建字典参数
indexParams = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)   # 处理索引
searchParams = dict(checks=100) # 创建对象,用来指定索引树的遍历次数,值越高结果越准确

flann = cv.FlannBasedMatcher(indexParams, searchParams)
match = flann.knnMatch(descriptors1, descriptors2, k=2)

# DMATCH
# queryIdx: query descriptor index
# trainIdx: train image index
good = []
for i, (m, n) in enumerate(match):
    if (m.distance < 0.2 * n.distance):     # edit origin 0.75
        good.append(m)

if len(good) > MIN:
    srcList = []
    dstList = []
    for m in good:
        srcList.append(keypoints1[m.queryIdx].pt)
        dstList.append(keypoints2[m.trainIdx].pt)
    srcPoints = np.float32(srcList).reshape(-1, 1, 2)
    dstPoints = np.float32(dstList).reshape(-1, 1, 2)
    #srcPoints = np.float32([keypoints1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2) # 列表生成式
    #dstPoints = np.float32([keypoints2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    M, mask = cv.findHomography(srcPoints, dstPoints, cv.RANSAC)    # 求变换矩阵
    warpImg = cv.warpPerspective(img2, np.linalg.inv(M), (img1.shape[1] + img2.shape[1], img2.shape[0]))    # 进行透视变换
    feather = np.zeros([warpImg.shape[0], warpImg.shape[1]], np.uint8)
    for row in range(0, warpImg.shape[0]):
        for col in range(0, warpImg.shape[1]):
            if warpImg[row, col].all():
                feather[row, col] = 255
    #feather = cv.blur(feather, (300, 300))
    blur = cv.blur(warpImg, (300, 300))
    feather = cv.cvtColor(feather, cv.COLOR_GRAY2RGB)
    for row in range(0, warpImg.shape[0]):
        for col in range(0, warpImg.shape[1]):
            if warpImg[row, col].all():
                blur[row, col] = 0
    #masked = np.zeros([warpImg.shape[0], warpImg.shape[1]], np.uint8)
    masked = cv.addWeighted(warpImg, 1, blur, 2, 0)
    print(warpImg.dtype, feather.dtype, masked.dtype)
    warpImg1 = cv.cvtColor(warpImg, cv.COLOR_BGR2RGB)
    blur1 = cv.cvtColor(blur, cv.COLOR_BGR2RGB)
    masked1 = cv.cvtColor(masked, cv.COLOR_BGR2RGB)
    #plt.subplot(311)
    #plt.imshow(warpImg1, )
    #plt.subplot(312)
    #plt.imshow(blur1, )
    #plt.subplot(313)
    #plt.imshow(masked1, )
    #plt.show()
    direct = warpImg.copy()
    direct[0:img1.shape[0], 0:img1.shape[1]] = img1     # 直接加上左侧图片
    rows, cols = img1.shape[:2]

    for col in range(0, cols):
        for row in range(0, rows):
            if img1[row, col].any() and warpImg[row, col].any():  # 开始重叠的最左端
                right = col
                top = row
                break
    for col in range(cols - 1, 0, -1):
        for row in range(0, rows):
            if img1[row, col].any() and warpImg[row, col].any():  # 重叠的最右一列
                left = col
                bottom = row
                break

    res = np.zeros([rows, cols, 3], np.uint8)
    grada = np.zeros([rows, cols], np.uint8)
    gradb = np.zeros([rows, cols], np.uint8)
    for row2 in range(0, rows):
        print(int((bottom - top) / (right - left) * (right - left)))
        for row in range(0, int((bottom - top) / (right - left) * (right - left))):
            col = int((right - left) / (bottom - top) * (- row) + right + row2)
            if col > right:
                col = right
            print((row2, row, col))
            if not img1[row, col].any():  # 如果没有原图，用旋转的填充
                res[row, col] = warpImg[row, col]
            elif not warpImg[row, col].any():
                res[row, col] = img1[row, col]
            else:
                srcImgLen = float(abs(col - left))
                testImgLen = float(abs(col - right))
                beta = 255
                alpha = srcImgLen / (srcImgLen + testImgLen)
                print(alpha, row, col)
                print((left, right), (top, bottom))
                grada[row, col] = alpha * 255
                if - row > rows / (right - left) * col - rows / (right - left) * right:
                    srcImgHi = float(abs(row - top))
                    testImgHi = float(abs(row - bottom))
                    beta = srcImgHi / (srcImgHi + testImgHi)
                else:
                    beta = 255
                alpha = min(alpha, beta)
                gradb[row, col] = beta * 255

            res[row, col] = np.clip(img1[row, col] * (1 - alpha) + warpImg[row, col] * alpha, 0, 255)

    warpImg[0:img1.shape[0], 0:img1.shape[1]] = res
    img3 = cv.cvtColor(direct, cv.COLOR_BGR2RGB)
    plt.subplot(411)
    plt.imshow(img3, )
    plt.axis('off')
    plt.subplot(412)
    grada = cv.cvtColor(grada, cv.COLOR_GRAY2RGB)
    plt.imshow(grada, )
    plt.axis('off')
    plt.subplot(413)
    gradb = cv.cvtColor(gradb, cv.COLOR_GRAY2RGB)
    plt.imshow(gradb, )
    plt.axis('off')
    img4 = cv.cvtColor(warpImg, cv.COLOR_BGR2RGB)
    plt.subplot(414)
    plt.imshow(img4, )
    plt.axis('off')
    plt.show()

    cv.imwrite(path + "simplepanorma.png", direct)
    cv.imwrite(path + "bestpanorma.png", warpImg)

else:
    print("not enough matches!")