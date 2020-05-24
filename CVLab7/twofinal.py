import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt


def two(img1, img2):
    MIN = 10    # 设置最少匹配数
    # 使用SIFT检测匹配点并计算描述子
    # Detects keypoints and computes the descriptors using SIFT Detector
    sift = cv.xfeatures2d.SIFT_create()
    keypoints1, descriptors1 = sift.detectAndCompute(img1, mask=None)
    keypoints2, descriptors2 = sift.detectAndCompute(img2, mask=None)
    # FLANN (Fast_Library_for_Approximate_Nearest_Neighbors)快速最近邻搜索包
    # 对大数据集和高维特征进行最近邻搜索的算法的集合，面对大数据集时效果好于BFMatcher
    FLANN_INDEX_KDTREE = 0  # kd树（k-dimensional树的简称），是一种分割k维数据空间的数据结构。主要应用于多维空间关键数据的搜索（如：范围搜索和最近邻搜索）。
    indexParams = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)   # 处理索引
    searchParams = dict(checks=100)     # 创建对象,用来指定索引树的遍历次数,值越高结果越准确
    flann = cv.FlannBasedMatcher(indexParams, searchParams)
    match = flann.knnMatch(descriptors1, descriptors2, k=2)
    good = []
    for i, (m, n) in enumerate(match):
        if (m.distance < 0.5 * n.distance):     # 数值越小越优
            good.append(m)

    if len(good) > MIN:
        srcList = []
        dstList = []
        for m in good:
            # DMATCH：queryIdx: 待匹配的描述子；trainIdx: 被匹配的描述子
            srcList.append(keypoints1[m.queryIdx].pt)
            dstList.append(keypoints2[m.trainIdx].pt)
        srcPoints = np.float32(srcList).reshape(-1, 1, 2)
        dstPoints = np.float32(dstList).reshape(-1, 1, 2)
        M, mask = cv.findHomography(srcPoints, dstPoints, cv.RANSAC)    # 求变换矩阵
        warpImg = cv.warpPerspective(img2, np.linalg.inv(M), (img1.shape[1] + img2.shape[1], img2.shape[0]))    # 进行透视变换
        direct = warpImg.copy()
        direct[0:img1.shape[0], 0:img1.shape[1]] = img1     # 直接加上左侧图片 == 直接叠加

        rows, cols = img1.shape[:2]
        for col in range(0, cols):
            if img1[:, col].any() and warpImg[:, col].any():  # 开始重叠的最左端
                left = col
                break
        for col in range(cols - 1, 0, -1):
            if img1[:, col].any() and warpImg[:, col].any():  # 重叠的最右一列
                right = col
                break
        res = np.zeros([rows, cols, 3], np.uint8)
        grad = np.zeros([rows, cols], np.uint8)
        for row in range(0, rows):
            for col in range(0, cols):
                if not img1[row, col].any():  # 如果没有1图，用变换后2图的填充
                    res[row, col] = warpImg[row, col]
                elif not warpImg[row, col].any():
                    res[row, col] = img1[row, col]
                else:
                    srcImgLen = float(abs(col - left))
                    dstImgLen = float(abs(col - right))
                    alpha = srcImgLen / (srcImgLen + dstImgLen)
                    grad[row, col] = alpha * 255    # 从重叠最左至最右过渡的遮罩
                    res[row, col] = np.clip(img1[row, col] * (1 - alpha) + warpImg[row, col] * alpha, 0, 255)
        warpImg[0:img1.shape[0], 0:img1.shape[1]] = res
    else:
        print("not enough matches!")

    return direct, warpImg, grad


def crop(Img):
    global left     # 寻找全黑的最左列
    rows, cols = Img.shape[:2]
    for col in range(0, cols):
        if not Img[:, col].any():
            left = col
            break
    cropImg = Img[:, 0:left]
    return cropImg


if __name__ == "__main__":
    ph = 'two/Hanging/'
    pu = 'two/uttower/'
    path = pu
    h1 = cv.imread(path + 'Hanging1.png')
    h2 = cv.imread(path + 'Hanging2.png')
    u1 = cv.imread(path + 'uttower_left.jpg')
    u2 = cv.imread(path + 'uttower_right.jpg')
    direct, warpImg, grad = two(u1, u2)

    out1 = cv.cvtColor(direct, cv.COLOR_BGR2RGB)
    plt.subplot(311)
    plt.imshow(out1, )
    plt.axis('off')
    out3 = cv.cvtColor(grad, cv.COLOR_GRAY2RGB)
    plt.subplot(312)
    plt.imshow(out3, )
    plt.axis('off')
    out2 = cv.cvtColor(warpImg, cv.COLOR_BGR2RGB)
    plt.subplot(313)
    plt.imshow(out2, )
    plt.axis('off')
    plt.show()

    cropImgD = crop(direct)
    outD = cv.cvtColor(cropImgD, cv.COLOR_BGR2RGB)
    plt.subplot(211)
    plt.imshow(outD, )
    plt.axis('off')
    cropImgW = crop(warpImg)
    outW = cv.cvtColor(cropImgW, cv.COLOR_BGR2RGB)
    plt.subplot(212)
    plt.imshow(outW, )
    plt.axis('off')
    plt.show()

    cv.imwrite(path + "direct.png", cropImgD)
    cv.imwrite(path + "warpImg.png", cropImgW)