import os
from sklearn.decomposition import PCA
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt


path = './att_faces/s24'    # 文件路径
img = []                    # 图片数组
X = []                      # 图片转成列向量数组
cnt = 0                     # 图片数量
# 列出目录中的所有文件，逐个从文本文件中读取点
for filePath in sorted(os.listdir(path)):
    fileExt = os.path.splitext(filePath)[1]
    if fileExt in ['.pgm']:
        # 读取图片
        imagePath = os.path.join(path, filePath)
        im = cv.imread(imagePath, 0)
        s = im.shape
        dots = int(s[0] * s[1])
        img.append(im)
        X.append(im.reshape(1, dots)[0])
        cnt += 1

# OpenCV自带的PCA似乎没有Python支持
# 使用sklearn的PCA进行实验
for n_comp in range(1,11):  # 为对比结果建立循环
    X = np.asarray(X)       # 转换
    #print('X', X.shape)
    pca = PCA(n_components=n_comp)  # 建立PCA参数
    pca.fit(X)              # 进行PCA
    XS = pca.fit_transform(X)   # 降维
    #print('XS', XS.shape)
    mean = pca.mean_.reshape((s[0], s[1]))  # 均值并重设为原始大小
    #sizedeigenfaces = pca.components_.reshape((n_comp, s[0], s[1]))
    #eigenfaces = pca.components_
    #print(eigenfaces.shape)
    #plt.figure("eigenfaces")
    #for i in range(0, n_comp):
    #    plt.subplot(1, n_comp, i+1)
    #    plt.imshow(sizedeigenfaces[i].astype(np.float32), cmap='gray')
    #    plt.title(str(i+1))
    #    plt.xticks([])
    #    plt.yticks([])
    #plt.show()
    #cv.waitKey()

    rebuild = pca.inverse_transform(XS[0])  # 使用inverse_transform重建
    #print(rebuild)
    rebuild = rebuild.reshape((s[0], s[1])) # rebuild恢复原始大小
    plt.figure("PCA")
    plt.subplot(2, cnt, n_comp)
    plt.imshow(mean.astype(np.float32), cmap='gray')
    plt.title("mean")
    plt.xticks([])
    plt.yticks([])
    plt.subplot(2, cnt, n_comp+cnt)
    plt.imshow(rebuild.astype(np.float32), cmap='gray')
    plt.title("rebuild"+str(n_comp))
    plt.xticks([])
    plt.yticks([])

plt.show()
cv.waitKey()