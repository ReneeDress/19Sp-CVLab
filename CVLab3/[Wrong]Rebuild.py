from sklearn.decomposition import PCA
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

img = cv.imread('att_faces/s24/10.pgm', 0)
# 原始数据已经经过处理归一化 因此不需要再次归一化
plt.figure("PCA")
plt.subplot(1, 5, 1)
plt.imshow(img.astype(np.float32), cmap='gray')
plt.title("OriginImg")
plt.xticks([])
plt.yticks([])
for i in range(10,50,10):
    pca = PCA(i)   # 设置为降到i维
    pca.fit(img)   # 对原图进行PCA降维
    print(img)
    transImg = pca.fit_transform(img)   # 降维后的数据
    rebuild = np.rint(pca.inverse_transform(transImg))  # 重建
    plt.subplot(1, 5, i/10+1)
    plt.imshow(rebuild.astype(np.float32), cmap='gray')
    plt.title(str(i))
    plt.xticks([])
    plt.yticks([])

plt.show()
cv.waitKey()