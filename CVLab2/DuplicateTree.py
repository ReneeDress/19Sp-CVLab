import cv2 as cv
import numpy as np

# 读取原始图片
img = cv.imread("tree.jpeg")
# 选定右侧树图像范围
obj = img[188:519, 538:818]
# 为选定右侧树图像范围添加一个全白遮罩
mask = 255 * np.ones(obj.shape, obj.dtype)
# T选定右侧树图像将要放置在原图像位置的中心点
center = (320, 352)
# 无缝将选定图像复制进原始图片
normal_clone = cv.seamlessClone(obj, img, mask, center, cv.NORMAL_CLONE)
# 将结果图片保存并显示
cv.imwrite("normal-clone-example.jpg", normal_clone)
cv.imshow("normalclone", normal_clone)

cv.waitKey(0)