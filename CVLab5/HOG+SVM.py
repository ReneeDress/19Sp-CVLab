import cv2 as cv
# 使用OpenCV的HOG特征进行行人检测
#img = cv.imread("4.jpeg")
img = cv.imread("xiangwang2.JPG")
s = img.shape
walkmanhog = cv.HOGDescriptor()
# 返回为人类识别训练的分类器系数
hogdes = cv.HOGDescriptor_getDefaultPeopleDetector()
# 将生成的分类器系数置入线性SVM分类器
walkmanhog.setSVMDetector(hogdes)
# 识别输入图像中不同大小的目标，返回矩形列表 128*64
result, digit = walkmanhog.detectMultiScale(img)
#print(result.shape)
#print(result)
# 复制原图片并画出矩形框
resultimg = img
for i in range(0, result.shape[0]):
    topleft = (result[i][0], result[i][1])
    bottomright = (result[i][0] + result[i][2], result[i][1] + result[i][3])
    print('No.', i + 1, topleft, bottomright)
    cv.rectangle(img=resultimg, pt1=topleft, pt2=bottomright, color=255, thickness=2)
cv.imshow('Result', resultimg)
cv.waitKey()