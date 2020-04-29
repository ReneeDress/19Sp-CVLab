import numpy as np
import cv2 as cv
import os
import sys


def read_person():
    # 创建一个图片数组
    img = []
    # 创建一个标签数组
    lab = []
    for personNum in range(1, 41):
        i = 0
        path = './att_faces/s' + str(personNum)
        # 列出目录中的所有文件，逐个从文本文件中读取点
        for filePath in sorted(os.listdir(path)):
            fileExt = os.path.splitext(filePath)[1]
            if fileExt in ['.pgm']:
                # 读取图片
                imagePath = os.path.join(path, filePath)
                im = cv.imread(imagePath, 0)
                i += 1
                if im is None:
                    print('image:{} not read properly'.format(imagePath))
                else:
                    # 将读取的图片添加到图片数组
                    if i != 2:
                        img.append(im)
                        lab.append(personNum)
        print('Person ' + str(personNum) + ' ' + str(len(img)) + ' files read.')
    # 当没有图片时退出
    if len(img) == 0:
        print('No images found')
        sys.exit(0)
    # 返回一个tuple类型
    return img, lab


if __name__ == "__main__":
    retval = cv.face.EigenFaceRecognizer_create()
    # 读取并训练
    data = read_person()
    retval.train(np.asarray(data[0]), np.asarray(data[1]))
    # 创建一个测试图片数组
    ten = []
    # 读取测试用图片
    for personNum in range(1, 41):
        test = cv.imread('./att_faces/s' + str(personNum) + '/10.pgm', 0)
        ten.append(test)
        if personNum == 40:
            print('All test files read.')
    # 测试并输出结果与置信度
    print('Here is the RESULT:')
    for testPersonNum in range(0, 40):
        result = retval.predict(np.asarray(ten[testPersonNum]))
        print('Test Image Data is from Person '+str(testPersonNum + 1))
        print('The Recognition Result is Person '+str(result[0])+', with confidence '+str(round(result[1],4)))