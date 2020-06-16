import cv2 as cv
import numpy as np

cap = cv.VideoCapture('./fakeman.avi')
fourcc = cv.VideoWriter_fourcc(*'MJPG')
fps = cap.get(cv.CAP_PROP_FPS)
print('该视频的帧速率为：',fps)
size =(int(cap.get(cv.CAP_PROP_FRAME_WIDTH)),
       int(cap.get(cv.CAP_PROP_FRAME_HEIGHT)))
print('该视频每一帧的大小为：',size)
out1 = cv.VideoWriter('5_Gaussian_mask_normal.avi', fourcc, fps, size,0)
out2 = cv.VideoWriter('5_Gaussian_final_normal.avi', fourcc, fps, size)

# 背景检测框架
backSub = cv.createBackgroundSubtractorMOG2()
# 设置混合高斯模型数
backSub.setNMixtures(5)
backSub.setShadowValue(0)

while cap.isOpened():
       ret, frame = cap.read()
       # if frame is read correctly ret is True
       if not ret:
              print("Can't receive frame any more. Exiting ...")
              break

       # 每一帧背景检测都会作为背景检测的输入
       fgMask = backSub.apply(frame)
       background = backSub.getBackgroundImage()
       out1.write(fgMask)

       # 寻找检测区域
       # 高斯平滑
       blur = cv.GaussianBlur(fgMask, (5, 5), 0)
       contours, hierarchy = cv.findContours(image=blur, mode=cv.RETR_CCOMP, method=cv.CHAIN_APPROX_SIMPLE)
       image = frame;
       # image =	cv.drawContours(image,contours,-1,(255,0,0))
       for i in range(0, len(contours)):

              # #以下画的是最小外接矩形，可以不是标准的水平放置的矩形框
              # rect = cv.minAreaRect(contours[i])
              # box = np.int0(cv.boxPoints(rect))
              # s = np.square(box[1][0]-box[0][0])+np.square(box[1][1]-box[0][1])\
              #     +np.square(box[2][0]-box[1][0])+np.square(box[2][1]-box[1][1])
              # if s > 500:
              #     cv.line(image,tuple(box[0]),tuple(box[1]),(255,0,0))
              #     cv.line(image,tuple(box[1]),tuple(box[2]),(255,0,0))
              #     cv.line(image,tuple(box[2]),tuple(box[3]),(255,0,0))
              #     cv.line(image,tuple(box[3]),tuple(box[0]),(255,0,0))

              # 以下画的是标准水平放置的矩形框
              x, y, w, h = cv.boundingRect(contours[i])
              s = w * h
              if s > 500:
                     cv.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0))

       out2.write(image)

       if cv.waitKey(1) == ord('q'):
              break

cv.imwrite('BackGround.png',background)
cap.release()
out1.release()
out2.release()
cv.destroyAllWindows()