# 导入必要的软件包
import cv2 as cv

# 视频文件输入初始化
filename = "./station.avi"
video = cv.VideoCapture(filename)
fps = video.get(cv.CAP_PROP_FPS)
print('该视频的帧速率为：',fps)
width = int(video.get(cv.CAP_PROP_FRAME_WIDTH))
height = int(video.get(cv.CAP_PROP_FRAME_HEIGHT))
size =(int(video.get(cv.CAP_PROP_FRAME_WIDTH)), int(video.get(cv.CAP_PROP_FRAME_HEIGHT)))
doublesize =(int(video.get(cv.CAP_PROP_FRAME_WIDTH)) * 2, int(video.get(cv.CAP_PROP_FRAME_HEIGHT)))
print('该视频每一帧的大小为：',size)

# 视频文件输出参数设置
out_fps = fps  # 输出文件的帧率
fourcc = cv.VideoWriter_fourcc('M', 'P', '4', '2')
out1 = cv.VideoWriter('station1.avi', fourcc, out_fps, doublesize)
out2 = cv.VideoWriter('station2.avi', fourcc, out_fps, doublesize)

# 初始化当前帧的前两帧
lastFrame1 = None
lastFrame2 = None

# 遍历视频的每一帧
while video.isOpened():
    print('Reading.')

    # 读取下一帧
    (ret, frame) = video.read()

    # 如果不能抓取到一帧，说明我们到了视频的结尾
    if not ret:
        break

    # 调整该帧的大小
    frame = cv.resize(frame, size, interpolation=cv.INTER_CUBIC)

    # 如果第一二帧是None，进行填充，计算第一二帧的不同
    if lastFrame2 is None:
        if lastFrame1 is None:
            lastFrame1 = frame
        else:
            lastFrame2 = frame
            global frameDelta1
            frameDelta1 = cv.absdiff(lastFrame1, lastFrame2)  # 帧差一
        continue

    # 计算当前帧和前帧的不同,计算三帧差分
    frameDelta2 = cv.absdiff(lastFrame2, frame)  # 帧差二
    thresh = cv.bitwise_and(frameDelta1, frameDelta2)  # 图像与运算
    thresh2 = thresh.copy()

    # 当前帧设为下一帧的前帧,前帧设为下一帧的前前帧,帧差二设为帧差一
    lastFrame1 = lastFrame2
    lastFrame2 = frame.copy()
    frameDelta1 = frameDelta2

    # 结果转为灰度图
    thresh = cv.cvtColor(thresh, cv.COLOR_BGR2GRAY)

    # 图像二值化
    thresh = cv.threshold(thresh, 15, 255, cv.THRESH_BINARY)[1]

    # # 去除图像噪声,先腐蚀再膨胀(形态学开运算)
    # thresh = cv.dilate(thresh, None, iterations=3)
    # thresh = cv.erode(thresh, None, iterations=1)

    # # 阀值图像上的轮廓位置
    # binary, cnts, hierarchy = cv.findContours(thresh.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    #
    # # 遍历轮廓
    # for c in cnts:
    #     # 忽略小轮廓，排除误差
    #     if cv.contourArea(c) < 300:
    #         continue
    #
    #     # 计算轮廓的边界框，在当前帧中画出该框
    #     (x, y, w, h) = cv.boundingRect(c)
    #     cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # 显示当前帧
    # cv.imshow("frame", frame)
    thresh = cv.cvtColor(thresh, cv.COLOR_GRAY2BGR)
    result = cv.hconcat([frame, thresh])
    result2 = cv.hconcat([frame, thresh2])
    cv.imshow("thresh", thresh)
    cv.imshow("threst2", thresh2)


    # 保存视频
    out1.write(result)
    out2.write(result2)

# 清理资源并关闭打开的窗口
out1.release()
out2.release()
video.release()
cv.destroyAllWindows()