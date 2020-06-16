import cv2 as cv

filename = "./school.avi"
video = cv.VideoCapture(filename)
mog = cv.createBackgroundSubtractorMOG2()
fps = video.get(cv.CAP_PROP_FPS)
size =(int(video.get(cv.CAP_PROP_FRAME_WIDTH)), int(video.get(cv.CAP_PROP_FRAME_HEIGHT)))
doublesize =(int(video.get(cv.CAP_PROP_FRAME_WIDTH)) * 2, int(video.get(cv.CAP_PROP_FRAME_HEIGHT)))
out_fps = fps  # 输出文件的帧率
fourcc = cv.VideoWriter_fourcc('M', 'P', '4', '2')
out = cv.VideoWriter('school_bkg.avi', fourcc, out_fps, doublesize)

while video.isOpened():
    print('Reading.')
    # 读取下一帧
    (ret, frame) = video.read()
    # 如果不能抓取到一帧，说明我们到了视频的结尾
    if not ret:
        break
    fgmask = mog.apply(frame)
    fgmask = cv.threshold(fgmask, 250, 255, cv.THRESH_BINARY)[1]
    fgmask = cv.cvtColor(fgmask, cv.COLOR_GRAY2BGR)
    # 显示当前帧
    cv.imshow('frame', fgmask)
    result = cv.hconcat([frame, fgmask])

    # 保存视频
    out.write(result)

out.release()
video.release()
cv.destroyAllWindows()

