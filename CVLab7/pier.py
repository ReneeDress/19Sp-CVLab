from CVLab7.twofinal import two
from CVLab7.twofinal import crop
import cv2 as cv
from matplotlib import pyplot as plt

path = 'multi/pier/'
img1 = cv.imread(path + '1.JPG')
img2 = cv.imread(path + '2.JPG')
img3 = cv.imread(path + '3.JPG')
direct1, warpImg1, grad1 = two(img1, img2)
direct2, warpImg2, grad2 = two(img2, img3)
direct, warpImg, grad = two(warpImg1, warpImg2)
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

cropImg = crop(warpImg)
cv.imshow('crop', cropImg)
cv.waitKey()
cv.imwrite(path + 'pier.png', cropImg)