from CVLab7.twofinal import two
from CVLab7.twofinal import crop
import cv2 as cv
from matplotlib import pyplot as plt

path = 'multi/Rainier/'
img1 = cv.imread(path + 'Rainier1.png')
img2 = cv.imread(path + 'Rainier2.png')
img3 = cv.imread(path + 'Rainier3.png')
img4 = cv.imread(path + 'Rainier4.png')
img5 = cv.imread(path + 'Rainier5.png')
img6 = cv.imread(path + 'Rainier6.png')
direct1, warpImg1, grad1 = two(img1, img2)
direct2, warpImg2, grad2 = two(img2, img3)
direct3, warpImg3, grad3 = two(img3, img4)
direct4, warpImg4, grad4 = two(img4, img5)
direct5, warpImg5, grad5 = two(img5, img6)
direct11, warpImg11, grad11 = two(warpImg1, warpImg2)
direct22, warpImg22, grad22 = two(warpImg2, warpImg3)
direct33, warpImg33, grad33 = two(warpImg3, warpImg4)
direct44, warpImg44, grad44 = two(warpImg4, warpImg5)
left11 = crop(warpImg11)
warpImg11 = warpImg11[:, 0:left11]
left22 = crop(warpImg22)
warpImg22 = warpImg22[:, 0:left22]
left33 = crop(warpImg33)
warpImg33 = warpImg33[:, 0:left33]
left44 = crop(warpImg44)
warpImg44 = warpImg44[:, 0:left44]
direct111, warpImg111, grad111 = two(warpImg11, warpImg22)
direct222, warpImg222, grad222 = two(warpImg22, warpImg33)
direct333, warpImg333, grad333 = two(warpImg33, warpImg44)
left111 = crop(warpImg111)
warpImg111 = warpImg111[:, 0:left111]
left222 = crop(warpImg222)
warpImg222 = warpImg222[:, 0:left222]
left333 = crop(warpImg333)
warpImg333 = warpImg333[:, 0:left333]
direct1111, warpImg1111, grad1111 = two(warpImg111, warpImg222)
direct2222, warpImg2222, grad2222 = two(warpImg222, warpImg333)
left1111 = crop(warpImg1111)
warpImg1111 = warpImg1111[:, 0:left1111]
left2222 = crop(warpImg2222)
warpImg2222 = warpImg2222[:, 0:left2222]
direct, warpImg, grad = two(warpImg1111, warpImg2222)
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

left = crop(warpImg)
cropImg = warpImg[:, 0:left]
cv.imshow('crop', cropImg)
cv.waitKey()