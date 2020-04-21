import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt


def convolve_2d(img, kernel):
    '''Given a kernel of arbitrary m x n dimensions, with both m and n being
    odd, compute the convolution of the given image with the given
    kernel, such that the output is of the same dimensions as the image and that
    you assume the pixels out of the bounds of the image to be zero. Note that
    you need to apply the kernel to each channel separately, if the given image
    is an RGB image.

    Inputs:
        img:    Either an RGB image (height x width x 3) or a grayscale image
                (height x width) as a numpy array.
        kernel: A 2D numpy array (m x n), with m and n both odd (but may not be
                equal).

    Output:
        Return an image of the same dimensions as the input image (same width,
        height and the number of color channels)
    '''
    # TODO-BLOCK-BEGIN
    k_height = kernel.shape[0]
    k_width = kernel.shape[1]

    conv_height = img.shape[0] - k_height + 1
    conv_width = img.shape[1] - k_width + 1

    conv = np.zeros((conv_height, conv_width))

    for i in range(0, conv_height):
        for j in range(0, conv_width):
            conv[i][j] = wise_element_sum(img[i:i + k_height, j:j + k_width], kernel)
    return conv
    # TODO-BLOCK-END


def wise_element_sum(img, kernel):
    res = (img * kernel).sum()
    if (res < 0):
        res = 0
    elif res > 255:
        res = 255
    return res


def gaussian_blur_kernel_2d(sigma, height, width):
    '''Return a Gaussian blur kernel of the given dimensions and with the given
    sigma. Note that width and height are different.

    Input:
        sigma:  The parameter that controls the radius of the Gaussian blur.
                Note that, in our case, it is a circular Gaussian (symmetric
                across height and width).
        width:  The width of the kernel.
        height: The height of the kernel.

    Output:
        Return a kernel of dimensions height x width such that convolving it
        with an image results in a Gaussian-blurred image.
    '''
    # TODO-BLOCK-BEGIN
    kernel = np.zeros([height, width])
    # center is the origin
    center_h = height/2
    center_w = width/2

    pi = 3.1415926

    if sigma == 0:
        sigma = ((height-1) * 0.5 - 1) * 0.3 + 0.8

    for i in range(0, height):
        for j in range(0, width):
            x = i - center_h
            y = j - center_w
            kernel[i, j] = (np.exp(- (x**2 + y**2) / (2 * (sigma**2)))) / (2 * pi * (sigma**2))
    return kernel
    # TODO-BLOCK-END


def low_pass(img, sigma, size):
    '''Filter the image as if its filtered with a low pass filter of the given
    sigma and a square kernel of the given size. A low pass filter supresses
    the higher frequency components (finer details) of the image.

    Output:
        Return an image of the same dimensions as the input image (same width,
        height and the number of color channels)
    '''
    # TODO-BLOCK-BEGIN
    low_b = img[:, :, 0]
    low_g = img[:, :, 1]
    low_r = img[:, :, 2]

    kernel = gaussian_blur_kernel_2d(sigma, size, size)

    low_b = convolve_2d(low_b, kernel)
    low_g = convolve_2d(low_g, kernel)
    low_r = convolve_2d(low_r, kernel)

    lowimg = np.dstack([low_b, low_g, low_r])
    return lowimg
    # TODO-BLOCK-END


def high_pass(img, sigma, size):
    '''Filter the image as if its filtered with a high pass filter of the given
    sigma and a square kernel of the given size. A high pass filter suppresses
    the lower frequency components (coarse details) of the image.

    Output:
        Return an image of the same dimensions as the input image (same width,
        height and the number of color channels)
    '''
    # TODO-BLOCK-BEGIN
    lowimg = low_pass(img, sigma, size)
    dim = (lowimg.shape[1], lowimg.shape[0])
    resized = cv.resize(img, dim)

    highimg = resized - low_pass(img, sigma, size)
    return highimg
    # TODO-BLOCK-END


#high = cv.imread("cat.jpg")
#low = cv.imread("dog.jpg")
high = cv.imread("einstein.jpg")
low = cv.imread("marilyn.jpg")
high = high/255
low = low/255

# cat & dog size = 21
lowpass = low_pass(low, 0, 21)
highpass = high_pass(high, 0, 21)

plt.figure("Hybrid Image")
plt.subplot(1,3,1)
plt.imshow(cv.cvtColor(lowpass.astype(np.float32), cv.COLOR_BGR2RGB))
plt.xticks([])
plt.yticks([])
plt.subplot(1,3,2)
plt.imshow(cv.cvtColor((highpass+0.5).astype(np.float32), cv.COLOR_BGR2RGB))
plt.xticks([])
plt.yticks([])

hybrid = lowpass * 1 + highpass * 1
plt.subplot(1,3,3)
plt.imshow(cv.cvtColor(hybrid.astype(np.float32), cv.COLOR_BGR2RGB))
plt.xticks([])
plt.yticks([])
plt.show()

cv.waitKey(0)