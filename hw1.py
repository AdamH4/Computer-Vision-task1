"""
ITS8030: Homework 1

Please implement all functions below.

For submission create a project called its8030-2021-hw1 and put the solution in there.

Please note that NumPy arrays and PyTorch tensors share memory represantation, so when converting a
torch.Tensor type to numpy.ndarray, the underlying memory representation is not changed.

There is currently no existing way to support both at the same time. There is an open issue on
PyTorch project on the matter: https://github.com/pytorch/pytorch/issues/22402

There is also a deeper problem in Python with types. The type system is patchy and generics
has not been solved properly. The efforts to support some kind of generics for Numpy are
reflected here: https://github.com/numpy/numpy/issues/7370 and here: https://docs.google.com/document/d/1vpMse4c6DrWH5rq2tQSx3qwP_m_0lyn-Ij4WHqQqRHY
but there is currently no working solution. For Dicts and Lists there is support for generics in the
typing module, but not for NumPy arrays.
"""
import cv2
import numpy as np
import util
import math
import matplotlib.pyplot as plot

cactus_image = cv2.cvtColor(cv2.imread(
    'images/cactus.jpg', cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
festival_image = cv2.cvtColor(cv2.imread(
    'images/songfestival.jpg', cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
yosemit_image = cv2.cvtColor(cv2.imread(
    'images/yosemite.png', cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
virgintrains_image = cv2.cvtColor(cv2.imread(
    'images/virgintrains.jpg', cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)


def save_img(file_name, image):
    cv2.imwrite(file_name, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))


"""
Task 1: Convolution

Implement the function

convolution(image : np.ndarray, kernel : np.ndarray, kernel_width : int, kernel_height : int, add : bool, in_place:bool) -> np.ndarray

to convolve an image with a kernel of size kernel_height*kernel_width.
Use zero-padding around the borders for simplicity (what other options would there be?).
Here:

    image is a 2D matrix of class double
    kernel is a 2D matrix with dimensions kernel_width and kernel_height
    kernel_width and kernel_height are the width and height of the kernel respectively

(Note: in the general case, they are not equal and may not be always odd, so you have to ensure that they are odd.)

    if add is true, then 128 is added to each pixel for the result to get rid of negatives.
    if in_place is False, then the output image should be a copy of the input image. The default is False,
    i.e. the operations are performed on the input image.

Write a general convolution function that can handle all possible cases as mentioned above.
You can get help from the convolution part of the function mean_blur_image (to be implemented in a lab)
to write this function.
"""


def addPadding(image, zeros_left, zeros_right):
    image = np.append(zeros_left, image, axis=0)
    image = np.append(image, zeros_left, axis=0)
    image = np.append(zeros_right, image, axis=1)
    image = np.append(image, zeros_right, axis=1)
    return image


def convolution2D(image: np.ndarray, kernel: np.ndarray, kernel_width: int, kernel_height: int) -> np.ndarray:
    image_width = image.shape[0]
    image_height = image.shape[1]

    if kernel_width % 2 == 0 or kernel_height % 2 == 0:
        raise "Kernel has to have odd number of columns and rows"

    kernel_half_width = np.short(np.floor(kernel_width/2))
    kernel_half_height = np.short(np.floor(kernel_height/2))

    dst_image = np.zeros_like(image)
    normalized_kernel = util.normalize_kernel(kernel)
    src_image = image

    zeros_lr = np.zeros((kernel_half_width, image_height))
    zeros_tb = np.zeros(
        (image_width + kernel_width - 1, kernel_half_height))

    src_image = addPadding(src_image, zeros_lr, zeros_tb)

    for m in range(0, image_width):
        for n in range(0, image_height):
            sample_image = src_image[m:m + kernel_width, n:n + kernel_height]
            dst_image[m][n] = np.sum(normalized_kernel * sample_image)

    return dst_image


def convolution3D(image: np.ndarray, kernel: np.ndarray, kernel_width: int, kernel_height: int) -> np.ndarray:
    image_width = image.shape[0]
    image_height = image.shape[1]

    if kernel_width % 2 == 0 or kernel_height % 2 == 0:
        raise "Kernel has to have odd number of columns and rows"

    kernel_half_width = np.short(np.floor(kernel_width/2))
    kernel_half_height = np.short(np.floor(kernel_height/2))

    dst_image = np.zeros_like(image)
    normalized_kernel = util.normalize_kernel(kernel)
    src_image = image

    zeros_lr = np.zeros((kernel_half_width, image_height, 3))
    zeros_tb = np.zeros(
        (image_width + kernel_width - 1, kernel_half_height, 3))

    src_image = addPadding(src_image, zeros_lr, zeros_tb)

    for m in range(0, image_width):
        for n in range(0, image_height):
            sample_image = src_image[m:m + kernel_width, n:n + kernel_height]
            dst_image[m][n][0] = np.sum(
                normalized_kernel * sample_image[:, :, 0])
            dst_image[m][n][1] = np.sum(
                normalized_kernel * sample_image[:, :, 1])
            dst_image[m][n][2] = np.sum(
                normalized_kernel * sample_image[:, :, 2])

    return dst_image


def convolution(image: np.ndarray, kernel: np.ndarray, kernel_width: int,
                kernel_height: int, add: bool, in_place: bool = False) -> np.ndarray:
    if image.ndim == 2:
        dst_image = convolution2D(image, kernel, kernel_width, kernel_height)
    elif image.ndim == 3:
        dst_image = convolution3D(image, kernel, kernel_width, kernel_height)
    else:
        raise "Too much dimensions slow down"

    if add == True:
        dst_image += 128

    return dst_image


"""
Task 2: Gaussian blur

Implement the function

gaussian_blur_image(image : np.ndarray, sigma : float, in_place : bool) -> np.ndarray

to Gaussian blur an image. "sigma" is the standard deviation of the Gaussian.
Use the function mean_blur_image as a template, create a 2D Gaussian filter
as the kernel and call the convolution function of Task 1.
Normalize the created kernel using the function normalize_kernel() (to
be implemented in a lab) before convolution. For the Gaussian kernel, use
kernel size = 2*radius + 1 (same as the Mean filter) and radius = int(math.ceil(3 * sigma))
and the proper normalizing constant.

To do: Gaussian blur the image "songfestival.jpg" using this function with a sigma of 4.0,
and save as "task2.png".
"""


def gaussian_blur_image(image: np.ndarray, sigma: float, in_place: bool = False) -> np.ndarray:
    radius = int(math.ceil(3 * sigma))
    kernel_size = 2 * radius + 1
    kernel = util.gkern(kernel_size, sigma)
    return convolution(image, kernel, kernel.shape[0],
                       kernel.shape[1], False, in_place)


blured_image = gaussian_blur_image(festival_image, 4.0)
save_img('results/task2.png', blured_image)

"""

Task 3: Separable Gaussian blur

Implement the function

separable_gaussian_blur_image (image : np.ndarray, sigma : float, in_place : bool) -> np.ndarray

to Gaussian blur an image using separate filters. "sigma" is the standard deviation of the Gaussian.
The separable filter should first Gaussian blur the image horizontally, followed by blurring the
image vertically. Call the convolution function twice, first with the horizontal kernel and then with
the vertical kernel. Use the proper normalizing constant while creating the kernel(s) and then
normalize using the given normalize_kernel() function before convolution. The final image should be
identical to that of gaussian_blur_image.

To do: Gaussian blur the image "songfestival.jpg" using this function with a sigma of 4.0, and save as "task3.png".
"""


def separable_gaussian_blur_image(image: np.ndarray, sigma: float, in_place: bool = False) -> np.ndarray:
    radius = int(math.ceil(3 * sigma))
    kernel_size = 2 * radius + 1
    kernel = util.gkern(kernel_size, sigma)
    kernel_h = kernel[:, [kernel.shape[0] // 2]]
    kernel_v = kernel[[kernel.shape[1] // 2], :]

    pre_image = convolution(
        image, kernel_h, kernel_h.shape[0], kernel_h.shape[1], False, in_place)
    dst_image = convolution(
        pre_image, kernel_v, kernel_v.shape[0], kernel_v.shape[1], False, in_place)

    return dst_image


blured_image_again = separable_gaussian_blur_image(festival_image, 4.0)
save_img('results/task3.png', blured_image_again)

"""
Task 4: Image derivatives

Implement the functions

first_deriv_image_x(image : np.ndarray, sigma : float, in_place : bool = False) -> np.ndarray
first_deriv_image_y(image : np.ndarray, sigma : float, in_place : bool = False) -> np.ndarray and
second_deriv_image(image : np.ndarray, sigma : float, in_place : bool = False) -> np.ndarray

to find the first and second derivatives of an image and then Gaussian blur the derivative
image by calling the gaussian_blur_image function. "sigma" is the standard deviation of the
Gaussian used for blurring. To compute the first derivatives, first compute the x-derivative
of the image (using the horizontal 1*3 kernel: [-1, 0, 1]) followed by Gaussian blurring the
resultant image. Then compute the y-derivative of the original image (using the vertical 3*1
kernel: [-1, 0, 1]) followed by Gaussian blurring the resultant image.
The second derivative should be computed by convolving the original image with the
2-D Laplacian of Gaussian (LoG) kernel: [[0, 1, 0], [1, -4, 1], [0, 1, 0]] and then applying
Gaussian Blur. Note that the kernel values sum to 0 in these cases, so you don't need to
normalize the kernels. Remember to add 128 to the final pixel values in all 3 cases, so you
can see the negative values. Note that the resultant images of the two first derivatives
will be shifted a bit because of the uneven size of the kernels.

To do: Compute the x-derivative, the y-derivative and the second derivative of the image
"cactus.jpg" with a sigma of 1.0 and save the final images as "task4a.png", "task4b.png"
and "task4c.png" respectively.
"""


def first_deriv_image_x(image: np.ndarray, sigma: float, in_place: bool = False) -> np.ndarray:
    kernel = -1 * np.array([[-1, 0, 1]])
    newImage = convolution(
        image, kernel, kernel.shape[0], kernel.shape[1], True, in_place)
    return gaussian_blur_image(newImage, sigma, in_place)


def first_deriv_image_y(image: np.ndarray, sigma: float, in_place: bool = False) -> np.ndarray:
    kernel = np.array([[-1, 0, 1]]).transpose()
    newImage = convolution(
        image, kernel, kernel.shape[0], kernel.shape[1], True, in_place)
    return gaussian_blur_image(newImage, sigma, in_place)


def second_deriv_image(image: np.ndarray, sigma: float, in_place: bool = False) -> np.ndarray:
    kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
    newImage = convolution(
        image, kernel, kernel.shape[0], kernel.shape[1], True, in_place)
    return gaussian_blur_image(newImage, sigma, in_place)


first_deriv_x = first_deriv_image_x(cactus_image, 1.0)
first_deriv_y = first_deriv_image_y(cactus_image, 1.0)
second_deriv = second_deriv_image(cactus_image, 1.0)

save_img('results/task4a.png', first_deriv_x)
save_img('results/task4b.png', first_deriv_y)
save_img('results/task4c.png', second_deriv)

"""
Task 5: Image sharpening

Implement the function
sharpen_image(image : np.ndarray, sigma : float, alpha : float, in_place : bool = False) -> np.ndarray
to sharpen an image by subtracting the Gaussian-smoothed second derivative of an image, multiplied
by the constant "alpha", from the original image. "sigma" is the Gaussian standard deviation. Use
the second_deriv_image implementation and subtract back off the 128 that second derivative added on.

To do: Sharpen "yosemite.png" with a sigma of 1.0 and alpha of 5.0 and save as "task5.png".
"""


def sharpen_image(image: np.ndarray, sigma: float, alpha: float, in_place: bool = False) -> np.ndarray:
    blured_image = second_deriv_image(image, sigma) - 128.
    result = image - (blured_image * alpha)
    return np.clip(result, 0, 255).astype(np.uint8)


sharp_image = sharpen_image(yosemit_image, 1.0, 1.6)
save_img('results/task5.png', sharp_image)


"""
Task 6: Edge Detection

Implement
sobel_image(image : np.ndarray, in_place : bool = False) -> np.ndarray
to compute edge magnitude and orientation information. Convert the image into grayscale.
Use the standard Sobel masks in X and Y directions:
[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]] and [[1, 2, 1], [0, 0, 0], [-1, -2, -1]] respectively to compute
the edges. Note that the kernel values sum to 0 in these cases, so you don't need to normalize the
kernels before convolving. Divide the image gradient values by 8 before computing the magnitude and
orientation in order to avoid spurious edges. sobel_image should then display both the magnitude and
orientation of the edges in the image.

To do: Compute Sobel edge magnitude and orientation on "cactus.jpg" and save as "task6.png".
"""


def sobel_image(image: np.ndarray, in_place: bool = False) -> np.ndarray:
    image = image.astype("float32")
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    x_mask = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    y_mask = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])

    x_image = convolution(
        gray_image, x_mask, x_mask.shape[0], x_mask.shape[1], False, in_place) / 8
    y_image = convolution(
        gray_image, y_mask, y_mask.shape[0], y_mask.shape[1], False, in_place) / 8

    magnitude = np.sqrt(x_image ** 2 + y_image ** 2)
    direction = np.arctan2(x_image, y_image)
    return magnitude, direction


magnitude, direction = sobel_image(cactus_image)
result = magnitude - direction
save_img('results/task6.png', result)

"""
Task 7: Bilinear Interpolation

Implement the function
bilinear_interpolation(image : np.ndarray, x : float, y : float) -> np.ndarray

to compute the linearly interpolated pixel value at the point (x,y) using bilinear interpolation.
Both x and y are real values. Put the red, green, and blue interpolated results in the vector "rgb".

To do: The function rotate_image will be implemented in a lab and it uses bilinear_interpolation
to rotate an image. Rotate the image "yosemite.png" by 20 degrees and save as "task7.png".
"""


"Returns a  vector containing interpolated red green and blue values (a vector of length 3)"


def bilinear_interpolation(image: np.ndarray, x: float, y: float) -> np.ndarray:
    if x < 0 or x > image.shape[1] - 1 or y < 0 or y > image.shape[0] - 1:
        if len(image.shape) > 2:
            return np.zeros(image.shape[image.ndim - 1])
        else:
            return 0

    y1 = math.floor(y)
    y2 = math.ceil(y)

    x1 = math.floor(x)
    x2 = math.ceil(x)

    # create square of posible values
    value_q11 = image[y1][x1]
    value_q12 = image[y1][x2]
    value_q21 = image[y2][x1]
    value_q22 = image[y2][x2]

    dx = x - x1
    dy = y - y1

    R1 = (1 - dx) * value_q11 + dx * value_q21
    R2 = (1 - dx) * value_q12 + dx * value_q22
    P = (1 - dy) * R1 + dy * R2

    return P


def rotate_image(image: np.ndarray, rotation_angle: float, in_place: bool = False) -> np.ndarray:
    return util.rotate_image(bilinear_interpolation, image, rotation_angle, in_place)


rotated_image = rotate_image(yosemit_image, 20.0)
save_img('results/task7.png', rotated_image)
"""
Task 8: Finding edge peaks

Implement the function
find_peaks_image(image : np.ndarray, thres : float, in_place : bool = False) -> np.ndarray
to find the peaks of edge responses perpendicular to the edges. The edge magnitude and orientation
at each pixel are to be computed using the Sobel operators. The original image is again converted
into grayscale in the starter code. A peak response is found by comparing a pixel's edge magnitude
to that of the two samples perpendicular to the edge at a distance of one pixel, which requires the
bilinear_interpolation function
(Hint: You need to create an image of magnitude values at each pixel to send as input to the
interpolation function).
If the pixel's edge magnitude is e and those of the other two are e1 and e2, e must be larger than
"thres" (threshold) and also larger than or equal to e1 and e2 for the pixel to be a peak response.
Assign the peak responses a value of 255 and everything else 0. Compute e1 and e2 as follows:

(please check the separate task8.pdf)

To do: Find the peak responses in "virgintrains.jpg" with thres = 40.0 and save as "task8.png".
What would be a better value for thres?
"""


def find_peaks_image(image: np.ndarray, thres: float, in_place: bool = False) -> np.ndarray:
    magnitude, direction = sobel_image(image)

    image_width = image.shape[0]
    image_height = image.shape[1]

    dst_image = np.zeros_like(image)

    for c in range(0, image_width):
        for r in range(0, image_height):
            angle = direction[c][r]
            if math.isnan(angle):
                angle = 0

            e1x = c + 1 * np.cos(angle)
            e1y = r + 1 * np.sin(angle)
            e2x = c - 1 * np.cos(angle)
            e2y = r - 1 * np.sin(angle)

            e = magnitude[c][r]
            e1 = bilinear_interpolation(magnitude, e1x, e1y)
            e2 = bilinear_interpolation(magnitude, e2x, e2y)

            if e > thres and e > e1 and e > e2:
                dst_image[c][r] = 255.

    return dst_image


peaks_image = find_peaks_image(virgintrains_image, 40.0)
save_img('results/task8.png', peaks_image)

"""
Task 9 (a): K-means color clustering with random seeds (extra task)

Implement the function

random_seed_image(image : np.ndarray, num_clusters : int, in_place : bool = False) -> np.ndarray

to perform K-Means Clustering on a color image with randomly selected initial cluster centers
in the RGB color space. "num_clusters" is the number of clusters into which the pixel values
in the image are to be clustered. Use random.randint(0,255) to initialize each R, G and B value.
to create #num_clusters centers, assign each pixel of the image to its closest cluster center
and then update the cluster centers with the average of the RGB values of the pixels belonging
to that cluster until convergence. Use max iteration # = 100 and L1 distance between pixels,
i.e. dist = |Red1 - Red2| + |Green1 - Green2| + |Blue1 - Blue2|. The algorithm converges when
the sum of the L1 distances between the new cluster centers and the previous cluster centers
is less than epsilon*num_clusters. Choose epsilon = 30 (or anything suitable). Note: Your code
should account for the case when a cluster contains 0 pixels during an iteration. Also, since
this algorithm is random, you will get different resultant images every time you call the function.

To do: Perform random seeds clustering on "flowers.png" with num_clusters = 4 and save as "task9a.png".
"""


def random_seed_image(image: np.ndarray, num_clusters: int, in_place: bool = False) -> np.ndarray:
    "implement the function here"
    raise "not implemented yet!"


"""
Task 9 (b): K-means color clustering with pixel seeds (extra)

Implement the function
pixel_seed_image(image : np.ndarray, num_clusters: int, in_place : bool = False)
to perform K-Means Clustering on a color image with initial cluster centers sampled from the
image itself in the RGB color space. "num_clusters" is the number of clusters into which the
pixel values in the image are to be clustered. Choose a pixel and make its RGB values a seed
if it is sufficiently different (dist(L1) >= 100) from already-selected seeds. Repeat till
you get #num_clusters different seeds. Use max iteration # = 100 and L1 distance between pixels,
 i.e. dist = |Red1 - Red2| + |Green1 - Green2| + |Blue1 - Blue2|. The algorithm converges when
 the sum of the L1 distances between the new cluster centers and the previous cluster centers
is less than epsilon*num_clusters. Choose epsilon = 30.

To do: Perform pixel seeds clustering on "flowers.png" with num_clusters = 5 and save as "task9b.png".
"""


def pixel_seed_image(image: np.ndarray, num_clusters: int, in_place: bool = False) -> np.ndarray:
    "implement the function here"
    raise "not implemented yet!"
