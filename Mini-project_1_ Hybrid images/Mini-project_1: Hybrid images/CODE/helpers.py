# Project Image Filtering and Hybrid Images Stencil Code
# Based on previous and current work
# by James Hays for CSCI 1430 @ Brown and
# CS 4495/6476 @ Georgia Tech
import numpy as np
from numpy import pi, exp, sqrt
from skimage import io, img_as_ubyte, img_as_float32
from skimage.transform import rescale
from scipy import fftpack
from scipy import signal
import cv2


def my_imfilter(img: np.ndarray, kernal: np.ndarray):
  assert kernal.size // 2 != 0
  if np.size(img.shape) != 2:
    filtered = np.zeros((img.shape[0], img.shape[1], img.shape[2]))
    for r in range(img.shape[2]):
      z = np.lib.pad(img[:, :, r], (int(np.floor(kernal.shape[0] / 2)), int(np.floor(kernal.shape[1] / 2))), 'constant',
                     constant_values=(0, 0))
      for i in range(int(np.floor(kernal.shape[0] / 2)), img.shape[0]):
        for j in range(int(np.floor(kernal.shape[1] / 2)), img.shape[1]):
          filtered[i, j, r] = np.sum(np.multiply(kernal, z[i - int(np.floor(kernal.shape[0] / 2)):i + int(
            np.floor(kernal.shape[0] / 2)) + 1, j - int(np.floor(kernal.shape[1] / 2)):j + int(
            np.floor(kernal.shape[1] / 2)) + 1]))



  else:

    filtered = np.zeros(
      (img.shape[0] + int(np.floor(kernal.shape[0] / 2)), img.shape[1] + int(np.floor(kernal.shape[1] / 2))))
    z = np.lib.pad(img, (int(np.floor(kernal.shape[0] / 2)), int(np.floor(kernal.shape[1] / 2))), 'constant',
                   constant_values=(0, 0))
    for i in range(int(np.floor(kernal.shape[0] / 2)), img.shape[0]):
      for j in range(int(np.floor(kernal.shape[1] / 2)), img.shape[1]):
        filtered[i, j] = np.sum(np.multiply(kernal, z[i - int(np.floor(kernal.shape[0] / 2)):i + int(
          np.floor(kernal.shape[0] / 2)) + 1, j - int(np.floor(kernal.shape[0] / 2)):j + int(
          np.floor(kernal.shape[0] / 2)) + 1]))
    #             [np.nonzero(filtered)]
  return filtered

# This function can be implemented using a double loop but since we
# have numpy we can try it without loops
# check np.meshgrid https://www.geeksforgeeks.org/numpy-meshgrid-function/
def create_gaussian_filter(ksize, sigma):
    assert(ksize%2 == 1)
    center = ksize//2
    kernel = [np.exp(-(i-center)**2/sigma**2)*np.exp(-(j-center)**2/sigma**2) for i in range(ksize) for j in range(ksize)]
    kernel = np.array(kernel)/ksize**2
    kernel = kernel
    kernel.resize(ksize,ksize)
    return kernel

def gen_hybrid_image(image1: np.ndarray, image2: np.ndarray, cutoff_frequency: float):
    assert image1.shape == image2.shape

    # Steps:
    # (1) Remove the high frequencies from image1 by blurring it. The amount of
    #     blur that works best will vary with different image pairs
    # generate a gaussian kernel with mean=0 and sigma = cutoff_frequency,
    # Just a heads up but think how you can generate 2D gaussian kernel from 1D gaussian kernel
    kernel = create_gaussian_filter(ksize = 5, sigma =cutoff_frequency )
    # kernel = cv2.getGaussianKernel(5, cutoff_frequency)
    # kernel = np.outer(kernel, kernel.transpose())

    # Your code here:
    low_frequencies = my_imfilter(image1, kernel)  # Replace with your implementation

    # (2) Remove the low frequencies from image2. The easiest way to do this is to
    #     subtract a blurred version of image2 from the original version of image2.
    #     This will give you an image centered at zero with negative values.
    # Your code here #
    high_frequencies = image2 - my_imfilter(image2, kernel)  # Replace with your implementation

    # (3) Combine the high frequencies and low frequencies
    # Your code here #
    hybrid_image = low_frequencies + high_frequencies  # Replace with your implementation

    # (4) At this point, you need to be aware that values larger than 1.0
    # or less than 0.0 may cause issues in the functions in Python for saving
    # images to disk. These are called in proj1_part2 after the call to
    # gen_hybrid_image().
    np.clip(hybrid_image, -1, 1)
    # One option is to clip (also called clamp) all values below 0.0 to 0.0,
    # and all values larger than 1.0 to 1.0.
    # (5) As a good software development practice you may add some checks (assertions) for the shapes
    # and ranges of your results. This can be performed as test for the code during development or even
    # at production!

    return low_frequencies, high_frequencies, hybrid_image

def gen_hybrid_image_fft(image1: np.ndarray, image2: np.ndarray, cutoff_frequency: float):
    assert image1.shape == image2.shape

  # Steps:
  # (1) Remove the high frequencies from image1 by blurring it. The amount of
  #     blur that works best will vary with different image pairs
  # generate a gaussian kernel with mean=0 and sigma = cutoff_frequency,
  # Just a heads up but think how you can generate 2D gaussian kernel from 1D gaussian kernel
#     kernel = create_gaussian_filter(ksize = 5, sigma =cutoff_frequency )
    kernel = cv2.getGaussianKernel(5 , cutoff_frequency)
    kernel = np.outer(kernel, kernel.transpose())
    kernel_ft = fftpack.fft2(kernel, shape=image1.shape[:2], axes=(0, 1))

  # Your code here:
    img_ft = fftpack.fft2(image1, axes=(0, 1))
    low_frequencies = signal.fftconvolve(image1, kernel[:, :, np.newaxis], mode='same')

    #low_frequencies = my_imfilter(image1 , kernel) # Replace with your implementation

  # (2) Remove the low frequencies from image2. The easiest way to do this is to
  #     subtract a blurred version of image2 from the original version of image2.
  #     This will give you an image centered at zero with negative values.
  # Your code here #
    img2_ft = fftpack.fft2(image2, axes=(0, 1))
    low_frequencies_2 = signal.fftconvolve(image2, kernel[:, :, np.newaxis], mode='same')
    high_frequencies = image2 - low_frequencies_2 # Replace with your implementation

#     high_frequencies = image2 - my_imfilter(image2 , kernel) # Replace with your implementation

  # (3) Combine the high frequencies and low frequencies
  # Your code here #
    hybrid_image = low_frequencies + high_frequencies # Replace with your implementation

  # (4) At this point, you need to be aware that values larger than 1.0
  # or less than 0.0 may cause issues in the functions in Python for saving
  # images to disk. These are called in proj1_part2 after the call to
# gen_hybrid_image().
    np.clip(hybrid_image ,-1,1)
  # One option is to clip (also called clamp) all values below 0.0 to 0.0,
  # and all values larger than 1.0 to 1.0.
  # (5) As a good software development practice you may add some checks (assertions) for the shapes
  # and ranges of your results. This can be performed as test for the code during development or even
  # at production!

    return low_frequencies, high_frequencies, hybrid_image

def vis_hybrid_image(hybrid_image: np.ndarray):
  """
  Visualize a hybrid image by progressively downsampling the image and
  concatenating all of the images together.
  """
  scales = 5
  scale_factor = 0.5
  padding = 5
  original_height = hybrid_image.shape[0]
  num_colors = 1 if hybrid_image.ndim == 2 else 3

  output = np.copy(hybrid_image)
  cur_image = np.copy(hybrid_image)
  for scale in range(2, scales+1):
    # add padding
    output = np.hstack((output, np.ones((original_height, padding, num_colors),
                                        dtype=np.float32)))
    # downsample image
    cur_image = rescale(cur_image, scale_factor, mode='reflect'  , multichannel=True)
    # pad the top to append to the output
    pad = np.ones((original_height-cur_image.shape[0], cur_image.shape[1],
                   num_colors), dtype=np.float32)
    tmp = np.vstack((pad, cur_image))
    output = np.hstack((output, tmp))
  return output

def load_image(path):
  return img_as_float32(io.imread(path))

def save_image(path, im):
  return io.imsave(path, img_as_ubyte(im.copy()))
