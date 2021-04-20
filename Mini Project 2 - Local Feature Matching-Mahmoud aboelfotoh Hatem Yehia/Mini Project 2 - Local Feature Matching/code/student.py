import numpy as np
import cv2
import scipy
from scipy import ndimage


def get_interest_points(image, feature_width):
    """
    Returns interest points for the input image

    (Please note that we recommend implementing this function last and using cheat_interest_points()
    to test your implementation of get_features() and match_features())

    Implement the Harris corner detector (See Szeliski 4.1.1) to start with.
    You do not need to worry about scale invariance or keypoint orientation estimation
    for your Harris corner detector.
    You can create additional interest point detector functions (e.g. MSER)
    for extra credit.

    If you're finding spurious (false/fake) interest point detections near the boundaries,
    it is safe to simply suppress the gradients / corners near the edges of
    the image.

    Useful functions: A working solution does not require the use of all of these
    functions, but depending on your implementation, you may find some useful. Please
    reference the documentation for each function/library and feel free to come to hours
    or post on Piazza with any questions

        - skimage.feature.peak_local_max (experiment with different min_distance values to get good results)
        - skimage.measure.regionprops


    :params:
    :image: a grayscale or color image (your choice depending on your implementation)
    :feature_width:

    :returns:
    :xs: an np array of the x coordinates of the interest points in the image
    :ys: an np array of the y coordinates of the interest points in the image

    :optional returns (may be useful for extra credit portions):
    :confidences: an np array indicating the confidence (strength) of each interest point
    :scale: an np array indicating the scale of each interest point
    :orientation: an np array indicating the orientation of each interest point

    """

    # TODO: Your implementation here! See block comments and the project webpage for instructions

    # These are placeholders - replace with the coordinates of your interest points!
    # xs = np.zeros(1)
    # ys = np.zeros(1)
    # TODO: Your implementation here! See block comments and the project webpage for instructions
    xs = np.zeros(1)
    ys = np.zeros(1)
    # These are placeholders - replace with the coordinates of your interest points!
    kernel_X = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    kernel_Y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    Ix = scipy.signal.convolve2d(image, kernel_X)
    Iy = scipy.signal.convolve2d(image, kernel_Y)

    IXX = ndimage.gaussian_filter(Ix ** 2, sigma=1)
    IXY = ndimage.gaussian_filter(Iy * Ix, sigma=1)
    IYY = ndimage.gaussian_filter(Iy ** 2, sigma=1)

    k = .04  # adjust it for different values
    harris = IXX * IYY - IXY ** 2 - k * (IXX * IYY) ** 2
    height, width = image.shape
    window = 3  # adjust for diff results
    offset = window // 2
    print(xs, ys)
    for y in range(offset, height - offset):
        for x in range(offset, width - offset):
            sumXX = np.sum(IXX[y - offset:y + 1 + offset, x - offset:x + 1 + offset])
            sumYY = np.sum(IYY[y - offset:y + 1 + offset, x - offset:x + 1 + offset])
            sumXY = np.sum(IXY[y - offset:y + 1 + offset, x - offset:x + 1 + offset])
            r = sumXX * sumYY - sumXY ** 2 - k * (sumXX + sumYY) ** 2
            # print(r , x ,y )

            if r > 0:
                xs = np.append(xs , x)
                ys = np.append(ys , y)
                    # xs = xs.append(x)
                # ys = ys.append(y)
                # print(xs)
                # print(ys)
    print(xs)
    print(ys)

    # return xs, ys
    return xs, ys


def get_features(image, x, y, feature_width):
    """
    Returns feature descriptors for a given set of interest points.

    To start with, you might want to simply use normalized patches as your
    local feature. This is very simple to code and works OK. However, to get
    full credit you will need to implement the more effective SIFT-like descriptor
    (See Szeliski 4.1.2 or the original publications at
    http://www.cs.ubc.ca/~lowe/keypoints/)

    Your implementation does not need to exactly match the SIFT reference.
    Here are the key properties your (baseline) descriptor should have:
    (1) a 4x4 grid of cells, each feature_width / 4 pixels square.
    (2) each cell should have a histogram of the local distribution of
        gradients in 8 orientations. Appending these histograms together will
        give you 4x4 x 8 = 128 dimensions.
    (3) Each feature should be normalized to unit length

    You do not need to perform the interpolation in which each gradient
    measurement contributes to multiple orientation bins in multiple cells
    As described in Szeliski, a single gradient measurement creates a
    weighted contribution to the 4 nearest cells and the 2 nearest
    orientation bins within each cell, for 8 total contributions. This type
    of interpolation probably will help, though.

    You do not have to explicitly compute the gradient orientation at each
    pixel (although you are free to do so). You can instead filter with
    oriented filters (e.g. a filter that responds to edges with a specific
    orientation). All of your SIFT-like feature can be constructed entirely
    from filtering fairly quickly in this way.

    You do not need to do the normalize -> threshold -> normalize again
    operation as detailed in Szeliski and the SIFT paper. It can help, though.

    Another simple trick which can help is to raise each element of the final
    feature vector to some power that is less than one.

    Useful functions: A working solution does not require the use of all of these
    functions, but depending on your implementation, you may find some useful. Please
    reference the documentation for each function/library and feel free to come to hours
    or post on Piazza with any questions

        - skimage.filters (library)


    :params:
    :image: a grayscale or color image (your choice depending on your implementation)
    :x: np array of x coordinates of interest points
    :y: np array of y coordinates of interest points
    :feature_width: in pixels, is the local feature width. You can assume
                    that feature_width will be a multiple of 4 (i.e. every cell of your
                    local SIFT-like feature will have an integer width and height).
    If you want to detect and describe features at multiple scales or
    particular orientations you can add input arguments.

    :returns:
    :features: np array of computed features. It should be of size
            [len(x) * feature dimensionality] (for standard SIFT feature
            dimensionality is 128)

    """

    # TODO: Your implementation here! See block comments and the project webpage for instructions

    # This is a placeholder - replace this with your features!
    # features = np.zeros((1, 128))

    x = np.round(x , decimals=0)
    y =np.round(y , decimals=0)
    angles = [i for i in range(-180, 180, 45)]
    features = np.zeros((len(x), 128))

    # print(image.shape[0])
    for i in range(1, len(x)):

        # Patch 16*16
        # x1 = int(max(1, x[i] - ((feature_width / 2) + 1)))
        # x2 =int(min(image.shape[0], x[i] + ((feature_width / 2) + 1)))
        # y1 = int(max(1, y[i] - ((feature_width / 2) + 1)))
        # y2 = int(min( image.shape[0] , y[i] + ((feature_width / 2) + 1)))
        x1 = int(max(1, x[i] - (feature_width / 2)))
        x2 = int(min(image.shape[0], x[i] + (feature_width / 2)))
        y1 = int(max(1, y[i] - (feature_width / 2)))
        y2 = int(min(image.shape[0], y[i] + (feature_width / 2)))
        # print(int(x1) )
        # print(x2)
        # print(y1)
        # print(y2)
        patch = image[x1: x2 , y1:y2 ]
        # patch = image[i : i+8 , i : i+8]
        patch_padded = np.pad(patch, (
        (int(np.floor((feature_width - patch.shape[1]) / 2)), int(np.ceil((feature_width - patch.shape[1]) / 2))),
        (int(np.floor((feature_width - patch.shape[1]) / 2)), int(np.ceil((feature_width - patch.shape[1]) / 2)))),
                              'constant')

        # Orintations in the patch
        gx = cv2.Sobel(patch_padded, cv2.CV_64F, 1, 0)
        gy = cv2.Sobel(patch_padded, cv2.CV_64F, 0, 1)
        mag, ang = cv2.cartToPolar(gx, gy)

        # Descriptors
        width = int (feature_width / 4)
        descriptor = np.zeros(( 128 ,1))
        j = 0
        # print(width)
        # 1- 4*4 cell grid
        for r in range(0, 4):
            for c in range(0, 4):
                cellMag = np.reshape(mag[r * width + 1: r * width + width, c * width + 1:c * width + width], -1);
                cellDir = np.reshape(ang[r * width + 1:r * width + width, c * width + 1:c * width + width], -1);

                # 2- Histogram of the local distribution of gradients in 8 orientations.
                inds = np.digitize(cellMag, angles)
                inds = inds.flatten()
                orientations = np.zeros((8, 1))
                # print(cellMag)
                for k in range(len(inds)):
                    orientations[inds[k]] = orientations[inds[k]] + cellMag[k]

                # print(j)
                descriptor[(j * 8) : (j * 8) + 8] = orientations

                j = j + 1

        # 3- Normalization
        norm = np.linalg.norm(descriptor)
        # print(descriptor.flatten())
        features[i] = descriptor.flatten() / norm

    # indx = np.where(features > .2)
    # features = features[indx]
    # print(features)
    # features [features > .2] = 1
    # print(features)

    return features


def match_features(im1_features, im2_features):
    """
    Implements the Nearest Neighbor Distance Ratio Test to assign matches between interest points
    in two images.

    Please implement the "Nearest Neighbor Distance Ratio (NNDR) Test" ,
    Equation 4.18 in Section 4.1.3 of Szeliski.

    For extra credit you can implement spatial verification of matches.

    Please assign a confidence, else the evaluation function will not work. Remember that
    the NNDR test will return a number close to 1 for feature points with similar distances.
    Think about how confidence relates to NNDR.

    This function does not need to be symmetric (e.g., it can produce
    different numbers of matches depending on the order of the arguments).

    A match is between a feature in im1_features and a feature in im2_features. We can
    represent this match as a the index of the feature in im1_features and the index
    of the feature in im2_features

    :params:
    :im1_features: an np array of features returned from get_features() for interest points in image1
    :im2_features: an np array of features returned from get_features() for interest points in image2

    :returns:
    :matches: an np array of dimension k x 2 where k is the number of matches. The first
            column is an index into im1_features and the second column is an index into im2_features
    :confidences: an np array with a real valued confidence for each match
    """

    # TODO: Your implementation here! See block comments and the project webpage for instructions

    # These are placeholders - replace with your matches and confidences!
    num_freatures = min(len(im1_features), len(im2_features))
    matches = np.zeros((num_freatures, 2))
    confidences = np.ones(num_freatures)
    threshold = .8

    # print(num_freatures)
    # print(len(im2_features))
    # print(im1_features)
    for i in range (1,num_freatures):
        distances = np.zeros(( num_freatures ,1))

        for j in range (1,num_freatures):
            distances[j][0] = np.linalg.norm(im1_features[i] - im2_features[j])

        # distances = distances[np.logical_not(np.isnan(distances))]
        distances = distances.flatten()
        distances = np.nan_to_num(distances)
        distances_sorted = np.sort(distances)
        distances_index = distances.argsort()
        # print(i)

        # print(distances)
        # print(distances_index)

        NNDR = distances_sorted[-2]/distances_sorted[-1]
        print(NNDR)
        # print(distances_sorted)
        if NNDR < .95 :
            matches[i] = [ i , distances_index[-1] ]
            confidences[i] = 1/NNDR
    # print(confidences)
    indx = np.where(confidences > 0)
    # # print(indx)
    confidences = confidences[indx]
    confidences = np.sort(confidences)
    confidences_index = confidences.argsort()
    # #print(confidences_index)
    matches = matches[confidences_index]
    #print(confidences)
    #print(matches)
    return matches, confidences
