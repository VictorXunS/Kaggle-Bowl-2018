import cv2
import numpy as np
import os
from itertools import *
import csv
import matplotlib.pyplot as plt
from scipy import ndimage
from skimage.feature import peak_local_max
from skimage.morphology import watershed


def encode(l):
    encode = [(len(list(group)), name) for name, group in groupby(l)]
    output = ''
    runlength = 0
    if len(encode) == 0:
        return '1 1'
    for i in range(len(encode)):
        if encode[i][1] == 255:
            output += str(runlength + 1) + ' ' + str(encode[i][0]) + ' '
        runlength += encode[i][0]
    return output[:-1]


def encode_mask(mask):
    mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)[1]
    return encode(np.hstack(np.transpose(mask)))


cur_folder = os.path.dirname(os.path.realpath(__file__))
data_folder = os.path.join(cur_folder, "Data")
train_folder = os.path.join(data_folder, "Train")
test_folder = os.path.join(data_folder, "Test")


# # Show training images
# train_imgs_folder = os.listdir(train_folder)
# train_imgs_folder.sort()

# output_list = [['ImageId', 'EncodedPixels']]
# for img_id in train_imgs_folder:

#     # im = cv2.imread(os.path.join(train_folder,
#     #                              img_id, 'images/') + img_id + '.png')
#     # cv2.imshow(img_id, im)
#     # cv2.waitKey(0)

#     # Load ground truth mask image folder
#     mask_folder = os.path.join(train_folder, img_id, 'masks/')
#     mask_list = os.listdir(mask_folder)

#     print img_id
#     # Read mask images
#     for i in range(len(mask_list)):
#         mask = cv2.imread(mask_folder + mask_list[i],
#                           cv2.CV_LOAD_IMAGE_GRAYSCALE)
#         # Runlength encoding mask
#         encoded = encode_mask(mask)
#         output_list.append([img_id, encoded])

#         # cv2.imshow(img_id, mask)
#         # cv2.waitKey(0)

# # Write train output solution file
# with open(os.path.join(cur_folder, 'stage1_submission.csv'), 'wb') as f:
#     wr = csv.writer(f, quoting=csv.QUOTE_ALL)
#     for i in range(len(output_list)):
#         wr.writerow(output_list[i])


def is_grayscale(im):
    return np.array_equal(im[:, :, 0], im[:, :, 1])


output_list = [['ImageId', 'EncodedPixels']]
# Test images
test_imgs_folder = os.listdir(test_folder)
test_imgs_folder.sort()

output_list = [['ImageId', 'EncodedPixels']]
for img_id in test_imgs_folder:
    im = cv2.imread(os.path.join(test_folder,
                                 img_id, 'images/') + img_id + '.png')
    print os.path.join(test_folder,
                                 img_id, 'images/') + img_id + '.png'

    # Detect if image is grayscale
    is_gray = is_grayscale(im)

    if is_gray:
        im_gray = im[:, :, 0]

        # Histogram
        # cv2.imshow('grayscale', im_gray)
        # cv2.waitKey(0)
        # hist = cv2.calcHist(im_gray, [0], None, [256], [0, 256])
        # plt.figure()
        # plt.plot(hist)
        # plt.show(block=True)

        # Threshold gray image
        ret, th1 = cv2.threshold(im_gray, 30, 255, cv2.THRESH_BINARY)
        th2 = cv2.adaptiveThreshold(im_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                    cv2.THRESH_BINARY, 23, 2)
        th3 = cv2.adaptiveThreshold(im_gray, 255,
                                    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY, 23, 2)
        retval, th4 =\
            cv2.threshold(im_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        titles = ['Original Image', 'Global (v = 30)',
                  'Adaptive Mean', 'Adaptive Gaussian', 'Otsu ' + str(retval)]

        # plt.figure()
        # images = [im_gray, th1, th2, th3, th4]
        # for i in xrange(5):
        #     plt.subplot(2, 3, i + 1), plt.imshow(images[i], 'gray')
        #     plt.title(titles[i])
        #     plt.xticks([]), plt.yticks([])
        # plt.show()

        thresh = th4

        # Image morphology
        # noise removal
        kernel = np.ones((2, 2), np.uint8)
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN,
                                   kernel, iterations=2)
        # sure background area
        sure_bg = cv2.dilate(opening, kernel, iterations=2)

        # Finding sure foreground area
        dist_transform = cv2.distanceTransform(opening, cv2.cv.CV_DIST_L2, 5)
        ret, sure_fg = cv2.threshold(dist_transform,
                                     0.7 * dist_transform.max(), 255, 0)
        # Finding unknown region
        sure_fg = np.uint8(sure_fg)
        unknown = cv2.subtract(sure_bg, sure_fg)

        # cv2.imshow("Thresholded", thresh)
        # cv2.waitKey(0)
        # cv2.imshow("Sure bg", sure_bg)
        # cv2.waitKey(0)
        # cv2.imshow("Sure fg", sure_fg)
        # cv2.waitKey(0)
        # cv2.imshow("Unknown", unknown)
        # cv2.waitKey(0)

        # Connex regions

        # # Add one to all labels so that sure background is not 0, but 1
        # markers = markers + 1
        # # Now, mark the region of unknown with zero
        # markers[unknown == 255] = 0

        # # Watershed
        # markers = cv2.watershed(img_gray, markers)
        # img_gray[markers == -1] = [255, 0, 0]

        D = ndimage.distance_transform_edt(thresh)
        localMax = peak_local_max(D, indices=False, min_distance=20,
                                  labels=thresh)

        # perform a connected component analysis on the local peaks,
        # using 8-connectivity, then appy the Watershed algorithm
        markers = ndimage.label(localMax, structure=np.ones((3, 3)))[0]
        labels = watershed(-D, markers, mask=thresh)
        print("[INFO] {} unique segments found".format(len(np.unique(labels)) - 1))

        all_masks = []
        for label in np.unique(labels):
            # if the label is zero, we are examining the 'background'
            # so simply ignore it
            if label == 0:
                continue

            # otherwise, allocate memory for the label region and draw
            # it on the mask
            mask = np.zeros(im_gray.shape, dtype="uint8")
            mask[labels == label] = 255
            all_masks.append(mask)
            # cv2.imshow('gray', im_gray)
            # cv2.waitKey(0)
            # cv2.imshow("Mask", mask)
            # cv2.waitKey(0)
            encoded = encode_mask(mask)
            output_list.append([img_id, encoded])

    else:
        output_list.append([img_id, '1 1'])

# Write train output solution file
with open(os.path.join(cur_folder, 'stage1_submission.csv'), 'wb') as f:
    wr = csv.writer(f, quoting=csv.QUOTE_ALL)
    for i in range(len(output_list)):
        wr.writerow(output_list[i])
