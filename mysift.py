import cv2 as cv
import numpy as np

def getSIFTpoints(img_1, img_2, k):
    '''
    Uses the SIFT algorithm in OpenCV to return a list of matched feature
    points for each image

    img_1:  first input image
    img_2:  second input image
    k:      returns first 'k' key points
    '''

    # create SIFT object
    sift = cv.xfeatures2d.SIFT_create()

    # convert images to grayscale
    gray_1 = cv.cvtColor(img_1, cv.COLOR_BGR2GRAY)
    gray_2 = cv.cvtColor(img_2, cv.COLOR_BGR2GRAY)

    # detect SIFT features in both images
    keypoints_1, descriptors_1 = sift.detectAndCompute(gray_1, None)
    keypoints_2, descriptors_2 = sift.detectAndCompute(gray_2, None)

    # create feature matcher
    bf = cv.BFMatcher(cv.NORM_L1, crossCheck = True)

    # match descriptors of both images
    matches = bf.match(descriptors_1,descriptors_2)

    # sort matches by distance
    matches = sorted(matches, key = lambda x:x.distance)
    set1 = np.zeros((k, 2))
    set2 = np.zeros((k, 2))
    for i in range(k):
        set1[i] = keypoints_1[matches[i].queryIdx].pt
        set2[i] = keypoints_2[matches[i].trainIdx].pt
    return set1, set2
