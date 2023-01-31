import cv2
import numpy as np
# import matplotlib.pyplot as plt

def swap(a):
    a2 = a[:, 0].reshape((a.shape[0], 1))
    a1 = a[:, 1].reshape((a.shape[0], 1))
    return np.hstack((a1, a2))

def getORBpoints(img_1, img_2):
    # img_1 = cv2.imread("IMG_21743.jpg")
    # img_2 = cv2.imread("IMG_2173.jpg")

    MAX_FEATURES = 200
    GOOD_MATCH_PERCENT = 0.85
    img_1_gray = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)
    img_2_gray = cv2.cvtColor(img_2, cv2.COLOR_BGR2GRAY)
    orb = cv2.ORB_create(MAX_FEATURES)
    keypoints1, descriptors1 = orb.detectAndCompute(img_1_gray, None)
    keypoints2, descriptors2 = orb.detectAndCompute(img_2_gray, None)
    matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
    matches = matcher.match(descriptors1, descriptors2, None)
    matches.sort(key=lambda x: x.distance, reverse=False)
    numGoodMatches = int(len(matches) * GOOD_MATCH_PERCENT)
    matches = matches[:numGoodMatches]
    imMatches = cv2.drawMatches(img_1, keypoints1, img_2, keypoints2, matches, None)
    cv2.imwrite("matches.jpg", imMatches)
    points1 = np.zeros((len(matches), 2), dtype=np.float32)
    points2 = np.zeros((len(matches), 2), dtype=np.float32)
    for i, match in enumerate(matches):
      points1[i, :] = keypoints1[match.queryIdx].pt
      points2[i, :] = keypoints2[match.trainIdx].pt

    return swap(points1), swap(points2)
