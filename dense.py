'''
This program implements Quasi Dense Matching (Greedy) algorithm from scratch
'''
# external modules
# import time
import cv2 as cv
import numpy as np
from heapq import heappop, heappush

# internal modules
import mysift

# ------------------------------- Classes ------------------------------------ #

class imgGrid():
    '''
    Generates a grid of size of img to keep track of matched pixels.
    Used to check if a pixel has already been paired or not.
    '''
    def __init__(self, img):
        '''
        Initialiser for class imgGrid
        img:    image object (numpy array), the same size of which a boolean
                grid will be made
        '''
        self.rows, self.cols = img.shape[:2]
        self.grid = np.full((self.rows, self.cols), False)

    def __str__(self):
        '''
        Returns string represntation of instance of class imgGrid
        '''
        ret_str = '\nGrid structure for image: \n'
        ret_str += 'Matched pixels: ' + str(np.sum(self.grid)) + '\n'
        return ret_str

    def setMatched(self, point):
        '''
        set the co-ordinates at 'point' to True

        point:  pair of x/y co-ordinates (list, tuple)
        '''
        self.grid[point[0], point[1]] = True

    def isMatched(self, point):
        '''
        return True if pixel at co-ordinates 'point' aleady matched,
        False otherwise

        point:  pair of x/y co-ordinates (list, tuple)
        '''
        return self.grid[point[0], point[1]]

    def reset(self, point):
        '''
        set the co-ordinates at 'point' to False

        point:  pair of x/y co-ordinates (list, tuple)
        '''
        self.grid[point[0], point[1]] = False


class keyPair():
    def __init__(self, kp1, kp2, image_1, image_2, window_size, entry_count):
        '''
        Initialiser for class keyPair

        kp1         :co-ordinates of matched keypoint from 1st iamge
        kp2         :co-ordinates of matched keypoints from 2nd image
        image_1     :first image object (numpy array)
        image_2     :second image object (numpy array)
        window_size :for calculating zncc
        entry_count :unique identifier for each 'keyPair' instance
        '''

        # take inputs as type int
        self.u1, self.v1, self.u2, self.v2 = int(kp1[0]), int(kp1[1]), int(kp2[0]), int(kp2[1])
        #--------------------------------#
        assert self.u1 <= image_1.shape[0]
        assert self.u2 <= image_1.shape[0]
        assert self.v1 <= image_1.shape[1]
        assert self.v2 <= image_1.shape[1]
        #--------------------------------#
        self.zncc = zncc(image_1, image_2, self.u1, self.v1, self.u2, self.v2, window_size)
        # self.priority_score = 1.0 - self.zncc
        self.window_size = window_size
        self.entry_count = entry_count

    def __lt__(self, other):
        '''
        Sorting rule for instances of class 'keyPair'
        '''
        # ~ smallest element in heap will have highest zncc
        # return self.priority_score < other.priority_score
        return self.zncc > other.zncc

    def __eq__(self, other):
        '''
        Tie breaker
        '''
        return (self.entry_count == other.entry_count).all()

    def __str__(self):
        '''
        Returns string representation of instance of class 'keyPair'
        '''
        ret_str = '\n--Keypair object consisting 2 matched image co-ordinates--\n'
        ret_str += 'Entry no.: ' + str(self.entry_count) + '\n'
        ret_str += 'Window size: ' + str(self.window_size) + '\n'
        ret_str += 'keypoint 1: (' + str(self.u1) + ',  ' + str(self.v1) + ')\n'
        ret_str += 'keypoint 2: (' + str(self.u2) + ',  ' + str(self.v2) + ')\n'
        ret_str += 'zncc score: ' + str(self.zncc)
        ret_str += '\n'
        return ret_str

    def getKeypoints(self):
        '''
        Returns 2 numpy arrays containing matched co-ordinates from each image
        '''
        return [np.array([self.u1, self.v1]), np.array([self.u2, self.v2])]

# ---------------------------- Helper Functions ------------------------------ #

def swap(a):
    '''
    swaps the columns of a (n, 2) numpy array
    '''
    a2 = a[:, 0].reshape((a.shape[0], 1))
    a1 = a[:, 1].reshape((a.shape[0], 1))
    return np.hstack((a1, a2))

def getAverage(img, u, v, n):
    '''
    img :input image as a square matrix of numbers
    u   :x co-ordinate of point of interest within img
    v   :y co-ordinate of point of interest within img
    n   :the window size across which zncc score will be calculated and matched

    Returns the arithmetic mean of values in n x n window around (u, v)
    '''
    s = img[u-n:u+n+1, v-n:v+n+1]
    avoid_zero_divide = 1e-6                                                    # expect slight deviation (after 8th decimal) in result
    return np.sum(s) / (s.shape[0]*s.shape[1] + avoid_zero_divide)


def getStandardDeviation(img, u, v, n):
    '''
    img:    input image as a square matrix of numbers
    u:      x co-ordinate of point of interest within img
    v:      y co-ordinate of point of interest within img
    n:      the window size across which zncc score will be calculated and matched

    Returns the standard deviation of values in n x n window around (u, v)
    '''
    avg = getAverage(img, u, v, n)
    s = img[u-n:u+n+1, v-n:v+n+1] - avg
    s = np.sum(s**2)
    return s**0.5 / (2*n + 1)


def zncc(img1, img2, u1, v1, u2, v2, n):
    '''
    img1, img2: input images as square matrices of numbers
    u1, u2:     x co-ordinates of points of interest within img1, img2 in order
    v1, v2:     y co-ordinates of points of interest within img1, img2 in order
    n:          the window size across which zncc score will be calculated and matched

    Returns the 'Zero-mean Normalised Cross Co-relation' between (u1, v1) and (u2, v2)
    around a window size of n x n
    '''
    # take inputs as type int
    u1, u2, v1, v2 = int(u1), int(u2), int(v1), int(v2)

    # get required statistics
    avg1 = getAverage(img1, u1, v1, n)
    avg2 = getAverage(img2, u2, v2, n)
    stdDeviation1 = getStandardDeviation(img1, u1, v1, n)
    stdDeviation2 = getStandardDeviation(img2, u2, v2, n)

    # compute zncc in n x n window
    s1 = img1[u1-n:u1+n+1, v1-n:v1+n+1] - avg1
    s2 = img2[u2-n:u2+n+1, v2-n:v2+n+1] - avg2
    try:
        s = np.sum(s1*s2)
    except:
        # if dimension mismatch between s1 and s2
        s = 0.0
    avoid_zero_divide = 1e-6                                                    # expect slight deviation (after 8th decimal) in result
    zncc_score = float(s) / ((2 * n + 1) ** 2 * stdDeviation1 * stdDeviation2 + avoid_zero_divide)
    #--------------------------------------------#
    assert zncc_score >= 0.0 and zncc_score <= 1.0
    #--------------------------------------------#
    return zncc_score

# --------------------------- Main Function Call ----------------------------- #


def getDenseMatches(img1, img2, n_keypoints, zncc_thresh):
    '''
    Takes two images and finds best keypoints using SIFT algorith in OpenCV
    Then it implements Quasi Dense Matching (QDM) algorithm on the keypoints to
    gather exponentially higher numbers of keypoints with robust quality check.

    img1        :first input image
    img2        :second input image
    n_keypoints :how many inital keypoints to use to propagate by QDM
    '''

    # extract dimensions of the colour images
    m, n = img1.shape[:2]

    # generate grids for each image
    grid1 = imgGrid(img1)
    grid2 = imgGrid(img2)

    # get keypoints for img1 and img2 (sparse key points) using SIFT
    kps_1, kps_2 = mysift.getSIFTpoints(img1, img2, n_keypoints)
    kps_1 = swap(kps_1)
    kps_2 = swap(kps_2)

    # initialise empty list to build heap structure from sparse features
    seeds = []

    # set unique count for every new keyPair object; increment after each push
    entry = 0
    for i in range(n_keypoints):
        # create new keyPair object for each SIFT match
        some_pair = keyPair(kps_1[i], kps_2[i], img1, img2, 5, entry)
        # push the keyPair object into 'seeds'
        heappush(seeds, some_pair)
        entry += 1

    # initialise final lists
    final_keypairs1, final_keypairs2 = [], []

    # iterate through each keyPair in seeds
    while len(seeds) != 0:

        # pop the keyPair with highest zncc value
        best_seed = heappop(seeds)
        x, x_dash = best_seed.getKeypoints()

        # set co-responding grid values to True
        grid1.setMatched(x)
        grid2.setMatched(x_dash)

        # store the grid values to be returned at the end
        final_keypairs1.append([x[0], x[1]])
        final_keypairs2.append([x_dash[0], x_dash[1]])

        # neightbourhood of 5x5
        # row_min, row_max = max(x[0] - 2, 0), min(x[0] + 2 + 1, m)
        # col_min, col_max = max(x[1] - 2, 0), min(x[1] + 2 + 1, n)

        # find local matches in 5 x 5 neighbourhood
        local = []
        for row1 in range(max(x[0] - 2, 0), min(x[0] + 2 + 1, m)):
            for col1 in range(max(x[1] - 2, 0), min(x[1] + 2 + 1, n)):
                for row2 in range(max(x_dash[0] - 2, 0), min(x_dash[0] + 2 + 1, m)):
                    for col2 in range(max(x_dash[1] - 2, 0), min(x_dash[1] + 2 + 1, m)):

                        # -------------------------------- enforce uniqueness constraints ------------------------------ #
                        # 1) infinity norm < 1.0
                        if np.linalg.norm((np.array([row2, col2]) - np.array([row1, col1])) - (x_dash - x), np.inf) < 1.0:      # take diff of pixel value

                            # 2) zncc threshold
                            if zncc(img1, img2, row1, col1, row2, col2, 5) > zncc_thresh:

                                # 3) check if already matched
                                if not grid1.isMatched((row1, col1)) and not grid2.isMatched((row2, col2)):

                                    # if above conditions satisfied, push the keyPair to 'local'
                                    heappush(local, keyPair([row1, col1], [row2, col2], img1, img2, 1, entry))
                                    entry += 1
                        # ---------------------------------------------------------------------------------------------- #

        # take each keyPair from 'local' and push to 'seeds'
        # additional loop for checking extra conditions / do modifications
        while len(local) != 0:
            best_local = heappop(local)
            x, x_dash = best_local.getKeypoints()
            # always check
            if  not grid1.isMatched(x) and not grid2.isMatched(x_dash):
                grid1.setMatched(x)
                grid2.setMatched(x_dash)
                heappush(seeds, best_local)
        # print('len(seeds) = ', len(seeds))    # uncomment to keep track of greedy search

    #---------------------------------------------#
    assert np.sum(grid1.grid) == np.sum(grid2.grid)
    #---------------------------------------------#

    # convert to numpy arrays
    final_keypairs1 = np.array(final_keypairs1)
    final_keypairs2 = np.array(final_keypairs2)

    # reshape
    final_keypairs1 = final_keypairs1.reshape((final_keypairs1.shape[0], 2))
    final_keypairs2 = final_keypairs2.reshape((final_keypairs2.shape[0], 2))

    return swap(final_keypairs1), swap(final_keypairs2), grid1, grid2


# test cases for sanity check
# tic = time.time()
# A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
# B1 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
# B2 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 7]])
# print(zncc(A, B1, 1, 1, 1, 1, 1))       # 1.0000000000000002
# print(zncc(A, B2, 1, 1, 1, 1, 1))       # 0.97348012378367
# toc = time.time()
# print("{:.6f} s".format(toc - tic))

# ---------------------------------------------------------------------------- #

# babyimg1 = cv.imread('baby-left.png')
# babyimg2 = cv.imread('baby-right.png')
# list1, list2, grid1, grid2 = getDenseMatches(babyimg1, babyimg2, 2, 0.9999)
# print(grid1, '\n\n', grid2)
# Matched pixels: 10794
