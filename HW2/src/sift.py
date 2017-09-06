import numpy as np
import cv2
import math
from scipy.ndimage.filters import gaussian_filter
from scipy.signal import gaussian

def plot_pyramid(pyramid):
    for i in range(len(pyramid)):
        for j in range(len(pyramid[0])):
                cv2.imwrite('pyramid_'+str(i) + '_' + str(j) + '.jpg', pyramid[i][j])

def create_blur_level(img, sigma = math.sqrt(2), k_gauss_factor = math.sqrt(2), blur_levels = 5):
    level = []
    sigma_level = []
    for i in range(0, blur_levels):
        sigma_level.append(sigma)
        img_blurred = gaussian_filter(img, sigma)
        level.append(img_blurred)
        sigma = sigma * k_gauss_factor
    return level, sigma_level

def built_pyramid(img, sigma = math.sqrt(2)/2.0, k_gauss_factor = math.sqrt(2), octaves = 4, blur_levels = 5):
    middle_octave_k_factor = k_gauss_factor
    for i in range(1, blur_levels//2):
        middle_octave_k_factor = middle_octave_k_factor * k_gauss_factor
    pyramid = []
    pyramid_sigma = []
    for i in range(octaves):
        pyramid_level, sigma_level = create_blur_level(img.copy(), sigma, k_gauss_factor, blur_levels)
        pyramid.append(pyramid_level)
        pyramid_sigma.append(sigma_level)
        sigma = sigma * middle_octave_k_factor
        img = cv2.resize(img, (img.shape[1]//2, img.shape[0]//2))
    return pyramid, pyramid_sigma

def difference_of_gaussian(pyramid):
    diff_gauss = []
    for i in range(len(pyramid)):
        level = []
        for j in range(len(pyramid[0]) - 1):
            '''
            print(pyramid[i][j + 1])
            print(pyramid[i][j])
            print(sub)
            print(sub.min(), sub.max())
            print('______________________________________________________________________')
            '''
            sub = np.array(pyramid[i][j + 1]).astype(np.int) - np.array(pyramid[i][j]).astype(np.int)
            level.append(sub)

        diff_gauss.append(level)
    return diff_gauss

def find_key_points(pyramid, threshold_constrat = 1.5, threshold_edges = 30):
    d_rows = [-1, 0, 1, 0, -1, 1, 1, -1]
    d_cols = [0, -1, 0, 1, 1, -1, 1, -1]
    key_points = []
    for i in range(len(pyramid)): #foreach octave
        key_points_octave = []
        rows, cols = pyramid[i][0].shape
        for j in range(1, len(pyramid[0]) - 1): #find extremas between (j-1, j, j+1)
            interest_point_counter = [0, 0, 0, 0, 0, 0]
            for row in range(1, rows - 1):
                for col in range(1, cols - 1):
                    if pyramid[i][j][row, col] > pyramid[i][j][row - 1, col - 1]:
                        ok = pyramid[i][j][row, col] > pyramid[i][j-1][row, col] and pyramid[i][j][row, col] > pyramid[i][j+1][row, col]
                        if not ok:
             #               print(row, col, pyramid[i][j][row, col], '++>', pyramid[i][j - 1][row, col], pyramid[i][j + 1][row, col], pyramid[i][j][row - 1, col - 1])
                            continue
                        for level in range(-1, 2):
                            for direction in range(8):
                                if pyramid[i][j][row, col] <= pyramid[i][j + level][row + d_rows[direction], col + d_cols[direction]]:
                                    ok = False
                                    break
                            if not ok:
                                break
                        if not ok:
                            continue
                    elif pyramid[i][j][row, col] < pyramid[i][j][row - 1, col - 1]:
                        ok = pyramid[i][j][row, col] < pyramid[i][j-1][row, col] and pyramid[i][j][row, col] < pyramid[i][j+1][row, col]
                        if not ok:
             #               print(row, col, pyramid[i][j][row, col], '++>', pyramid[i][j - 1][row, col], pyramid[i][j + 1][row, col], pyramid[i][j][row - 1, col - 1])
                            continue
                        for level in range(-1, 2):
                            for direction in range(8):
                                if pyramid[i][j][row, col] >= pyramid[i][j + level][row + d_rows[direction], col + d_cols[direction]]:
                                    ok = False
                                    break
                            if not ok:
                                break
                        if not ok:
                            continue
                    #current pixel is a filter extremas candidate
                    interest_point_counter[0] += 1
                    image_norm_factor = 1.0/255.0 # convert image to range [0, 1]
                    first_der_factor = image_norm_factor * 0.5
                    second_der_factor = image_norm_factor
                    cross_der_factor = image_norm_factor * 0.5

                    #compute real extremas using taylor extension
                    #reference https://dsp.stackexchange.com/questions/10403/sift-taylor-expansion

                    der_D = np.array([pyramid[i][j][row, col + 1] - pyramid[i][j][row, col -1],
                                        pyramid[i][j][row + 1, col] - pyramid[i][j][row - 1, col],
                                        pyramid[i][j + 1][row, col] - pyramid[i][j - 1][row, col]]).astype(np.float) * first_der_factor
                    
                    Dxx = (pyramid[i][j][row, col + 1] + pyramid[i][j][row, col - 1] - (2.0 * pyramid[i][j][row, col])) * second_der_factor
                    Dyy = (pyramid[i][j][row + 1, col] + pyramid[i][j][row - 1, col] - (2.0 * pyramid[i][j][row, col])) * second_der_factor
                    Dss = (pyramid[i][j + 1][row, col] + pyramid[i][j - 1][row, col] - (2.0 * pyramid[i][j][row, col])) * second_der_factor
                    Dxy = ((pyramid[i][j][row + 1, col + 1] - pyramid[i][j][row + 1, col - 1]) - (pyramid[i][j][row - 1, col + 1] - pyramid[i][j][row - 1, col - 1])) * cross_der_factor
                    Dxs = ((pyramid[i][j + 1][row, col + 1] - pyramid[i][j + 1][row, col - 1]) - (pyramid[i][j - 1][row, col + 1] - pyramid[i][j][row, col - 1])) * cross_der_factor
                    Dys = ((pyramid[i][j + 1][row + 1, col] - pyramid[i][j + 1][row + 1, col]) - (pyramid[i][j - 1][row - 1, col] - pyramid[i][j][row - 1, col])) * cross_der_factor

                    H_x = np.array([[Dxx, Dxy, Dxs],
                                    [Dxy, Dyy, Dys],
                                    [Dxs, Dys, Dss]])
                    
                    if np.linalg.det(H_x) == 0.0:
                        interest_point_counter[1] += 1
                        continue

                    offset_col, offset_row, offset_sig =  -1.0 * np.dot(np.linalg.inv(H_x), der_D)

                    if offset_row >= 0.5 and offset_row >= 0.5 or offset_sig >= 0.5: 
                        correct_row  = (int)(row + round(offset_row, 0))
                        correct_col  = (int)(col + round(offset_col, 0))
                        correct_sig  = (int)(j + round(offset_sig, 0))
                    else:
                        correct_row  = row
                        correct_col  = col
                        correct_sig  = j
                        #break #current candidate is far from origina point
                    
                    if correct_row <= 0 or correct_row >= rows - 1 or correct_col <= 0 or correct_col >= cols - 1 or correct_sig <= 0 or correct_sig >= len(pyramid[0]) - 1:
                        interest_point_counter[3] += 1
                        continue #it is not possible to do next step
                    
                    #eliminate low constrast points using taylor expansion evalutated in correct_row, correct_col, correct_sig
                    
                    der_D = np.array([pyramid[i][correct_sig][correct_row, correct_col + 1] - pyramid[i][correct_sig][correct_row, correct_col - 1],
                                        pyramid[i][correct_sig][correct_row + 1, correct_col] - pyramid[i][correct_sig][correct_row - 1, correct_col],
                                        pyramid[i][correct_sig + 1][correct_row, correct_col] - pyramid[i][correct_sig - 1][correct_row, correct_col]]).astype(np.float) * first_der_factor

                    Dxx = (pyramid[i][correct_sig][correct_row, correct_col + 1] + pyramid[i][correct_sig][correct_row, correct_col - 1] - (2.0 * pyramid[i][correct_sig][correct_row, correct_col])) * second_der_factor
                    Dyy = (pyramid[i][correct_sig][correct_row + 1, correct_col] + pyramid[i][correct_sig][correct_row - 1, correct_col] - (2.0 * pyramid[i][correct_sig][correct_row, correct_col])) * second_der_factor
                    Dss = (pyramid[i][correct_sig + 1][correct_row, correct_col] + pyramid[i][correct_sig - 1][correct_row, correct_col] - (2.0 * pyramid[i][correct_sig][correct_row, correct_col])) * second_der_factor
                    Dxy = ((pyramid[i][correct_sig][correct_row + 1, correct_col + 1] - pyramid[i][correct_sig][correct_row + 1, correct_col - 1]) - (pyramid[i][correct_sig][correct_row - 1, correct_col + 1] - pyramid[i][correct_sig][correct_row - 1, correct_col - 1])) * cross_der_factor
                    Dxs = ((pyramid[i][correct_sig + 1][correct_row, correct_col + 1] - pyramid[i][correct_sig + 1][correct_row, correct_col - 1]) - (pyramid[i][correct_sig - 1][correct_row, correct_col + 1] - pyramid[i][correct_sig][correct_row, correct_col - 1])) * cross_der_factor
                    Dys = ((pyramid[i][correct_sig + 1][correct_row + 1, correct_col] - pyramid[i][correct_sig + 1][correct_row + 1, col]) - (pyramid[i][correct_sig - 1][correct_row - 1, correct_col] - pyramid[i][correct_sig][correct_row - 1, correct_col])) * cross_der_factor
                    
                    H_x = np.array([[Dxx, Dxy, Dxs],
                                    [Dxy, Dyy, Dys],
                                    [Dxs, Dys, Dss]])
                    
                    if np.linalg.det(H_x) == 0.0:
                        continue

                    vect_val = np.array([correct_col, correct_row, correct_sig])
                    
                    contrast_value = abs(image_norm_factor * pyramid[i][correct_sig][correct_row, correct_col] +  np.dot(der_D, vect_val) + 0.5 * np.dot(vect_val, np.dot(np.linalg.inv(H_x), der_D)))
                    #contrast_value = abs(image_norm_factor * pyramid[i][correct_sig][correct_row, correct_col] + np.dot(der_D, vect_val)) * len(pyramid[0])
                    if abs(contrast_value) < threshold_constrat:
                        interest_point_counter[4] += 1
                        continue
                    #eliminate edge points
                    
                    Dxx = (pyramid[i][correct_sig][correct_row, correct_col + 1] + pyramid[i][correct_sig][correct_row, correct_col - 1] - (2.0 * pyramid[i][correct_sig][correct_row, correct_col])) * second_der_factor
                    Dyy = (pyramid[i][correct_sig][correct_row + 1, correct_col] + pyramid[i][correct_sig][correct_row - 1, correct_col] - (2.0 * pyramid[i][correct_sig][correct_row, correct_col])) * second_der_factor
                    Dxy = ((pyramid[i][correct_sig][correct_row + 1, correct_col + 1] - pyramid[i][correct_sig][correct_row + 1, correct_col - 1]) - (pyramid[i][correct_sig][correct_row - 1, correct_col + 1] - pyramid[i][correct_sig][correct_row - 1, correct_col - 1])) * cross_der_factor

                    trace = Dxx + Dyy
                    determinant = Dxx * Dyy - Dxy**2

                    if determinant == 0 or trace*trace*threshold_edges >= (threshold_edges + 1) * (threshold_edges + 1) * determinant:
                        interest_point_counter[5] += 1
                        continue
                    
                    interest_point_counter[2] += 1
                    key_points_octave.append((correct_row, correct_col, correct_sig))
                    
            #for ind in range(1, len(interest_point_counter)):
            #    interest_point_counter[ind] += interest_point_counter[ind - 1]
            print('cnt interest:', interest_point_counter)
            #key_points_octave.append(extrema_image)
        print('\t', len(key_points_octave))
        key_points.append(key_points_octave)
    return key_points

def detect(img, sigma = 1.5):
    #img = cv2.resize(img, (img.shape[1]*2, img.shape[0]*2))
    pyramid, pyramid_sigma = built_pyramid(img.copy(), sigma = sigma, octaves = 3, blur_levels = 4)
    laplacian_of_gaussian = difference_of_gaussian(pyramid)
    #plot_pyramid(pyramid)
    #plot_pyramid(laplacian_of_gaussian)

    key_points = find_key_points(laplacian_of_gaussian)
    return key_points, pyramid, pyramid_sigma

def sift_feature(img, kpt, sigma = 1.5):
    #find keypoint direction
    hist = np.zeros((36))
    gaussian_kernel = np.outer(gaussian((int)(sigma*1.5), std = sigma*1.5), gaussian((int)(sigma*1.5), std = sigma*1.5))
    r, c = gaussian_kernel.shape
    cent_r, cent_c = (int)(np.ceil(r/2)), (int)(np.ceil(c/2))
    for i in range(r):
        for j in range(0, c):
            offset_r = i - cent_r
            offset_c = j - cent_c
            #print(i, j, img.shape, kpt, kpt[0] + offset_r, kpt[1] + offset_c)
            if kpt[0] + offset_r - 1>=0 and kpt[0] + offset_r + 1< img.shape[0] and kpt[1] + offset_c - 1>=0 and kpt[1] + offset_c + 1< img.shape[1] and img[kpt[0] + 1, kpt[1]] != img[kpt[0] - 1, kpt[1]]:
                mag = np.sqrt((0.0 + img[kpt[0] + 1, kpt[1]] - img[kpt[0] - 1, kpt[1]])**2 + (0.0 + img[kpt[0], kpt[1] + 1] - img[kpt[0], kpt[1] - 1])**2)
                ang = np.arctan((img[kpt[0], kpt[1] + 1]*1.0 - img[kpt[0], kpt[1] - 1]*1.0) / (img[kpt[0] + 1, kpt[1]]*1.0 - 1.0*img[kpt[0] - 1, kpt[1]])*1.0)

            #    print (r, c, i, j, mag, ang, (np.pi*ang)/180.0, (np.pi*ang)/1800.0)
               
    return 0
    #describe keypoint neighbourhood


def describe(key_points, pyramid, pyramid_sigma):
    descriptor = []
    for i in range(len(pyramid)): #for each octave
        for kpt in key_points[i]:
            descriptor.append(sift_feature(pyramid[i][kpt[2]], kpt, sigma = pyramid_sigma[i][kpt[2]]))
    return descriptor

def detect_and_compute(img, sigma = 1.5):
    key_points, pyramid, pyramid_sigma = detect(img, sigma = sigma)
    print(pyramid_sigma)
    features = describe(key_points, pyramid, pyramid_sigma)
    return key_points, features

if __name__ == '__main__':
    img = cv2.imread('input/p2-1-0.jpg')
    gray= cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    kp, des = detect_and_compute(gray)
    '''
    cv2.drawKeypoints(gray, kp, img, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    print(des)

    cv2.imwrite('sift_keypoints.jpg', img)
    '''