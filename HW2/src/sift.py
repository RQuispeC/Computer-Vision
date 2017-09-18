import numpy as np
import cv2
import math
from scipy.signal import gaussian
from scipy.ndimage.filters import gaussian_filter
from random import randint

def plot_pyramid(pyramid, name = 'pyramid_'):
    for i in range(len(pyramid)):
        for j in range(len(pyramid[0])):
                pyramid[i][j][pyramid[i][j] < 0 ] = 255
                cv2.imwrite( name + str(i) + '_' + str(j) + '.jpg', pyramid[i][j])
'''
def create_blur_level(img, sigma = [], k_gauss_factor = math.sqrt(2), blur_levels = 5):
    level = [img]
    for i in range(1, blur_levels):
        img_blurred = cv2.GaussianBlur(img, (7, 7), sigma[i], sigma[i])
        level.append(img_blurred)
    return level

def built_pyramid(img, sigma = math.sqrt(2)/2.0, k_gauss_factor = math.sqrt(2), octaves = 4, blur_levels = 5):
    #precompute sigma values
    sigmas = [sigma]
    k_gauss_factor = math.pow(2, 1.0 / blur_levels)
    for i in range(1, blur_levels):
        prev_sigma = math.pow(k_gauss_factor, i - 1) * sigma
        cur_sigma = prev_sigma * k_gauss_factor
        sigmas.append(np.sqrt(cur_sigma * cur_sigma - prev_sigma * prev_sigma))
    print(sigmas)
    pyramid = []
    for i in range(octaves):
        pyramid_level = create_blur_level(img.copy(), sigmas, k_gauss_factor, blur_levels)
        pyramid.append(pyramid_level)
        img = pyramid_level[blur_levels - 1]
        img = cv2.resize(img, (img.shape[1]//2, img.shape[0]//2))
    return pyramid
'''

def create_blur_level(img, sigma, k_gauss_factor = math.sqrt(2), blur_levels = 5):
    level = []
    for i in range(0, blur_levels):
        img_blurred = cv2.GaussianBlur(img, (0, 0), sigma, sigma)
        level.append(img_blurred)
        sigma *= k_gauss_factor
    return level

def built_pyramid(img, sigma = math.sqrt(2)/2.0, k_gauss_factor = math.sqrt(2), octaves = 4, blur_levels = 5):
    #precompute sigma values
    k_gauss_factor = math.pow(2, 1.0 / blur_levels)
    mid_factor = k_gauss_factor**(blur_levels//2)
    pyramid = []
    for i in range(octaves):
        pyramid_level = create_blur_level(img.copy(), sigma, k_gauss_factor, blur_levels)
        pyramid.append(pyramid_level)
        img = pyramid_level[blur_levels - 1]
        img = cv2.resize(img, (img.shape[1]//2, img.shape[0]//2))
        sigma *= mid_factor
    return pyramid


def difference_of_gaussian(pyramid):
    diff_gauss = []
    for i in range(len(pyramid)):
        level = []
        for j in range(len(pyramid[0]) - 1):
            diff = pyramid[i][j + 1].astype(np.float) - pyramid[i][j].astype(np.float)
            #level.append(cv2.subtract(pyramid[i][j + 1].astype(np.float), pyramid[i][j].astype(np.float)))
            level.append(diff)
        diff_gauss.append(level)
    return diff_gauss

def find_key_points(pyramid, threshold_constrat = 0.04, threshold_edges = 15):
    image_norm_factor = 1.0/255.0 # convert image to range [0, 1]
    first_der_factor = image_norm_factor * 0.5
    second_der_factor = image_norm_factor
    cross_der_factor = image_norm_factor * 0.25
    converge_iterations = 5
    d_rows = [-1, 0, 1, 0, -1, 1, 1, -1]
    d_cols = [0, -1, 0, 1, 1, -1, 1, -1]
    threshold = np.floor(0.5 * threshold_constrat / (len(pyramid[0])) * 255.0);
    key_points = []
    for i in range(len(pyramid)): #foreach octave
        key_points_octave = []
        rows, cols = pyramid[i][0].shape
        for j in range(1, len(pyramid[0]) - 1): #find extremas between (j-1, j, j+1)
            interest_point_counter = [0, 0, 0, 0, 0, 0]
            for row in range(1, rows - 1):
                for col in range(1, cols - 1):
                    cur_val = pyramid[i][j][row, col]
                    if abs(cur_val) <= threshold:
                        continue
                    if cur_val > 0:
                        ok = cur_val >= pyramid[i][j][row - 1, col - 1] and cur_val >= pyramid[i][j-1][row, col] and cur_val >= pyramid[i][j+1][row, col]
                        if not ok:
             #               print(row, col, pyramid[i][j][row, col], '++>', pyramid[i][j - 1][row, col], pyramid[i][j + 1][row, col], pyramid[i][j][row - 1, col - 1])
                            continue
                        for level in range(-1, 2):
                            for direction in range(8):
                                if pyramid[i][j][row, col] < pyramid[i][j + level][row + d_rows[direction], col + d_cols[direction]]:
                                    ok = False
                                    break
                            if not ok:
                                break
                        if not ok:
                            continue
                    elif cur_val < 0:
                        ok = cur_val <= pyramid[i][j][row - 1, col - 1] and cur_val <= pyramid[i][j-1][row, col] and cur_val <= pyramid[i][j+1][row, col]
                        if not ok:
             #               print(row, col, pyramid[i][j][row, col], '++>', pyramid[i][j - 1][row, col], pyramid[i][j + 1][row, col], pyramid[i][j][row - 1, col - 1])
                            continue
                        for level in range(-1, 2):
                            for direction in range(8):
                                if cur_val > pyramid[i][j + level][row + d_rows[direction], col + d_cols[direction]]:
                                    ok = False
                                    break
                            if not ok:
                                break
                        if not ok:
                            continue
                    else:
                        continue
                    interest_point_counter[0] += 1
                    converge = True
                    rr, cc, jj = row, col, j
                    for it in range(converge_iterations): #iterate to converge
                        #compute real extremas using taylor extension
                        #reference https://dsp.stackexchange.com/questions/10403/sift-taylor-expansion

                        der_D = np.array([pyramid[i][j][row, col + 1] - pyramid[i][j][row, col -1],
                                            pyramid[i][j][row + 1, col] - pyramid[i][j][row - 1, col],
                                            pyramid[i][j + 1][row, col] - pyramid[i][j - 1][row, col]]).astype(np.float) * first_der_factor
                        
                        Dxx = (pyramid[i][j][row, col + 1] + pyramid[i][j][row, col - 1] - (2.0 * pyramid[i][j][row, col])) * second_der_factor
                        Dyy = (pyramid[i][j][row + 1, col] + pyramid[i][j][row - 1, col] - (2.0 * pyramid[i][j][row, col])) * second_der_factor
                        Dss = (pyramid[i][j + 1][row, col] + pyramid[i][j - 1][row, col] - (2.0 * pyramid[i][j][row, col])) * second_der_factor
                        Dxy = ((pyramid[i][j][row + 1, col + 1] - pyramid[i][j][row + 1, col - 1]) - (pyramid[i][j][row - 1, col + 1] - pyramid[i][j][row - 1, col - 1])) * cross_der_factor
                        Dxs = ((pyramid[i][j + 1][row, col + 1] - pyramid[i][j + 1][row, col - 1]) - (pyramid[i][j - 1][row, col + 1] - pyramid[i][j - 1][row, col - 1])) * cross_der_factor
                        Dys = ((pyramid[i][j + 1][row + 1, col] - pyramid[i][j + 1][row + 1, col]) - (pyramid[i][j - 1][row + 1, col] - pyramid[i][j - 1][row - 1, col])) * cross_der_factor

                        H_x = np.array([[Dxx, Dxy, Dxs],
                                        [Dxy, Dyy, Dys],
                                        [Dxs, Dys, Dss]])
                        
                        if np.linalg.det(H_x) == 0.0:
                            converge = it > 0
                            break

                        offset_col, offset_row, offset_sig =  -1.0 * np.dot(np.linalg.inv(H_x), der_D)
                        #print (offset_col, offset_row, offset_sig)
                        if abs(offset_row) < 0.5 and abs(offset_col) < 0.5 and abs(offset_sig) < 0.5:
                            break #current candidate is far from real extrema point
                        
                        if abs(offset_row) > (1<<31)/3 or abs(offset_col) > (1<<31)/3 and abs(offset_sig) > (1<<31)/3:
                            converge = False
                            break

                        row += (int)(round(offset_row, 0))
                        col += (int)(round(offset_col, 0))
                        j += (int)(round(offset_sig, 0))


                        if row <= 0 or row >= rows - 1 or col <= 0 or col >= cols - 1 or j <= 0 or j >= len(pyramid[0]) - 1:
                            converge = False
                            break
                        
                        if it == converge_iterations - 1: #last iteration yield
                            converge = False

                    correct_row  = (int)(row)
                    correct_col  = (int)(col)
                    correct_sig  = (int)(j)
                    row, col, j = rr, cc, jj

                    if not converge:
                        interest_point_counter[1] += 1
                        continue
                
                    #eliminate low constrast points using taylor expansion evalutated in correct_row, correct_col, correct_sig
                    der_D = np.array([pyramid[i][correct_sig][correct_row, correct_col + 1] - pyramid[i][correct_sig][correct_row, correct_col - 1],
                                        pyramid[i][correct_sig][correct_row + 1, correct_col] - pyramid[i][correct_sig][correct_row - 1, correct_col],
                                        pyramid[i][correct_sig + 1][correct_row, correct_col] - pyramid[i][correct_sig - 1][correct_row, correct_col]]).astype(np.float) * first_der_factor

                    vect_val = np.array([correct_col, correct_row, correct_sig])
                    dot_prod = np.dot(der_D, vect_val)

                    #contrast_value = abs(image_norm_factor * pyramid[i][correct_sig][correct_row, correct_col] +  np.dot(der_D, vect_val) + 0.5 * np.dot(vect_val, np.dot(np.linalg.inv(H_x), der_D)))
                    contrast_value = abs(dot_prod * 0.5 + image_norm_factor * pyramid[i][correct_sig][correct_row, correct_col]) * len(pyramid[0])
                    if contrast_value < threshold_constrat:
                        interest_point_counter[2] += 1
                        continue

                    Dxx = (pyramid[i][correct_sig][correct_row, correct_col + 1] + pyramid[i][correct_sig][correct_row, correct_col - 1] - (2.0 * pyramid[i][correct_sig][correct_row, correct_col])) * second_der_factor
                    Dyy = (pyramid[i][correct_sig][correct_row + 1, correct_col] + pyramid[i][correct_sig][correct_row - 1, correct_col] - (2.0 * pyramid[i][correct_sig][correct_row, correct_col])) * second_der_factor
                    Dxy = ((pyramid[i][correct_sig][correct_row + 1, correct_col + 1] - pyramid[i][correct_sig][correct_row + 1, correct_col - 1]) - (pyramid[i][correct_sig][correct_row - 1, correct_col + 1] - pyramid[i][correct_sig][correct_row - 1, correct_col - 1])) * cross_der_factor
                    trace = Dxx + Dyy
                    determinant = Dxx * Dyy - Dxy**2

                    if determinant == 0 or trace*trace*threshold_edges >= (threshold_edges + 1) * (threshold_edges + 1) * determinant:
                        interest_point_counter[3] += 1
                        continue
                    
                    interest_point_counter[4] += 1
                    key_points_octave.append((correct_row, correct_col, correct_sig, i))
                    
            print(pyramid[i][0].shape, 'cnt interest:', interest_point_counter)
        print('\t', len(key_points_octave))
        key_points.append(key_points_octave)
    return key_points

def keypoints_refinement(keypoints):
    pyramid = []
    

def detect(img, sigma = 1.5):
    #doublesize original imag to get more keypoints
    img = cv2.resize(img, (img.shape[1]*2, img.shape[0]*2))
    init_sigma = np.sqrt(max(sigma * sigma - 0.5 * 0.5 *  4, 0.01))
    print(init_sigma)
    img = cv2.GaussianBlur(img, (0, 0), init_sigma, init_sigma)

    #compute pyramid and dog
    pyramid = built_pyramid(img.copy(), sigma = sigma, octaves = 3, blur_levels = 4)
    laplacian_of_gaussian = difference_of_gaussian(pyramid)

    #plot_pyramid(pyramid, name = 'pyramid_')
    #plot_pyramid(laplacian_of_gaussian, name = 'dog_')

    #find keypoints
    key_points = find_key_points(laplacian_of_gaussian)

    #refine keypoints
    #key_points = keypoints_refinement(key_points)

    #plot keypoints
    img_kpt = draw_key_points(img.copy(), key_points, pyramid)
    cv2.imwrite('my_sift.jpg', img_kpt)

    return key_points, pyramid

def keypoint_orientation(img, kpt, sigma = 1.5, threshold_kpt_direction = 0.8):
    hist = np.zeros((36))
    gaussian_kernel = np.outer(gaussian((int)(sigma*1.5), std = sigma*1.5), gaussian((int)(sigma*1.5), std = sigma*1.5))
    gaussian_kernel /= gaussian_kernel.sum()
    r, c = gaussian_kernel.shape
    cent_r, cent_c = (int)(np.ceil(r/2)), (int)(np.ceil(c/2))
    for i in range(r):
        for j in range(c):
            offset_r = i - cent_r
            offset_c = j - cent_c
            #print(i, j, img.shape, kpt, kpt[0] + offset_r, kpt[1] + offset_c)
            if kpt[0] + offset_r - 1>=0 and kpt[0] + offset_r + 1< img.shape[0] and kpt[1] + offset_c - 1>=0 and kpt[1] + offset_c + 1< img.shape[1] and img[kpt[0] + offset_r, kpt[1] + offset_c + 1] != img[kpt[0] + offset_r, kpt[1] + offset_c - 1]:
                mag = np.sqrt((0.0 + img[kpt[0] + offset_r + 1, kpt[1] + offset_c] - img[kpt[0] - 1 + offset_r, kpt[1] + offset_c])**2 + (0.0 + img[kpt[0] + offset_r, kpt[1] + 1 + offset_c] - img[kpt[0] + offset_r, kpt[1] - 1 + offset_c])**2)
                ang = np.arctan2((img[kpt[0] + 1 + offset_r, kpt[1]+offset_c]*1.0 - 1.0*img[kpt[0] - 1 + offset_r, kpt[1] + offset_c])*1.0, (img[kpt[0] + offset_r, kpt[1] + 1 + offset_c]*1.0 - img[kpt[0] + offset_r, kpt[1] - 1 + offset_c]*1.0))
                if ang < 0.0 :
                    ang = (ang + 2 * np.pi)
                ang = (ang*180.0) / np.pi

                hist[(int)(ang/10.0)] += gaussian_kernel[i, j] * mag
                # print (kpt[0] + offset_r, kpt[1] + offset_c, ang, mag, gaussian_kernel[i, j])

    
    max_gradient = hist.max()
    gradient_directions = []
    for i in range(len(hist)):
        if hist[i] >= threshold_kpt_direction * max_gradient:
            gradient_directions.append(i * 10 + 5)
    return gradient_directions

def compute(img, kpt_list, kpt_orientation_list):
    init_sigma = np.sqrt(max(1.6 * 1.6 - 0.5 * 0.5 *  4, 0.01))
    cv2.imwrite('blurred.jpg', img)
    img = cv2.GaussianBlur(img, (0, 0), init_sigma, init_sigma)
    features = []
    for kpt, orientation in zip(kpt_list, kpt_orientation_list):
        feature = sift_feature(img, kpt, orientation)
        features.append(feature)
    return features

def sift_feature(img, kpt, kpt_orientation, sigma = 1.5, threshold_histogram_maxima = 0.2, threshold_kpt_direction = 0.8, hist_bin = 8, cell_len = 4, block_len = 16):
    #find keypoint direction
    gaussian_kernel = np.outer(gaussian(block_len, std = 0.5*block_len), gaussian(block_len, std = 0.5*block_len))
    gaussian_kernel /= gaussian_kernel.sum()
    features = []
    for direction in [kpt_orientation]:
        hist = np.zeros((cell_len**2 * hist_bin))
        for i in range(-block_len//2, block_len //2):
            for j in range(-block_len//2, block_len //2):
                if kpt[0] + i - 1>=0  and kpt[0] + i + 1< img.shape[0] and kpt[1] + j - 1>=0  and kpt[1] + j  + 1< img.shape[1] and img[kpt[0] + i, kpt[1] + j + 1] != img[kpt[0] + i, kpt[1] + j - 1]:
                    pi = i + block_len//2
                    pj = j + block_len//2
                    i_cell = pi % cell_len
                    j_cell = pj % cell_len
                    cell_num = (pi // cell_len) * (block_len // cell_len) + (pj//cell_len)
                    mag = np.sqrt((0.0 + img[kpt[0] + 1 + i, kpt[1] + j] - img[kpt[0] - 1 + i, kpt[1] + j])**2 + (0.0 + img[kpt[0] + i, kpt[1] + 1 + j] - img[kpt[0] + i, kpt[1] - 1 + j])**2)
                    ang = np.arctan2((img[kpt[0] + 1 + i, kpt[1] + j]*1.0 - 1.0*img[kpt[0] - 1 + i, kpt[1] + j])*1.0, (img[kpt[0] + i, kpt[1] + 1 + j]*1.0 - img[kpt[0] + i, kpt[1] - 1 + j]*1.0))
                    if ang < 0.0 :
                        ang = (ang + 2.0 * np.pi)
                    ang = ((ang*180.0) / np.pi) - direction
                    if ang < 0.0 :
                        ang += 360.0
                    # print(pi, pj, i_cell, j_cell, cell_num, ang, (int)(ang/(360.0 / hist_bin)), cell_num * hist_bin + (int)(ang/(360.0 / hist_bin)))
                    hist[cell_num * hist_bin + (int)(ang/(360.0 / hist_bin))] += mag * gaussian_kernel[pi, pj]
                
                #else:
                #    print('SIFT features for point', kpt, ' get out of image bounds', img.shape)
        
        #normalize to unit vertor
        norm = np.sqrt((hist * hist).sum())
        hist /= norm
        fact = 0
        #remove large responses
        for i in range(len(hist)):
            #if hist[i] >= threshold_histogram_maxima * norm:
            hist[i] = min(hist[i], threshold_histogram_maxima * norm)
            fact += hist[i] * hist[i]
        #normalize again
        #hist /= np.sqrt((hist * hist).sum())
        fact = max(fact, 1e-20)
        hist *= 512.0/fact
        hist /= np.sqrt((hist * hist).sum())

        features.append(hist)
    if len(features) == 0:
        exit('Sift feature vector was not calculated')
    elif len(features) == 1:
        return features[0]
    return features

def describe(key_points, pyramid, pyramid_sigma):
    descriptor = []
    for i in range(len(pyramid)): #for each octave
        for kpt in key_points[i]:
            descriptor += sift_feature(pyramid[i][kpt[2]], kpt, keypoint_orientation(pyramid[i][kpt[2]], kpt, sigma = pyramid_sigma[i][kpt[2]]), sigma = pyramid_sigma[i][kpt[2]])
    return np.array(descriptor)

def draw_key_points(img, key_points, pyramid):
    if len(img.shape)==2 or (len(img.shape)==3 and img.shape[2] == 1):
        img = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
    for i in range(len(key_points)):
        scale = (int)(np.ceil(img.shape[0]//pyramid[i][0].shape[0]))
        for kpt in key_points[i]:
            cv2.circle(img, (scale * kpt[1], scale * kpt[0]), 2**scale, thickness = 1, color = (randint(0, 255), randint(0, 255), randint(0, 255)))
    return img

def detect_and_compute(img, sigma = 1.5):
    key_points, pyramid= detect(img, sigma = sigma)
    #features = describe(key_points, pyramid, pyramid_sigma)
    #print (features)
    #return key_points, features
    return key_points, 0


if __name__ == '__main__':
    #img = cv2.imread('input/p2-1-0.jpg')
    img = cv2.imread('dbg/frame0.jpg')
    
    gray= cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    kp, des = detect_and_compute(gray)
    '''
    cv2.drawKeypoints(gray, kp, img, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    print(des)

    cv2.imwrite('sift_keypoints.jpg', img)
    '''
