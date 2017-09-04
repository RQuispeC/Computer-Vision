import numpy as np
import cv2
import math
from scipy.ndimage.filters import gaussian_filter

def plot_pyramid(pyramid):
    for i in range(len(pyramid)):
        for j in range(len(pyramid[0])):
                cv2.imwrite('pyramid_'+str(i) + '_' + str(j) + '.jpg', pyramid[i][j])

def create_blur_level(img, sigma = math.sqrt(2), k_gauss_factor = math.sqrt(2), blur_levels = 5):
    level = [gaussian_filter(img, sigma)]
    #print (sigma)
    for i in range(1, blur_levels):
        sigma = sigma * k_gauss_factor
        #print (sigma)
        img_blurred = gaussian_filter(level[i-1], sigma)
        level.append(img_blurred)
    return level

def built_pyramid(img, sigma = math.sqrt(2)/2.0, k_gauss_factor = math.sqrt(2), octaves = 4, blur_levels = 5):
    middle_octave_k_factor = k_gauss_factor
    for i in range(1, blur_levels//2):
        middle_octave_k_factor = middle_octave_k_factor * k_gauss_factor
    pyramid = []
    for i in range(octaves):
        pyramid.append(create_blur_level(img, sigma, k_gauss_factor, blur_levels))
        sigma = sigma * middle_octave_k_factor
        img = cv2.resize(img, (img.shape[1]//2, img.shape[0]//2))
    return pyramid

def difference_of_gaussian(pyramid):
    diff_gauss = []
    for i in range(len(pyramid)):
        level = []
        for j in range(len(pyramid[0]) - 1):
            level.append(cv2.subtract(pyramid[i][j + 1], pyramid[i][j]))
        diff_gauss.append(level)
    return diff_gauss

def find_key_points(pyramid, threshold_constrat = 0.3, threshold_edges = 10):
    d_rows = [-1, 0, 1, 0, -1, 1, 1, -1]
    d_cols = [0, -1, 0, 1, 1, -1, 1, -1]
    key_points = []
    for i in range(len(pyramid)): #foreach octave
        key_points_octave = []
        rows, cols = pyramid[i][0].shape
        for j in range(1, len(pyramid[0]) - 1): #find extremas between (j-1, j, j+1)
            extrema_image = np.zeros((rows, cols))
            interest_point_counter = [0, 0, 0, 0, 0]
            for row in range(1, rows - 1):
                for col in range(1, cols - 1):
                    if pyramid[i][j][row, col] > pyramid[i][j][row - 1, col - 1]:
                        ok = pyramid[i][j][row, col] > pyramid[i][j-1][row, col] and pyramid[i][j][row, col] > pyramid[i][j+1][row, col]
                        if not ok:
                            break
                        for level in range(-1, 2):
                            for direction in range(8):
                                if pyramid[i][j][row, col] <= pyramid[i][j + level][row + d_rows[direction], col + d_cols[direction]]:
                                    ok = False
                                    break
                            if not ok:
                                break
                        if not ok:
                            break
                        else:
                            extrema_image[row, col] = 255 #is extrema point
                    elif pyramid[i][j][row, col] < pyramid[i][j][row - 1, col - 1]:
                        ok = pyramid[i][j][row, col] < pyramid[i][j-1][row, col] and pyramid[i][j][row, col] < pyramid[i][j+1][row, col]
                        if not ok:
                            break
                        for level in range(-1, 2):
                            for direction in range(8):
                                if pyramid[i][j][row, col] >= pyramid[i][j + level][row + d_rows[direction], col + d_cols[direction]]:
                                    ok = False
                                    break
                            if not ok:
                                break
                        if not ok:
                            break
                        else: #filter extremas
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

                            H_x = np.array([[Dxx, Dxy, Dxs]
                                            [Dxy, Dyy, Dys]
                                            [Dxs, Dys, Dss]])
                            offset_col, offset_row, offset_sig =  -1.0 * np.outer(np.linalg.inv(H_x), np.transpose(der_D))

                            if offset_row < 0.5 or offset_row < 0.5 or offset_sig < 0.5: 
                                break #current candidate is far from origina point
                            
                            interest_point_counter[1] += 1

                            correct_row  = row + round(offset_row, 0)
                            correct_col  = col + round(offset_col, 0)
                            correct_sig  = sig + round(offset_sig, 0)

                            if correct_row <= 0 or correct_row >= rows - 1 or correct_col <= 0 or correct_col >= cols - 1 or correct_sig <= 0 or correct_sig >= len(pyramid[0]) - 1:
                                break #it is not possible to do next step

                            interest_point_counter[2] += 1
                            #eliminate low constrast points using taylor expansion evalutated in correct_row, correct_col, 
                            
                            der_D = np.array([pyramid[i][correct_sig][correct_row, correct_col + 1] - pyramid[i][correct_sig][correct_row, correct_col -1],
                                              pyramid[i][correct_sig][correct_row + 1, correct_col] - pyramid[i][correct_sig][correct_row - 1, correct_col],
                                              pyramid[i][correct_sig + 1][correct_row, correct_col] - pyramid[i][correct_sig - 1][correct_row, correct_col]]).astype(np.float) * first_der_factor
                            
                            constrast_value = len(pyramid[0])  *  abs(image_norm_factor * pyramid[i][correct_sig][correct_row, correct_col] +  (der_D * np.array([coorect_col, correct_row, correct_sig])).sum() * 0.5)

                            if abs(constrast_value) < threshold_constrat:
                                break
                            
                            interest_point_counter[3] += 1

                            #eliminate edge points
                            
                            Dxx = (pyramid[i][correct_sig][correct_row, correct_col + 1] + pyramid[i][correct_sig][correct_row, correct_col - 1] - (2.0 * pyramid[i][correct_sig][correct_row, correct_col])) * second_der_factor
                            Dyy = (pyramid[i][correct_sig][correct_row + 1, correct_col] + pyramid[i][correct_sig][correct_row - 1, correct_col] - (2.0 * pyramid[i][correct_sig][correct_row, correct_col])) * second_der_factor
                            Dxy = ((pyramid[i][correct_sig][correct_row + 1, correct_col + 1] - pyramid[i][correct_sig][correct_row + 1, correct_col - 1]) - (pyramid[i][correct_sig][correct_row - 1, correct_col + 1] - pyramid[i][correct_sig][correct_row - 1, correct_col - 1])) * cross_der_factor

                            trace = Dxx + Dyy
                            determinant = Dxx * Dyy - Dxy**2

                            if determinant == 0 or trace*trace*threshold_edges >= (threshold_edges + 1) * (threshold_edges + 1) * determinant:
                                break

                            interest_point_counter[4] += 1
                            extrema_image[row, col] = 255 #is extrema point

            for ind in range(1, len(interest_point_counter)):
                interest_point_counter[i] += interest_point_counter[i - 1]
            print(interest_point_counter)    
            key_points_octave.append(extrema_image)
        key_points.append(key_points_octave)    
    
    return key_points

def detect(img):
    pyramid = built_pyramid(img.copy())
    laplacian_of_gaussian = difference_of_gaussian(pyramid)
    #plot_pyramid(pyramid)
    #plot_pyramid(laplacian_of_gaussian)

    key_points = find_key_points(laplacian_of_gaussian)
    return key_points

def describe(img, key_points):
    return 0

def detect_and_compute(img):
    key_points = detect(img)
    features = describe(img, key_points)
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