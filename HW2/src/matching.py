import cv2
import numpy as np
from scipy.spatial.distance import cosine
from scipy.spatial.distance import hamming
import sys
from random import randint
import time
import fast
import ORB
import sift

def similarity(first_feature, first_pos, second_feature, second_pos, distance_metric = 'l2-norm', spacial_weighting = 0.0):
    if distance_metric == 'l1-norm':
        diff = abs(first_feature - second_feature)
        distance = (diff * diff).sum()
    elif distance_metric == 'l2-norm':
        distance = np.linalg.norm(first_feature - second_feature)
    elif distance_metric == 'chebyshev':
        diff = abs(first_feature - second_feature)
        distance = diff.max()
    elif distance_metric == 'cosine':
        distance = cosine(first_feature, second_feature)
    elif distance_metric == 'hamming':
        distance = hamming(first_feature, second_feature)
    else:
        exit('Error: invalid distance metric')

    spacial_distance = np.linalg.norm(np.array(first_pos) - np.array(second_pos))
    distance = (1.0 - spacial_weighting) * distance + spacial_weighting * spacial_distance

    return distance

def find_matches(firsts_feature, firsts_pos, seconds_feature, seconds_pos, threshold = 0.8, distance_metric = 'l2-norm', spacial_weighting = 0.0, hard_match = False):
    matches = []
    hard_matches = []
    invalid_matches = 0
    for i in range(len(firsts_feature)):
        distances = []
        min_dist = sys.float_info.max
        closest = -1
        for j in range(len(seconds_feature)):
            diff = similarity(firsts_feature[i], firsts_pos[i], seconds_feature[j], seconds_pos[j], distance_metric = distance_metric, spacial_weighting = spacial_weighting)
            distances.append(diff)
            if diff < min_dist:
                min_dist = min(min_dist, diff)
                closest = j
        
        hard_matches.append(closest)
        if hard_match:
            continue

        distances.sort()
        if distances[0] / distances[1] >= threshold: #compare closest neighbours ratio
            invalid_matches += 1
            closest = -1
        matches.append(closest)

    if len(firsts_feature) - invalid_matches < 4 or hard_match: #there is not enought matches to compute
        print('Using hard matching')
        return hard_matches
    return matches

def joint_matches(img_fst, first_pos, img_scd, second_pos, match, file_name = 'matches', plot = True):
    if len(img_fst.shape)==2 or (len(img_fst.shape)==3 and img_fst.shape[2] == 1):
        img_fst = cv2.cvtColor(img_fst,cv2.COLOR_GRAY2RGB)
    
    if len(img_scd.shape)==2 or (len(img_scd.shape)==3 and img_scd.shape[2] == 1):
        img_scd = cv2.cvtColor(img_scd,cv2.COLOR_GRAY2RGB)

    for point in first_pos:
        cv2.circle(img_fst, (point[1], point[0]), 2, thickness = 1, color = (randint(0, 255), randint(0, 255), randint(0, 255)))
    
    for point in second_pos:
        cv2.circle(img_scd, (point[1], point[0]), 2, thickness = 1, color = (randint(0, 255), randint(0, 255), randint(0, 255)))
        
    #print(img_fst.shape, img_scd.shape)
    img_final = np.hstack((img_fst, img_scd))
    for i in range(len(first_pos)):
        if match[i] == -1:
            continue
        point_left = (first_pos[i][1], first_pos[i][0])
        point_right = (second_pos[match[i]][1] + img_fst.shape[1], second_pos[match[i]][0])
        cv2.line(img_final, point_left, point_right, thickness = 1, color = (randint(0, 255), randint(0, 255), randint(0, 255)))
    
    if plot:
        cv2.imwrite(file_name, img_final)
    return img_final

def convert_opencv_keypoint_format(kpts):
    kpts_ans = []
    for i in kpts:
        kpts_ans.append(((int)(i.pt[1]), (int)(i.pt[0])))
    return kpts_ans
def convert_opencv_keypoint_format_angle(kpts):
    kpts_ans = []
    or_list = []
    for i in kpts:
        kpts_ans.append(((int)(i.pt[1]), (int)(i.pt[0])))
        or_list.append(i.angle)
    return kpts_ans, or_list
    
def opencv_kpts_des(img, kpt_method, des_method):
    fast_opencv = cv2.FastFeatureDetector_create()
    sift_opencv = cv2.xfeatures2d.SIFT_create()
    orb_opencv = cv2.ORB_create()

    if kpt_method == 'orb':
        kpt = orb_opencv.detect(img, None)
        #kpt = fast.interest_points(img, 20, 8)
        #kpt = ORB.harris_measure_and_orientation(img, kpt, min(500, len(kpt)))
        kpt, angles = convert_opencv_keypoint_format_angle(kpt)
    elif kpt_method == 'fast':
        kpt = fast_opencv.detect(img, None)    
    else:
        exit('Invalid opencv kpt method')
    
    if des_method == 'sift':
        des = sift.compute(img, kpt, angles)
        #kpt, des = sift_opencv.compute(img, kpt)
    elif des_method == 'orb':
        kpt, des = orb_opencv.compute(img, kpt)
    else:
        exit('Invalid opencv des method')
    
    #kpt = convert_opencv_keypoint_format(kpt)
    
    return kpt, des

def find_keypoints_descriptors(img, kpt_method, des_method):
    
    if kpt_method == 'orb':
        kpt = fast.interest_points(img, 20, 8)
        kpt = ORB.harris_measure_and_orientation(img, kpt, min(500, len(kpt)))
        kpt, angles = convert_opencv_keypoint_format_angle(kpt)
    elif kpt_method == 'fast':
        kpt = fast.interest_points(img, 20, 8)
    else:
        exit('Invalid opencv kpt method')
    
    if des_method == 'sift':
    
        des = sift.compute(img, kpt, angles)
    #elif des_method == 'orb':
     #   kpt, des = orb.compute(img, kpt)
    else:
        exit('Invalid opencv des method')
    
    return kpt, des
    
def perf_match_images(file_name_fst, file_name_scnd): #evaluation using opencv methods
    img_a = cv2.imread(file_name_fst)
    img_a = cv2.cvtColor(img_a, cv2.COLOR_BGR2GRAY)
    img_b = cv2.imread(file_name_scnd)
    img_b = cv2.cvtColor(img_b, cv2.COLOR_BGR2GRAY)


    start = time.time()
    kpt_a, des_a = opencv_kpts_des(img_a, 'orb', 'sift')
    kpt_b, des_b = opencv_kpts_des(img_b, 'orb', 'sift')
    end = time.time()
    print('keypoint description', end - start)

    start = time.time()  
    matches = find_matches(des_a, kpt_a, des_b, kpt_b, hard_match = False, distance_metric = 'hamming', spacial_weighting = 0.0, threshold = 0.9)
    end = time.time()
    print('find matches', end - start)
    plot_matches(img_a, kpt_a, img_b, kpt_b, matches, file_name = 'test_opencv')

    print ('descriptors shape', np.array(des_a).shape, np.array(kpt_a).shape, np.array(des_b).shape, np.array(kpt_b).shape)


if __name__ == '__main__':
    # match_perf_images('input/p2-1-0.jpg', 'input/p2-1-1.jpg')
    perf_match_images('dbg/frame0.jpg', 'dbg/frame1.jpg')
