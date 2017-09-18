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

def brute_force_matching(firsts_feature, firsts_pos, seconds_feature, seconds_pos, threshold = 0.8, distance_metric = 'l2-norm', spacial_weighting = 0.0, hard_match = False):
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

def knn_matching(firsts_feature, seconds_feature):
    from sklearn.neighbors import KNeighborsClassifier
    for i in range(len(firsts_feature)):
        firsts_feature[i] =  np.array(firsts_feature[i]).reshape(-1, 1)
        print(firsts_feature[i].shape)
    
    for i in range(len(seconds_feature)):
        seconds_feature[i] =  np.array(seconds_feature[i]).reshape(-1, 1)
    
    knn_first = KNeighborsClassifier(n_neighbors = 1)
    knn_second = KNeighborsClassifier(n_neighbors = 1)
    y_fir = np.arange(len(firsts_feature))
    y_sec = np.arange(len(seconds_feature))
    knn_first.fit(firsts_feature, y_fir)
    knn_second.fit(seconds_feature, y_sec)
    matches = []
    for i in range(len(firsts_feature)):
        right_pred = knn_second.predict(firsts_feature[i].reshape(-1, 1))
        if knn_first.predict(seconds_feature[right_pred].reshape(-1, 1)) == i:
            matches.append(right_pred)
        else:
            matches.append(-1)
    return matches

def find_matches(firsts_feature, seconds_feature, firsts_pos = [], seconds_pos = [], threshold = 0.8, distance_metric = 'l2-norm', spacial_weighting = 0.0, hard_match = False, approach = 'brute_force'):
    if approach == 'brute_force':
        return brute_force_matching(firsts_feature, firsts_pos, seconds_feature, seconds_pos, threshold, distance_metric, spacial_weighting , hard_match)
    elif approach == 'knn':
        return knn_matching(firsts_feature, seconds_feature)
    else:
        exit('Error: invalid approach for find_matches')

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

def unpack_kpts_angle(kpts):
    kpts_ans = []
    or_list = []
    for i in kpts:
        kpts_ans.append(((int)(i.pt[1]), (int)(i.pt[0])))
        or_list.append(i.angle)
    return kpts_ans, or_list

def find_keypoints_descriptors(img, kpt_method, des_method, orb_thr = 30, orb_N = 8, orb_nms = False):
    
    if kpt_method == 'orb':
        kpt = fast.interest_points(img, orb_thr, orb_N, orb_nms)
        kpt = ORB.harris_measure_and_orientation(img, kpt, min(500, len(kpt)))
        kpt, angles = unpack_kpts_angle(kpt)
    elif kpt_method == 'fast':
        kpt = fast.interest_points(img, orb_thr, orb_N, orb_nms)
    else:
        exit('Invalid opencv kpt method')
    
    if des_method == 'sift':
        des = sift.compute(img, kpt, angles)
    else:
        exit('Invalid opencv des method')
    
    return kpt, des

