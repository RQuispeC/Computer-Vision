import numpy
import cv2
import ORB
import sift
import fast
import matching

def print_kpts(kpts, des):
    for i in range(len(kpts)):
        print(kpts[i].pt, kpts[i].angle, des[i])
    return kpts

def update_angle(kpts):
    for i in range(len(kpts)):
        kpts[i].size = 20
        #print(kpts[i].pt, kpts[i].angle)
    return kpts

def convert_opencv_keypoint_format(kpts):
    kpts_ans = []
    or_list = []
    for i in kpts:
        kpts_ans.append(((int)(i.pt[1]), (int)(i.pt[0])))
        or_list.append(i.angle)
    return kpts_ans, or_list

def draw_key_points(img, key_points, pyramid):
    if len(img.shape)==2 or (len(img.shape)==3 and img.shape[2] == 1):
        img = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
    for i in range(len(key_points)):
        scale = (int)(np.ceil(img.shape[0]//pyramid[i][0].shape[0]))
        for kpt in key_points[i]:
            cv2.circle(img, (scale * kpt[1], scale * kpt[0]), 2**scale, thickness = 1, color = (randint(0, 255), randint(0, 255), randint(0, 255)))
    return img

if __name__ == '__main__':
    img_a = cv2.imread('input/p2-1-0.jpg')
    img_b = cv2.imread('input/p2-1-1.jpg')
    
    #img_a = cv2.imread('input/fumar.jpg')
    #img_b = cv2.imread('input/fumar_rota.jpg')
    
    gray_a = cv2.cvtColor(img_a, cv2.COLOR_BGR2GRAY)
    gray_b = cv2.cvtColor(img_b, cv2.COLOR_BGR2GRAY)
  
    fast_kpt_a = fast.interest_points(gray_a, 40, 9,True)
    fast_kpt_b = fast.interest_points(gray_b, 40, 9,True)
  
    print(len(fast_kpt_a), len(fast_kpt_b))
  
    orb_kpt_a = ORB.harris_measure_and_orientation(gray_a, fast_kpt_a, min(200, len(fast_kpt_a)))
    orb_kpt_b = ORB.harris_measure_and_orientation(gray_b, fast_kpt_b, min(200, len(fast_kpt_b)))
  
    #pos_a = convert_opencv_keypoint_format(orb_kpt_a)
    #pos_b = convert_opencv_keypoint_format(orb_kpt_b)
    sift_opencv = cv2.xfeatures2d.SIFT_create()
    
    #des_a = sift.compute(gray_a, pos_a[0], pos_a[1])
    #des_b = sift.compute(gray_b, pos_b[0], pos_b[1])
    
    #orb_kpt_a, des_a = sift_opencv.compute(gray_a, orb_kpt_a)
    #orb_kpt_b, des_b = sift_opencv.compute(gray_b, orb_kpt_b)
        
    #matches = matching.find_matches(des_a, pos_a[0], des_b, pos_b[0], hard_match = True, distance_metric = 'cosine', spacial_weighting = 0.0, threshold = 0.9)
  
    #matching.joint_matches(img_a, pos_a[0], img_b, pos_b[0], matches, file_name = 'mymatch.jpg')
    
    kpts_a = orb_kpt_a
    kpts_b = orb_kpt_b
    
    orb_kpt_a = update_angle(orb_kpt_a)
    orb_kpt_b = update_angle(orb_kpt_b)
    
    cv2.drawKeypoints(gray_a, orb_kpt_a, img_a, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv2.drawKeypoints(gray_b, orb_kpt_b, img_b, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    cv2.imwrite('kpt_dir_a.jpg', img_a)
    cv2.imwrite('kpt_dir_b.jpg', img_b)

    pos_a = convert_opencv_keypoint_format(orb_kpt_a)
    pos_b = convert_opencv_keypoint_format(orb_kpt_b)

    des_a = sift.compute(gray_a, pos_a[0], pos_a[1])
    des_b = sift.compute(gray_b, pos_b[0], pos_b[1])

    #print_kpts(kpts_a, des_a)
    #print(' --->')
    #print_kpts(kpts_b, des_b)

    matches = matching.find_matches(des_a, des_b, pos_a[0], pos_b[0], hard_match = True, distance_metric = 'cosine', spacial_weighting = 0.0, threshold = 0.9, approach = 'brute_force')
    matching.joint_matches(gray_a, pos_a[0], gray_b, pos_b[0], matches, file_name = 'match-40-9-true.jpg')

    
    kpts_a, des_a = sift_opencv.compute(gray_a, kpts_a)
    kpts_b, des_b = sift_opencv.compute(gray_b, kpts_b)

    print_kpts(kpts_a, des_a)
    print(' --->')
    print_kpts(kpts_b, des_b)

    matches = matching.find_matches(des_a, des_b, pos_a[0], pos_b[0], hard_match = True, distance_metric = 'cosine', spacial_weighting = 0.0, threshold = 0.9, approach = 'brute_force')
    matching.joint_matches(gray_a, pos_a[0], gray_b, pos_b[0], matches, file_name = 'expected.jpg')
