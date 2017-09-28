import cv2
import numpy as np
from random import randint

def load_video(file_name = 'input/p2-1-1.mp4', fps = 30, original_fps = 30):
    if fps > original_fps or fps < 1:
        exit('Error: not valid frames per second')
    vidcap = cv2.VideoCapture(file_name)
    cnt = 0
    video = []
    success, img = vidcap.read()
    factor_fps = original_fps // fps
    while success:
        if cnt % factor_fps == 0:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            #img = cv2.resize(image, (640, 480))
            img = np.float32(img)
            video.append(img)
        cnt += 1
        success, img = vidcap.read()
    return video
    
def plot_flow(video, kpt, trans_params, file_name = 'dbg/flow_'):
    mask = np.zeros((video[0].shape[0], video[0].shape[1], 3))
    colors = np.random.randint(0, 255, (1000, 3 ))
    for i in range(len(video) - 1): #plot kpts for each frame
        if len(video[i].shape)==2 or (len(video[i].shape)==3 and video[i].shape[2] == 1):
            video[i] = cv2.cvtColor(video[i], cv2.COLOR_GRAY2RGB)
        for j in range(len(kpts[i])):
            if trans_params[i][j] == []:
              continue
            kpt = kpts[i][j]
            point_left = ((int)(kpt.pt[0]), (int)(kpt.pt[1]))
            point_right = ((int)(kpt.pt[0]) + (int)(np.ceil(trans_params[i][j][0])), (int)(kpt.pt[1]) + (int)(np.ceil(trans_params[i][j][1])))
            
            if point_left != point_right:
                print(point_left, point_right, ' ----------------')
            #else:
                #print(point_left, point_right)
            #cv2.circle(video[i], point_left, 1, thickness = 1, color = (randint(0, 255), randint(0, 255), randint(0, 255)))
            mask = cv2.line(mask, point_left, point_right, thickness = 2, color = colors[i].tolist())
        
        print(mask.shape , ' -- ', video[i].shape)
        added = cv2.add(video[i].astype('u1'), mask.astype('u1'))    
        print('Flow', i)
        cv2.imwrite(file_name + str(i) + '.jpg', added)
   
    
def plot_match(video, kpts, trans_params, file_name = 'dbg/match_'):
    for i in range(len(video)): #plot kpts for each frame
        if len(img_fst.shape)==2 or (len(img_fst.shape)==3 and img_fst.shape[2] == 1):
            video[i] = cv2.cvtColor(video[i],cv2.COLOR_GRAY2RGB)
        for point in kpts[i]:
            cv2.circle(video[i], (point[1], point[0]), 2, thickness = 1, color = (randint(0, 255), randint(0, 255), randint(0, 255)))
    
    for i in range(len(video - 1)):
        if trans_params[i] == []:
            continue    
        img_final = np.hstack((video[i], video[i + 1]))
        for j in range(len(kpts[i])):
            kpt = ktps[i][j]
            point_left = ((int)(kpt.pt[0]), (int)(kpt.pt[1]))
            point_right = (kpt.pt[0] + trans_params[i][j][0], kpt.pt[1] + trans_params[i][j][1])
            
            rand_color = (randint(0, 255), randint(0, 255), randint(0, 255))
            cv2.line(img_final, point_left, point_right, thickness = 1, color = rand_color)
        print('Match', i)
        cv2.imwrite(file_name + str(i) + '.jpg', img_final)

#find keypoints 
def keyPoints(video, method = 'harris', harris_block_sz = 2, harris_aperture_sz = 3, harris_k = 0.04):
    kpts = []
    ind = 1
    for frame in video:
        if method == 'harris':
            empty_frame = np.zeros(frame.shape)
            kpt = cv2.cornerHarris(np.float32(frame), harris_block_sz, harris_aperture_sz, harris_k)
            empty_frame[kpt > 0.01 * kpt.max()] = 1
            #kpt = [cv2.KeyPoint(j, i, 1) if empty_frame[i, j] == 1 for i in range(frame.shape[0]) for j in range(frame.shape[1])]
            kpt = []
            
            for i in range((int)(frame.shape[0] * 0.25), (int)(frame.shape[0] * 0.75)):
                for j in range((int)(frame.shape[1] * 0.25), (int)(frame.shape[1] * 0.75)):
                    if empty_frame[i, j] == 1:
                      kpt.append(cv2.KeyPoint(j, i, 1))
            
        elif method == 'sift':
            sift = cv2.xfeatures2d.SIFT_create()
            kpt = sift.detect(frame.astype('u1'), None)
        else:
            exit('Error: Invalid method for keypoints')
        kpts.append(kpt)
        print(ind, ' --> ', len(kpt))
        #print(kpt)
        ind += 1
    return kpts

#compute U, V
def parameters(curr_img, next_img, curr_kpts, neigh_size = 15):
    Ix = cv2.Sobel(curr_img,cv2.CV_64F,1,0,ksize=3)
    Iy = cv2.Sobel(curr_img,cv2.CV_64F,0,1,ksize=3)
    It = next_img - curr_img
    
    opt_flow=[]
    for index in range(len(curr_kpts)):
        S_Ixx,S_Iyy,S_Ixy,S_Iyx = 0,0,0,0
        S_Ixt,S_Iyt = 0,0
        row = int(curr_kpts[index].pt[1]) - (neigh_size-1)//2
        col = int(curr_kpts[index].pt[0]) - (neigh_size-1)//2
        space = (neigh_size-1)//2
        for i in range(0,space):
            new_row = row + i
            for j in range(space):              
                new_col = col + j
                S_Ixx = S_Ixx + Ix[new_row][new_col] * Ix[new_row][new_col]
                S_Iyy = S_Iyy + Iy[new_row][new_col] * Iy[new_row][new_col]
                S_Ixy = S_Ixy + Ix[new_row][new_col] * Iy[new_row][new_col]
                S_Iyx = S_Iyx + Iy[new_row][new_col] * Ix[new_row][new_col]
                S_Ixt = S_Ixt + Ix[new_row][new_col] * It[new_row][new_col]
                S_Iyt = S_Iyt + Iy[new_row][new_col] * It[new_row][new_col]
        X = np.array([[S_Ixx, S_Ixy], [S_Iyx, S_Iyy]])
        #print('--> ', np.linalg.eig(X)[0])
        Y = -1 * np.array([[S_Ixt],[S_Iyt]])
        x_transpose = np.matrix.transpose(X)
        A = np.dot(x_transpose, X)
        if np.linalg.det(A) == 0:
          print('Points', curr_kpts[index].pt, 'are not suitable for the transformation')
          opt_flow.append([])
        else:
          A = np.dot(np.linalg.inv(A), np.dot(x_transpose, Y))
          if A[0]**2 + A[1]**2 > 25:
            opt_flow.append([])
          else:
            opt_flow.append(A)
    return opt_flow
   
def computeTransformation(video, kpts, neigh_size = 15):
    trans_params = [] 
    for i in range(len(video) - 1):
        trans_params.append(parameters(video[i], video[i+1], kpts[i], neigh_size))
    return trans_params

#Structure from motion 
def structureFromMotion(video, kpts, trans_params):
    print('con fe!')

if __name__ == '__main__':
    file_name = 'input/p2-1-1.avi'
    neigh_size = 3
    kpts_method = 'harris'
    frame_per_sec = 30
     
    video = load_video(file_name, frame_per_sec)
    video = video[:120]
    kpts = keyPoints(video, kpts_method)
    trans_params = computeTransformation(video, kpts, neigh_size)
    
    plot_flow(video, kpts, trans_params)
    #structureFromMotion(video, kpts, trans_params)
