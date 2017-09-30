import cv2
import numpy as np
from random import randint

def load_video(file_name = 'input/p2-1-1.mp4', fps = 30, original_fps = 30):
    #kernel = np.ones((5,5),np.float32)/25
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
            #img = cv2.resize(img, (640, 360))
            #img = cv2.filter2D(img,-1,kernel)
            img = np.float32(img)
            video.append(img)
        cnt += 1
        success, img = vidcap.read()
    return video

#compute U, V
def interpolation(x, y, img, isfirst):
    if isfirst:
        return img[int(x)][int(y)]
        
    new_x = int(np.floor(x))
    new_y = int(np.floor(y))
    
    i = [new_x, new_x, new_x + 1, new_x + 1]
    j = [new_y, new_y + 1, new_y, new_y + 1]
    
    r1 =  ((j[3]-y)/(j[3]-j[2]))*img[i[2]][j[2]] +((y-j[2])/(j[3]-j[2]))*img[i[3]][j[3]]
    r2 =  ((j[3]-y)/(j[3]-j[2]))*img[i[0]][j[0]] +((y-j[2])/(j[3]-j[2]))*img[i[1]][j[1]]
    
    p = ((i[0]-x)/(i[0]-i[2]))*r1 + ((x-i[2])/(i[0]-i[2]))*r2   
    
    return p
    
def derivate(img, direct='X'):
    kernelX=np.array([[0,0,0],[-0.5,0,0.5],[0,0,0]])
    kernelY=np.array([[0,-0.5,0],[0,0,0],[0,0.5,0]])
    if direct=='X':
        return cv2.filter2D(img, -1, kernelX)
    else :
        return cv2.filter2D(img, -1, kernelY)

def parameters(curr_img, next_img, curr_kpts, neigh_size = 15, isfirst=True):
    Ix = derivate(curr_img, direct='X')
    Iy = derivate(curr_img, direct='Y')
    It = next_img.astype(int) - curr_img.astype(int)
    
    opt_flow=[]
    for index in range(len(curr_kpts)):
        if curr_kpts[index]==[]:
            opt_flow.append([])
            continue
        S_Ixx,S_Iyy,S_Ixy,S_Iyx = 0,0,0,0
        S_Ixt,S_Iyt = 0,0
        row = curr_kpts[index][0] - (neigh_size-1)//2
        col = curr_kpts[index][1] - (neigh_size-1)//2
        space = (neigh_size-1)//2   
        for i in range(0,neigh_size):
            new_row = row + i
            for j in range(0,neigh_size):          
                new_col = col + j
                if new_row<0 or new_col<0 or new_row>=curr_img.shape[0]-1 or new_col>=curr_img.shape[1]-1:
                    continue
                Ix_ = interpolation(new_row,new_col,Ix, isfirst)
                Iy_ = interpolation(new_row,new_col,Iy, isfirst)
                It_ = interpolation(new_row,new_col,It, isfirst)
                S_Ixx = S_Ixx + pow(Ix_,2)
                S_Iyy = S_Iyy + pow(Iy_,2)
                S_Ixy = S_Ixy + Ix_ * Iy_
                S_Iyx = S_Iyx + Ix_ * Iy_
                S_Ixt = S_Ixt + It_ * Ix_
                S_Iyt = S_Iyt + It_ * Iy_ 

        X = np.array([[S_Ixx, S_Ixy], [S_Iyx, S_Iyy]])
        #print('--> ', np.linalg.eig(X)[0])
        Y = -1 * np.array([[S_Ixt],[S_Iyt]])
        x_transpose = np.matrix.transpose(X)
        A = np.dot(x_transpose, X)
        if np.linalg.det(A) == 0:
          print('Points', curr_kpts[index] , 'are not suitable for the transformation')
          opt_flow.append([])
        else:
          A = np.dot(np.linalg.inv(A), np.dot(x_transpose, Y))
          A = np.array([A[0][0],A[1][0]])
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

#find keypoints 
def interest_point(frame, method='shiTomosi', harris_block_sz = 2, harris_aperture_sz = 3, harris_k = 0.04):
    kpt=[]
    if method == 'shiTomosi':
        # params for ShiTomasi corner detection
        feature_params = dict( maxCorners = 100, qualityLevel = 0.3, minDistance = 7, blockSize = 7 )
        kpt_aux = cv2.goodFeaturesToTrack(frame, mask = None, **feature_params)
        kpt = []
        for index in range(len(kpt_aux)):
            kpt.append([kpt_aux[index][0][0], kpt_aux[index][0][1]])
    elif method == 'Harris':
        empty_frame = np.zeros(frame.shape)
        kpt = cv2.cornerHarris(np.float32(frame), harris_block_sz, harris_aperture_sz, harris_k)
        empty_frame[kpt > 0.01 * kpt.max()] = 1
        kpt = []
        for i in range((int)(frame.shape[0] * 0.25), (int)(frame.shape[0] * 0.75)):
            for j in range((int)(frame.shape[1] * 0.25), (int)(frame.shape[1] * 0.75)):
                if empty_frame[i, j] == 1:
                    kpt.append([i, j])
    else:
        exit('Error: Invalid method for keypoints')
        
    return kpt

def validate_points(key_points):
    for index in range(0,len(key_points)):
        for i in range(0, len(key_points[index])):
            if key_points[index][i]==[]:
                for j in range(0, len(key_points)):
                    key_points[j][i]=[]
    
    new_keypoints=[]
    for index in range(0,len(key_points)):
        aux=[]
        for i in range(0, len(key_points[index])):
            if key_points[index][i]!=[]:
                aux.append(key_points[index][i])
        new_keypoints.append(aux)
        
    return new_keypoints
    
def optical_flow(video, level=2,max_keypoints=100, file_name = 'dbg/flow_'):
    keypoints=[]
    pyramid = obtaining_pyramid(video,level)
    kpts = interest_point(pyramid[0], kpts_method)[0:max_keypoints]
    mask = np.zeros((video[0].shape[0], video[0].shape[1], 3))
    color = np.random.randint(0,255,(max_keypoints,3))
    for index in range(1, len(pyramid)-1):
        keypoints.append(kpts)
        frame = video[index]
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        print("frame: ", index)
        if index==1:
            flow = parameters(pyramid[index-1],pyramid[index],kpts, neigh_size = 15, isfirst=True)
        else :
            flow = parameters(pyramid[index-1],pyramid[index],kpts, neigh_size = 15, isfirst=False)
            
        for i in range(0, min(max_keypoints,len(kpts))):
            if flow[i]!=[]:
                a,b = int(np.ceil(kpts[i][0]+flow[i][0])), int(np.ceil(kpts[i][1]+flow[i][1]))
                c,d = int(np.floor(kpts[i][0])), int(np.floor(kpts[i][1]))
                factor = pow(2,level)
                mask = cv2.line(mask, (factor*a,factor*b),(factor*c,factor*d), color[i].tolist(), 2)
                frame = cv2.circle(frame,(factor*a,factor*b),5,color[i].tolist(),-1)
                kpts[i][0] = kpts[i][0] + flow[i][0]
                kpts[i][1] = kpts[i][1] + flow[i][1]
            else :
                kpts[i]=[]
        mask_prev=mask.copy()
        img = cv2.add(frame.astype('u1'),mask.astype('u1'))
        cv2.imwrite(file_name + str(index) + '.jpg', img)
    return validate_points(keypoints)
    
def obtaining_pyramid(video,level=1):
    pyramid=[]
    kernel = np.ones((5,5),np.float32)/25
    for frame in video:
        for index in range(0,level):
            frame = cv2.resize(frame, (frame.shape[1]//2, frame.shape[0]//2))
        frame = cv2.filter2D(frame,-1,kernel)
        pyramid.append(frame)    
    return pyramid

if __name__ == '__main__':
    file_name = 'input/rodo3.mp4'
    neigh_size = 15
    kpts_method = 'shiTomosi'
    frame_per_sec = 30 
    video = load_video(file_name, frame_per_sec)
    print(len(video))
    a = optical_flow(video, level=1)
