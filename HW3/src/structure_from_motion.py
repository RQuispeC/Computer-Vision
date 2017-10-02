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
        if curr_kpts[index][0]==-100:
            opt_flow.append([])
            continue
        S_Ixx,S_Iyy,S_Ixy,S_Iyx = 0,0,0,0
        S_Ixt,S_Iyt = 0,0
        row = curr_kpts[index][1] - (neigh_size-1)//2
        col = curr_kpts[index][0] - (neigh_size-1)//2
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
                    kpt.append([j, i])
    elif method == 'sift':
        sift = cv2.xfeatures2d.SIFT_create()
        points = sift.detect(frame.astype('u1'), None)
        for i in range(len(points)):
            if points[i].pt[0]>=frame.shape[1] * 0.25 and points[i].pt[0]<frame.shape[1] * 0.75 and points[i].pt[1]>=frame.shape[0] * 0.25 and points[i].pt[1]<frame.shape[0] * 0.75:
                kpt.append([points[i].pt[0],points[i].pt[1]]) 
    else:
        exit('Error: Invalid method for keypoints')
        
    return kpt

def validate_points(key_points,status,level=1):
    new_keypoints=[]
    factor = pow(2,level)
    for index in range(0,len(key_points)):
        aux=[]
        for i in range(0, len(key_points[index])):
            if status[i]==True:
                aux.append(np.array([factor*key_points[index][i][0], factor*key_points[index][i][1]]))
        new_keypoints.append(np.array(aux))
        
    return np.array(new_keypoints)
    
def optical_flow(video, color, level=2,max_keypoints=100, file_name = 'dbg/flow_',kpts_method = 'sift',):
    keypoints=[]
    pyramid = obtaining_pyramid(video,level)
    kpts = interest_point(pyramid[0], kpts_method)
    kpts = kpts[0:min(max_keypoints,len(kpts)-1)]
    status=[True for i in range(len(kpts))]
    mask = np.zeros((video[0].shape[0], video[0].shape[1], 3))
    for index in range(1, len(pyramid)):
        keypoints.append(np.array(kpts))
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
                if kpts[i][0]<0 or kpts[i][0]>=frame.shape[0] or kpts[i][1]<0 or kpts[i][1]>=frame.shape[1]:
                    status[i]=False
                    kpts[i]=[-100,-100]
            else :
                status[i]=False
                kpts[i]=[-100,-100]
        mask_prev=mask.copy()
        img = cv2.add(frame.astype('u1'),mask.astype('u1'))
        cv2.imwrite(file_name + str(index) + '.jpg', img)
        
    return validate_points(keypoints,status,level)
    
def obtaining_pyramid(video,level=1):
    pyramid=[]
    kernel = np.ones((5,5),np.float32)/25
    for frame in video:
        for index in range(0,level):
            frame = cv2.resize(frame, (frame.shape[1]//2, frame.shape[0]//2))
        frame = cv2.filter2D(frame,-1,kernel)
        pyramid.append(frame)    
    return pyramid

def g_t(a_f, b_f):
    return np.array([a_f[0]*b_f[0], a_f[0]*b_f[1] + a_f[1]*b_f[0], a_f[0]*b_f[2] + a_f[2]*b_f[0], a_f[1]*b_f[1], a_f[1]*b_f[2] + a_f[2]*b_f[1], a_f[2]*b_f[2] ])

ply_header = '''ply
format ascii 1.0
element vertex %(vert_num)d
property float x
property float y
property float z
property uchar blue
property uchar green
property uchar red
end_header
'''

def write_ply(fn, verts, colors):
    verts = verts.reshape(-1, 3)
    colors = colors.reshape(-1, 3)
    print(verts.shape, colors.shape)
    verts = np.hstack([verts, colors])
    with open(fn, 'wb') as f:
        f.write((ply_header % dict(vert_num=len(verts))).encode('utf-8'))
        np.savetxt(f, verts, fmt='%f %f %f %d %d %d ')

def structure_from_motion(kpts):
    P = len(kpts[0])
    F = len(kpts)
    #compute w
    X = []
    Y = []
    for frame in kpts:
        centroit = np.array([(float)(frame[:, 0].sum()), (float)(frame[:, 1].sum())])/ P
        if X == []:
            X = np.array(frame[:, 0] - centroit[0])
            Y = np.array(frame[:, 1] - centroit[1])
        else:
            X = np.vstack((X, np.array(frame[:, 0] - centroit[0])))
            Y = np.vstack((Y, np.array(frame[:, 1] - centroit[1])))
    W = np.vstack((X, Y))
    
    #apply SVD
    U, SIG, V_T = np.linalg.svd(W)
    U=U[:,:3]

    SIG=np.diag(SIG[:3])
    V_T=V_T[:,V_T.shape[1]-3:]
    V_T = np.matrix.transpose(V_T)

    
    M_hat = U
    S_hat = np.dot(SIG,V_T)
    #compute A
    c = np.append(np.ones(2 * F), np.zeros(F))
    Gii = []
    Gjj = []
    Gij = []
    for i in range(F):
        if Gii == []:
            Gii = g_t(M_hat[i], M_hat[i])
            Gjj = g_t(M_hat[i + F], M_hat[i + F])
            Gij = g_t(M_hat[i], M_hat[i + F])
        else:
            Gii = np.vstack((Gii, g_t(M_hat[i], M_hat[i])))
            Gjj = np.vstack((Gjj, g_t(M_hat[i + F], M_hat[i + F])))
            Gij = np.vstack((Gij, g_t(M_hat[i], M_hat[i + F])))
    G = np.vstack((Gii, Gjj, Gij))
    G_trans = np.matrix.transpose(G)
    G_t_G = np.dot(G_trans, G)
    if np.linalg.det(G_t_G) == 0:
          exit('Matrix G_t_G i not invertible')
    l = np.dot(np.linalg.inv(G_t_G), np.dot(G_trans, c))
    l = l.ravel()
    L = np.array([[l[0], l[1], l[2]], [l[1], l[3], l[4]], [l[2], l[4], l[5]]])
    A = np.linalg.cholesky(L) 
    M = np.dot(M_hat, A)
    S = np.dot(np.linalg.inv(A),S_hat)
    print(M.shape, S.shape)
    print(M)
    print(S)
    return M, S

if __name__ == '__main__':
    file_name = 'input/video3.mp4'
    neigh_size = 15
    kpts_method_ = 'Harris'
    frame_per_sec = 30
    video = load_video(file_name, frame_per_sec)[0:800]
    color = np.random.randint(0,255,(1000,3))    
    keypoints = optical_flow(video,color, max_keypoints=1000, level=0,kpts_method = kpts_method_)
    print(len(keypoints),color[:len(keypoints)].shape)
    M,S = structure_from_motion(keypoints)
    write_ply("keypoints.ply", S, color[:len(keypoints[0])])

