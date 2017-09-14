import cv2
import numpy as np 
import matching
import transformation

def load_video(file_name = 'input/p2-1-1.mp4', fps = 30, original_fps = 30):
    if fps > original_fps or fps < 1:
        exit('Error: not valid frames per second')
    vidcap = cv2.VideoCapture(file_name)
    cnt = 0
    video = []
    video_grayscale = []
    success, img = vidcap.read()
    factor_fps = original_fps // fps
    while success:
        if cnt % factor_fps == 0:
            video.append(img)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            video_grayscale.append(img)
        cnt += 1
        success, img = vidcap.read()
    return video, video_grayscale

def apply_transformation(frame_rgb, frame_gray, params, trans_meth = 'affine'):
    rows, cols = frame_rgb.shape[:2]
    trans_frame_rgb = np.full(frame_rgb.shape, -1)
    trans_frame_gray = np.full(frame_gray.shape, -1)
    cnt = 0
    for y in range(rows):
        for x in range(cols):
            if trans_meth == 'affine':
                x_p = params[0] * x + params[1] * y + params[2]
                y_p = params[3] * x + params[4] * y + params[5]
            elif trans_meth == 'projective':
                den = params[6] * x + params[7] * y + 1
                if den == 0.0:
                    continue
                x_p = (params[0] * x + params[1] * y + params[2]) / den
                y_p = (params[3] * x + params[4] * y + params[5]) / den
            else:
                exit('Error apply_transformation: Invalid transformation')
            if x_p >= 0 and y_p >= 0 and x_p < cols and y_p < rows:
                cnt += 1
                y_p = (int)(y_p)
                x_p = (int)(x_p)
                trans_frame_rgb[y_p, x_p] = frame_rgb[y, x]
                trans_frame_gray[y_p, x_p] = frame_gray[y, x]

    #interpolate missing points    
    print(rows * cols, cnt)
    dx = [-1, 1, 1, -1]
    dy = [1, -1, 1, -1]
    for y in range(rows): #interpolate misssing points
        for x in range(cols):
            if trans_frame_gray[y, x] == -1:
                cnt = 0
                ac_rgb = [0, 0, 0]
                ac_gray = 0
                for xx, yy in zip(dx, dy):
                    if x + xx >= 0 and y + yy >= 0 and x + xx < cols and y + yy < rows and trans_frame_gray[y + yy, x + xx] != -1:
                        ac_rgb += trans_frame_rgb[y + yy, x + xx]
                        ac_gray += trans_frame_gray[y + yy, x + xx]
                        cnt += 1
                if cnt > 0:
                    trans_frame_gray[y, x] = ac_gray/cnt
                    trans_frame_rgb[y, x] = ac_rgb/cnt
                #else:
                    #print('INTERPOLATION FAILS')
                    
    trans_frame_rgb[trans_frame_rgb == -1] = 0
    trans_frame_gray[trans_frame_gray == -1] = 0
    return trans_frame_rgb.astype('u1'), trans_frame_gray.astype('u1')

def save_video(video, file_name = 'output/p2-1-1.avi', fps = 30, by_frame = False):
    if by_frame:
        cnt = 0
        for frame in video:
            cv2.imwrite((file_name + 'frame%d.jpg') % cnt, frame)
            cnt += 1

    else:
        fourcc = cv2.VideoWriter_fourcc(*'DIVX')
        out = cv2.VideoWriter(file_name, fourcc, fps, (video[0].shape[1], video[0].shape[0]))
        for frame in video:
            out.write(frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        out.release()

def joint_video(file_name_or, file_name_sta, save_file_name, fps):
    video_or, _ = load_video(file_name_or, fps = fps)
    video_sta, _ = load_video(file_name_sta, fps = fps)
    video_joint = []
    for left, right in zip(video_or, video_sta):
        img_final = np.hstack((left, right))
        video_joint.append(img_final)
    save_video(video_joint, save_file_name, fps = fps)

def stabilize(video_rgb, video, transformat = 'affine', save_name = 'new_changed_frame'):
    last = 0
    video_kpt = []
    for i in range(1, len(video)):
        kpt_cur, des_cur = matching.opencv_kpts_des(video[i], 'orb', 'sift')
        kpt_prev, des_prev = matching.opencv_kpts_des(video[i - 1], 'orb', 'sift')

        #kpt_cur, des_cur, kpt_prev, des_prev =  kpt_cur[200:250], des_cur[200:250], kpt_prev[200:250], des_prev[200:250]
        ''' 
        print('+++++++++++++++', i, '+++++++++++++++', len(kpt_cur), len(kpt_prev))
        left_cur, right_cur = max(len(kpt_cur)//2 - 25, 0), min(len(kpt_cur)//2 + 25, len(kpt_cur))
        left_prev, right_prev = max(len(kpt_prev)//2 - 25, 0), min(len(kpt_prev)//2 + 25, len(kpt_prev))
        kpt_cur = kpt_cur[left_cur: right_cur]
        des_cur = des_cur[left_cur: right_cur]
        kpt_prev = kpt_prev[left_prev: right_prev]
        des_prev = des_prev[left_prev: right_prev] '''
        #compute transformation frame to frame
        print('+++++++++++++++', i, '+++++++++++++++', len(kpt_cur), len(kpt_prev))
        if len(kpt_prev) == 0:
            print('WE COUNT FIND INTEREST POINTS AT FRAME ', i-1)
            exit()
        matches = matching.find_matches(des_cur, kpt_cur, des_prev, kpt_prev, hard_match = True, distance_metric = 'cosine', spacial_weighting = 0.0, threshold = 0.9)
        img_kpt = matching.joint_matches(video[i], kpt_cur, video[i-1], kpt_prev, matches, file_name = 'dbg/' + transformat + '_' + save_name + '_{}-{}.jpg'.format(i-1, i))
        video_kpt.append(img_kpt)
        print(i, len(matches), '-------------------------------------------------')
        best_fit, params = transformation.ransac(kpt_cur, kpt_prev, matches, threshold = 2, S = 140, transformation = transformat)
        if best_fit == []: #ransac has not reach a solution
            print('RANSAC CANNOT REACH AN INITIAL TRANSFORAMTION AT FRAME', i)
            best_fit, params = transformation.ransac(kpt_cur, kpt_prev, matches, threshold = 10, S = 140, transformation = transformat)
        if best_fit == []:
            print('RANSAC CANNOT REACH A TRANSFORAMTION MATRIX AT FRAME ', i)
            exit()
        print(i, len(best_fit))
        params_complete = transformation.least_square(kpt_cur, kpt_prev, matches, best_fit, transformation = transformat)
        if params_complete == []: #avoid runtime error
            params_complete = np.array(params)
        video_rgb[i], video[i] = apply_transformation(video_rgb[i], video[i], params_complete, trans_meth = transformat)
        #cv2.imwrite(('dbg/sol%d.jpg') % i, video_rgb[i])
    return video_rgb, video_kpt

if __name__ == '__main__':
    frame_per_sec = 10
    file_name = 'p2-1-0'
    transformat = 'affine' #projective or affine

    load_filename = 'input/' + file_name  + '.mp4'
    original_save_filename = 'output/' + transformat + '_' + file_name + '_ori.avi'
    stabilized_save_filename = 'output/' + transformat + '_' + file_name + '_sta.avi'
    joint_file_name = 'output/' + transformat + '_' + file_name + '_joint.avi'
    kpt_file_name = 'output/' + transformat + '_' + file_name + '_kpt.avi'

    video, video_grayscale = load_video(load_filename, fps = frame_per_sec) #original video has 30
    save_video(video, original_save_filename, fps = frame_per_sec)
    print('Video has ', len(video), 'frames')
    
    video, video_kpt = stabilize(video, video_grayscale, transformat, file_name)
    save_video(video, stabilized_save_filename, fps = frame_per_sec)
    
    #save_video(video, 'dbg/', by_frame = True)
    joint_video(original_save_filename, stabilized_save_filename, joint_file_name, frame_per_sec)
    save_video(video_kpt, kpt_file_name, fps = frame_per_sec)