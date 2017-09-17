import cv2
import numpy as np

def load_video(file_name = 'input/p2-1-1.mp4'):
    vidcap = cv2.VideoCapture(file_name)
    cnt = 0
    video = []
    success, img = vidcap.read()
    while success:
        video.append(img)
        success, img = vidcap.read()
    return video

def save_video(video, file_name = 'output/p2-1-1.avi'):
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    out = cv2.VideoWriter(file_name, fourcc, 30.0, (video[0].shape[1], video[0].shape[0]))
    for frame in video:
        out.write(frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    out.release()

if __name__ == '__main__':
    file_name = 'p2-1-2'

    load_filename = 'input/' + file_name  + '.mp4'
    original_save_filename = 'output/' + file_name + '.avi'

    video = load_video(load_filename) #original video has 30
    
    #video = video[:310]

    save_video(video, original_save_filename)
