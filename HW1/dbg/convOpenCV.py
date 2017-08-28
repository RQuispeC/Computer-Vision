import numpy as np
import cv2
import time

def load_kernels():
    kernels = []
    #horizontal line detector
    kernels.append(np.array([[-1, -1, -1],
                             [2,   2,  2],
                             [-1, -1, -1]]).astype(np.float)/16.0)
    
    #gaussian
    kernels.append(np.array([[0, 0,   0,   5,   0,  0, 0],
                            [0,  5,  18,  32,  18,  5, 0],
                            [0, 18,  64, 100,  64, 18, 0],
                            [5, 32, 100, 100, 100, 32, 5],
                            [0, 18,  64, 100,  64, 18, 0],
                            [0,  5,  18,  32,  18,  5, 0],
                            [0,  0,   0,   5,   0,  0, 0]]).astype(np.float)/1068.0)
    #media
    kernels.append(np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]).astype(np.float)/225.0)                         
    #sobel
    kernels.append(np.array([[-1,0,1],
                            [-2,0,2],
                            [-1,0,1]]).astype(np.float))
    #laplacian of gaussian
    kernels.append(np.array([[0, 0, -1, 0, 0],
                            [0, -1, -2, -1, 0],
                            [-1, -2, 16, -2, -1],
                            [0, -1, -2, -1, 0],
                            [0, 0, -1, 0, 0]]).astype(np.float))
    return kernels

if __name__ == "__main__":
    image_names = ['input/p1-1-0.jpg', 'input/p1-1-1.jpg', 'input/p1-1-2.png', 'input/p1-1-3.png']
    kernels = load_kernels()
    for image_name in image_names:
        image = cv2.imread(image_name, 0)
        for kernel in kernels:
            start = time.time()
            ans=cv2.filter2D(image,-1,kernel)
            end = time.time()
            print('Kernel dimensions ' + str(kernel.shape[0])+'x'+str(kernel.shape[1]) + ', time execution: '+str(end-start)) 
