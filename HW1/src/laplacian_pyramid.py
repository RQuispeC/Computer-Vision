import numpy as np
import cv2
import convolution
import scipy as sp
from scipy.interpolate import interp2d
import gaussian_pyramid as gp

class laplacian_pyramid:
    pyramid = []
    def __init__(self, img, levels, kernel_a = 0.3):
        self.img = img
        self.levels = levels
        self.gaussian_pyramid = gp.gaussian_pyramid(img, levels + 1, kernel_a)
    
    def diff_gauss(self, gauss_cur, gauss_down):
        if len(gauss_cur.shape) == 3:
            gauss_down[:, :, 0] = self.diff_gauss(gauss_cur[:, :, 0], gauss_down[:, :, 0])
            gauss_down[:, :, 1] = self.diff_gauss(gauss_cur[:, :, 1], gauss_down[:, :, 1])
            gauss_down[:, :, 2] = self.diff_gauss(gauss_cur[:, :, 2], gauss_down[:, :, 2])
            return gauss_down
        if gauss_cur.shape[0] != gauss_down.shape[0]:
                gauss_cur = gauss_cur[0:gauss_cur.shape[0]-1, :]
        if gauss_cur.shape[1] != gauss_down.shape[1]:
                gauss_cur = gauss_cur[:, 0:gauss_cur.shape[1]-1]
        return gauss_cur - gauss_down
    
    def build(self):
        self.pyramid = []
        self.gaussian_pyramid.build()
        for i in range(self.levels):
            print(i)
            gauss_cur = self.gaussian_pyramid.get(i)
            gauss_down = self.gaussian_pyramid.down(i + 1)           
            self.pyramid.append(self.diff_gauss(gauss_cur, gauss_down))
    
    def get(self, level):
        if(level >=0 and level < self.levels):
            return self.pyramid[level]
        exit('Parameter error: invalid level')
        
    def show(self, name = 'laplace_pyramid'):
        for i in range(self.levels):
            print(self.get(i).shape)
            cv2.imwrite(name + str(i) + '.jpg', self.get(i))
    

if __name__ == "__main__":
    file_name_input = 'p0-1-0.jpg'
    image = cv2.imread(file_name_input,0) 
    pyramide=laplacian_pyramid(image,6)
    pyramide.build()
    pyramide.show()
