import numpy as np
import cv2
import utils
import scipy as sp
from scipy.interpolate import interp2d
class gaussian_pyramid:
    pyramid = []              
    def __init__(self, img, levels, kernel_a = 0.3):
        self.img = img
        self.levels = levels
        self.kernel = utils.create_kernel(kernel_a)
    
    def interpolation(self,x,y,v):
        return 0
        
    def up_sample(self, image, size):
        return 1
    
    def down(self, level): #upsample
		return 1
        
    def down_sample(self, image,size):
        new_image=np.empty(size)
        filter_image=utils.convolve(image,self.kernel,normalize=False)

        for i in range(0,(image.shape[0]-image.shape[0]%2),2):
            for j in range(0,(image.shape[1]-image.shape[1]%2),2):
                new_image[i/2,j/2] = filter_image[i,j]
        return new_image
              
    def up(self, level): #downsample
        axis_x=self.pyramid[level].shape[0]//2
        axis_y=self.pyramid[level].shape[1]//2
        if len(self.img.shape) == 3 :
            new_image = np.empty((axis_x,axis_y, 3))
            new_image[:, :, 0] = self.down_sample(self.pyramid[level][:, :, 0],(axis_x,axis_y))
            new_image[:, :, 1] = self.down_sample(self.pyramid[level][:, :, 1],(axis_x,axis_y))
            new_image[:, :, 2] = self.down_sample(self.pyramid[level][:, :, 2],(axis_x,axis_y))
            return new_image
        else:
            new_image = self.down_sample(self.pyramid[level],(axis_x,axis_y))
            return new_image
            
    def build(self):
        self.pyramid = [self.img]
        for i in range(self.levels - 1):
            self.pyramid.append(self.up(i))
    
    def get(self, level):
        if(level >=0 and level < self.levels):
            return self.pyramid[level]
        exit('Parameter error: invalid level')
        
    def show(self, name = 'gauss_pryramid'):
        for i in range(self.levels):
            print(self.get(i).shape)
            cv2.imwrite(name + str(i) + '.jpg', self.get(i))
    

if __name__ == "__main__":
    file_name_input = 'p0-1-0.jpg'
    image = cv2.imread(file_name_input,1) 
    pyramide=gaussian_pyramid(image,7)
    pyramide.build()
    pyramide.show()
    pyramide.down(0)
