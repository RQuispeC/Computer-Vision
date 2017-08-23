import numpy as np
import cv2
import convolution
import scipy as sp

class gaussian_pyramid:
    pyramid = []              
    def __init__(self, img, levels, kernel_a = 0.3):
        self.img = img
        self.levels = levels
        self.kernel = convolution.create_kernel(kernel_a)
    
    def interpolation(self,x,y,v, interp = 'bilinear'):
        if interp=='bilinear':  
            if (x[0]==x[1]==x[2]==x[3]) or (y[0]==y[1]==y[2]==y[3]):
                return v[0]
            r1=(((x[1]-x[4])/(x[1]-x[0]))*v[2]+(((x[4]-x[0])/(x[1]-x[0])))*v[3])
            r2=(((x[1]-x[4])/(x[1]-x[0]))*v[0]+(((x[4]-x[0])/(x[1]-x[0])))*v[1])
            p=(((y[1]-y[4])/(y[0]-y[2]))*r1+(((y[4]-y[2])/(y[0]-y[2])))*r2)
            return p
        return 1

    def up_sample(self, image, size, interp = 'bilinear'):
        new_image=np.empty(size)
        for i in range(size[0]):
            for j in range(size[1]):
                if(interp == 'bilinear'):
                    i_min = min(i//2, image.shape[0]-1)
                    j_min = min(j//2, image.shape[1]-1)
                    i_lim = min(i_min +1, image.shape[0]-1)
                    j_lim = min(j_min +1, image.shape[1]-1)
                    y = [i_min, i_min, i_lim, i_lim, (i_min + i_lim)/2.0]
                    x = [j_min, j_lim, j_min, j_lim, (j_min + j_lim)/2.0]
                    z = [image[i_min, j_min], image[i_min, j_lim], image[i_lim, j_min], image[i_lim, j_lim]]
                    new_image[i, j] = self.interpolation(x, y, z, interp)
        return new_image
    
    def down(self, level): #upsample
        axis_x=self.pyramid[level].shape[0] * 2
        axis_y=self.pyramid[level].shape[1] * 2
        if len(self.img.shape) == 3 :
            new_image = np.empty((axis_x,axis_y, 3))
            new_image[:, :, 0] = self.up_sample(self.pyramid[level][:, :, 0],(axis_x,axis_y))
            new_image[:, :, 1] = self.up_sample(self.pyramid[level][:, :, 1],(axis_x,axis_y))
            new_image[:, :, 2] = self.up_sample(self.pyramid[level][:, :, 2],(axis_x,axis_y))
            return new_image
        else:
            new_image = self.up_sample(self.pyramid[level],(axis_x,axis_y))
            return new_image
        
    def down_sample(self, image,size):
        new_image=np.empty(size)
        filter_image=convolution.convolve(image,self.kernel,normalize=False)

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
        exit('Parameter error: invalid gaussian level')
        
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
    cv2.imwrite("juan.jpg",pyramide.down(1))
