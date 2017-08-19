import numpy as np
import cv2

def create_kernel(kernel_a):
    vect = np.array([1/4.0 - kernel_a/2.0, 1/4.0, kernel_a, 1/4.0, 1/4.0 - kernel_a/2.0])
    return np.outer(vect, np.transpose(vect))
        
def padding(img, padding_width, padding_type = 'zero', padding_color = 0):
    if padding_type == 'zero' or padding_type == 'constant':
        img = np.pad(img, padding_width, mode = 'constant', constant_values = padding_color)
    elif padding_type == 'mirror':
        img = np.pad(img, padding_width, mode = 'reflect')
    else:
        exit('Invalid argument: padding type')
    return img

def flip(kernel):
    return np.flip(np.flip(kernel, 0), 1)

def validate(img_shape, kernel_shape):
    if img_shape[0] < kernel_shape[0] or img_shape[1] < kernel_shape[1]:
        exit('Error: kernel must be smaller')
    if kernel_shape[0] % 2 == 0 or kernel_shape[1] % 2 == 0 or kernel_shape[0] != kernel_shape[1]:
        exit('Error: kernel dimensions must be odd and equal')
    return True

def norm(img):
    img /= img.max()
    img = (img * 255).astype("uint8")
    return img

def convolve(img_input, kernel, padding_type = 'zero', padding_color = 0, normalize = True):
    img_ans = img_input.copy()
    if len(img_input.shape) == 3 :

        img_input[:, :, 0] = convolve(img_input[:, :, 0], kernel, padding_type, padding_color, normalize)
        img_input[:, :, 1] = convolve(img_input[:, :, 1], kernel, padding_type, padding_color, normalize)
        img_input[:, :, 2] = convolve(img_input[:, :, 2], kernel, padding_type, padding_color, normalize)
        return img_input
    
    #    validate data
    validate(img_input.shape, kernel.shape)
    #   flit kernel
    kernel = flip(kernel)
    #    padding
    img = padding(img_input, kernel.shape[0]//2, padding_type, padding_color=padding_color)
    #    colvolve
    space = kernel.shape[0]//2
    for row in range(space,img_ans.shape[0]+space):
        for col in range(space, img_ans.shape[1]+space): 
            aux_matrix = img[row-space:row+space + 1,col-space:col+space + 1]
            new_value = (int)((aux_matrix*kernel).sum())
            img_ans[row-space,col-space]=new_value

    if normalize == True:
        img_ans = norm(img_ans)
    return img_ans

if __name__ == "__main__":
    file_name_input = 'p0-1-0.jpg'
    image = cv2.imread(file_name_input,1)
    kernel = np.array([[-1,0,1],
              [-2,0,2],
              [-1,0,1]]).astype(np.float)
  #  kernel=np.ones((5,5))/25
    ans=convolve(image,kernel,normalize=False)
    cv2.imwrite("3.jpg",ans)
