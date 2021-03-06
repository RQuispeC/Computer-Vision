import cv2
import numpy as np
import gaussian_pyramid as gp
import laplacian_pyramid as lp
import fourier

def inverse(mask, value = 255):
    return np.full(mask.shape, value) - mask

def change_dim(polar_img):
    ans = np.empty((polar_img.shape[1], polar_img.shape[2], 3))
    for i in range(polar_img.shape[1]):
      for j in range(polar_img.shape[2]):
        ans[i,j, :] = polar_img[:, i, j]
    return ans
     
def set_mask(mask_type = 'left_right', img_shape = None, value = 255, mask_img_filename = ''):
    if mask_type == 'left_right': #left and right blending mask
        mid = img_shape[1] // 2
        if len(img_shape) == 3:
            return np.hstack((np.zeros((img_shape[0], mid, 3)), np.full((img_shape[0], img_shape[1] - mid, 3), value)))
        else:
            return np.hstack((np.zeros((img_shape[0], mid)), np.full((img_shape[0], img_shape[1] - mid), value)))
    elif mask_type == 'bottom_up': #bottom up blending mask
        mid = img_shape[0] // 2
        if len(img_shape) == 3:
            return np.vstack((np.zeros((mid, img_shape[1], 3)), np.full((img_shape[0] - mid, img_shape[1], 3), value)))
        else:
            return np.vstack((np.zeros((mid, img_shape[1])), np.full((img_shape[0] - mid, img_shape[1]), value)))        
    elif mask_type == 'file': #load mask form an image
        mask = cv2.imread(mask_img_filename, 0)
        if len(img_shape)==3:
            tmp_mask=mask.copy()
            mask=[]
            mask.append(tmp_mask)
            mask.append(tmp_mask)
            mask.append(tmp_mask)
            mask=np.asarray(mask)
            mask=change_dim(mask)
        return np.asarray(mask);
    else:
        exit('Error in mask definition')

def combine(first_img, second_img, mask):
    return (first_img*mask)/255.0 + (second_img*inverse(mask))/255.0

def blending(img_a, img_b, mask, level = 2):
    lp_a = lp.laplacian_pyramid(img_a,level)
    lp_b = lp.laplacian_pyramid(img_b,level)
    gp_mask = gp.gaussian_pyramid(mask,level)
    lp_a.build()
    lp_b.build()
    gp_mask.build()
    
    mid = []
    for i in range(level - 1, -1, -1):
        mask = gp_mask.get(i)
        joint = combine(lp_a.get(i), lp_b.get(i), mask)
        mid.append(joint)
    
    img_blend = mid[0]
    for i in range(1, len(mid)):
       img_blend = lp_a.down(img_blend, mid[i])
    return img_blend
    
if __name__ == "__main__":
    image_names = ['input/p1-1-2.png', 'input/p1-1-3.png', 'input/p1-1-0.jpg', 'input/p1-1-1.jpg','input/p1-1-4.jpg','input/p1-1-5.jpg','input/p1-1-7.jpg']
    name_it = 0
    
    for i in range(3):
        image1 = cv2.imread(image_names[i], 1)
        image2 = cv2.imread(image_names[i+1], 1)
        image2 = cv2.resize(image2, (image1.shape[1],image1.shape[0]))
        mask = set_mask(mask_type = 'left_right', img_shape = image1.shape)
        cv2.imwrite('output/p1-2-4-' + str(name_it) + '.jpg', blending(image1, image2, mask, level = 6))
        print(image_names[i], image_names[i+1], 'output/p1-2-4-' + str(name_it) + '.jpg')
        name_it += 1
    
    image1 = cv2.imread(image_names[5], 1)
    image2 = cv2.imread(image_names[4], 1)
    image1 = cv2.resize(image1, (image2.shape[1],image2.shape[0]))
    mask = set_mask(mask_type='file', img_shape =image2.shape, mask_img_filename = image_names[6])
    cv2.imwrite('output/blending.jpg', blending(image1, image2, mask, level = 7))
