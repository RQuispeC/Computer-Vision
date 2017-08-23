import cv2
import numpy as np
import gaussian_pyramid as gp
import laplacian_pyramid as lp

def comple(mask):
    print(np.ones(mask.shape) - mask,mask,mask.shape)
    return np.ones(mask.shape) - mask
    
def create_mask(image):
    return np.hstack((np.zeros((image.shape[0], image.shape[1]//2,3)), np.ones((image.shape[0], image.shape[1] - image.shape[1]//2,3))))

def up_sample(image, size):
    print(size)
    gp_mask=gp.gaussian_pyramid(image,1)
    image_ans=np.empty((size[0],size[1],3))
    image_ans[:,:,0]= gp_mask.up_sample(image[:,:,0], size)  
    image_ans[:,:,1]= gp_mask.up_sample(image[:,:,1], size)  
    image_ans[:,:,2]= gp_mask.up_sample(image[:,:,2], size)   
    return image_ans
    
def blending(img_a, img_b, mask,level = 2):
    lp_a= lp.laplacian_pyramid(img_a,level)
    lp_a.build()
    lp_b= lp.laplacian_pyramid(img_b,level)
    lp_b.build()
    gp_mask=gp.gaussian_pyramid(mask,level)
    gp_mask.build()
    
    c = lp_a.get(level-1).shape[1]
    mid = [np.hstack((lp_a.gaussian_pyramid.get(level-1)[:,0:c//2], lp_b.gaussian_pyramid.get(level-1)[:, c//2:]))]
    cv2.imwrite('b.jpg', mid[0])
    for i in range(level -1, 0, -1):
        c = lp_a.get(i-1).shape[1]
        joint = np.hstack((lp_a.get(i-1)[:,0:c//2], lp_b.get(i-1)[:, c//2:]))
        cv2.imwrite(str(i)+'b.jpg', joint)
        mid.append(joint)
    
    img_ans = mid[0]
    for i in range(1, len(mid)):
       cv2.imwrite(str(i)+'c.jpg', up_sample(img_ans, (mid[i].shape[0],mid[i].shape[1])))
       img_ans = lp_a.operation_gauss(up_sample(img_ans, (mid[i].shape[0],mid[i].shape[1])).astype(int), mid[i].astype(int), '+')
    return img_ans
    
if __name__ == "__main__":
    file_name_input1 = 'p0-1-0.jpg'
    file_name_input2 = 'imagen.jpg'
    image1 = cv2.imread(file_name_input1,1) 
    image2 = cv2.imread(file_name_input2,1)
    image2 = cv2.resize(image2, (image1.shape[1],image1.shape[0])) 
    mask = np.hstack((np.zeros((image1.shape[0], image1.shape[1]//2,3)), np.ones((image1.shape[0], image1.shape[1] - image1.shape[1]//2,3))))
    cv2.imwrite('blending.jpg', blending(image1, image2, mask, level = 6))
    
