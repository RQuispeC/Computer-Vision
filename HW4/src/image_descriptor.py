import numpy as np
import cv2
from skimage.feature import greycomatrix
from skimage.feature import greycoprops
import random
from scipy.spatial.distance import cosine
from scipy.spatial.distance import hamming

class imageDescriptor():
    def __init__(self, img, components, features = []):
        self.img = img
        self.components = components
        self.masks = self.builtGrayMasks()
        self.color_components = self.builtColorComponents()
        self.descriptor = self.builtDescriptor(features)
        self.components = None
        self.masks = None
        self.color_components = None

        
    def builtColorComponents(self):
        color_comp = []
        i = 0
        for component in self.components:
            i+= 1
            comp = []
            for pixel in component:
                comp.append(self.img[pixel[0], pixel[1]])
            color_comp.append(np.array(comp))
        return color_comp

    def builtGrayMasks(self):
        masks = []
        gray_img = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        for component in self.components:
            mask = np.zeros((self.img.shape[0], self.img.shape[1])).astype(int)
            for pixel in component:
                mask[pixel[0], pixel[1]] = gray_img[pixel[0], pixel[1]]
            masks.append(mask)
        return masks
        
    def builtDescriptor(self, features):
        descriptor = []
        for color_component, component, mask in zip(self.color_components, self.components, self.masks):
            des_comp = []
            for feature in features:
                if feature  == 'region_size':
                    des_comp.append(self.region_size(component))
                elif feature == 'mean_color':
                    des_comp.append(self.mean_color(color_component))
                elif feature == 'contrast': 
                    des_comp.append(self.contrast(mask))
                elif feature == 'correlation':
                    des_comp.append(self.correlation(mask))
                elif feature == 'entropy':
                    des_comp.append(self.entropy(mask))
                elif feature == 'centroid':
                    des_comp.append(self.centroid(mask))
                elif feature == 'bound_box':
                    des_comp.append(self.bounding_box(component))
                else:
                    exit('Error: not valid feature ' + feature)
            descriptor.append(des_comp)
        return descriptor
        
    def region_size(self, component):
        return len(component)
 
    def mean_color(self, component):
        return np.mean(component, 0)
    
    def contrast(self, mask):
        comat = greycomatrix(mask, [1], [0, np.pi/2], levels = 256, normed = True, symmetric = True)
        return greycoprops(comat, 'contrast')
    
    def correlation(self, mask):
        comat = greycomatrix(mask, [1], [0, np.pi/2], levels = 256, normed = True, symmetric = True)
        return greycoprops(comat, 'correlation')
        
    def entropy(self, mask):
        comat = greycomatrix(mask, [1], [0, np.pi/2], levels = 256)
        count, _ = np.histogram(comat.ravel(), bins = np.max(comat) + 1, density = True)
        entropy = 0
        count = np.array(count)
        for p in count:
            if p != 0:
                entropy -= p * np.log2(p)
        return entropy
        
    def centroid(self, mask):
        ret,thresh = cv2.threshold(mask.astype(np.uint8), 1, 255, cv2.THRESH_BINARY)

        ind = (int)(random.random() * 100000)
        ''' 
        cv2.imwrite('dbg/centroid_'+ str(ind) +'.jpg', mask.astype(np.uint8))
        cv2.imwrite('dbg/thre_'+ str(ind) +'.jpg', thresh.astype(np.uint8))
        '''
        im2, contours, hierarchy = cv2.findContours(thresh, 1, 2)
        cnt = contours[0]
        M = cv2.moments(cnt)
        if M['m00'] == 0:
            #print(ind, 'mask without centroid')
            return []
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])
        return [cy, cx]
    
    def bounding_box(self, component):
        vect = np.min(component, 0) - np.max(component, 0)
        return np.linalg.norm(vect)


def similarity(left, right, features_to_use, distance_metric = 'l2-norm'):
    tot_distance = 0
    best_comp_match = []
    for left_component in left.descriptor:
        distances = []
        i = 0
        for right_component in right.descriptor:
            distance_sum = 0
            ac_dist = []
            for ind in features_to_use:
                left_elem, right_elem = left_component[ind], right_component[ind]
                if left_elem == [] or right_elem == []:
                    continue
                left_elem = np.array(left_elem)
                right_elem = np.array(right_elem)
                if distance_metric == 'l1-norm':
                    diff = abs(left_elem -  right_elem)
                    distance = (diff * diff).sum()
                elif distance_metric == 'l2-norm':
                    distance = np.linalg.norm(left_elem - right_elem).sum()
                elif distance_metric == 'chebyshev':
                    diff = abs(left_elem - right_elem)
                    distance = diff.max()
                elif distance_metric == 'cosine':
                    distance = cosine(left_elem, right_elem).sum()
                elif distance_metric == 'hamming':
                    distance = hamming(left_elem, right_elem).sum()
                else:
                    exit('Error: invalid distance metric')
                distance_sum += distance
                ac_dist.append(distance)
            ''' 
            print('+', left_component)
            print('-', right_component)
            print(distance_sum) 
            '''
            distances.append((distance_sum, i, ac_dist))
            i += 1
        distances.sort()
        tot_distance += np.min(distances[0][0])
        best_comp_match.append(distances[0][2])
    return tot_distance
    
