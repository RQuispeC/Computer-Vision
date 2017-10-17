import numpy as np
import cv2
import sklearn
import image_descriptor as img_des
import os
import utils

#returns an image with clusters
def cluster(img, K=4, max_iters=10):
    shape_img=img.shape;
    kernel = np.ones((3,3),np.float32)/9
    img = cv2.filter2D(img,-1,kernel)
    #img = cv2.blur(img,(5,5))
    Z = img.reshape((-1,3))
    # convert to np.float32
    Z = np.float32(Z)
    # define criteria, number of clusters(K) and apply kmeans()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, max_iters, 1.0)
    ret,label,center=cv2.kmeans(Z,K, None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)

    label=label.reshape((shape_img[0],shape_img[1]))
    return label

#return a vector of vectors(1 for each cluster)
def conectedComponents(img, img_clustered):
    visited = np.zeros(img_clustered.shape)
    components = []
    ac = 0
    for i in range(img_clustered.shape[0]):
        for j in range(img_clustered.shape[1]):
            if visited[i, j] == 1:
                continue
            comp, visited = utils.bfs(img_clustered, i, j, img_clustered[i, j], visited)
            components.append(comp)
            ac += len(comp)
    return components


#builts data if not exiss
def builtStructure(K = 10, features = [], save_filename = 'input/data.npz', overwrite = False):
    #check if data exists
    if os.path.exists(save_filename) and not overwrite:
        npz_file = np.load(outfile)
        data = npz_file['arr_0']
        features = npz_file['arr_1']
    else:
        img_names = os.listdir('input/')
        img_names.sort()
        #img_names = img_names[:3]
        data = []
        for img_name in img_names:
            if img_name[len(img_name) - 4:] != '.jpg':
                continue
            img = cv2.imread('input/' + img_name)
            #clusterize
            clusters = cluster(img, K = K)
            #find conected components
            components = conectedComponents(img, clusters)
            #built
            data.append(img_des.imageDescriptor(img, components, features))
        #save data
        np.savez(save_filename, data, features)
    return data, features
   
#return matches
def findMatch(query, data_structure):
    return 1

if __name__ == '__main__':
    features = ['region_size', 'mean_color', 'contrast', 'correlation', 'entropy', 'centroid', 'bound_box']
    K = 10
    save_filename = 'input/data.npz'
    
    #built structure
    data, features = builtStructure(K, features, save_filename)
    #querie the structure 


