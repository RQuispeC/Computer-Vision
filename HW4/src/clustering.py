import numpy
import cv2
import sklearn
import imageDescriptor
import os
import utils

#returns an image with clusters
def clusterize(img, K=4, max_iters=10):
    shape_img=img.shape;
    kernel = np.ones((3,3),np.float32)/9
    img = cv2.filter2D(img,-1,kernel)
    #img = cv2.blur(img,(5,5))
    Z = img.reshape((-1,3))
    # convert to np.float32
    Z = np.float32(Z)
    # define criteria, number of clusters(K) and apply kmeans()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, max_iters, 1.0)
    ret,label,center=cv2.kmeans(Z,K,criteria,10,cv2.KMEANS_RANDOM_CENTERS)

    label=label.reshape((shape_img[0],shape_img[1]))
    return label

#return a vector of vectors(1 for each cluster)
def conectedComponents(img, img_clustered):
    visited = np.zeros(img_clustered.shape)
    components = []
    for i in range(img_clustered.shape[0]):
        for j in range(img_clustered.shape[1]):
            if visited[i, j]:
                continue
            comp, visited = utils.bfs(img_clustered, i, j, img_clustered[i, j], img, visited)
            components.append(comp)
    return components

#save in ouutput directory
def saveData(data)

#builts data if not exiss
def builtStructure(imgs, clusters, over_write = False)
    #check if data exists
    #clusterize
    #find conected components
    #built
    #save data
   
#return matches
def findMatch(query, data_structure)

if __name__ == '__main__':
    #built structure
        
    #querie the structure 


