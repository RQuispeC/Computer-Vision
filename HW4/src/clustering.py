import numpy
import cv2
import sklearn
import imageDescriptor
import os
import utils

#returns an image with clusters
def clusterize(img, K, max_iters):


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


