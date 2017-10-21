import numpy as np
import cv2
import image_descriptor as img_des
import os
import utils
import random
import pickle

#returns an image with clusters
top_3=[]
def cluster(img, K=4, max_iters=10, img_name = '', plot = False):
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

    if plot:    
        center = np.uint8(center)
        res = center[label.flatten()]
        res2 = res.reshape((img.shape))
        cv2.imwrite('output/' + img_name + '_clust.jpg', res2)

    return label

#return a vector of vectors(1 for each cluster)
def conectedComponents(img, img_clustered, size_threshold = 300, img_name = '', plot = False):
    visited = np.zeros(img_clustered.shape)
    components = []
    ac = 0
    img_rgb=np.zeros(img.shape)
    for i in range(img_clustered.shape[0]):
        for j in range(img_clustered.shape[1]):
            if visited[i, j] == 1:
                continue

            color=[random.randint(0,255),random.randint(0,255),random.randint(0,255)]
            comp, visited = utils.bfs(img_clustered, i, j, img_clustered[i, j], visited)
            if len(comp) <= size_threshold:
                continue
            for pixel in comp:
                img_rgb[pixel[0],pixel[1]]=color;
            components.append(comp)
            ac += len(comp)
    if plot:    
        cv2.imwrite('output/' + img_name + '_comp.jpg', img_rgb)
    return components


#builts data if not exiss
def builtStructure(K = [2,4,8], features = [], save_filename = 'input/data.npz', overwrite = False):
    npz_file_names = []
    img_file_names = []
    img_names = os.listdir('input/')
    img_names.sort()
    alreadyCompt = True
    print("building files with features of each component......")
    for img_name in img_names:
        if img_name[len(img_name) - 4:] != '.jpg':
            continue
        img_file_names.append('input/' + img_name)
        npz_file_name = 'input/' + img_name + '.ob1'
        npz_file_names.append(npz_file_name)
        if not os.path.exists(npz_file_name):
            alreadyCompt = False
        
    #check if data exists
    if not alreadyCompt or overwrite:
        #img_names = img_names[:3]
        for img_name, npz_name in zip(img_file_names, npz_file_names):
            print("creating file ",npz_name,"...")
            img = cv2.imread(img_name)
            for K_ in K:
                img_name_= img_name[6:img_name.find('-')-3]+'_K-'+str(K_)
		        #clusterize
                clusters = cluster(img, K_, img_name = img_name_, plot=True)
		        #find conected components
                components = conectedComponents(img, clusters, img_name = img_name_, plot=True)
                    
            print('Nro components : ', len(components))
            #built
            des = img_des.imageDescriptor(img, components, features)
            #save data
            pickle.dump(des, open(npz_name, "wb"))
        
    return npz_file_names

def loadNpzFile(filename):
    data = pickle.load(open(filename, "rb"))
    return data

#return matches
def findMatch(query, data_structure, features_use):
    distance = []
    left = loadNpzFile(query)
    for data_name in data_structure.ravel():
        distance.append((img_des.similarity(left, loadNpzFile(data_name),features_use,distance_metric = 'l2-norm'), data_name,))
        
    distance.sort()
    name_img = query[6:query.find('.')+4]
    name_match=[name_img]
    for element in distance[:3]:
        name_match.append(element[1][6:element[1].find('.jpg')+4])
    top_3.append(name_match)
    return distance

def metrics(distances, npz_file_names, k = 5):
    #source: https://stats.stackexchange.com/questions/127041/mean-average-precision-vs-mean-reciprocal-rank
    position_match = []
    for name, distance in zip(npz_file_names, distances):
        new_positions = []
        idx = 1;
        name = name[0:name.find('_')]
        for pos in distance:
            if name == pos[1][0:pos[1].find('_')]:
                new_positions.append(idx)
            idx += 1
        position_match.append([name[6:len(name)],new_positions])

    #print(position_match)

    # source: https://en.wikipedia.org/wiki/Mean_reciprocal_rank
    print('********** Mean Reciprocal Rank **********')
    MRR=0.0;
    for element in position_match:
        #print(element[0]+" : {0:2f}".format(1.0/element[1][0]))
        MRR += 1.0/element[1][0]
    print('MRR : {0:2f}'.format(MRR/len(position_match)))

    # source: https://en.wikipedia.org/wiki/Information_retrieval#Mean_average_precision
    print('********** Mean Average Precision **********')
    MAP=0.0;
    print('Average Precision : ')
    for element in position_match:
        AP = 0.0
        number_relevant_document = 0.0
        for idx in element[1]:
            if idx < k:
                number_relevant_document += 1
                AP += number_relevant_document/idx
        if number_relevant_document != 0 :
            AP = AP/number_relevant_document
        else:
            AP = 0
        #print(element[0]+" : {0:2f}".format(AP))
        MAP += AP
    print('MAP : {0:2f}'.format(MAP/len(position_match)))

if __name__ == '__main__':
    features = ['region_size', 'mean_color', 'contrast', 'correlation', 'entropy', 'centroid', 'bound_box']
    #features_to_use= [[0,1,2,3,4,5,6],[1,2,3,4,5,6],[0,1,2,3,4,5],[1,2,3,4,5],[1,2,3,4]]
    features_to_use= [[1,2,3,4,5,6]]
    K = [2,4,8]
    save_filename = 'input/data'
    print('***********************************************************')
    print('***********************************************************\n')
    #built structure for different K's
    npz_file_names = builtStructure(K, features , save_filename, overwrite = True)
    #querie the s   tructure
    for features_use in features_to_use:
        print('\nfeatures => ',features[min(features_use): max(features_use)])
        i = 0
        distances=[]
        for npz_file in npz_file_names:
            print('find match for '+npz_file[6:npz_file.find('.')+4]+'...')
            distance = findMatch(npz_file, np.append(npz_file_names[:i], npz_file_names[i+1:]),features_use)
            distances.append(distance)        
            i += 1
        metrics(distances, npz_file_names, k=5)
    
    # Write top 3 for each test
    print('writing file the top 3 results of querying.....')
    file_ = open('output/top_3.txt','w')
    file_.write('#format: query ===> (first) (second) (third)\n\n')
    for top_3_element in top_3:
        new_line = top_3_element[0]+' ===> '
        for element in top_3_element[1:]:
            new_line = new_line + ' ('+element+') '
        file_.write(new_line+'\n')
    file_.close()   
            
