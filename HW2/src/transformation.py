'''
References:
http://homepages.inf.ed.ac.uk/rbf/CVonline/LOCAL_COPIES/BEARDSLEY/node3.html
http://homepages.inf.ed.ac.uk/rbf/CVonline/LOCAL_COPIES/EPSRC_SSAZ/node11.html
'''
import numpy as np
def least_square(src, dst, matches, k_points, transformation = 'affine'):
    X = []
    for pnt in k_points: #create X, Y matrixes
        #print('--> ', matches[pnt])
        if transformation == 'affine':
            if X == []:
                X = np.array([src[pnt][1], src[pnt][0], 1, 0, 0, 0])
                X = np.vstack((X, np.array([0, 0, 0, src[pnt][1], src[pnt][0], 1])))
                Y = np.array([dst[matches[pnt]][1], dst[matches[pnt]][0]])
            else:
                X = np.vstack((X, np.array([src[pnt][1], src[pnt][0], 1, 0, 0, 0])))
                X = np.vstack((X, np.array([0, 0, 0, src[pnt][1], src[pnt][0], 1])))
                Y = np.append(Y, [dst[matches[pnt]][1], dst[matches[pnt]][0]])
        elif transformation == 'projective':
            if X == []:
                X = np.array([src[pnt][1], src[pnt][0], 1, 0, 0, 0, - src[pnt][1] * dst[matches[pnt]][1], - src[pnt][0] * dst[matches[pnt]][1]])
                X = np.vstack((X, np.array([0, 0, 0, src[pnt][1], src[pnt][0], 1, - src[pnt][1] * dst[matches[pnt]][0], - src[pnt][0] * dst[matches[pnt]][0]])))
                Y = np.array([dst[matches[pnt]][1], dst[matches[pnt]][0]])
            else:
                X = np.vstack((X, np.array([src[pnt][1], src[pnt][0], 1, 0, 0, 0, - src[pnt][1] * dst[matches[pnt]][1], - src[pnt][0] * dst[matches[pnt]][1]])))
                X = np.vstack((X, np.array([0, 0, 0, src[pnt][1], src[pnt][0], 1, - src[pnt][1] * dst[matches[pnt]][0], - src[pnt][0] * dst[matches[pnt]][0]])))
                Y = np.append(Y, [dst[matches[pnt]][1], dst[matches[pnt]][0]])
        else:
            exit('Error least_square: invalid transformation ')

    #print(X.shape, Y.shape)

    x_transpose = np.matrix.transpose(X)
    A = np.dot(x_transpose, X)
    if np.linalg.det(A) == 0:
        print('Points', k_points, 'are not suitable for the transformation')
        return []
    A = np.dot(np.linalg.inv(A), np.dot(x_transpose, Y))
    return A
    

def evaluate_transformation(src, dst, matches, trans_params, threshold = 1, evaluation_method = 'ramsac', transformation = 'affine'):
    correct_fit = []
    for i in range(len(src)):
        if transformation == 'affine':
            x_comp = src[i][1] * trans_params[0] + src[i][0] * trans_params[1] + trans_params[2]
            y_comp = src[i][1] * trans_params[3] + src[i][0] * trans_params[4] + trans_params[5]
        elif transformation == 'projective':
            den = src[i][1] * trans_params[6] + src[i][0] * trans_params[7] + 1
            if den == 0.0:
                continue
            x_comp = (src[i][1] * trans_params[0] + src[i][0] * trans_params[1] + trans_params[2]) / den
            y_comp = (src[i][1] * trans_params[3] + src[i][0] * trans_params[4] + trans_params[5]) / den
        else:
            exit('Error evaluate_transformation: Invalid transformation method')
        if evaluation_method == 'ramsac':
            error = abs(x_comp - dst[matches[i]][1]) + abs(y_comp - dst[matches[i]][0])
        else:
            exit('Error : Invalid evaluation method')
        # print(x_comp, y_comp, error)
        if  error <= threshold:
            correct_fit.append(i)
    return correct_fit

def ransac(src, dst, matches, k = 3, S = 35, threshold = 1, transformation = 'affine'):
    if transformation == 'affine':
        k = max(k, 3)
    elif transformation == 'projective':
        k = max(k, 4)
    else:
        exit('Error ransac: Invalid transformation method')
    best_fit = []
    best_params = []
    for i in range(S):
        ind_rand = np.arange(len(src))
        np.random.shuffle(ind_rand)
        ind_rand = ind_rand[:k]
        #print(ind_rand)
        trans_params = least_square(src, dst, matches, ind_rand, transformation = transformation)
        if trans_params == []:
            continue
        correct_fit = evaluate_transformation(src, dst, matches, trans_params, threshold, transformation = transformation)
        if len(correct_fit) > len(best_fit):
            best_fit = correct_fit
            best_params = trans_params
    # print('asdf', len(best_fit), cnt_kpt, best_params)
    return best_fit, best_params

