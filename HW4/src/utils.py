import Queue
import numpy as np

dx = (0, 1, 0, -1, 1, -1, 1, -1)
dy = (1, 0, -1, 0, -1, 1, 1, -1)

def inside(i, j, bfsImg):
    return i>=0 and j>=0 and i < bfsImg.shape[0] and j < bfsImg.shape[1]

def bfs(bfsImg, i, j, label, img_data, visited):
    visited[i][j] = 1
    q = Queue.Queue(maxsize = 0)
    q.put((i, j))
    component = []
    while(not q.empty()):
        cur = q.get()
        visited[cur[0]][cur[1]] = 1
        if component == []:
            component = np.array(img_data[i, j])
        else:
            component = np.vstack((component, img_data[i, j]))

        for k in range(8):
            ni = cur[0] + dx[k]
            nj = cur[1] + dy[k]
            if(inside(ni, nj, bfsImg) and visited[ni][nj] == 0 and bfsImg[ni, nj] == label):
                visited[ni][nj] = 1
                q.put((ni, nj))
    return component, visited
    
