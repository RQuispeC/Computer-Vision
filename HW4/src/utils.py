import queue
import numpy as np

dx = (0, 1, 0, -1, 1, -1, 1, -1)
dy = (1, 0, -1, 0, -1, 1, 1, -1)

def inside(i, j, bfsImg_shape):
    return i>=0 and j>=0 and i < bfsImg_shape[0] and j < bfsImg_shape[1]

def bfs(bfsImg, i, j, label, visited):
    q = []
    q.append([i, j])
    comp = []
    while q != []:
        cur = q.pop()
        visited[cur[0], cur[1]] = 1
        comp.append(cur)    
        for k in range(8):
            ni = cur[0] + dx[k]
            nj = cur[1] + dy[k]
            if inside(ni, nj, bfsImg.shape) and visited[ni, nj] == 0 and bfsImg[ni, nj] == label:
                visited[ni, nj] = 1
                q.append([ni, nj])
    comp = np.array(comp)
    return comp, visited


