import numpy as np

def transpose(m, n_i, n_j):
    for i in range(n_i):
        for j in range(i+1, n_j):
            m[i*n_i+j], m[j*n_i+i] = m[j*n_i+i], m[i*n_i+i]
    return m

M = [1,2,3,4,5,6] 
n_i=3
n_j=2
print np.reshape(M, (n_i, n_j))
MT = transpose(M, n_i, n_j)
print np.reshape(MT, (n_j, n_i))
