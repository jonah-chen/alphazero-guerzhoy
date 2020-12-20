from concurrent.futures import ProcessPoolExecutor

import numpy as np


if __name__ == '__main__':
    x = np.zeros((1000,3,3,))
    for i in range(1000):
        x[i,0,0] = 1
    
    x[1] = np.rot90(x[1], 1)
    print(x[1])
        
