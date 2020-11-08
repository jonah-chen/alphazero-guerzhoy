import nptrain
import numpy as np

x = np.zeros((8,8,2,),dtype=np.float64)
x[6,7,0] = 1.0
x[0,0,1] = 1.0
x[0,1,1] = 1.0
x[0,2,1] = 1.0
x[0,3,1] = 1.0
x[0,4,1] = 1.0
print(nptrain.is_win(x))
