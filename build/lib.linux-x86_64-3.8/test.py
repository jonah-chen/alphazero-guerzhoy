from nptrain import is_win, move_from_policy
import numpy as np

x = np.zeros((8,8,), dtype=float)
x[2,3] = 0.2
x[1,4] = 0.2
x[5,1] = 0.2
x[6,6] = 0.4
for _ in range(10):
    print(move_from_policy(x, np.random.rand()))