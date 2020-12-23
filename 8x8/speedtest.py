'''A speedtest of the efficiency of different functions with a control feature, 
and an error estimite assuming poission distribtuion.'''

import tensorflow as tf
import time
import numpy as np
from nptrain import is_win
import random
ops = 1e8
    
''' Do not declare functions if not absolutely necessary'''
x = np.random.randint(0,2, size=(8,8,2))


k = 0
cstart = time.time_ns()
while k < ops:
    k += 1
    '''Control Code'''
cend = time.time_ns()

k = 0    
start = time.time_ns()
while k < ops:
    k += 1
    '''Test Code'''
    y = is_win(x)
    

end = time.time_ns()

print(u'Operation Time: {:.4f}\u03bcs/op'.format((end - start) / (1e3 * ops)))
print(u'Control Time: {:.4f}\u03bcs/op'.format((cend - cstart) / (1e3 * ops)))
print()
op_rate = (1e9 * ops) / (end - start)
ctrl_rate = (1e9 * ops) / (cend - cstart)
count_rate = 1e9 * ops / ((end - start) - (cend - cstart))
print(u'Net Operation Time: {:.4f}\u03bcs/op'.format((end - start) / (1e3 * ops) - (cend - cstart) / (1e3 * ops)))
print('Operation Rate: ' + str(int(count_rate)) + u' \u00B1 ' + '{:.1f}ops/s'.format(np.sqrt(1e9 * (op_rate / (end - start) * op_rate / (end - start) + ctrl_rate / (cend - cstart) * ctrl_rate / (cend - cstart)))))