import numpy as np


partitionable_index = np.where(
    1921 % np.array([1,4,2,1]) == 0)
print('partitionable_index:', partitionable_index)