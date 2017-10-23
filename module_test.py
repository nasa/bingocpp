import numpy as np
from build import bingocpp
a = np.ones((3,3), dtype=float)
b = [(1, (0,)), (0, (1,)), (2, (0, 1)), (2, (1, 2)), (2, (1, 2))]
print(bingocpp.evauluate(b,a, [3.3,]))

