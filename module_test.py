import numpy as np
from build import bingocpp
a = np.array([[1, 4, 7], [2, 5, 8], [3, 6, 9]], dtype=float)
b = [(0, (0,)), 
     (0, (1,)), 
     (1, (0,)), 
     (1, (1,)), 
     (5, (3, 1)),
     (5, (3, 1)),
     (2, (4, 2)),
     (2, (4, 2)),
     (4, (6, 0)),
     (4, (5, 6)),
     (3, (7, 6)),
     (3, (8, 0))]
print(a)
print(bingocpp.evauluate(b,a, [3.14, 10.]))
print(bingocpp.evauluate_with_derivative(b,a, [3.14, 10.]))

