import numpy as np

y=np.array([0,0,0,1,1])

for i in y:
    print(i)

    g = 1/(1+np.exp(-i))
    print(g)