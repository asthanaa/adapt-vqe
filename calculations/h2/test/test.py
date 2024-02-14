import numpy as np
a,b=np.triu_indices(4)
for idx, _ in enumerate(a):
    print(a[idx],b[idx])
