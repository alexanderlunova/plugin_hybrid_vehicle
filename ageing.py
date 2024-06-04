import matplotlib.pyplot as plt
import numpy as np
import random


soc_ts = []
N = 100
for i in range(N):
    soc_ts.append(random.uniform(0.2,0.9))

soc_limits = [0.2, 0.9]
weights = np.ones_like(soc_ts)

axs = plt.plot()

plt.hist(soc_ts, bins = 25, weights = weights)
plt.axvline(soc_limits[0], color='k', linestyle='dashed', linewidth=1)
plt.axvline(soc_limits[1], color='g', linestyle='dashed', linewidth=1)
plt.show()