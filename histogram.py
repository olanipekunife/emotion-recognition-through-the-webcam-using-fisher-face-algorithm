import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
bins = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
metascore = [80, 82, 70, 60, 90, 30]
ids = [x for x in range(len(metascore))]
id = [x for x in range(len(bins))]
print("\n\nend score:", np.mean(metascore), "percent right!")
plt.xlabel('Percentage')
plt.ylabel('Accuracy')
plt.title('Accuracy(100%)')
plt.legend()
plt.show()
