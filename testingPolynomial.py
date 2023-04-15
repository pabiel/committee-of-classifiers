import matplotlib.pyplot as plt
import random
data = [random.randint(1, 100) for _ in range(100)]
plt.hist(data, edge)

plt.show()