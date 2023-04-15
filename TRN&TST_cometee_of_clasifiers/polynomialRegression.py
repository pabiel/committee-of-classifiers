import numpy
import matplotlib.pyplot as plt
import random
from sklearn.metrics import r2_score

x = [1, 2, 3, 5, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 18, 19, 21, 22]
y = [100, 90, 80, 60, 60, 55, 60, 65, 70, 70, 75, 76, 78, 79, 90, 99, 99, 100]
mymodel = numpy.poly1d(numpy.polyfit(x, y, 3))
myline = numpy.linspace(1, 22, 100)

plt.scatter(x, y)
plt.plot(myline, mymodel(myline))
plt.show()

tab_x = []
for i in range(100):
    x = round(random.uniform(0, 22), 2)
    tab_x.append(x)
tab_x.sort()
print(tab_x)

# tab.append(3.5)
# tab.append(4.5)
# tab.append(5.5)
# tab.append(6.5)

tab_y = []
for i in range(100):
    tab_y.append(mymodel(tab_x[i]))
print(tab_y)

# print(r2_score(y, mymodel(x)))
# print('mymodel(3.5) =', mymodel(3.5))
# print('mymodel(4.5) =', mymodel(4.5))
# print('mymodel(5.5) =', mymodel(5.5))
# print('mymodel(6.5) =', mymodel(6.5))
# mymodel.coef


plt.scatter(tab_x, tab_y)
plt.plot(myline, mymodel(myline))
# plt.plot(tab_x, tab_y)
plt.show()
