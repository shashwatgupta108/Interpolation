import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate

# basic 1-d interpolation
x = np.linspace(0, 10, 11)
y = np.sin(x)
plt.figure(1)
plt.clf()
plt.plot(x, y, 'ro')
xnew = np.linspace(0, 10, 100)
for kind in ['nearest', 'zero', 'linear', 'slinear', 'quadratic', 'cubic', ]:
    f = interpolate.interp1d(x, y, kind=kind)
    ynew = f(xnew)
    plt.plot(xnew, ynew, label=kind)

plt.legend()
plt.show()
