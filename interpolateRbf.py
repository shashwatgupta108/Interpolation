import pylab as py
import numpy as np
from scipy import interpolate


def func(x):
    return x * np.exp(-5.0 * x ** 2)


x = np.random.uniform(-1.0, 1.0, size=10)
fvals = func(x)
py.figure(1)
py.clf()
py.plot(x, fvals, 'ro')
xnew = np.linspace(-1, 1, 100)

for kind in ['multiquadric', 'inverse multiquadric', 'gaussian', 'linear', 'cubic', 'quintic', 'thin-plate']:
    newfunc = interpolate.Rbf(x, fvals, function=kind)
    fnew = newfunc(xnew)
    py.plot(xnew, fnew, label=kind,linewidth = 1)

py.plot(xnew, func(xnew), label='ture',linewidth = 0.5)
py.legend()
py.show()
