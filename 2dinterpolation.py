import pylab as py
import numpy as np
from scipy.ndimage import map_coordinates
import gdal


def func(x, y):
    return (x + y) * np.exp(-5.0 * (x ** 2 + y ** 2))


def map_to_index(x, y, bounds, N, M):
    xmin, xmax, ymin, ymax = bounds
    i1 = (x - xmin) / (xmax - xmin) * N
    i2 = (y - ymin) / (ymax - ymin) * M
    return i1, i2


x, y = np.mgrid[-1:1:10j, -1:1:10j]
fvals = func(x, y)
xnew, ynew = np.mgrid[-1:1:100j, -1:1:100j]
i1, i2 = map_to_index(xnew, ynew, [-1, 1 - 1, 1], *x.shape)
fnew = map_coordinates(fvals, [i1, i2])
true = func(xnew, ynew)

# Create image plot
py.figure(1)
py.clf()
py.imshow(fvals, extent=[-1, 1, -1, 1], cmap=py.cm.jet)
py.figure(2)
py.clf()
py.imshow(fnew, extent=[-1, 1, -1, 1], cmap=py.cm.jet)
py.show()
