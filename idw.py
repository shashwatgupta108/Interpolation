import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


def interpolation(Xc, Yc, V, Xi, Yi, w, r=1, c=1):
    if r == 1 and c == 1:
        value = 0
        Sum = 0
        weight = 0
        for k in range(len(Xc)):
            d = np.sqrt(((Xi - Xc[k]) ** 2) + ((Yi - Yc[k]) ** 2))
            Sum = Sum + V[k] * (d ** w)
            weight = weight + (d ** w)
            value = Sum / weight
        return value
    else:
        values = np.zeros([r, c], dtype=float)
        for i in range(r):
            for j in range(c):
                Sum = 0
                weight = 0
                for k in range(len(Xc)):
                    d = np.sqrt(((Xi[i] - Xc[k]) ** 2) + ((Yi[j] - Yc[k]) ** 2))
                    Sum = Sum + V[k] * (d ** w)
                    weight = weight + (d ** w)
                    values[i][j] = Sum / weight
        return values


# Example
data = pd.read_csv("D:\Interpolation\data.csv")
x_coordinates = data['LON']
y_coordinates = data['LAT']
rainfall = data['ANN']
# Xmin = float(input("Enter minimum X coordinate: "))
# Xmax = float(input("Enter maximum X coordinate: "))
# Ymin = float(input("Enter minimum Y coordinate: "))
# Ymax = float(input("Enter maximum Y coordinate: "))
# r = int(input("Rows: "))
# c = int(input("Columns: "))
# Xi = np.linspace(Xmin, Xmax, r)
# Yi = np.linspace(Ymin, Ymax, c)
# w = -2

## Split, train and test
X_train, X_test, Y_train, Y_test, rainfall_train, rainfall_test = train_test_split(x_coordinates, y_coordinates,
                                                                                   rainfall, test_size=0.2)
X_train = list(X_train)
Y_train = list(Y_train)
rainfall_train = list(rainfall_train)
Vi = interpolation(X_train, Y_train, rainfall_train, X_test, Y_test, -2)
error = abs(rainfall_test-Vi)
accuracy = (1-((error)/rainfall_test))*100
print(accuracy)
print(Vi)
print(rainfall_test)

# Vi = interpolation(x_coordinates, y_coordinates, rainfall, Xi, Yi, -2, r, c)
# Vi = np.transpose(Vi)
# print(Vi)
# plt.imshow(Vi,origin='lower')
# plt.colorbar()
# plt.jet()
# plt.show()
