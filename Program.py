from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import datasets, linear_model

data = pd.read_csv("https://raw.githubusercontent.com/nnh92/Linear-regression-Gold/main/Sources/Data.csv").values

# Nam-Thang
x = data[:,0].reshape(-1,1)

# Chi so VNIndex
y1 = data[:,1].reshape(-1,1)

# Chi so VN30
y2 = data[:,2].reshape(-1,1)

# Gia cp MBB
y3 = data[:,3].reshape(-1,1)

# Gia vang
y4 = data[:,4].reshape(-1,1)

# Gia Bitcoine
y5 = data[:,5].reshape(-1,1)

plt.scatter(x,y1)
plt.xlabel('Time (month)')
plt.ylabel('Chi so')

x1 = np.hstack((x,np.ones((data.shape[0],1))))

regr = linear_model.LinearRegression()

regr.fit(x,y1)

y_pred = regr.predict()

w = np.array([regr.coef_[0],regr.intercept_])

#X = np.dot(x,w)

print(x)
print("w1 =", regr.coef_[0])
print("w0 =", regr.intercept_)

#plt.plot((x[0],y1[0]),(X[0],X[1]),"go")
plt.show()