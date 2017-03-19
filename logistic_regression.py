from sklearn import preprocessing  # import StandardScaler
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import numpy as np

X = [4.0,5.0,5.6,6.8,7.0,7.2,8.0,0.8,1.0,1.2,2.5,2.6,3.0,4.3]
X = np.matrix(X).transpose()
t = [1,1,1,1,1,1,1,0,0,0,0,0,0,0]
t = np.matrix(t).transpose()

standardized_X = preprocessing.scale(X)
print(standardized_X)

plt.scatter(standardized_X,t,color='green')
plt.show()

