from sklearn import preprocessing  # import StandardScaler
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model

# (a)
X = [4.0,5.0,5.6,6.8,7.0,7.2,8.0,0.8,1.0,1.2,2.5,2.6,3.0,4.3]
X = np.asarray(X).transpose()
t = [1,1,1,1,1,1,1,0,0,0,0,0,0,0]
t = np.asarray(t).transpose().reshape(len(t),1)

standardized_X = preprocessing.scale(X)

plt.scatter(standardized_X,t,color='green')

# (b)
xtest = np.arange(-1.5,1.5,0.1)
xtest_matrix = xtest.reshape(len(xtest),1)

# ridge regression
ridgeModel = linear_model.Ridge(alpha=0.1)
ridgeModel.fit(standardized_X.reshape(len(X),1),t)
predRidge = ridgeModel.predict(xtest_matrix)



# logistic regression L2 regurlization

lr = LogisticRegression(C=10)
lrModel = lr.fit(standardized_X.reshape(len(X),1),t)

predLog = lrModel.predict(xtest_matrix)
plt.plot(xtest,predRidge,'r')
plt.plot(xtest,predLog,'y')
plt.title('problem2 part(b) plot')
plt.show()

# (c)
new_a = [12]
new_a = preprocessing.scale(new_a)
new_a = np.asarray(new_a).transpose()
x_new = np.hstack((standardized_X,new_a))

new_b = [1]
new_b = np.asarray(new_b).transpose()
print('new_b--',new_b)
t_new = np.vstack((t,new_b))
print('t_new--',t_new)

### new ridge regression
ridgeModel2 = linear_model.Ridge(alpha=0.1)
ridgeModel2.fit(x_new.reshape(len(X)+1,1),t_new)

predRidge2 = ridgeModel2.predict(xtest_matrix)
plt.plot(xtest,predRidge2,'b')


# logistic regression L2 regurlization

lr2 = LogisticRegression(C=10)
lrModel2 = lr2.fit(x_new.reshape(len(X)+1,1),t_new)

predLog2 = lrModel2.predict(xtest_matrix)
plt.scatter(x_new,t_new,color='green')
plt.plot(xtest,predLog2,'g')
plt.title('problem2 part(c) plot')
plt.show()

