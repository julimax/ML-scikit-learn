import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
     

data = load_diabetes()
     

print(data.data.shape)
print(data.target.shape)
     
(442, 10)
(442,)

data.feature_names
     
['age', 'sex', 'bmi', 'bp', 's1', 's2', 's3', 's4', 's5', 's6']

print(data.data)

print(data.target)

x_train, x_test, y_train, y_test = train_test_split(data.data, data.target)
     

clf = KNeighborsRegressor()
     

clf.fit(x_train, y_train)
     
KNeighborsRegressor()

y_pred = clf.predict(x_test)
     

df_y_test = pd.DataFrame(y_test, columns=['y_test'])
df_y_pred = pd.DataFrame(y_pred, columns=['y_pred'])
pd.concat([df_y_test, df_y_pred], axis=1)

plt.plot(y_test)
plt.plot(y_pred)
plt.grid()
plt.xlabel('x')
plt.ylabel('y')
plt.legend(['y_test', 'predicciones'])
plt.show()
     
mean_squared_error(y_test, y_pred)