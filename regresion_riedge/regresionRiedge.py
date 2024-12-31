import pandas as pd
import matplotlib.pyplot as plt
from keras.datasets import boston_housing
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
     
(x_train, y_train), (x_test, y_test) = boston_housing.load_data()

print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

clf = Ridge()

clf.fit(x_train, y_train)

y_pred = clf.predict(x_test)

df_y_test = pd.DataFrame(y_test, columns=['y_test'])
df_y_pred = pd.DataFrame(y_pred, columns=['y_pred'])
pd.concat([df_y_test, df_y_pred], axis=1)

plt.plot(y_test)
plt.plot(y_pred)
plt.grid()
plt.xlabel('N_casa')
plt.ylabel('Precio')
plt.legend(['y_test', 'predicciones'])
plt.show()

mean_squared_error(y_test, y_pred)