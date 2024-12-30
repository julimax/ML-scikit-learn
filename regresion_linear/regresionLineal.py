import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression

# Leer el CSV y mostrarlo
df = pd.read_csv('reduccion_datos.csv')
print(df)
 
x = df[["Reduccion de solidos"]]
y = df[["Reduccion demanda de oxigeno"]]

# grafica de datos
plt.scatter(x,y)
plt.xlabel("Reduccion de solidos")
plt.ylabel("Reduccion demanda de oxigeno")
plt.grid() 

# Convertir dataframe a numpy
matriz = df[["Reduccion de solidos", "Reduccion demanda de oxigeno"]].to_numpy()
print(matriz)

n = len(matriz)
suma_X = np.sum(matriz[:,0])
suma_y = np.sum(matriz[:,1])
suma_producto_yx = np.sum(matriz[:,0] * matriz[:,1])
suma_cuadrado_x = np.sum(matriz[:,0] * matriz[:,0])

print("n:", n, "sum_X:", suma_X, "sum_Y:", suma_y, "suma_producto_yx:", suma_producto_yx, "suma_cuadrado_x", suma_cuadrado_x)

# Calcular la pendiente b1
pendiente = (n * suma_producto_yx - suma_X * suma_y) / (n * suma_cuadrado_x - suma_X ** 2)

print("Pendiente:", pendiente)

# Calcular la intersección b0
interseccion = (suma_y - pendiente * suma_X) / n

print("Intersección:", interseccion)

recta_regresion = pendiente * matriz[:,0] + interseccion

plt.plot(matriz[:,0], recta_regresion, color="red")

# scikit-learm
clf = LinearRegression()

#entrenar el modelo
clf.fit(x, y)

# pendiente b1
clf.coef_ 

print("Pendiente:", clf.coef_)

# intersección b0
clf.intercept_

print("Intersección:", clf.intercept_)

# Predecir para x=" "
x_valor_prediccion = 5  # Variable para el valor de x
y_valor_prediccion = clf.predict(pd.DataFrame([[x_valor_prediccion]], columns=["Reduccion de solidos"]))

# Mostrar la predicción
print(f"Predicción para x={x_valor_prediccion}: {y_valor_prediccion}")

# Graficar la predicción
plt.scatter(x_valor_prediccion, y_valor_prediccion, color="purple", s=200, zorder=5)

