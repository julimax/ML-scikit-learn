
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix

x, y = make_classification(n_samples = 200)

plt.scatter(x[:,0], x[:,1], c = y)
plt.grid()
plt.show()

x_train, x_test, y_train, y_test = train_test_split(x, y)

plt.scatter(x_train[:,0], x_train[:,1], c = y_train)
plt.grid()
plt.show()

plt.scatter(x_test[:,0], x_test[:,1])
plt.grid()
plt.show()

clf = KNeighborsClassifier()

clf.fit(x_train, y_train)

clf.score(x_test, y_test)
     
y_pred = clf.predict(x_test)
print(y_pred)

print(y_test)

matriz_confusion = confusion_matrix(y_test, y_pred)
print(matriz_confusion)

plt.scatter(x_train[:,0], x_train[:,1], c = y_train)
plt.scatter(x_test[2,0], x_test[2,1], s=120)
plt.grid()
plt.show()
     