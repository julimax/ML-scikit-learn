import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, classification_report)

# x, y = make_moons(n_samples=200)
x, y = make_moons(n_samples=(150, 50))

plt.scatter(x[:,0], x[:,1], c = y)
plt.grid()
plt.show()
print(x)

x_train, x_test, y_train, y_test = train_test_split(x, y)

clf = MLPClassifier(max_iter=3500) 

clf.fit(x_train, y_train)

# Accuracy (TP + TN)/(TP + FP + TN + fN)
score = clf.score(x_test, y_test)

print("score:", score)


y_pred = clf.predict(x_test)
print(y_pred)

accuracy = accuracy_score(y_test, y_pred)
print("accuracy",accuracy)


matriz_confusion = confusion_matrix(y_test, y_pred)
print("matriz_confusion:", matriz_confusion)


precision = precision_score(y_test, y_pred)
print("precision:",precision)

recall = recall_score(y_test, y_pred)
print("recall:",recall)

f1 = f1_score(y_test, y_pred)
print("f1:",f1)

report = classification_report(y_test, y_pred)
print("repor:",report)



