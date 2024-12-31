from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
     
iris = load_iris()

x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target)

clf = LogisticRegression()
     
clf.fit(x_train, y_train)

score = clf.score(x_test, y_test)
     
print("score:", score)

y_pred = clf.predict(x_test)
     
matriz_confusion = confusion_matrix(y_test, y_pred)
print("matriz_confusion:",matriz_confusion)