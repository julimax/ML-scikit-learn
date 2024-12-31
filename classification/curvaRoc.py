import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix
import seaborn as sns

x, y = make_moons(n_samples = 128)

plt.scatter(x[:,0], x[:,1], c=y)
plt.grid()
plt.show()

x_train, x_test, y_train, y_test = train_test_split(x, y)

clf = MLPClassifier()

clf.fit(x_train, y_train)

clf.score(x_test, y_test)

probabilidades = clf.predict_proba(x_test)
     
probabilidades=probabilidades[:, 1]

auc = roc_auc_score(y_test, probabilidades)
print("Area bajo la curva ROC",auc)

fpr, tpr, thresholds = roc_curve(y_test, probabilidades)

y_pred = clf.predict(x_test)
print(y_pred)

matriz_confusion = confusion_matrix(y_test, y_pred)
print("matriz_confusion:", matriz_confusion)

plt.plot(fpr, tpr, marker='.', label='MLP')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.grid()
plt.show()

# Plot confusion matrix
labels = ['Class 0', 'Class 1']
plt.figure(figsize=(10, 7))
sns.heatmap(matriz_confusion, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()