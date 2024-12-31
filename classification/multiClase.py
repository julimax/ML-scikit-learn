
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (accuracy_score, confusion_matrix, precision_score,
                             recall_score, f1_score, classification_report)
import matplotlib.pyplot as plt
import seaborn as sns

wine = load_wine()

x_train, x_test, y_train, y_test = train_test_split(wine.data, wine.target)

clf = MLPClassifier()

clf.fit(x_train, y_train)

clf.score(x_test, y_test)

y_pred = clf.predict(x_test)
print(y_pred)

accuracy = accuracy_score(y_test, y_pred)
print(accuracy)

matriz_confusion = confusion_matrix(y_test, y_pred)
print(matriz_confusion)

precision = precision_score(y_test, y_pred, average='macro')
print(precision)

recall = recall_score(y_test, y_pred, average='macro')
print(recall)

f1 = f1_score(y_test, y_pred, average='macro')
print(f1)

report = classification_report(y_test, y_pred)
print(report)


# Plot confusion matrix
plt.figure(figsize=(10, 7))
sns.heatmap(matriz_confusion, annot=True, fmt='d', cmap='Blues', xticklabels=wine.target_names, yticklabels=wine.target_names)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

# Plot accuracy, precision, recall, and f1-score
metrics = [accuracy, precision, recall_score(y_test, y_pred, average='macro'), f1_score(y_test, y_pred, average='macro')]
metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1 Score']

plt.figure(figsize=(10, 5))
plt.bar(metrics_names, metrics, color=['blue', 'green', 'red', 'purple'])
plt.ylim(0, 1)
plt.title('Classification Metrics')
plt.show()

# Plot the actual vs predicted values
plt.figure(figsize=(10, 5))
plt.scatter(range(len(y_test)), y_test, color='blue', label='Actual')
plt.scatter(range(len(y_pred)), y_pred, color='red', label='Predicted')
plt.xlabel('Sample Index')
plt.ylabel('Class')
plt.title('Actual vs Predicted Values')
plt.legend()
plt.show()