from sklearn.datasets import load_iris
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.metrics import plot_confusion_matrix
import matplotlib.pyplot as plt
from GetCOVID_19Data1 import get_data
from sklearn.metrics import plot_confusion_matrix
import matplotlib.pyplot as plt

x_train, x_test, y_train, y_test = get_data(percent=0.9)
clf = MLPClassifier(hidden_layer_sizes=(256, 128, 64, 32),
                    activation="relu", random_state=1).fit(x_train, y_train)
y_pred=clf.predict(x_test)
#print(clf.score(x_test, y_test))
scores = clf.score(x_test, y_test)
print('准确度: %.2f%%' % (100 * scores))
fig=plot_confusion_matrix(clf, x_test, y_test, display_labels=["Setosa", "Versicolor", "Virginica"])
fig.figure_.suptitle("Confusion Matrix for Iris Dataset")
plt.show()