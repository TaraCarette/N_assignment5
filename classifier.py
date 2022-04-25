from sklearn.ensemble import AdaBoostClassifier
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve

wine = load_wine()

X = wine.data
y = wine.target
print(len(X))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

train_sizes, train_scores, valid_scores = learning_curve(AdaBoostClassifier(n_estimators=50, learning_rate=1), X, y, train_sizes=[0.2], shuffle=True, cv=20)
print(train_scores)
print(valid_scores)


plt.plot(range(1, len(valid_scores[0]) + 1), valid_scores[0])
plt.ylim(0,1.2)
plt.show()

