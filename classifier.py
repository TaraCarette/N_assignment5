from sklearn.ensemble import AdaBoostClassifier
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn import metrics

wine = load_wine()

X = wine.data
y = wine.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

clf = AdaBoostClassifier(n_estimators=50, learning_rate=1)
model = clf.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))