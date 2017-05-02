# As with the previous exercises, let's look at the performance of a couple of classifiers
# on the familiar Titanic dataset. Add a train/test split, then store the results in the
# dictionary provided.

import numpy as np
import pandas as pd

# Load the dataset
X = pd.read_csv('titanic_data.csv')

X = X._get_numeric_data()
y = X['Survived']
del X['Age'], X['Survived']


from sklearn.tree import DecisionTreeClassifier
# from sklearn.metrics import recall_score as recall
# from sklearn.metrics import precision_score as precision
from sklearn.metrics import recall_score as f1_score
from sklearn.naive_bayes import GaussianNB

from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = train_test_split(X, y)

clf1 = DecisionTreeClassifier()
clf1.fit(features_train, labels_train)
# preds_r, preds_p = recall(labels_test, clf1.predict(features_test)),precision(labels_test, clf1.predict(features_test))
# print "Decision Tree recall: {:.2f} and precision: {:.2f}".format(preds_r, preds_p)
f1 = f1_score(labels_test, clf1.predict(features_test))
print "Decision Tree F1 score: {:.2f}".format(f1)

clf2 = GaussianNB()
clf2.fit(features_train, labels_train)
# preds_r2, preds_p2 = recall(labels_test, clf2.predict(features_test)),precision(labels_test, clf2.predict(features_test))
# print "GaussianNB recall: {:.2f} and precision: {:.2f}".format(preds_r2, preds_p2)
f12 = f1_score(labels_test, clf2.predict(features_test))
print "GaussianNB F1 score: {:.2f}".format(f12)

# results = {
#   "Naive Bayes Recall": 0,
#   "Naive Bayes Precision": 0,
#   "Decision Tree Recall": 0,
#   "Decision Tree Precision": 0
# }
F1_scores = {
 "Naive Bayes": f12,
 "Decision Tree": f1
}