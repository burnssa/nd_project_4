#!/usr/bin/python


"""
    Starter code for the evaluation mini-project.
    Start by copying your trained/tested POI identifier from
    that which you built in the validation mini-project.

    This is the second step toward building your POI identifier!

    Start by loading/formatting the data...
"""

import pickle
import sys
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit
from sklearn.metrics import accuracy_score
from sklearn.cross_validation import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

data_dict = pickle.load(open("../final_project/final_project_dataset.pkl", "r") )

### add more features to features_list!
features_list = ["poi", "salary"]

data = featureFormat(data_dict, features_list)
labels, features = targetFeatureSplit(data)

feature_train, feature_test, label_train, label_test = train_test_split(features, labels, test_size = 0.3, random_state=42)

clf = DecisionTreeClassifier()
clf.fit(feature_train, label_train)
pred = clf.predict(feature_test)

pred_pois = [1 for el in pred if el == 1.]
act_pois = [1 for el in label_test if el == 1.]

print accuracy_score(label_test, pred)

print "This is the number of fingered POIs: {0}".format(len(pred_pois))
print "This is the number of total people in test set: {0}".format(len(pred))
print "This is the number of actual pois {0}".format(len(act_pois))

print(confusion_matrix(label_test, pred))
print(classification_report(label_test, pred))
print(precision_score(label_test, pred))


### your code goes here 


