#!/usr/bin/python

import matplotlib.pyplot as plt
from prep_terrain_data import makeTerrainData
from class_vis import prettyPicture

features_train, labels_train, features_test, labels_test = makeTerrainData()


### the training data (features_train, labels_train) have both "fast" and "slow"
### points mixed together--separate them so we can give them different colors
### in the scatterplot and identify them visually
grade_fast = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==0]
bumpy_fast = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==0]
grade_slow = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==1]
bumpy_slow = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==1]


#### initial visualization
plt.xlim(0.0, 1.0)
plt.ylim(0.0, 1.0)
plt.scatter(bumpy_fast, grade_fast, color = "b", label="fast")
plt.scatter(grade_slow, bumpy_slow, color = "r", label="slow")
plt.legend()
plt.xlabel("bumpiness")
plt.ylabel("grade")
plt.show()
################################################################################

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score
import numpy as np
from time import time

#Random Forest
clf = RandomForestClassifier(oob_score = True, n_jobs = 20)

t0 = time()
clf.fit(features_train, labels_train)
print "Random Forest training time:", round(time()-t0, 3), "s"

t0 = time()
pred = clf.predict(features_test)
print "Random Forest prediction time:", round(time()-t0, 3), "s"

accuracy = accuracy_score(labels_test, pred)
print {"Random Forest Accuracy":round(accuracy,3)}

#Adaboost
a_clf = AdaBoostClassifier(learning_rate = 1)

t0 = time()
a_clf.fit(features_train, labels_train)
print "Adaboost training time:", round(time()-t0, 3), "s"

t0 = time()
a_pred = a_clf.predict(features_test)
print "Adaboost prediction time:", round(time()-t0, 3), "s"

a_accuracy = accuracy_score(labels_test, a_pred)
print {"Adaboost Accuracy":round(a_accuracy,3)}


### your code here!  name your classifier object clf if you want the 
### visualization code (prettyPicture) to show you the decision boundary

# try:
#     prettyPicture(clf, features_test, labels_test)
# except NameError:
#     pass
