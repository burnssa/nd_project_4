#!/usr/bin/python

import sys
import os
import pickle
sys.path.append("../tools/")

import matplotlib.pyplot as plt
from matplotlib import gridspec
from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data, test_classifier
from sklearn import linear_model, svm, tree, naive_bayes, ensemble
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.neighbors import KNeighborsClassifier
from poi_email_addresses import poiEmails
import numpy as np
import copy

###Utility functions
def get_feature_values(feature, dictionary):
	return [(float(d[feature]) if d[feature] != 'NaN' else 0.) for d in dictionary.values()]

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
poi_label = ['poi']
financial_features_list = ['salary', 'deferral_payments', 'deferred_income', 
'bonus', 'total_stock_value', 'expenses', 'exercised_stock_options', 'long_term_incentive',
'restricted_stock']
#Removed: 'director_fees', 'restricted_stock_deferred', 'loan_advances', 'total_payments'
email_features_list = ['to_messages', 'from_poi_to_this_person', 'from_messages', 'from_this_person_to_poi', 
'shared_receipt_with_poi'] 
features_list = poi_label + financial_features_list + email_features_list

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
   data_dict = pickle.load(data_file)

#Create new features - emails to/from pois as share of total emails (also part of Task 3)
for person in data_dict.keys():
	if data_dict[person]['from_messages'] not in ['NaN', 0]:
		data_dict[person]['percent_of_emails_from_poi'] = float(data_dict[person]['from_poi_to_this_person']) / float(data_dict[person]['from_messages'])
	else:
		data_dict[person]['percent_of_emails_from_poi'] = 'NaN'
for person in data_dict.keys():
	if data_dict[person]['to_messages'] not in ['NaN', 0]:
		data_dict[person]['percent_of_emails_to_poi'] = float(data_dict[person]['from_this_person_to_poi']) / float(data_dict[person]['to_messages'])
	else:
		data_dict[person]['percent_of_emails_to_poi'] = 'NaN'

#Including new percent features
features_list.extend(['percent_of_emails_to_poi', 'percent_of_emails_from_poi'])

### Task 2: Remove outliers
#First get a baseline of the initial number of datapoints in set, for each feature
print "Starting with {0} employees in dataset".format(len(data_dict))
pois = [(1 if d['poi'] == True else 0) for d in data_dict.values()]
print "... and {0} POIs".format(sum(pois))

#Remove the 'TOTAL' data_dict entry
data_dict.pop('TOTAL',0)

# ### Task 3: Create new feature(s)
#Scale training and test features
scaler = MinMaxScaler()
rescaled_data_dict = copy.deepcopy(data_dict)
for feature in features_list[1:]:
	feature_values = get_feature_values(feature, data_dict)
	feature_array = np.reshape(np.array(feature_values), (len(feature_values) ,1))
	rescaled_feature_array = scaler.fit_transform(feature_array)
	for i, person in enumerate(data_dict.keys()):
		rescaled_data_dict[person][feature] = rescaled_feature_array[i]
print "Rescaled_data_dict has {0} elements".format(len(rescaled_data_dict))

#Function to clean away the 10% of points that have the largest residual errors...
#...(different between the prediction and actual)
#return a list of tuples named cleaned_data where... 
#each tuple is of the form (salary, feature, error)
def outlierFinder(keys, x_features, y_features):
	reg = linear_model.LinearRegression()
	reg.fit(x_features, y_features)
	predictions = reg.predict(x_features)

  #Value to include x percentile of observations
	outlier_cutoff_value = 0.9
	residual_errors = np.abs(np.subtract(predictions, y_features))
	
	variables = zip(keys, residual_errors)
	sorted_errors = sorted(variables, key = lambda x: x[1])

	cleaner_cutoff = int(outlier_cutoff_value * len(sorted_errors) - 1)
	obs_to_remove = sorted_errors[cleaner_cutoff:]
	keys_to_remove = [k[0] for k in obs_to_remove]
	return keys_to_remove

#Trying various features for which to assess datapoint's outlier status
outlier_test_features = ['bonus', 'exercised_stock_options']
non_poi_outlier_dict = {}
poi_outlier_dict = {}
for feature in outlier_test_features:
	print "Removing outliers for {0}".format(feature)
	salaries = [(d['salary'] if d['salary'] != 'NaN' else 0) for d in rescaled_data_dict.values()]
	feature_values = get_feature_values(feature, rescaled_data_dict)
	keys = np.reshape( np.array(rescaled_data_dict.keys()), (len(rescaled_data_dict.keys()), 1))
	x_features = np.reshape( np.array(salaries), (len(salaries), 1))
	y_features = np.reshape( np.array(feature_values), (len(feature_values), 1))
	outliers = outlierFinder(keys, x_features, y_features)
	outlier_list = [outlier[0] for outlier in outliers]
	for outlier in outlier_list:
		if rescaled_data_dict[outlier]['poi'] == False:
			non_poi_outlier_dict[outlier] = rescaled_data_dict.pop(outlier,0)
		else:
			poi_outlier_dict[outlier] = rescaled_data_dict.pop(outlier,0)
	print "Rescaled_data_dict has {0} elements".format(len(rescaled_data_dict))

#Print scatterplots for financial feature vs salary to visually inspect for residual outliers
#Note - must skip first element of financial_features_list, which is salary
for index, feature in enumerate(financial_features_list[1:]):
	features = ['salary', feature]
	financial_data = featureFormat(data_dict, features)
	for point in financial_data:
		salary = point[0]
		fin_feature = point[1]
		plt.scatter( salary, fin_feature )

	plt.xlabel("salary")
	plt.ylabel(feature)
	plt.subplot(4, 3, index)
plt.show()

#Check the completeness of data for each feature
print "With a current data set of {0} enron employees".format(len(rescaled_data_dict))
for feature in features_list:
	actual_values = [(1 if d[feature] != [ 0.] else 0) for d in rescaled_data_dict.values()]
	print "There are {0} actual values for feature {1}".format(sum(actual_values), feature)

print "After removing outliers, we have {0} employees in the dataset".format(len(rescaled_data_dict))
pois = [(1 if d['poi'] == True else 0) for d in rescaled_data_dict.values()]
print "... and {0} POIs".format(sum(pois))

#Create feature: 'exclusive_poi_exchange' - all emails per user to or from a POI with no other recipients 
# print data_dict.keys()

###FIXME: Revisit when have more time - add features assessing non-cc emails to and from POIs, and text mentions of SPVs
# for path, dirs, files in os.walk('../maildir'):
# 	for file in files[0:2]:
# 		if file == ".DS_Store":
# 			continue
# 		filepath = os.path.join(path, file)
# 		with open(filepath, "r") as f:
# 			content = f.readlines()
# 			for c in content:
# 				if 'From:' in c:
# 					from_address = c.replace(" ", "").split(':')[1][:-2].split(',')[0]
# 					print from_address
# 					# if from_address in all_names:
# 					# 	print from_address[0]

# 					# if from_address in data_dict
# 					#print from_address
# 				# if 'To:' in c:
# 				# 	to_addresses = c.replace(" ", "").split(':')[1][:-2].split(',')
# 				# 	if len(to_addresses) == 1 and to_addresses[0] in poi_emails
					
# #			print to_emails

### Task 4: Try a variety of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

data = featureFormat(rescaled_data_dict, features_list, sort_keys = True)
poi_data = featureFormat(poi_outlier_dict, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)
poi_labels, poi_features = targetFeatureSplit(poi_data)

feature_train, feature_test, label_train, label_test = train_test_split(features, labels, test_size = 0.3, random_state=42)
feature_test = feature_test + poi_features
label_test = label_test + poi_labels

#Select the k-best features for training set, using outlier-free dataset
best_features = SelectKBest(chi2, 5)
best_feature_train = best_features.fit_transform(feature_train, label_train)
best_feature_test = best_features.transform(feature_test)
best_feature_set = best_features.transform(features) #Transform features in aggregate
print "These features have been selected:"
#Omit 'poi' from feature list to be displayed
best_feature_list = [features_list[1:][i] for i in best_features.get_support(indices = True)]
print best_feature_list

sv_clf = svm.SVC(kernel='linear', C=200)
nb_clf = naive_bayes.GaussianNB()
dt_clf = tree.DecisionTreeClassifier(min_samples_split=5)
ad_clf = ensemble.AdaBoostClassifier(n_estimators=10, random_state=42)
rf_clf = ensemble.RandomForestClassifier()
kn_clf = KNeighborsClassifier()

classifiers = [sv_clf, dt_clf, ad_clf, kn_clf, rf_clf, nb_clf] 

classifier_scores = {}
for classifier in classifiers:
	classifier.fit(best_feature_train, label_train)
	pred = classifier.predict(best_feature_test)

	print "This is the f1 score for a {0} classifier".format(type(classifier))
	classifier_scores[classifier] = f1_score(label_test, pred)
	print classifier_scores[classifier]
	
max_score = max(classifier_scores.values())

#Note - in final output, selecting AdaBoost classifier directly
clf = ad_clf
#clf = classifier_scores.keys()[classifier_scores.values().index(max_score)]
print clf

#Outputting the importances of the 'best features'
importances = clf.feature_importances_
indices = np.argsort(importances)[::-1][:len(best_feature_list)]
for f in range(len(best_feature_list)):
  print("%d. feature %s (%f)" % (f + 1, best_feature_list[indices[f]], importances[indices[f]]))

#Using GridSearchCV to find optimal AdaBoost classifier parameters
#Including only the best features
folds = 1000
cv = StratifiedShuffleSplit(
     labels, folds, random_state=42)
parameters = {'n_estimators':[10,20,30]}
grid = GridSearchCV(clf, parameters, cv = cv, scoring='f1')
grid.fit(best_feature_set, labels)

print("The best parameters are %s with a score of %0.2f"
      % (grid.best_params_, grid.best_score_))

## Task 5: Tune your classifier to achieve better than .3 precision and recall 
## using our testing script. Check the tester.py script in the final project
## folder for details on the evaluation method, especially the test_classifier
## function. Because of the small size of the dataset, the script uses
## stratified shuffle split cross validation. For more info: 
## http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

## Task 6: Dump your classifier, dataset, and features_list so anyone can
## check your results. You do not need to change anything below, but make sure
## that the version of poi_id.py that you submit can be run on its own and
## generates the necessary .pkl files for validating your results.

non_poi_outlier_data = featureFormat(non_poi_outlier_dict, features_list, sort_keys = True)
non_poi_labels, non_poi_features = targetFeatureSplit(non_poi_outlier_data)
best_non_poi_features = best_features.transform(non_poi_features)

true_negatives = 0
false_positives = 0

#Reusing some code from 'tester.py'
### test the classifier on non-POI outliers 
predictions = clf.predict(best_non_poi_features)
for prediction, truth in zip(predictions, non_poi_labels):
	if prediction == 0 and truth == 0:
		true_negatives += 1
	elif prediction == 1 and truth == 0:
		false_positives += 1
	else:
		print "Warning: Found a predicted label not == 0 or 1."
		print "All predictions should take value 0 or 1."
		print "Evaluating performance for processed predictions:"

total_predictions = true_negatives + false_positives
accuracy = 1.0*(true_negatives)/float(total_predictions)

print "Classifier correctly predicted non-POI status of {:.4f} of all non-POI outlier datapoints".format(accuracy, 2)
print "Total number of excluded outlier non-POIs: {0}".format(total_predictions)

#Adding back POIs to rescaled_data_dict
rescaled_data_dict.update(poi_outlier_dict)
#Adding back 'poi' to best_feature_list to ensure tester runs properly
best_feature_list.insert(0,'poi')

dump_classifier_and_data(clf, rescaled_data_dict, best_feature_list)
test_classifier(clf, rescaled_data_dict, best_feature_list)
