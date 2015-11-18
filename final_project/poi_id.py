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
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.neighbors import KNeighborsClassifier
from poi_email_addresses import poiEmails
import numpy as np
import copy

###Utility functions
def get_feature_values(feature):
	return [(float(d[feature]) if d[feature] != 'NaN' else 0.) for d in data_dict.values()]

def get_original_feature_values(feature):
	return [(float(d[feature]) if d[feature] != 'NaN' else 0.) for d in original_data_dict.values()]

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
poi_label = ['poi']
financial_features_list = ['salary', 'deferral_payments', 'deferred_income', 'total_payments', 
'bonus', 'total_stock_value', 'expenses', 'exercised_stock_options', 'long_term_incentive',
'restricted_stock']
#Removed: 'director_fees', 'other', 'restricted_stock_deferred', 'loan_advances'
email_features_list = ['to_messages', 'from_poi_to_this_person', 'from_messages', 'from_this_person_to_poi', 
'shared_receipt_with_poi'] 
features_list = poi_label + financial_features_list + email_features_list

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    original_data_dict = pickle.load(data_file)

#Create new features - emails to/from pois as share of total emails (also part of Task 3)
for person in original_data_dict.keys():
	if original_data_dict[person]['from_messages'] not in ['NaN', 0]:
		original_data_dict[person]['percent_of_emails_from_poi'] = float(original_data_dict[person]['from_poi_to_this_person']) / float(original_data_dict[person]['from_messages'])
	else:
		original_data_dict[person]['percent_of_emails_from_poi'] = 'NaN'
for person in original_data_dict.keys():
	if original_data_dict[person]['to_messages'] not in ['NaN', 0]:
		original_data_dict[person]['percent_of_emails_to_poi'] = float(original_data_dict[person]['from_this_person_to_poi']) / float(original_data_dict[person]['to_messages'])
	else:
		original_data_dict[person]['percent_of_emails_to_poi'] = 'NaN'

#Including new percent features
features_list.extend(['percent_of_emails_to_poi', 'percent_of_emails_from_poi'])
# features_list = [feature for feature in features_list if feature not in email_features_list]

### Task 2: Remove outliers
#First get a baseline of the initial number of datapoints in set, for each feature
print "Starting with {0} employees in dataset".format(len(original_data_dict))
pois = [(1 if d['poi'] == True else 0) for d in original_data_dict.values()]
print "... and {0} POIs".format(sum(pois))

#Remove the 'TOTAL' data_dict entry
original_data_dict.pop('TOTAL',0)

#Function to clean away the 5% of points that have the largest residual errors...
#...(different between the prediction and actual)
#return a list of tuples named cleaned_data where... 
#each tuple is of the form (salary, feature, error)
def outlierFinder(keys, x_features, y_features):
	reg = linear_model.LinearRegression()
	reg.fit(x_features, y_features)
	predictions = reg.predict(x_features)

  #Value to include x percentile of observations
	outlier_cutoff_value = 0.95
	residual_errors = np.abs(np.subtract(predictions, y_features))
	
	variables = zip(keys, residual_errors)
	sorted_errors = sorted(variables, key = lambda x: x[1])

	cleaner_cutoff = int(outlier_cutoff_value * len(sorted_errors) - 1)
	obs_to_remove = sorted_errors[cleaner_cutoff:]
	keys_to_remove = [k[0] for k in obs_to_remove]
	return keys_to_remove

#Trying various features for which to assess datapoint's outlier status
outlier_test_features = ['bonus', 'exercised_stock_options']
outlier_dict = {}
data_dict = copy.deepcopy(original_data_dict)
for feature in outlier_test_features:
	print "Removing outliers for {0}".format(feature)
	salaries = [(d['salary'] if d['salary'] != 'NaN' else 0) for d in data_dict.values()]
	feature_values = get_feature_values(feature)
	keys = np.reshape( np.array(data_dict.keys()), (len(data_dict.keys()), 1))
	x_features = np.reshape( np.array(salaries), (len(salaries), 1))
	y_features = np.reshape( np.array(feature_values), (len(feature_values), 1))
	outliers = outlierFinder(keys, x_features, y_features)
	outlier_list = [outlier[0] for outlier in outliers]
	for outlier in outlier_list:
		outlier_dict[outlier] = data_dict.pop(outlier,0)

#First print scatterplots for financial feature vs salary to visually inspect for residual outliers
#Note - must skip first element of financial_features_list, which is salary
# for index, feature in enumerate(financial_features_list[1:]):
# 	features = ['salary', feature]
# 	financial_data = featureFormat(data_dict, features)
# 	for point in financial_data:
# 		salary = point[0]
# 		fin_feature = point[1]
# 		plt.scatter( salary, fin_feature )

# 	plt.xlabel("salary")
# 	plt.ylabel(feature)
# 	plt.subplot(4, 3, index)
# plt.show()

#Check the completeness of data for each feature
print "With a current data set of {0} enron employees".format(len(data_dict))
for feature in features_list:
	actual_values = [(1 if d[feature] != 'NaN' else 0) for d in data_dict.values()]
	print "There are {0} actual values for feature {1}".format(sum(actual_values), feature)

print "After removing outliers, we have {0} employees in dataset".format(len(data_dict))
pois = [(1 if d['poi'] == True else 0) for d in data_dict.values()]
print "... and {0} POIs".format(sum(pois))

# ### Task 3: Create new feature(s)
#Scale training and test features
scaler = MinMaxScaler()
rescaled_data_dict = copy.deepcopy(data_dict)
#Rescale features with outlier-removed dataset
for feature in features_list[1:]:
	feature_values = get_feature_values(feature)
	feature_array = np.reshape(np.array(feature_values), (len(feature_values) ,1))
	rescaled_feature_array = scaler.fit_transform(feature_array)
	for i, person in enumerate(data_dict.keys()):
		rescaled_data_dict[person][feature] = rescaled_feature_array[i]

rescaled_original_data_dict = copy.deepcopy(original_data_dict)
#Rescale features with original dataset
for feature in features_list[1:]:
	orig_feature_values = get_original_feature_values(feature)
	orig_feature_array = np.reshape(np.array(orig_feature_values), (len(orig_feature_values) ,1))
	rescaled_orig_feature_array = scaler.fit_transform(orig_feature_array)
	for i, person in enumerate(original_data_dict.keys()):
		rescaled_original_data_dict[person][feature] = rescaled_orig_feature_array[i]

#Create dict of outliers scaled according to original data
rescaled_outlier_dict = {}
for person in outlier_dict.keys():
	rescaled_outlier_dict[person] = rescaled_original_data_dict[person]

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
outlier_data = featureFormat(rescaled_outlier_dict, features_list, sort_keys = True)

labels, features = targetFeatureSplit(data)
outlier_labels, outlier_features = targetFeatureSplit(outlier_data)

feature_train, feature_test, label_train, label_test = train_test_split(features, labels, test_size = 0.2, random_state=42)
feature_test = feature_test + outlier_features
label_test = label_test + outlier_labels

#Select the k-best features for training set, using outlier-free dataset
best_features = SelectKBest(chi2, 11)
best_feature_train = best_features.fit_transform(feature_train, label_train)
best_feature_test = best_features.transform(feature_test)
print "These features have been selected:"
best_features = [features_list[i] for i in best_features.get_support(indices = True)]
print best_features

sv_clf = svm.SVC(kernel='linear', C=1, random_state=42)
nb_clf = naive_bayes.GaussianNB()
dt_clf = tree.DecisionTreeClassifier(min_samples_split=5)
ad_clf = ensemble.AdaBoostClassifier(n_estimators=20)
rf_clf = ensemble.RandomForestClassifier()
kn_clf = KNeighborsClassifier(n_neighbors = 5)

classifiers = [sv_clf, dt_clf, ad_clf, kn_clf, nb_clf, rf_clf] 

classifier_scores = {}
for classifier in classifiers:
	classifier.fit(best_feature_train, label_train)
	pred = classifier.predict(best_feature_test)

	print "This is the f1 score for a {0} classifier".format(type(classifier))
	classifier_scores[classifier] = f1_score(label_test, pred)
	print classifier_scores[classifier]
	
max_score = max(classifier_scores.values())

clf = ad_clf
# clf = classifier_scores.keys()[classifier_scores.values().index(max_score)]
print clf

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, rescaled_original_data_dict, best_features)
test_classifier(clf, rescaled_original_data_dict, best_features)
