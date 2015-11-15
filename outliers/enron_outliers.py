#!/usr/bin/python

import operator
import pickle
import sys
import matplotlib.pyplot
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit


### read in data dictionary, convert to numpy array
data_dict = pickle.load( open("../final_project/final_project_dataset.pkl", "r") )
data_dict.pop('TOTAL',0)
features = ["salary", "bonus"]
data = featureFormat(data_dict, features)

for point in data:
    salary = point[0]
    bonus = point[1]
    matplotlib.pyplot.scatter( salary, bonus )

matplotlib.pyplot.xlabel("salary")
matplotlib.pyplot.ylabel("bonus")
matplotlib.pyplot.show()


# max_salary

max_salary = max([(d['salary'] if d['salary'] != 'NaN' else 0) for d in data_dict.values()])
print max_salary

salaries = [(d['salary'] if d['salary'] != 'NaN' else 0) for d in data_dict.values()]

sorted_salaries = sorted(salaries)

max_salary = [d for d in data_dict.items() if d[1]['salary'] == max_salary]

top_real_salary = sorted_salaries[-1]
second_real_salary = sorted_salaries[-2]

top_name = [d for d in data_dict.items() if d[1]['salary'] == top_real_salary]
second_name = [d for d in data_dict.items() if d[1]['salary'] == second_real_salary]

print top_name
print second_name


### your code below



