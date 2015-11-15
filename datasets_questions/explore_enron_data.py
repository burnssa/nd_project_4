#!/usr/bin/python

""" 
    Starter code for exploring the Enron dataset (emails + finances);
    loads up the dataset (pickled dict of dicts).

    The dataset has the form:
    enron_data["LASTNAME FIRSTNAME MIDDLEINITIAL"] = { features_dict }

    {features_dict} is a dictionary of features associated with that person.
    You should explore features_dict as part of the mini-project,
    but here's an example to get you started:

    enron_data["SKILLING JEFFREY K"]["bonus"] = 5600000
    
"""

import pickle

enron_data = pickle.load(open("../final_project/final_project_dataset.pkl", "r"))

#How many people in dataset?
print "Total people in dataset: {0}".format(len(enron_data)) 

#How many features for each person?
print "Number of features for each person: {0}".format(len(enron_data.itervalues().next()))

#How many persons of interests are defined in dataset?
pois = (1 for p in enron_data if enron_data[p]['poi'] == True)
print "Number of POIs: {0}".format(sum(pois)) 

#What was value of Enron stock did James Prentice own?
prentice_stock = enron_data["PRENTICE JAMES"]['total_stock_value']
print "James Prentice's total stock value: {0}".format(prentice_stock) 

#How many email messages do we have from Wesley Colwell to persons of interest?
colwell_to_pois = enron_data["COLWELL WESLEY"]['from_this_person_to_poi']
print "Wesley Colwell's messages to POIs: {0}".format(colwell_to_pois) 

#Value of stock options exercised by Jeffrey Skilling?
skilling_name = [p for p in enron_data if "SKILL" in p][0]
skilling_options = enron_data[skilling_name]['exercised_stock_options']
print "Jeff Skilling's exercised options: {0}".format(skilling_options) 

#Of top threee - who took home highest total payments
lay_name = [p for p in enron_data if "LAY" in p][0]
fastow_name = [p for p in enron_data if "FASTOW" in p][0]

print "Jeff Skilling's payments: {0}".format(enron_data[skilling_name]['total_payments'])
print "Ken Lay's payments: {0}".format(enron_data[lay_name]['total_payments'])
print "Andrew Fastow's payments: {0}".format(enron_data[fastow_name]['total_payments'])

#How many people have quantified salary?
salaries = [1 for p in enron_data if enron_data[p]['salary'] != 'NaN']
print "Total number of quantified salaries: {0}".format(sum(salaries)) 

#How many people have valid email address?
emails = [1 for p in enron_data if enron_data[p]['email_address'] != 'NaN']
print "Total number of valid email addresses: {0}".format(sum(emails))

#How many people have NaN total payments fields? What percentage of the total dataset is this?
total_payments = [1 for p in enron_data if enron_data[p]['total_payments'] != 'NaN']
print "Total number of valid total payments fields: {0}".format(sum(total_payments))
print 1. - float(sum(total_payments)) / len(enron_data)

#How many POIS have NaN total payments fields? What percentage of the total dataset is this?
#How many people have NaN total payments fields? What percentage of the total dataset is this?
nan_poi_total_payments = [1 for p in enron_data if (enron_data[p]['total_payments'] == 'NaN' and enron_data[p]['poi'] == True)]
print "Total number of POIs with NaN total payments: {0}".format(sum(nan_poi_total_payments))


#print float(sum(nan_poi_total_payments)) / float(sum(pois))
