#!/usr/bin/python

import poi_plot, poi_explore, poi_select, poi_tune
import pickle
import pprint
pp = pprint.PrettyPrinter(indent=4)

import sys
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

from sklearn.ensemble import AdaBoostClassifier

# Switch to True to display the main features of the Exploration
print_plot = False
print_explore = False
print_select = False
print_tune = False

# Load the dictionary containing the dataset
with open("../data/final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)


# PART 1: Data Exploration
# Related file: poi_explore.py

if print_explore:
    # Get the size of the dictionary containing the dataset
    print "* Dataset length:", poi_explore.get_size(data_dict)
    # Get the number of POI in the dataset
    print "* Number of POI:", poi_explore.count_poi(data_dict)

if print_explore:
    # Get a list of all the features and the number of NaNs each one contains
    print "* List of NaNs per feature:"
    pp.pprint(poi_explore.count_nan(data_dict))

# Replace all the NaNs in financial features with zeros
data_dict = poi_explore.nan_replacer(data_dict)

if print_explore:
    print "* List of NaNs per feature once replaced NaNs for zeros:"
    pp.pprint(poi_explore.count_nan(data_dict))

if print_plot:
    # Scatterplot: bonus vs. salary
    fig1 = poi_plot.scatterplot(data_dict, "salary", "bonus")

# Find the outlier with the highest salary
outlier = poi_explore.sort_data(data_dict, "salary", 1, reverse=True)

if print_explore:
    outlier_name = poi_explore.get_name(data_dict, "salary", outlier[0])
    print "* Outlier found:", outlier_name

# Remove outlier TOTAL
data_dict.pop('TOTAL', None)

if print_explore:
    # Find persons with little or no information
    print "* List of persons with more than 90% of data missing:"
    pp.pprint(poi_explore.get_incompletes(data_dict, 0.90))

# Remove outliers from get_incompletes()
more_outliers = ['WHALEY DAVID A',
                 'WROBEL BRUCE',
                 'LOCKHART EUGENE E',
                 'THE TRAVEL AGENCY IN THE PARK',
                 'GRAMM WENDY L']

for o in more_outliers:
    data_dict.pop(o, None)


# PART 2: Selected Features
# Related file: poi_select.py

# Default dataset features to test base line performance
features_list = ["poi",
                 "salary",
                 "deferral_payments",
                 "total_payments",
                 "loan_advances",
                 "bonus",
                 "restricted_stock_deferred",
                 "deferred_income",
                 "total_stock_value",
                 "expenses",
                 "exercised_stock_options",
                 "other",
                 "long_term_incentive",
                 "restricted_stock",
                 "director_fees",
                 "to_messages",
                 "from_poi_to_this_person",
                 "from_messages",
                 "from_this_person_to_poi",
                 "shared_receipt_with_poi"
                 ]

# Get Accuracy, Precision and Recall from the three algorithms
if print_select:
    poi_select.test_clf(data_dict, features_list, 0.2)

# Create new engineered features
# f_bonus
data_dict = poi_select.create_feature(data_dict,
                                      "bonus",
                                      "total_payments",
                                      "f_bonus")
# f_salary
data_dict = poi_select.create_feature(data_dict,
                                      "salary",
                                      "total_payments",
                                      "f_salary")
# f_stock
data_dict = poi_select.create_feature(data_dict,
                                      "total_stock_value",
                                      "total_payments",
                                      "f_stock")
# r_from
data_dict = poi_select.create_feature(data_dict,
                                      "from_this_person_to_poi",
                                      "from_messages",
                                      "r_from")
# r_to
data_dict = poi_select.create_feature(data_dict,
                                      "from_poi_to_this_person",
                                      "to_messages",
                                      "r_to")

# New dataset features with engineered features
features_list = ["poi",
                 "salary",
                 "total_payments",
                 "bonus",
                 "deferred_income",
                 "total_stock_value",
                 "expenses",
                 "exercised_stock_options",
                 "other",
                 "long_term_incentive",
                 "restricted_stock",
                 "to_messages",
                 "from_poi_to_this_person",
                 "from_messages",
                 "from_this_person_to_poi",
                 "shared_receipt_with_poi",
                 "f_bonus",
                 "f_salary",
                 "f_stock",
                 "r_from",
                 "r_to"
                 ]

# Get Accuracy, Precision and Recall from the three algorithms
if print_select:
    poi_select.test_clf(data_dict, features_list, 0.2)

# Get the importance of each feature for AB and RF
if print_select:
    poi_select.feature_importances(data_dict, features_list, 0.2)

# Optimized feature set for AdaBoost
features_list_AB = ["poi",
                    "salary",
                    "bonus",
                    "deferred_income",
                    "total_stock_value",
                    "expenses",
                    "exercised_stock_options",
                    "long_term_incentive",
                    "restricted_stock",
                    "from_poi_to_this_person",
                    "from_this_person_to_poi",
                    "shared_receipt_with_poi",
                    "f_bonus",
                    "f_stock",
                    "r_from",
                    "r_to"
                    ]

# Optimized feature set for RandomForest
features_list_RF = ["poi",
                    "salary",
                    "total_payments",
                    "deferred_income",
                    "total_stock_value",
                    "expenses",
                    "other",
                    "long_term_incentive",
                    "from_poi_to_this_person",
                    "from_this_person_to_poi",
                    "shared_receipt_with_poi",
                    "f_bonus",
                    "f_salary",
                    "f_stock",
                    "r_from",
                    "r_to"
                    ]

# Get Accuracy, Precision and Recall with the optimized dataset
if print_select:
    poi_select.test_clf(data_dict, features_list_AB, 0.2)
    poi_select.test_clf(data_dict, features_list_RF, 0.2)


# PART 3: Algorithm Tuning
# Related file: poi_tune.py

# Fine tune the classifiers
if print_tune:
    poi_tune.rf_tune(data_dict, features_list, True)
    poi_tune.ab_tune(data_dict, features_list, True)
    poi_tune.svc_tune(data_dict, features_list)

# Code Review: store to my_dataset and feature_list for easy export below
clf = AdaBoostClassifier(n_estimators=50)
my_dataset = data_dict
features_list = ["poi",
                 "salary",
                 "total_payments",
                 "bonus",
                 "deferred_income",
                 "total_stock_value",
                 "expenses",
                 "exercised_stock_options",
                 "other",
                 "long_term_incentive",
                 "restricted_stock",
                 "to_messages",
                 "from_poi_to_this_person",
                 "from_messages",
                 "from_this_person_to_poi",
                 "shared_receipt_with_poi",
                 "f_bonus",
                 "f_salary",
                 "f_stock",
                 "r_from",
                 "r_to"
                 ]

dump_classifier_and_data(clf, my_dataset, features_list)
