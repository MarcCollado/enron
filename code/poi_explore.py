#!/usr/bin/python

import sys
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit


def get_size(dataset):
    """
    Returns the size of a list.
    """

    return len(dataset)


def count_poi(dataset):
    """
    Returns the number of POI from the Enron Dataset.
    """
    n = 0
    for person in dataset:
        if dataset[person]["poi"]:
            n += 1

    return n


def count_nan(dataset):
    """
    Returns a dictionary with these key-value pairs:
        key = feature name
        value = number of NaNs the feature has across the dataset
    """
    d = {}
    for person in dataset:
        for key, value in dataset[person].iteritems():
            if value == "NaN":
                if key in d:
                    d[key] += 1
                else:
                    d[key] = 1

    return d


def nan_replacer(dataset):
    """
    Sets all the NaN values for financial features to zero.
    """
    ff = [
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
        "director_fees"
    ]
    for f in ff:
        for person in dataset:
            if dataset[person][f] == "NaN":
                dataset[person][f] = 0

    return dataset


def sort_data(dataset, feature, results, reverse=False):
    """
    Returns an array of sorted data.

    Args:
    dataset: a dictionary containing the data
    feature: the dictionary key indicating the feature to be sorted
    results: an integer indicating the number of results to be output
    reverse: a boolean indicating the order of the results (default: False)

    Output: an array with the sorted results
    """
    features = [feature]
    data = featureFormat(dataset, features)

    s = sorted(data, key=lambda x: x[0], reverse=reverse)[:int(results)]

    return s


def get_name(dataset, feature, value):
    """
    Returns the matching name of a person, given a feature and its value.
    """
    for p in dataset:
        if dataset[p][feature] == value:

            return p

    return


def get_incompletes(dataset, threshold):
    """
    Returns an array of person names that have no information (NaN) in a
    percentage of features above a given threshold (between 0 and 1).
    """
    incompletes = []
    for person in dataset:
        n = 0
        for key, value in dataset[person].iteritems():
            if value == 'NaN' or value == 0:
                n += 1
        fraction = float(n) / float(21)
        if fraction > threshold:
            incompletes.append(person)

    return incompletes
