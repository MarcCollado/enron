#!/usr/bin/python

import matplotlib.pyplot

import sys
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit


def scatterplot(dataset, var1, var2):
    """
    Creates and shows a scatterplot given a dataset and two features.
    """
    features_name = [str(var1), str(var2)]
    features = [var1, var2]
    data = featureFormat(dataset, features)

    for point in data:
        var1 = point[0]
        var2 = point[1]
        matplotlib.pyplot.scatter(var1, var2)

    matplotlib.pyplot.xlabel(features_name[0])
    matplotlib.pyplot.ylabel(features_name[1])
    matplotlib.pyplot.show()

    return
