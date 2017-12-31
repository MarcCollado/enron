#!/usr/bin/python

import poi_select, poi_tune
import tester

import sys
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit

from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC


def rf_tune(d, features_list, scaler):
    """
    Prints the key metrics of an algorithm after it has gone through the
    tuning process with Pipleine and GridSearchCV.
    """
    # Keep only the values from features_list
    data = featureFormat(d, features_list, sort_keys=True)
    # Split between labels (poi) and the rest of features
    labels, features = targetFeatureSplit(data)

    if scaler:
        rf = Pipeline([('scaler', StandardScaler()),
                       ('rf', RandomForestClassifier())])
    else:
        rf = Pipeline([('rf', RandomForestClassifier())])

    param_grid = ([{'rf__n_estimators': [4, 5, 10, 500]}])

    rf_clf = GridSearchCV(rf,
                          param_grid,
                          scoring='f1').fit(
                            features, labels).best_estimator_

    tester.test_classifier(rf_clf, d, features_list)

    return


def ab_tune(d, features_list, scaler):
    """
    Prints the key metrics of an algorithm after it has gone through the
    tuning process with Pipleine and GridSearchCV.
    """
    # Keep only the values from features_list
    data = featureFormat(d, features_list, sort_keys=True)
    # Split between labels (poi) and the rest of features
    labels, features = targetFeatureSplit(data)

    if scaler:
        ab = Pipeline([('scaler', StandardScaler()),
                       ('ab', AdaBoostClassifier())])
    else:
        ab = Pipeline([('ab', AdaBoostClassifier())])

    param_grid = ([{'ab__n_estimators': [1, 5, 10, 50]}])

    ab_clf = GridSearchCV(ab,
                          param_grid,
                          scoring='recall').fit(
                            features, labels).best_estimator_

    tester.test_classifier(ab_clf, d, features_list)

    return


def svc_tune(d, features_list, scaler=True):
    """
    Prints the key metrics of an algorithm after it has gone through the
    tuning process with Pipleine and GridSearchCV.
    """
    # Keep only the values from features_list
    data = featureFormat(d, features_list, sort_keys=True)
    # Split between labels (poi) and the rest of features
    labels, features = targetFeatureSplit(data)

    svm = Pipeline([('scaler', StandardScaler()), ('svm', SVC())])

    param_grid = ([{'svm__C': [1, 50, 100, 1000],
                    'svm__gamma': [0.5, 0.1, 0.01],
                    'svm__degree': [1, 2],
                    'svm__kernel': ['rbf', 'poly', 'linear'],
                    'svm__max_iter': [1, 100, 1000]}])

    svm_clf = GridSearchCV(svm,
                           param_grid,
                           scoring='f1').fit(
                           features, labels).best_estimator_

    tester.test_classifier(svm_clf, d, features_list)

    return


def get_svc(d, features_list):
    """
    Generates the classifier for final submission.
    """
    # Keep only the values from features_list
    data = featureFormat(d, features_list, sort_keys=True)
    # Split between labels (poi) and the rest of features
    labels, features = targetFeatureSplit(data)

    svm = Pipeline([('scaler', StandardScaler()), ('svm', SVC())])

    param_grid = ([{'svm__C': [50],
                    'svm__gamma': [0.1],
                    'svm__degree': [2],
                    'svm__kernel': ['poly'],
                    'svm__max_iter': [100]}])

    svm_clf = GridSearchCV(svm,
                           param_grid,
                           scoring='f1').fit(
                           features, labels).best_estimator_

    return svm_clf
