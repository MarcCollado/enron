#!/usr/bin/python

# Default feature set
feat_1 = [
    "poi",
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


# feat_1 - features w/ +60% NaNs
feat_2 = [
    "poi",
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
    "shared_receipt_with_poi"
    ]


# feat_1 + engineered features
feat_3 = [
    "poi",
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
    "shared_receipt_with_poi",
    "f_bonus",
    "f_salary",
    "f_stock",
    "r_from",
    "r_to"
    ]


# feat_K - 10 most important features from SelectKBest
feat_K = [
    "poi",
    "director_fees",
    "other",
    "shared_receipt_with_poi",
    "long_term_incentive",
    "total_payments",
    "salary",
    "bonus",
    "total_stock_value",
    "loan_advances",
    "exercised_stock_options"
]


# feat_3 - AB non imporant features from feature_importances_
feat_4AB = [
    "poi",
    "deferral_payments",
    "loan_advances",
    "restricted_stock_deferred",
    "deferred_income",
    "total_stock_value",
    "expenses",
    "exercised_stock_options",
    "other",
    "long_term_incentive",
    "from_this_person_to_poi",
    "shared_receipt_with_poi",
    "f_bonus",
    "f_salary",
    "f_stock",
    "r_to"
    ]


# feat_3 - RF non imporant features from feature_importances_
feat_4RF = [
    "poi",
    "salary",
    "deferral_payments",
    "loan_advances",
    "restricted_stock_deferred",
    "deferred_income",
    "expenses",
    "exercised_stock_options",
    "other",
    "long_term_incentive",
    "director_fees",
    "to_messages",
    "from_poi_to_this_person",
    "from_messages",
    "from_this_person_to_poi",
    "f_bonus",
    "f_salary",
    "f_stock",
    "r_from",
    "r_to"
    ]


# feat_3 - SVC non imporant features from feature_importances_
feat_4SVC = [
    "poi",
    "salary",
    "deferral_payments",
    "loan_advances",
    "restricted_stock_deferred",
    "total_stock_value",
    "exercised_stock_options",
    "other",
    "long_term_incentive",
    "director_fees",
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
