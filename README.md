# Identify Fraud from Enron Email
#### Udacity Data Analyst Nanodegree — Project 7

## Introduction
In 2000, Enron was one of the largest companies in the United States. By 2002, it had collapsed into bankruptcy due to widespread corporate fraud. In the resulting Federal investigation, a significant amount of typically confidential information entered into the public record, including tens of thousands of emails and detailed financial data for top executives.

Using all the concepts and ideas learned through the *Introduction to Machine Learning* module within the Udacity's Data Analyst NanoDegree, this project will use the `scikit-learn` [Python library](http://scikit-learn.org/stable/) and machine learning techniques to predict and spot culpable persons of the Enron scandal.

## How The Project Works
The project contains three main folders:

* code
* data
* misc
* tools

The `code` folder contains all the scripts that run the project.

* poi_id.py
  * poi_explore.py
  * poy_select.py
  * poy_tune.py

All the project questions are answered directly in the README.md document. Despite README.md contains plenty of code snippets, the project flow has been designed to jump between README.md and poi_*.py files for further reference and in order to better understand the code being discussed at any given time.

In order to facilitate readability, both README.md and poi_*.py files share the same structure:

* Data Exploration: related to *data exploration* and *outlier detection*.
* Selected Features: related to *create new features*, *intelligently select features* and *properly scale features*.
* Algorithm Tuning: related to *parameter tuning* and *tune the algorithm*.

### Other Things You Should Know
* `print_*` functions


## Data Exploration
> Summarize for us the goal of this project and how machine learning is useful in trying to accomplish it. As part of your answer, give some background on the dataset and how it can be used to answer the project question. Were there any outliers in the data when you got it, and how did you handle those?

The main goal of this project is to use both financial and email data from Enron to build a predictive model that could potentially identify a "person of interest" (POI), i.e. Enron employees who may have committed fraud, based on the aforementioned public data.

### Data Structure
Here's how this data looks like: as part of the preprocessing for this project, the email and financial data from Enron has been combined into a dictionary, where each key-value pair corresponds to one person.

Therefore, the dictionary key is the person's name, and the value is another dictionary, which contains the names of all the features and their values for that person.

Additionally, the features in the data fall into three major types:

* Financial features (all units in US dollars):
  * `salary`
  * `deferral_payments`
  * `total_payments`
  * `loan_advances`
  * `bonus`
  * `restricted_stock_deferred`
  * `deferred_income`
  * `total_stock_value`
  * `expenses`
  * `exercised_stock_options`
  * `other`
  * `long_term_incentive`
  * `restricted_stock`
  * `director_fees`
* Email features (units are number of emails messages; except ‘email_address’, which is a text string):
  * `to_messages`
  * `email_address`
  * `from_poi_to_this_person`
  * `from_messages`
  * `from_this_person_to_poi`
  * `shared_receipt_with_poi`
* POI labels (boolean, represented as integer):
  * `poi`

Here's a real example extracted from the dataset.
```
"SKILLING JEFFREY K":
  {'salary': 1111258,
  'to_messages': 3627,
  'deferral_payments': 'NaN',
  'total_payments': 8682716,
  'exercised_stock_options': 19250000,
  'bonus': 5600000,
  'restricted_stock': 6843672,
  'shared_receipt_with_poi': 2042,
  'restricted_stock_deferred': 'NaN',
  'total_stock_value': 26093672,
  'expenses': 29336,
  'loan_advances': 'NaN',
  'from_messages': 108,
  'other': 22122,
  'from_this_person_to_poi': 30,
  'poi': True,
  'director_fees': 'NaN',
  'deferred_income': 'NaN',
  'long_term_incentive': 1920000,
  'email_address': 'jeff.skilling@enron.com',
  'from_poi_to_this_person': 88}
```

A quick overview of the dataset provides some highlights about its structure and main characteristics:

* Financial features: 14
* Email features: 6
* POI label: 1
* Dataset length: 146
* Number of POI: 18
* Percentage of POI: 12.3%

### NaNs
Despite having really valuable information in order to identify POIs, the dataset contained a lot of missing values — NaN.

Here's a table showing the amount of NaN values per feature.

| Feature | NaNs |
|:-------:|:----:|
| Loan advances | 142 |
| Director fees | 129 |
| Restricted stock deferred | 128 |
| Deferred payment | 107 |
| Deferred income | 97 |
| Long term incentive | 80 |
| Bonus | 64 |
| Emails sent also to POI | 60 |
| From messages (emails sent) | 60 |
| To messages (emails received) | 60 |
| Emails from POI (emails received from POI) | 60 |
| Emails to POI (emails sent to POI) | 60 |
| Other | 53 |
| Expenses | 51 |
| Salary | 51 |
| Exercised stock options | 44 |
| Restricted stock | 36 |
| Email address | 35 |
| Total payment | 21 |
| Total stock value | 20 |

The data definitely has a lot of NaNs, the most concerning features are `Loan advances`, `Director fees` and `Restricted stock deferred`, which have more than 85% of their values missing.

At this point no data won't be discarded, waiting until the Feature Selection through `SelectKBest` to spot the most influential features. Despite, for the financial features, one way to interpret the NaNs is that they are equivalent to a zero.

There is a case to be made that if an employee did not receive restricted stock, the value could be zero. On top of that, this hypothesis is also supported by the `insider-pay.pdf`, that can be found in the `misc` folder, in how the totals are calculated.

For these reasons, NaN values in only financial features will be replaced by zeros.

### Outliers
The outlier explorations starts by plotting two of the most telling features when it comes to uncover the relation between the data and POIs: `salary` and `bonus`.

![fig1](misc/img/fig1.png)

The plot clearly shows an outlier in the top right of the plot. Ordering the list by salary, the outlier is called `TOTAL`, which represents the sum of all the salaries, as shown in the `insider-pay.pdf`, therefore cannot be considered a person.

After removing `TOTAL` here's the same plot again.

![fig2](misc/img/fig2.png)

It looks like there still are four additional points with higher salary and bonus, in a range that could potentially consider them as outliers.

| Person Name | Salary | isPOI |
|:-----------:|:------:|:-----:|
| SKILLING JEFFREY K | 1111258 | True |
| LAY KENNETH L | 1072321 | True |
| FREVERT MARK A | 1060932 | False |
| PICKERING MARK R | 655037 | False |

After closer inspection, and despite not all of them were POIs, the rest of their data seemed consistent across the board and all of them looked like valid and meaningful data points.

**Incomplete Data**

Another potential source of outliers are the ones that don't add meaningful information to the mix, such as persons with little or no relevant information at all.

In order to spot these data points, the `get_incompletes()` function returns a list of the names with no feature data above a certain threshold.

With `get_incompletes()` set at 90%, which means that the persons returned by the function have only less than 10% of the data completed, it returns this list.

```
['WHALEY DAVID A',
 'WROBEL BRUCE',
 'LOCKHART EUGENE E',
 'THE TRAVEL AGENCY IN THE PARK',
 'GRAMM WENDY L']
```

After inspecting closely each person one by one, there's no meaningful information we can derive from these persons and on top of that, none of each is a POI, therefore, they will be removed from the dataset.

**Additional Plots**

The image below plots the amount of email `from_messages` and `to_messages` sent or received from each person.

![fig3](misc/img/fig3.png)

This plot does appear to contain some outliers. There are three suspicious persons that send / receive way more email than what would be considered average.

For example, the farthest point on the right sent over 14000 emails, with the closest being less then a half. The points outside the main cluster are identified below.

```
  ['KAMINSKI WINCENTY J',
   'KEAN STEVEN J',
   'SHAPIRO RICHARD S']
```

Besides sending / receiving a ton of email, it seems there is apparently nothing strange about these persons. In an enterprise like Enron, it is totally possible to have certain jobs / roles that involve heavier email use. For this reason these three person won't be discarded.

Additionally below are included other interesting plots from the exploration.

`exercised_stock_options` vs. `total_stock_value`

![fig4](misc/img/fig4.png)

All the data seems clustered between a certain range, except for four outliers. Two of them are already usual suspects: `LAY KENNETH L` and `SKILLING JEFFREY K`, the other two, `HIRKO JOSEPH` and `PAI LOU L` (POI), don't look like outliers after careful inspection of their records.

`from_poi_to_this_person` vs. `from_this_person_to_poi`

![fig5](misc/img/fig5.png)

Despite they can't be considered outliers, since the values stay within a reasonable range, there are four persons: `DELAINEY DAVID W` (POI himself), `LAVORATO JOHN J`, `KEAN STEVEN J`, `BECK SALLY W`, that send a huge amount of email to POIs.

After close inspection of the rest of the features, the same names keep showing up. If we wouldn't know part of the story, some persons — like `LAY KENNETH L` and `SKILLING JEFFREY K`, would probably classify as outliers. Even after removing `TOTAL`, they keep showing up at the higher band of all the financial related plots, like `total_payments`. But since they played a key role in the development of the story, they must stay, and with that the outlier inspection should be considered completed.

Finally, the updated dataset that will be used for the upcoming sections has the following characteristics:

* Dataset length: 140
* Number of POI: 18
* Percentage of POI: 12.9%


## Selected Features
> What features did you end up using in your POI identifier, and what selection process did you use to pick them? Did you have to do any scaling? Why or why not? As part of the assignment, you should attempt to engineer your own feature that does not come ready-made in the dataset -- explain what feature you tried to make, and the rationale behind it.

### Pre Assessment
Before jumping straight into engineering and selecting features or even scaling algorithms, lets assess first out of the box the performance of three different algorithms (decided upon [documentation](http://scikit-learn.org/stable/tutorial/machine_learning_map/) from `sklearn`). The goal is to use the results yielded in this section as a base line to benchmark once the new features are added or the scaling is performed.

The three selected algorithms are:

* [Adaboost](http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html)
* [Random Forest](http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html#sklearn.ensemble.RandomForestClassifier)
* [SVC](http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC)

#### Out of the Box Performance
These are the Accuracy, Precision and Recall values for the three algorithms with the default features and no scaling. Test size was set at 20%.

**AdaBoost**
* Accuracy: 0.857142857143
* Precision: 0.666666666667
* Recall: 0.4

**RandomForest**
* Accuracy: 0.857142857143
* Precision: 1.0
* Recall: 0.2

**SVM**
* Accuracy: 0.821428571429
* Precision: 0.0
* Recall: 0.0

Both AdaBoost and RandomForest got the best accuracy, closely followed by SVM. Precision was also hire in the case of AdaBoost and RandomForest, but Recall of AdaBoost was better by a factor of x2. SVM didn't get neither Precision nor Recall scores. The issue will be further explored.

The idea here was mainly to set these values as a benchmark to see how feature selection, feature engineering and scaling affected the results.

### Feature Pre Selection
Going back to the table that displayed the amount of NaNs per feature, is clear that no features have information for each employee in the dataset. Up to five features have more than 60% missing values, but if that was not enough, `restricted_stock_deferred`, `director_fees`, `loan_advances` and `deferral_payments` features are missing for more than 50% of the POI segment.

As a result, these four features and `restricted_stock_deferred` — with over 85% of NaN, will be omitted from the selection process.

Additionally, the `email_address` feature will also be left out since it is text based, and hardly provides any predictive value.

These are the base features that will be used alongside `scikit-learn`'s `SelectKBest` module to compute the 10 most influential features.

* Financial features:
  * `salary`
  * `total_payments`
  * `bonus`
  * `deferred_income`
  * `total_stock_value`
  * `expenses`
  * `exercised_stock_options`
  * `other`
  * `long_term_incentive`
  * `restricted_stock`
* Email features:
  * `to_messages`
  * `from_poi_to_this_person`
  * `from_messages`
  * `from_this_person_to_poi`
  * `shared_receipt_with_poi`

### Feature Engineering
The combination of several features, both financial and email, provide a richer view of the whole situation. After analyzing and plotting different possibilities, these are the new features created that derived the most value:

**New financial features**
* `f_bonus` = `bonus` / `total_payments`
* `f_salary` = `salary` / `total_payments`
* `f_stock` = `total_stock_value` / `total_payments`

The first set of engineered features relates to the fraction (f stands for fraction) of the type of financial incentives received. Employees usually can be rewarded mainly through three mechanisms: salary, bonus or stock.

The goal here is to understand if higher fractions of payments in certain modalities led to POI. Say, for example, all the involved persons in the scandal were to be payed mostly through bonuses. Having the fraction of each type of payment could potentially spot the ones under such circumstances.

Since all three sources are available and have little missing values, they seemed like an interesting choice.

**New email features**
* `r_from` = `from_this_person_to_poi` / `from_messages`
* `r_to` = `from_poi_to_this_person` / `to_messages`

The second set of engineered features relates to the ratio (r stands for ratio) of email sent to or received from a POI. Since the total data for to and from is available, getting the ratio is rather easy.

These features will reveal the persons with higher communications with POI in a percentage basis. The reason behind it is because it could be the case that somebody went unnoticed with lower email volume, but almost all of it directed or from POI. This new feature would help surface these edge cases.

### Feature Selection
After both the Pre Selection and Feature Engineering, Accuracy, Precision and Recall for the three algorithms with the same test sizes, will be measured again.

**AdaBoost**
* Accuracy: 0.785714285714
* Precision: 0.333333333333
* Recall: 0.2

**RandomForest**
* Accuracy: 0.892857142857
* Precision: 1.0
* Recall: 0.4

**SVM**
* Accuracy: 0.821428571429
* Precision: 0.0
* Recall: 0.0

Despite it didn't show radical changes, overall, the work perform on the features got mixed results, with the notable exception of RandomForest, that saw improvements in both Accuracy (topping at 0.89) and Recall (doubling its past mark).

On the other hand, AdaBoost didn't show any improvements, on the contrary, it lowered its Accuracy.

Finally, SVM, surprisingly, stayed the same, still not showing results for neither Precision nor Recall.

Since RandomForest showed the best overall results with the new engineered features, the feature selection and scaling with the new feature set.

> What algorithm did you end up using? What other one(s) did you try? How did model performance differ between algorithms?

#### feature_importances_
In order to get in depth with feature selection, both AdaBoost and RandomForest support the `feature_importances_` attribute. It will be used in order to understand which features are having the most impact in the algorithm.

**AdaBoost**
```
['exercised_stock_options', 0.16],
['poi', 0.10],
['total_stock_value', 0.10],
['expenses', 0.10],
['salary', 0.08],
['deferred_income', 0.08],
['long_term_incentive', 0.08],
['f_stock', 0.08],
['bonus', 0.04],
['from_this_person_to_poi', 0.04],
['shared_receipt_with_poi', 0.04],
['r_from', 0.040000000000000001],
['restricted_stock', 0.02],
['from_poi_to_this_person', 0.02],
['f_bonus', 0.02],
['total_payments', 0.0],
['other', 0.0],
['to_messages', 0.0],
['from_messages', 0.0],
['f_salary', 0.0]]
```

**RandomForest**
```
[['total_payments', 0.11809056026359306],
['r_from', 0.10841421686525166],
['expenses', 0.097773397551924268],
['f_stock', 0.088581161152790952],
['total_stock_value', 0.072250318742140995],
['long_term_incentive', 0.059523653362006511],
['from_this_person_to_poi', 0.058841579541288026],
['f_bonus', 0.054445532261694699],
['f_salary', 0.046918866633150175],
['deferred_income', 0.041328686283820346],
['other', 0.040321738483114725],
['salary', 0.035952732800633068],
['from_poi_to_this_person', 0.03001131221719457],
['shared_receipt_with_poi', 0.029355558275385229],
['bonus', 0.027469125883525229],
['exercised_stock_options', 0.026319297318460506],
['from_messages', 0.026027903415596621],
['poi', 0.015816326530612247],
['to_messages', 0.011621302815261837],
['restricted_stock', 0.010936729602555303]]
```

Despite not getting similar results for both algorithms, some patterns do emerge. For example, `from_messages` and `to_messages` rank really low in both algorithms.

On the other hand, some contradictions seem to arise as well. For example, `exercised_stock_options` is the highest ranking feature in AdaBoost, but one of the lowest in RandomForest.

Another positive note is that almost all the new engineered features are ranked on the top on terms of importance for both algorithms, therefore it validates the rationale behind its inclusion.

#### Select and Evaluate Again
As further experimentation, the five less important features will be removed for each algorithm and evaluated again, with the optimized feature set.

**AdaBoost**
* Accuracy: 0.857142857143
* Precision: 0.5
* Recall: 0.25

**RandomForest**
* Accuracy: 0.857142857143
* Precision: 0.666666666667
* Recall: 0.4

Removing the less important features had a great positive impact on the AdaBoost algorithm, with more than a 10% higher accuracy and more than double its precision.

On the other hand, RandomForest almost didn't see any changes, but a small drop in Accuracy, which was already the highest using the prior feature set.

So far, the best performing algorithm has been RandomForest with the addition  of engineered features set and dropping the ones with the most NaNs, but not after dropping more features based on `feature_importances_`.

Regardless, the Precision under such circumstances seem utterly unrealistic, and due to chance as a consequence of a small test set. Changing the test size can have a great impact on the precision, ranging from values from 0.66 to 1.

The performance metrics of this algorithm under these conditions has been:

**RandomForest**
* Accuracy: 0.857142857143
* Precision: 0.666666666667
* Recall: 0.4

Therefore, RandomForest, as the most solid and best performing algorithm (close call with AdaBoost here), will be fine tuned and used in the final analysis.


## Algorithm Tuning
> What does it mean to tune the parameters of an algorithm, and what can happen if you don’t do this well?  How did you tune the parameters of your particular algorithm? What parameters did you tune?

First of all, in order to support feature scaling and parameter tuning, `sklearn`'s `Pipeline` and `GridSearchCV` modules will be implemented.

Every machine learning algorithm has several parameters at its disposal in order to model the results depending on the case to be studied. Parameter tuning is a really important step when it comes to analyze data sets through machine learning algorithms. Since each data set has its own uniqueness, it has to be tackled under a specific set of conditions or parameters in order to get optimal results.

Then the goal of researchers is to find the parameters that yield optimal performance for each circumstance. Of course this task can be done manually, but in some occasions the the amount of parameter combinations can be overwhelming. Here's where `GridSearchCV` enters the picture.

`GridSearchCV` module automates this process by recursively trying several parameter combination and provide the most optimal result.




As mentioned in the Pre Selection section, both preselected and engineered features will be combined and put into `scikit-learn`'s `SelectKBest` module to compute the 10 most influential features.




[relevant rubric items: ]



What is validation, and what’s a classic mistake you can make if you do it wrong? How did you validate your analysis?  [relevant rubric items: “discuss validation”, “validation strategy”]



Give at least 2 evaluation metrics and your average performance for each of them.  Explain an interpretation of your metrics that says something human-understandable about your algorithm’s performance. [relevant rubric item: “usage of evaluation metrics”]
