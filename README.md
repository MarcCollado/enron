# Identify Fraud From Enron Email
Machine Learning Project — Udacity Data Analyst Nanodegree

This project is part of the Data Analyst Nanodegree. Below you'll find the rest of the projects and I also wrote a [short post](https://collado.io/blog/2018/udacity-dand) about the experience.

* [Exploratory data analysis](https://github.com/MarcCollado/wine)
* [Data wrangling](https://github.com/MarcCollado/open-street-map)
* [Machine learning](https://github.com/MarcCollado/enron)
* [Data visualization](https://public.tableau.com/profile/marccollado#!/vizhome/TitanicFinal_6/Titanic)

## Introduction
In 2000, Enron was one of the largest companies in the United States. By 2002, it had collapsed into bankruptcy due to widespread corporate fraud. In the resulting Federal investigation, a significant amount of typically confidential information entered into the public record, including tens of thousands of emails and detailed financial data for top executives.

Using all the concepts and ideas learned through the *Introduction to Machine Learning* module within the Udacity's Data Analyst NanoDegree, this project will use the `scikit-learn` [Python library](http://scikit-learn.org/stable/) and machine learning techniques to predict and spot culpable persons of the Enron scandal.

## How The Project Works
The project contains four main folders:

* `code`: Python main scripts
* `data`: data sets used throughout the project
* `misc`: several assets, such as images, generated to illustrate the `README.md`
* `tools`: support functions for the main scripts

The `code` folder contains all the scripts that run the project. The main script is `poi_id.py` which needs to be run first in order to generate the models for `tester.py` to yield the right results.

Besides `poi_id.py` all the `poi_*.py` scripts contain the functions called from the main script. Their "surname" describe which section each file tackles and which stage of the project the code belongs.

> i.e. the `poi_plot.py` contains all the plotting functions, while the `poi_explore.py` contains all the functions used during the feature exploration phase.

The final structure of the code is the following:

* `poi_id.py`: main script, which shares the structure with `README.md`
  * `poi_plot.py`: contains all the plotting functions
  * `poi_explore.py`: contains all the functions related to the data exploration
  * `poy_select.py`: contains all the functions related to the feature selection
  * `poy_tune.py`: contains all the functions related to algorithm tuning
  * `features.py`: stores different iterations of features lists
* `tester.py`: script used to test the algorithm performance

All the project questions are answered directly in the `README.md` document. Despite `README.md` contains plenty of code snippets, the project flow has been designed to jump between `README.md` and `poi_*.py` scripts for further reference and in order to better understand the code being discussed at any given time.

In order to facilitate readability, both `README.md` and `poi_*.py` files share the same structure:

* Data Exploration: related to *data exploration* and *outlier detection*.
* Selected Features: related to *create new features*, *intelligently select features* and *properly scale features*.
* Algorithm Tuning: related to *parameter tuning* and *validation strategy*.
* Final Model

### Other Things You Should Know
In the main script, `poi_id.py`, line 15, you'll find the `print_*` knobs. Their main use is to turn on/off different printing functions of the code, in order to avoid dozens of lines appearing in the console when running `poi_id.py`.

All the printing functions called from `poi_id.py` are wrapped inside a conditional statement that points to these Booleans.

The printing variables themselves are also segmented. The gist of it being that if you wanted to print only the code from the Data Exploration phase, you should only "turn" the switch `print_explore` to `True`.

From a functionality perspective, all the work is done regardless of these switches, then its use is merely cosmetic and should not affect the final results.

## Data Exploration
> Summarize for us the goal of this project and how machine learning is useful in trying to accomplish it. As part of your answer, give some background on the data set and how it can be used to answer the project question. Were there any outliers in the data when you got it, and how did you handle those?

The main goal of this project is to use both financial and email data from Enron to build a predictive model that could potentially identify a "person of interest" (POI), i.e. Enron employees who may have committed fraud, based on the aforementioned public data.

### Data Structure
As part of the preprocessing for this project, the email and financial data from Enron has been combined into a dictionary, where each key-value pair corresponds to one person.

The dictionary key is the person's name, and the value is another dictionary, which contains the names of all the features and their values for that person.

Additionally, the features always fall into three major types:

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
* Email features (units are number of emails messages; except `email_address`, which is a text string):
  * `to_messages`
  * `email_address`
  * `from_poi_to_this_person`
  * `from_messages`
  * `from_this_person_to_poi`
  * `shared_receipt_with_poi`
* POI labels (boolean, represented as integer):
  * `poi`

Here's a real example extracted from the data set.
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

A quick overview of the data set provides some highlights about its structure and main characteristics:

* Financial features: 14
* Email features: 6
* POI label: 1
* Dataset length: 146
* Number of POI: 18
* Percentage of POI: 12.3%

### NaNs
Despite overall having really valuable information in order to identify POIs, the data set also contained a lot of missing values — NaN.

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

At this point no data will be discarded, waiting until further Feature Selection to spot the most influential features. Despite, for the financial features, one way to interpret the NaNs is that they are equivalent to a zero.

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

After inspecting closely each person one by one, there's no meaningful information we can derive from these persons and on top of that, none of each is a POI, therefore, they will be removed from the data set.

**Additional Plots**

The image below plots the amount of email `from_messages` and `to_messages` sent or received from each person.

![fig3](misc/img/fig3.png)

This plot does appear to contain some outliers. There are three suspicious persons that send / receive way more email than what would be considered average.

For example, the farthest point on the right sent over 14000 emails, with the closest being less than a half. The points outside the main cluster are identified below.

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

Finally, the updated data set that will be used for the upcoming sections has the following characteristics:

* Dataset length: 140
* Number of POI: 18
* Percentage of POI: 12.9%


## Selected Features
> What features did you end up using in your POI identifier, and what selection process did you use to pick them? Did you have to do any scaling? Why or why not? As part of the assignment, you should attempt to engineer your own feature that does not come ready-made in the data set -- explain what feature you tried to make, and the rationale behind it.

### Pre Assessment
Before jumping straight into engineering and selecting features or even scaling algorithms, lets assess first out of the box the performance of three different algorithms (decided upon [documentation](http://scikit-learn.org/stable/tutorial/machine_learning_map/) from `sklearn`). The goal is to use the results yielded in this section as a base line to benchmark once the new features are added or the scaling is performed.

The three selected algorithms are:

* [AdaBoost](http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html)
* [Random Forest](http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html#sklearn.ensemble.RandomForestClassifier)
* [SVC](http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC)

#### Performance Review: out of the box
These are Accuracy, Precision and Recall values for the three algorithms with the default parameters, features and no scaling. They have been calculated using the `test_classifier()` function found in `tester.py`.

**AdaBoost**
* Accuracy: 0.83193
* Precision: 0.38098
* Recall: 0.28250

**RandomForest**
* Accuracy: 0.84943
* Precision: 0.41290
* Recall: 0.12800

**SVC**
* Accuracy: 0.49636
* Precision: 0.14762
* Recall: 0.52900

Both AdaBoost and RandomForest got a really good Accuracy out of the box. They also got similar values for Precision, around 40%, but both felt short when it comes to Recall, under 30%.

This means that out of all the items that are truly positive, i.e. POI, how many were correctly classified as positive. Or simply put, how many positive items were 'recalled' from the data set.

Finally, SVC threw out this error out of the box with `kernel:'rbf'`:

```
Precision or recall may be undefined due to a lack of true positive predictions.

```

Twisting a little bit some parameters by hand showed that changing the kernel to `linear` got rid of the error but entered into a infinite loop that yielded no results.

Without using `GridSearchCV` to fine tune the algorithm, which will be addressed later in the project, some variables have been manually adjusted until it got meaningful results.

The key parameter responsible for the infinite loop once solved the error of lack of true positives turned out to be `max_iter`, which according to `scikit-learn` documentation is the *hard limit on iterations within solver*. The default value `-1`, sets the algorithm with no limit, but rolling it back to `1000`, yielded valid results.

The idea here was mainly to set these values as a benchmark to see how feature selection, feature engineering and scaling affected the results.

### Feature Pre Selection
Going back to the table that displayed the amount of NaNs per feature, is clear that there are no features that have information for all the employees in the data set. Despite, up to five features have more than 60% missing values. If that was not enough, `restricted_stock_deferred`, `director_fees`, `loan_advances` and `deferral_payments` features are missing for more than 50% of the POI segment.

As a result, these four features and `restricted_stock_deferred` — with over 85% of NaN, will be omitted from the selection process.

Additionally, the `email_address` feature will also be left out since it is text based, and hardly provides any predictive value.

These are the updated set of features, removing the ones with higher rate of incompleteness, its respective list can be found at `features_db.py`, under the name of `feat_2`.

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

#### Performance Review: remove features with more than 60% NaNs
Here are the new metrics obtained by removing the features with more than 60% missing values, that were not adding value and could potentially cause noise in the results.

**AdaBoost**
* Accuracy: 0.83279
* Precision: 0.38580
* Recall: 0.28800

**RandomForest**
* Accuracy: 0.84536
* Precision: 0.36967
* Recall: 0.11700

**SVC**
* Accuracy: 0.50900
* Precision: 0.15763
* Recall: 0.56100

The removal of these features yielded almost identical results for AdaBoost, but it saw a drop in both Precision and Recall for RandomForest. Which suggests that it is not a good idea to get rid the aforementioned features.

Along the same lines, similar results were shown in the SVC classifier, where all the metrics saw a minimal increase, definitely not enough to justify the removal of the features.

### Feature Engineering
The combination of several features, both financial and email, provide a richer view of the whole picture. After analyzing and plotting different possibilities, these are the new features created that derived the most value out of the existing set:

**New financial features**
* `f_bonus` = `bonus` / `total_payments`
* `f_salary` = `salary` / `total_payments`
* `f_stock` = `total_stock_value` / `total_payments`

The first set of engineered features relates to the fraction (f stands for fraction) from the type of financial incentives received. Employees usually can be rewarded mainly through three mechanisms: salary, bonus or stock.

The goal here is to understand if higher fractions of payments in certain modalities led to POI. Say, for example, all the involved persons in the scandal were to be payed mostly through bonuses. Having the fraction of each type of payment could potentially spot the ones under such circumstances.

Since all three sources are available and have little missing values, they seemed like an interesting choice.

**New email features**
* `r_from` = `from_this_person_to_poi` / `from_messages`
* `r_to` = `from_poi_to_this_person` / `to_messages`

The second set of engineered features relates to the ratio (r stands for ratio) of email sent to or received from a POI. Since the total data for to and from is available, getting the ratio is rather easy.

These features will reveal the persons with higher communications with POI in a percentage basis. The reason behind it is because it could be the case that somebody went unnoticed with lower email volume, but almost all of it directed or from POI. This new feature would help surface these edge cases.

#### Performance Review: add engineered features
After both the Pre Selection and Feature Engineering, Accuracy, Precision and Recall for the three algorithms with the same test sizes, will be measured again.

**AdaBoost**
* Accuracy: 0.83729
* Precision: 0.40479
* Recall: 0.29550

**RandomForest**
* Accuracy: 0.85271
* Precision: 0.44404
* Recall: 0.12300

**SVC**
* Accuracy: 0.49636
* Precision: 0.14772
* Recall: 0.52950

The addition of new engineered features yielded really good results, specially for the AdaBoost classifier. It boosted both Precision and Recall, almost crossing the 0.3 mark for Recall.

RandomForest also saw improvements across the board, with more than 10% increase in Precision, compared to the default feature set.

Finally, things were no better for SVC, which is still struggling with Accuracy and Precision, with the worst results among the three.

### SelectKBest
Yet another approach to automate the feature selection process is using the `scikit` library `SelectKBest`. This function basically select features according to the k highest scores, passed as a parameter.

The `kbest()` function implements this method and returns a list with the ten most important features according to `SelectKBest`. Here are the results sorted by its importances or `pscore` — the 3 decimal number — output from the function:

```
[
('director_fees', '1.41', '0.235'),
('other', '1.64', '0.200'),
('shared_receipt_with_poi', '2.28', '0.131'),
('long_term_incentive', '2.42', '0.120'),
('total_payments', '2.69', '0.101'),
('salary', '2.86', '0.091'),
('bonus', '4.90', '0.027'),
('total_stock_value', '5.28', '0.022'),
('loan_advances', '6.53', '0.011'),
('exercised_stock_options', '6.61', '0.010')
]
```

#### Performance Review: SelectKBest
After running `kbest()`, its feature list was tested with the three classifiers and yielded the following results:

**AdaBoost**
* Accuracy: 0.82036
* Precision: 0.29284
* Recall: 0.18200

**RandomForest**
* Accuracy: 0.84764
* Precision: 0.40459
* Recall: 0.14100

**SVC**
* Accuracy: 0.50757
* Precision: 0.15241
* Recall: 0.53650

The new set of features `feat_K` didn't perform well in AdaBoost or RandomForest, which both saw drops in Accuracy and Precision, and mixed results for Recall. But in any case, boosting its values in a meaningful way.

SVC was almost agnostic to the implementation of the features from `SelectKBest` and its performance metrics stayed almost the same across the board.

### feature_importances_
It seems that nor a more manual approach, with the notable exception of Feature Engineering, neither an algorithm based selection have had a significant impact on the performance so far.

This time around feature selection will be tackled from the other end, removing the less important features. `feature_importances_` will be used in order to understand which features are having the most impact in the final results. Below the results for each algorithm:

**AdaBoost**
```
[['total_stock_value', 0.20000000000000001], ['expenses', 0.12], ['poi', 0.10000000000000001], ['loan_advances', 0.10000000000000001], ['exercised_stock_options', 0.080000000000000002], ['deferral_payments', 0.059999999999999998], ['from_this_person_to_poi', 0.059999999999999998], ['f_salary', 0.059999999999999998], ['f_stock', 0.059999999999999998], ['long_term_incentive', 0.040000000000000001], ['f_bonus', 0.040000000000000001], ['restricted_stock_deferred', 0.02], ['deferred_income', 0.02], ['other', 0.02], ['shared_receipt_with_poi', 0.02], ['salary', 0.0], ['total_payments', 0.0], ['bonus', 0.0], ['restricted_stock', 0.0], ['director_fees', 0.0], ['to_messages', 0.0], ['from_poi_to_this_person', 0.0], ['from_messages', 0.0], ['r_from', 0.0]]
```
**RandomForest**
```
[['long_term_incentive', 0.087231488852846803], ['other', 0.086489110652360215], ['from_this_person_to_poi', 0.083918100933557296], ['loan_advances', 0.081534798458218094], ['f_stock', 0.078852851155577122], ['f_bonus', 0.064538698118749904], ['exercised_stock_options', 0.063228665160139613], ['deferred_income', 0.062298629930743733], ['f_salary', 0.057618663961196914], ['deferral_payments', 0.04473987317520272], ['expenses', 0.035961330096999977], ['from_messages', 0.035197628458498019], ['from_poi_to_this_person', 0.03471824699572075], ['r_from', 0.033778861120719469], ['restricted_stock_deferred', 0.032478491937300021], ['poi', 0.030407177609787794], ['to_messages', 0.026947578830846863], ['salary', 0.022443181818181831], ['director_fees', 0.013847376788553264], ['total_stock_value', 0.012979950551845624], ['shared_receipt_with_poi', 0.010789295392953927], ['total_payments', 0.0], ['bonus', 0.0], ['restricted_stock', 0.0]]
```
**SVC**
```
[['poi', 0.13580494485781833], ['loan_advances', 0.12296931461314901], ['f_stock', 0.10842863477155748], ['to_messages', 0.078275084950569535], ['exercised_stock_options', 0.074449954982917732], ['from_poi_to_this_person', 0.057058803959956786], ['f_bonus', 0.053522644352031523], ['other', 0.050747989298544106], ['shared_receipt_with_poi', 0.04758094580036528], ['deferral_payments', 0.041461506713625348], ['r_from', 0.040070334762602647], ['from_messages', 0.039263624135758304], ['long_term_incentive', 0.034725457543239219], ['salary', 0.020891345289300357], ['total_stock_value', 0.01940072817213228], ['from_this_person_to_poi', 0.017809343434343435], ['restricted_stock_deferred', 0.016442894432291043], ['f_salary', 0.014698143222239609], ['director_fees', 0.013989213798467029], ['deferred_income', 0.012409090909090911], ['total_payments', 0.0], ['bonus', 0.0], ['expenses', 0.0], ['restricted_stock', 0.0]]
```

Despite not getting similar results for all algorithms, some patterns do emerge. For example, surprisingly `loan_advances` and `exercised_stock_options` rank really low in all algorithms despite not having a lot of completeness in the data set.

Also surprising how there is an unanimous agreement across `total_payments`, `bonus` (intuitively thought of one of the most important drivers for POI) and `restricted_stock`, which were classified as not important at all.

On the other hand, some contradictions seem to arise as well. For example, `total_stock_value` is the highest ranking feature in AdaBoost, but one of the lowest in both RandomForest and SVC.

Another positive note is that almost all the new engineered features are ranked on the top on terms of importance for all algorithms, which validates the rationale behind their inclusion.

#### Performance Review: feature_importances_
As further experimentation, the five less important features will be removed from the engineered set `feat_3` for each algorithm and evaluated again, with the optimized feature set.

**AdaBoost**
* Accuracy: 0.85836
* Precision: 0.50606
* Recall: 0.35500

**RandomForest**
* Accuracy: 0.85550
* Precision: 0.47950
* Recall: 0.13450

**SVC**
* Accuracy: 0.39936
* Precision: 0.12769
* Recall: 0.54950

This time each algorithm had a specific, distinct, feature set crafted for itself. And overall, removing the less important features for each classifier had a great positive impact, specially on AdaBoost, which saw almost 20% increase in Recall, its weakest metric at the time.

On the other hand, RandomForest almost didn't see any changes, but a small bump in Accuracy, which was already at par with AdaBoost. On the other hand, this classifier is still showing a lot of trouble when it comes to Recall.

Finally SVC didn't show improvements either. For this classifier should be considered pre-processing the data with `StandardScaler`, which be performed in the upcoming section.

That being said, so far the best performing algorithm has been AdaBoost, featuring the tailor made feature set with the engineered ones and dropping the less relevant based on `feature_importances_`.

## Algorithm Tuning and Validation
> What does it mean to tune the parameters of an algorithm, and what can happen if you don’t do this well?  How did you tune the parameters of your particular algorithm? What parameters did you tune?

First of all, in order to support feature scaling and parameter tuning, `sklearn`'s `Pipeline` and `GridSearchCV` modules will be implemented.

Every machine learning algorithm has several parameters at its disposal in order to model the results depending on the situation at hand. Some of it has been already discussed in the first performance review where an infinite loop was fixed by tweaking `max_iter` in the SVC classifier.

Parameter tuning is a really important craft when it comes to analyze data sets through machine learning algorithms. Since each data set has its own uniqueness, it has to be tackled under a specific set of conditions or combination of parameters in order to yield optimal results.

### The Dangers of Overfitting
But parameter tuning can be tricky, though. If not done well, we can end up with an algorithm that correctly classifies our data, but is erratic at many other places. That happens when the data is taken too literally and our machine learning algorithm produces an extremely complex model that doesn't generalize well. This phenomena is called overfitting and it is something any machine learning practitioner should always keep an eye on.

For example, in its fundamental nature, an SVC classifier works by recursively drawing a boundary between different classes on the data, that maximizes the distance to the nearest point, also called the margin.

Of course this boundary can be as simplistic as a straight line, which generalizes well but doesn't yield optimal results out of a particular situation, or as twisted and complex as we can imagine, which could perfectly match our current model, but it wouldn't fit other situations.

Doubling down on the SVC example, a classifier tuned with a linear kernel and high values of gamma (which defines how far the influence of a single point reaches), would produce seemingly linear cuts to the data, that could have the potential to work on many sets, but hardly yield optimal performance in any particular situation.

On the other hand, if tuning an SVC with an RBF kernel and lower values of gamma, will inevitably produce a twisted, wiggled boundary with an exceptional fit for that particular situation, that won't be useful if changes in the data structure are enforced. Our classifier would be somehow biased by the data and that's a clear example of overfitting.

Then the our goal here is to find and tune the parameters that yield optimal performance for this particular case. Of course this task can be done manually, as seen in the previous sections, but in some occasions the the amount of parameter combinations can be overwhelming and the process can take a lot of time. Here's where `GridSearchCV` enters the picture.

`GridSearchCV` module automates the testing by recursively trying several parameter combination and finally, pick the most optimal result.

### RandomForest and AdaBoost
The tune process for both algorithms has been similar, and the results almost identical.

They were tested using `rf_tune()` and `ab_tune()` respectively. Tests were performed with and without scaler using `StandardScaler`, it made little difference in RandomForest and no difference in AdaBoost. Which shouldn't come as a surprise since it was stated during the class materials.

`n_estimators` parameter was iterated, with values ranging from 1 to 200. The sweet spot was found around 3 and 10, being 5 the value that yielded the best results. Here's a code snippet of the best estimator derived from Pipeline, both for RandomForest and AdaBoost.

```
# RandomForestClassifier
Pipeline(steps=[('rf', RandomForestClassifier(bootstrap=True,
            class_weight=None, criterion='gini',
            max_depth=None, max_features='auto', max_leaf_nodes=None,
            min_impurity_split=1e-07, min_samples_leaf=1,
            min_samples_split=2, min_weight_fraction_leaf=0.0,
            n_estimators=5, n_jobs=1, oob_score=False, random_state=None,
            verbose=0, warm_start=False))])

# AdaBoostClassifier
Pipeline(steps=[('ab', AdaBoostClassifier(algorithm='SAMME.R', base_estimator=None, learning_rate=1.0, n_estimators=5, random_state=None))])
```

Finally, several scores were tested. The one that produced the best results was `F1`, which stroke a healthy balance between Recall and Precision. This fact shouldn't come as a surprise since this is its main definition, according to Wikipedia:

> F1 score considers both the precision p and the Recall r of the test to compute the score. [...] Is the harmonic average of the precision and Recall, where an F1 score reaches its best value at 1 (perfect precision and Recall) and worst at 0.

Curiously, after the iteration in the search of the best parameters with `GridSearchCV`, AdaBoost didn't show better results than its "out of the box" behavior with a customized feature set.

### SVC
SVC went through a similar optimization process, but with more focus on the parameters, since `Pipeline` allowed more customization when it came to this algorithm.

Also high expectations were held for how the pre-processing through feature scaling would affected its final results, since it had little to no effect to AdaBoost and RandomForest.

After several tries with mixed luck, here's the `param_grid` with the combinations that yield the best overall results.

```
param_grid = ([{'svm__C': [1, 50, 100, 1000],
                'svm__gamma': [0.5, 0.1, 0.01],
                'svm__degree': [1, 2],
                'svm__kernel': ['rbf', 'poly', 'linear'],
                'svm__max_iter': [1, 100, 1000]}])
```

As expected, the scaler and parameter tuning in SVC had a huge positive impact.

**SVC**
* Accuracy: from 0.39936 to 0.80336
* Precision: from 0.12769 to 0.32545
* Recall: from 0.54950 to 0.35100

Incredibly, both Accuracy and Precision more than doubled its score, on the other hand Recall got more in line with Precision. This was also a byproduct of optimizing for `F1` instead of Recall. If the classifier was optimized for Recall instead, the overall results were not that balanced.

The parameters that yielded the best results were the following:

```
C=50
cache_size=200
class_weight=None
coef0=0.0,
decision_function_shape=None
degree=2,
gamma=0.1,
kernel='poly',
max_iter=100,
probability=False,
random_state=None,
shrinking=True,
tol=0.001,
verbose=False
```

## Final Model
> What algorithm did you end up using? What other one(s) did you try? How did model performance differ between algorithms?

Finally, here's the table that sums everything up:

| Metric    | RF | RF w/ scaler | AB | AB w/ scaler | SVC w/ scaler |
|:------:|:--:|:------------:|:--:|:------------:|:-------------:|
| Accuracy  | 0.84  | 0.85 | 0.85 | 0.86 | 0.80 |
| Precision | 0.38  | 0.47 | 0.48 | 0.50 | 0.33 |
| Recall    | 0.19  | 0.13 | 0.22 | 0.22 | 0.35 |
| F1        | 0.25  | 0.20 | 0.30 | 0.31 | 0.34 |
| F2        | 0.21  | 0.15 | 0.24 | 0.25 | 0.25 |
|   True +  | 383   | 256  | 430  | 448  | 702 |
|   False + | 616   | 288  | 457  | 454  | 1455 |
|   False - | 1617  | 1744 | 1570 | 1552 | 1298 |
|   True -  | 11384 | 11712 | 11543 | 11546 | 10545 |

Interestingly enough, SVC, the worst performer along all the project, turned the situation in its head by using a scaler and fine tuning its parameters.

As the table above shows, SVC has the most healthy balance across Precision and Recall, with a Accuracy almost at par with AdaBoost and RandomForest.

On top of that, SVC is also the best performer when it comes to identify True Positives, with more than 50% success rate than its contenders. The ability of identify True positives can have a major impact when having imbalanced classes (many more non-POIs than POIs), because you want your classifier a little bit biased towards getting the maximum amount of "suspicious" persons, despite declared innocent afterwards.

That being said, after the tuning process and comparing the performance of the different algorithms, under a different set of conditions and parameters, SVC, in a close call, appears to be the best performing in most of the measurements.

Therefore, SVC will be the candidate for submission in the final analysis with the aforementioned parameters and the `feat_4SVC` feature list.

Under these conditions the algorithm performed at its best, getting the following metrics:

* Accuracy: 80%
* Precision: 33%
* Recall: 35%

> What is validation, and what’s a classic mistake you can make if you do it wrong? How did you validate your analysis?

Validation is the set of techniques to make sure the model performs well in a wide range of situations, and it's not just optimized for a particular data set or conditions.

As seen during the DAND, a classic mistake if your model is not validated and only trained in one set of data, is overfitting.

This phenomena can be studied adjusting the amount of data points assigned to both training and testing sets. The most common way to test for it is with cross validation, a technique that dynamically assigns a percentage to the different sets.

The function `test_clf()` in `poi_select.py` dynamically assigns different ratios to the train / test sets given a classifier. It uses `train_test_split` from `sklearn.cross_validation` a module that assists in creating different sets of data to test. This allows the researcher to understand how the algorithm behaves under different set of conditions.

The validation was performed with test sets of 20%, 40% and 60% to ensure the classifier was behaving properly across a wide range of situations.

| Test size | 0.2 | 0.4 | 0.6 |
|:---------:|:---:|:---:|:---:|
| Accuracy | 0.785714285714 | 0.839285714286 | 0.821428571429 |
| Precision | 0.333333333333 | 0.6 | 0.4375 |
| Recall | 0.2 | 0.3 | 0.538461538462 |

As shown in the table above, the fine tuned classifier is consistent across different test sizes, which ensures it is not overfit and particularly train for this type of circumstances.

## Evaluation Metrics
Against all the odds judging by its performance at the very beginning, our SVC classifier got great results: above the 0.3 mark — in both Precision and Recall — required for the submission of the project. But let's dive deeper into each of these metrics to really understand their meaning in the Enron situation.

When it comes to evaluate an algorithm, there are several evaluation metrics at our disposal, as seen during the whole project. It follows that each metric is evaluating the classifier's performance in a certain way, so different metrics should be used where you care for some outcomes more than others.

In other words, each performance metrics favor one type of error over another, this allows our algorithm to be tuned and optimized for very specific outcomes.

#### Why Accuracy is flawed
As an example, Accuracy is a perfectly valid metric. From a technical standpoint it describes the ratio between the number of items in a class labeled correctly and all the items in a class.

In the Enron case, it means the algorithm is able to guess correctly whether a person is or is not a POI.

But sometimes this metric can be extremely misleading, particularly when dealing with imbalanced classes, or in other words, when the data is really skewed towards one class.

This is the case with the Enron set — since there are many more non-POIs than POIs, which introduces some special challenges, namely that you can just guess the more common class label for every point, which is not a very insightful strategy, but still get pretty good Accuracy.

#### Recall
That's the ultimate reason why the classifier has been also optimized for other metrics, more attuned to the nature of the data and the specifics of the social framework it described.

Another metric, Recall, describes the ability of the algorithm to correctly identify a POI provided that the person is a POI. Topping at a 0.35, means that 35% of the POI won't go unnoticed by the algorithm.

35% might seem low, but this metric is particularly insightful for the Enron case. Since we are dealing with a criminal situation, we want our classifier to err on the side of guessing guilty — higher levels of scrutiny — so it makes sure as many people get flagged as POI, maybe at a cost of identifying some innocent people along the way.

Boosting its Recall metric the classifier ensures that is correctly identifying every single POI. The tradeoff is that the algorithm will be biased towards "overdoing" it. In this particular situation this exactly what we are looking for: guarantee that no POIs will go unnoticed and (hope) the misclassified innocents will be declared innocent by the justice later on.

#### Precision
On the other hand, Precision topped at more than 32%. What this number is telling, is the chances that every time the algorithm is flagging somebody as POI, this person truly is a POI.

Unlike the previous situation, if the classifier doesn't have have great Recall, but it does have good Precision, it means that whenever a POI gets flagged in the test set, there's a lot of confidence that it’s very likely to be a real POI and not a false alarm.

On the other hand, the tradeoff is that sometimes real POIs are missed, since the classifier is effectively reluctant to pull the trigger on edge cases. Which in the case of Enron is definitely something we don't want.

#### F1 Score
It seems that neither Accuracy nor Precision were helping much in terms of assessing the results. For this reason, as a final note and despite not widely covered during class, I wanted to talk about the F1 score.

In some way, the F1 score can be thought of "the best of both worlds."

In its pure definition F1 *"considers both the Precision and the Recall of the test to compute the score [...] The F1 score is the harmonic average of the precision and recall, where an F1 score reaches its best value at 1 (perfect precision and recall) and worst at 0."*

Technically it ensures that both False Positives an False Negatives rates are low, which translated to the Enron set, means that I can identify POIs reliably and accurately. If the identifier finds a POI then the person is almost certainly to be a POI, and if the identifier does not flag someone, then they are almost certainly not a POI.

To wrap it up, it is clear that in this context, Recall is way more important than both Accuracy and Precision. If further work ought to be performed to the final algorithm, given the specific data set and social framework, re-tuning the classifier to yield a better Recall score — even at the cost of lower Precision — would be the most effective way to ensure all POIs are prosecuted.
