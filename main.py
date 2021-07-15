# %%
import os
import time

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import pandas as pd
import seaborn as sns

from sklearn import metrics  # evaluation of models
from sklearn.linear_model import LogisticRegression  # logistic regression algorithm
from sklearn.model_selection import train_test_split  # data splitting
from sklearn.neighbors import KNeighborsClassifier  # k-nearest neighbor algorithm
from sklearn.preprocessing import StandardScaler  # data normalisation
from sklearn.tree import DecisionTreeClassifier  # decision tree algorithm
from sklearn.tree import plot_tree  # function to display decision tree
from sklearn.svm import SVC  # svm algorithm
from sklearn.ensemble import RandomForestClassifier  # random forest algorithm
from sklearn.model_selection import cross_val_score  # cross validation method
from sklearn.model_selection import GridSearchCV  # parameter tuning method using cv

from termcolor import colored as cl  # text customization

from operator import itemgetter  # used to get key with max val in dict

sns.set_style('dark')

# %% MODEL RESULTS

all_model_metrics = {}  # empty dictionary to store model metric results
all_model_times = {}  # empty dictionary to store model times


# %% CUSTOM FUNCTIONS

def get_metrics(model_yhat):
    model_acc = metrics.accuracy_score(y_test, model_yhat)
    model_f1 = metrics.f1_score(y_test, model_yhat)
    return model_acc, model_f1


def show_metrics(model):
    header = model.upper() + ' MODEL METRICS'
    model_metrics = all_model_metrics[model]
    print(cl(header, attrs=['bold']))
    print(cl('------------------------------', attrs=['bold']))
    print('Accuracy Score : ', end='')
    print(cl(model_metrics['accuracy_score'], 'white'))
    print('F1 Score : ', end='')
    print(cl(model_metrics['f1_score'], 'white'))
    print(cl('------------------------------', attrs=['bold']))


def show_times(model):
    header = model.upper() + ' MODEL TIMES'
    model_times = all_model_times[model]
    print(cl(header, attrs=['bold']))
    print(cl('------------------------------', attrs=['bold']))
    print('Build Time: ', end='')
    print(cl(str(model_times['build_time']) + 's', 'white'))
    print('Train Time : ', end='')
    print(cl(str(model_times['train_time']) + 's', 'white'))
    print('Predict Time: ', end='')
    print(cl(str(model_times['predict_time']) + 's', 'white'))
    print(cl('------------------------------', attrs=['bold']))


# %% IMPORT DATA

data_path = os.getcwd() + os.sep + 'input' + os.sep + 'creditcard.csv'
df = pd.read_csv(data_path)
df = df.drop(columns='Time')
# dataframe columns:
# V1-28 --> principal features from PCA
# Amount --> amount of money transferred
# Class --> fraud identifier (0:Not Fraud, 1:Fraud)

# %% TRANSACTION COUNTS

transactions_count = len(df)
nonfraud_count = len(df[df.Class == 0])
fraud_count = len(df[df.Class == 1])
fraud_percentage = round(fraud_count / transactions_count * 100, 2)

print(cl('TRANSACTIONS', attrs=['bold']))
print(cl('------------------------------', attrs=['bold']))
print('Total transactions: ', end='')
print(cl(transactions_count, 'white'))
print('Non-fraudulent transactions: ', end='')
print(cl(nonfraud_count, 'white'))
print('Fraudulent transactions: ', end='')
print(cl(fraud_count, 'white'))
print('Percentage of fraudulent transactions: ', end='')
print(cl('{}%'.format(fraud_percentage), 'white'))
print(cl('------------------------------', attrs=['bold']))

# %% AMOUNT STATISTICS

df_nonfraud = df[df.Class == 0]
df_fraud = df[df.Class == 1]

print(cl('TRANSACTION AMOUNT STATISTICS', attrs=['bold']))
print(cl('------------------------------', attrs=['bold']))
print(cl('NON-FRAUDULENT TRANSACTIONS', attrs=['bold']))
print(df_nonfraud.Amount.describe())
print(cl('------------------------------', attrs=['bold']))
print(cl('FRAUDULENT TRANSACTIONS', attrs=['bold']))
print(df_fraud.Amount.describe())
print(cl('------------------------------', attrs=['bold']))

# %% NORMALISE AMOUNT VALUES

sc = StandardScaler()  # initialize StandardScaler

amount_values = df['Amount'].values
df['Amount'] = sc.fit_transform(amount_values.reshape(-1, 1))

# %% STATIC DATA SPLITTING

X = df.drop(columns='Class').values  # independent variables
y = df.Class.values  # dependent variable

# randomly splits the X and y data into 80% training data and 20% test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# %% DECISION TREE MODELLING

# build, train, and test decision tree algorithm
tree_build_start = time.time()
tree_model = DecisionTreeClassifier(max_depth=4, criterion='entropy')
tree_build_stop = time.time()
tree_model.fit(X_train, y_train)
tree_train_stop = time.time()
tree_model_yhat = tree_model.predict(X_test)
tree_predict_stop = time.time()

# calculate times for build, train, and test
tree_build_time = tree_build_stop - tree_build_start
tree_train_time = tree_train_stop - tree_build_stop
tree_predict_time = tree_predict_stop - tree_train_stop

# draw decision tree visualization
fig = plt.figure(figsize=(25, 20))
_ = plot_tree(tree_model,
              filled=True,
              rounded=True,
              class_names=['non-fraud', 'fraud'])
plot_dir = os.getcwd() + os.sep + 'visualizations'
if not os.path.exists(plot_dir):
    os.mkdir(plot_dir)
plot_path = plot_dir + os.sep + 'decision_tree.png'
fig.savefig(plot_path)  # saves figure to visualizations dir

# store metrics and times
tree_metrics = get_metrics(tree_model_yhat)
all_model_metrics['tree'] = {'accuracy_score': tree_metrics[0],
                             'f1_score': tree_metrics[1]}
all_model_times['tree'] = {'build_time': tree_build_time,
                           'train_time': tree_train_time,
                           'predict_time': tree_predict_time}

# report decision tree metrics and times
show_metrics('tree')
show_times('tree')

# %% K-NEAREST NEIGHBORS MODELLING

# determine optimal k-value
# knn_results = {}
# for i in range(1, 16):
#     knn_model = KNeighborsClassifier(n_neighbors=i)
#     knn_model.fit(X_train, y_train)
#     knn_model_yhat = knn_model.predict(X_test)
#     acc = metrics.accuracy_score(y_test, knn_model_yhat)
#     f1 = metrics.f1_score(y_test, knn_model_yhat)
#     knn_results[i] = [acc + f1]
# k_optimal = max(knn_results.items(), key=itemgetter(1))[0]

# build, train, and test k-nearest neighbors algorithm
knn_build_start = time.time()
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_build_stop = time.time()
knn_model.fit(X_train, y_train)
knn_train_stop = time.time()
knn_model_yhat = knn_model.predict(X_test)
knn_predict_stop = time.time()

# calculate times for build, train, and test
knn_build_time = knn_build_stop - knn_build_start
knn_train_time = knn_train_stop - knn_build_stop
knn_predict_time = knn_predict_stop - knn_train_stop

# store metrics
knn_metrics = get_metrics(knn_model_yhat)
all_model_metrics['knn'] = {'accuracy_score': knn_metrics[0],
                            'f1_score': knn_metrics[1]}
all_model_times['knn'] = {'build_time': knn_build_time,
                          'train_time': knn_train_time,
                          'predict_time': knn_predict_time}

# report knn metrics and times
show_metrics('knn')
show_times('knn')

# %% LOGISTIC REGRESSION MODELLING

# build, train, and test logistic regression algorithm
lr_build_start = time.time()
lr_model = LogisticRegression()
lr_build_stop = time.time()
lr_model.fit(X_train, y_train)
lr_train_stop = time.time()
lr_model_yhat = lr_model.predict(X_test)
lr_predict_stop = time.time()

# calculate times for build, train, and test
lr_build_time = lr_build_stop - lr_build_start
lr_train_time = lr_train_stop - lr_build_stop
lr_predict_time = lr_predict_stop - lr_train_stop

# store metrics
lr_metrics = get_metrics(lr_model_yhat)
all_model_metrics['lr'] = {'accuracy_score': lr_metrics[0],
                           'f1_score': lr_metrics[1]}
all_model_times['lr'] = {'build_time': lr_build_time,
                         'train_time': lr_train_time,
                         'predict_time': lr_predict_time}

# report lr metrics and times
show_metrics('lr')
show_times('lr')

# %% SUPPORT VECTOR MACHINE MODELLING

# build, train, and test svm algorithm
svm_build_start = time.time()
svm_model = SVC()
svm_build_stop = time.time()
svm_model.fit(X_train, y_train)
svm_train_stop = time.time()
svm_model_yhat = svm_model.predict(X_test)
svm_predict_stop = time.time()

# calculate times for build, train, and test
svm_build_time = svm_build_stop - svm_build_start
svm_train_time = svm_train_stop - svm_build_stop
svm_predict_time = svm_predict_stop - svm_train_stop

# store metrics
svm_acc, svm_f1 = get_metrics(svm_model_yhat)
all_model_metrics['svm'] = {'accuracy_score': svm_acc,
                            'f1_score': svm_f1}
all_model_times['svm'] = {'build_time': svm_build_time,
                          'train_time': svm_train_time,
                          'predict_time': svm_predict_time}

# report lr metrics and times
show_metrics('svm')
show_times('svm')

# %% RANDOM FOREST TREE MODELLING

# build, train, and test svm algorithm
rf_build_start = time.time()
rf_model = RandomForestClassifier()
rf_build_stop = time.time()
rf_model.fit(X_train, y_train)
rf_train_stop = time.time()
rf_model_yhat = rf_model.predict(X_test)
rf_predict_stop = time.time()

# calculate times for build, train, and test
rf_build_time = rf_build_stop - rf_build_start
rf_train_time = rf_train_stop - rf_build_stop
rf_predict_time = rf_predict_stop - rf_train_stop

# store metrics
rf_acc, rf_f1 = get_metrics(rf_model_yhat)
all_model_metrics['rf'] = {'accuracy_score': rf_acc,
                           'f1_score': rf_f1}
all_model_times['rf'] = {'build_time': rf_build_time,
                         'train_time': rf_train_time,
                         'predict_time': rf_predict_time}

# report lr metrics and times
show_metrics('rf')
show_times('rf')

# %% XGBOOST MODELLING

# %% METRICS DATAFRAME

# create pandas dataframe from metrics dictionary
df_metric = pd.DataFrame.from_dict(all_model_metrics).T
df_metric = df_metric.reset_index(drop=False).rename(columns={'index': 'model'})

# %% TIMES DATAFRAME

# create pandas dataframe from time dictionary
df_time = pd.DataFrame.from_dict(all_model_times).T
df_time = df_time.reset_index(drop=False).rename(columns={'index': 'model'})

# %% COLOR PALETTES

# get color palettes
custom_pal = sns.color_palette('tab20c') + sns.color_palette('tab20b')
core_pal = [custom_pal[1], custom_pal[5], custom_pal[9],
            custom_pal[13], custom_pal[21], custom_pal[34]]

# %% ACCURACY SCORE PLOT

# draw plot
fig, ax = plt.subplots()
_ = sns.barplot(ax=ax,
                data=df_metric,
                x='model',
                y='accuracy_score',
                palette='Set1')

# set plot title
fig.suptitle('Accuracy Score Results of ML Algorithms')

# set y-axis limits
ax.set_ylim(0.998, 1)

# set axis and tick labels
ax.set_ylabel('Accuracy')
yticks = [i / 1000000 for i in range(998000, 1000500, 500)]
ax.set_yticks(yticks)
ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
ax.set_xlabel('Algorithm')

# adjust figure borders
plt.gcf().subplots_adjust(left=0.2, bottom=0.14)

# display plot
plt.show()

# %% F1 SCORE PLOT

# draw plot
fig, ax = plt.subplots()
_ = sns.barplot(ax=ax,
                data=df_metric,
                x='model',
                y='f1_score',
                palette='Set1')

# set plot title
fig.suptitle('F1 Score Results of ML Algorithms')

# set y-axis limits
y_min = round(df_metric.f1_score.min() - 0.1, 1)
ax.set_ylim(y_min, 1)

# set axis and tick labels
ax.set_ylabel('F1 Score')
ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
ax.set_xlabel('Algorithm')

# adjust figure borders
plt.gcf().subplots_adjust(left=0.15, bottom=0.14)

# display plot
plt.show()

# %% TIMES PLOT

# reshape df_time dataframe
df_time_melt = pd.melt(df_time,
                       id_vars='model',
                       value_vars=['build_time',
                                   'train_time',
                                   'predict_time'],
                       var_name='stage',
                       value_name='time')

# create palette for time plot
time_pal = [custom_pal[0], custom_pal[1], custom_pal[2],
            custom_pal[4], custom_pal[5], custom_pal[6],
            custom_pal[8], custom_pal[9], custom_pal[10],
            custom_pal[12], custom_pal[13], custom_pal[14],
            custom_pal[20], custom_pal[21], custom_pal[22],
            custom_pal[24], custom_pal[25], custom_pal[26]]

# draw plot
fig, ax = plt.subplots()
_ = sns.barplot(ax=ax,
                data=df_time_melt,
                x='model',
                y='time',
                hue='stage',
                palette=time_pal)

# set plot title
fig.suptitle('Time to Build, Train, and Test ML Algorithms')

# set axis and tick labels
plt.yscale('log')
ax.set_ylabel('log(time) (s)')
ax.set_xlabel('Algorithm')

# adjust figure borders
plt.gcf().subplots_adjust(left=0.15, bottom=0.14)

# display plot
plt.show()

# %% DECISION TREE MODELLING - CROSS VALIDATION AND HYPER-PARAM TUNING

# although the above tests proved mostly successful for predicting outputs (classifying)
# the hyper-parameters of each algorithm could be improved by using cross-validation methods
# such as k-fold cross validation, e.g. is the depth of the decision tree optimal?
# let's try that out...

depth_scores = {}  # empty dictionary to store results
for i in range(2, 21):
    tree_model = DecisionTreeClassifier(max_depth=i)
    # perform 7 fold cross validation
    scores = cross_val_score(estimator=tree_model, X=X_train, y=y_train, cv=7, n_jobs=4)
    depth_scores[i] = scores.mean()
# get depth with maximum cross validation mean score
optimal_depth = max(depth_scores.items(), key=itemgetter(1))[0]

# print the optimal depth and it's associated 7 fold cross validation mean score
print(cl('DECISION TREE MODELLING - CV AND HYPER-PARAM TUNING', attrs=['bold']))
print(cl('------------------------------', attrs=['bold']))
print('Optimal Tree Depth : ', end='')
print(cl(optimal_depth, 'white'))
print('7 Fold Cross Validation Score : ', end='')
print(cl(depth_scores[optimal_depth], 'white'))
print(cl('------------------------------', attrs=['bold']))

# this loop worked well for the decision tree as we only optimised one hyper-parameter
# what about when we want to optimise multiple hyper-parameters?

# %% SUPPORT VECTOR MACHINE MODELLING - EXHAUSTIVE GRID SEARCH

# print a list of hyper-parameters for the support vector classifier
svm_model = SVC()
hyper_params = svm_model.get_params()
print(cl('SVC HYPER-PARAMETERS', attrs=['bold']))
print(cl('------------------------------', attrs=['bold']))
for key in hyper_params:
    print(key)
print(cl('------------------------------', attrs=['bold']))
print('Number of Hyper-Parameters : ', end='')
print(cl(len(hyper_params), 'white'))
print(cl('------------------------------', attrs=['bold']))

# hyper-parameters we want to test
tuned_params = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4], 'C': [1, 10, 100, 1000]},
                {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]

# scores to maximise
scores = ['accuracy', 'f1']

print(cl('TUNING HYPER-PARAMETERS OF \nSUPPORT VECTOR CLASSIFIER', attrs=['bold']))
for score in scores:
    print(cl('------------------------------', attrs=['bold']))
    print('Tuning hyper-parameters to \nmaximise {} score...'.format(score), end=' ')
    # tries each combination of tuned hyper-parameters with cv to get mean score
    svm_model = GridSearchCV(SVC(), tuned_params, scoring=score)
    svm_model.fit(X_train, y_train)
    print('Done!')
    print('Optimal hyper-parameters found using training data : ', end='')
    print(cl(svm_model.best_params_, 'white'))
    print('Mean score using optimal hyper-parameters : ', end='')
    print(cl(svm_model.best_score_, 'white'))
    print(cl('------------------------------', attrs=['bold']))

# update model to use best hyper-parameters
svm_model = svm_model.best_estimator_

# %% CROSS VALIDATION PLOTS

