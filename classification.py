# Import Class Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import purity
import seaborn as sns
import warnings
import seaborn as sns
from sklearn import metrics

warnings.filterwarnings("ignore")

from scipy.stats import stats
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler, MaxAbsScaler
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectKBest, f_regression, RFE
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from imblearn.over_sampling import SMOTE
from sklearn.metrics import precision_recall_fscore_support, matthews_corrcoef, accuracy_score, make_scorer, f1_score, \
    classification_report, roc_curve, confusion_matrix, plot_confusion_matrix
from sklearn.tree import DecisionTreeClassifier

sns.set()

from scipy.stats import stats

#####################################################################
# Dataset = Online Shoppers Purchasing Intention
# Feature = Administrative, Administrative Duration, Informational,
#           Informational Duration, Product Related, Product Related Duration,
#           Bounce Rate, Exit Rate, Page Value, Special Day, Browser, Region,
#           Traffic Type, Visitor Type, Weekend, Operating Systems, Month
# Target  = Revenue

# Number of dataset = 12,330
# Numerical value   = Administrative, Administrative Duration, Informational,
#                     Informational Duration, Product Related, Product Related Duration,
#                     Bounce Rate, Exit Rate, Page Value, Special Day
# Categorical value = Browser, Region, Traffic Type, Visitor Type, Weekend,
#                     Operating Systems, Month, Revenue

df = pd.read_csv('/Users/kohyojin/Desktop/SoftWare/software2021/2021-2/machine learning/term_project/online_shoppers_intention.csv')

feature_label = ['Administrative', 'Administrative_Duration', 'Informational',
                 'Informational_Duration', 'ProductRelated', 'ProductRelated_Duration',
                 'BounceRates', 'ExitRate', 'PageValue', 'SpecialDay', 'Browser', 'Region',
                 'TrafficType', 'VisitorType', 'Weekend', 'OperatingSystems', 'Month']
target_label = ['Revenue']

scale_col = ['Administrative', 'Administrative_Duration', 'Informational', 'Informational_Duration',
             'ProductRelated', 'ProductRelated_Duration', 'BounceRates',
             'ExitRates', 'PageValues', 'SpecialDay']
encode_col = ['Browser', 'Region', 'TrafficType', 'VisitorType','Weekend', 'OperatingSystems', 'Month']

# Print dataset's information
# print("\n***** Online Shoppers Intention *****")
# print(df.head())

# print("\n************ Description *************")
# print(df.describe())

# print("\n************ Information *************")
# print(df.info())

# Check null value
# print("\n************ Check null *************")
# print(df.isna().sum())

# Fill null value - Using ffill
df = df.fillna(method='ffill')
# Check null value (Cleaned Data)
# print("\n***** Check null (Cleaned Data) *****")
# print(df.isna().sum())

# Remove Outliers with z-score
# Description = Use the z-score to handle outlier over mean +- 3SD
# Input  = dataframe's column
# Output = index
def find_outliers(col):
    z = np.abs(stats.zscore(col))
    idx_outliers = np.where(z > 3, True, False)
    return pd.Series(idx_outliers, index=col.index)

# Remove outliers (Numerical value)
for n in range(len(scale_col)):
    idx = None
    idx = find_outliers(df.iloc[:, n])
    df = df.loc[idx == False]

# print("\n****** Removed Outlier (Numerical value) *****")
# print(df.info())

# Remove outliers (Categorical value)
# print("\n***** Check outlier of categorical values *****")
# for n in encode_col:
#     print(df[n].value_counts(), "\n")

for n in [11, 12, 14]:
    idx = None
    idx = find_outliers(df.iloc[:, n])
    df = df.loc[idx == False]
df = df[df['VisitorType'] != 'Other']

# print("\n***** Removed Outlier (Categorical value) *****")
# for n in encode_col:
#     print(df[n].value_counts(), "\n")

# print("\n***** Cleaned Dataset *****")
# print(df.info())

########################################################
# Data Inspection (After Outlier detection)
# Whole data
# plt.figure(figsize=(30,10))
# df.boxplot(column=encode_col)
# plt.show()

# Numerical column
# df.hist(column=scale_col, figsize=(20,20))
# plt.show()

# Categorical column
# df.hist(column=encode_col, figsize=(20,20))
# plt.show()

# Target Value
# plt.figure(figsize=(10,10))
# df['Revenue'].value_counts().plot(kind='pie',autopct='%1.1f', textprops={'fontsize': 15},startangle=90,explode =(0.1,0),colors=['slategray','cornflowerblue'])
# plt.title('Revenue', fontsize = 18)
# plt.ylabel('')
# plt.show()

# Compare plot
# column1l=['Administrative','Informational','ProductRelated','SpecialDay','OperatingSystems','Browser','Region','TrafficType','Month','VisitorType','Weekend']
# plt.figure(figsize=(30,30))
# plot_number = 0
# for i in column1l:
#    plot_number = plot_number + 1
#    ax = plt.subplot(6, 2, plot_number,adjustable='datalim')
#    sns.countplot(df[i],hue=df['Revenue'])
#    ax.set_title('Customers adding Revenue based on '+ i,fontdict=None)
#    plt.tight_layout()
# plt.show()

# Set X, y data
y_data = df.loc[:, target_label]
X_data = df.drop(target_label, axis=1)

##############################################################################
# Description = When actual and prediction vales are put in, the mean of prediction + recall + f1 score + accuracy is output
# Input  = actual value, predicted value
# Output = (accuracy + prediction + recall + f1 score) / 5
# Scoring function
def overall_average_score(actual,prediction):
    precision, recall, f1_score, _ = precision_recall_fscore_support(actual, prediction, average='binary')
    total_score = matthews_corrcoef(actual, prediction)+accuracy_score(actual, prediction)+precision+recall+f1_score
    return total_score/5
df.columns = df.columns.to_series().apply(lambda x: x.strip())

#####################################################################
# FindBestAccuracy(X, y, scale_col, encode_col, scalers=None, encoders=None,
#                       models=None, model_param=None, cv=None, n_jobs=None)
# Description = When parameters are put in, the highest score and the best model are printed
#Input = X: Data Feature
#        Y: Data Target
#        Scale_col = columns to scaled
#        Encode_col = columns to encoded
#        Scalers: list of scalers
#            None: [StandardScaler(), RobustScaler(), MinMaxScaler(), MaxAbsScaler()]
#        Encoder: list of encoders
#            None: [OrdinalEncoder(), LabelEncoder()]
#        Models: list of models
#            None: [LogisticRegression(), SVC(), GradientBoostingClassifier()]
#
#         Model_param: list of model’s hyperparameter
#         LogisticRegression()’s None: [penalty:(none,l2), random_state:(0,1), C:(0.01, 0.1, 1.0, 10.0, 100.0),
#                                       solver:(lbfgs, sag, saga], max_iter:(10, 50, 100)]
#         SVC()’s None: [random_state: (0,1), kernel: (linear, rbf, sigmoid),
#                        C: (0.01, 0.1, 1.0, 10.0, 100.0), gamma: (scale, auto)]
#         GradientBoostingClassifier()’s None: [loss:(deviance, exponential),learning_rate:(0.001, 0.1, 1),
#                                               n_estimators:(1, 10,100,1000),subsample: (0.0001,0.001, 0.1),
#                                               min_samples_split:(10,50, 100, 300),min_samples_leaf:(5, 10, 15,50)]
#
#         If you want to set other ways, then put the hyperparameter in list.
#
#         Cv = cross validation’s K
#            None: 5
#         N_jobs = number of jobs to run in parallel. Training the estimator and computing score are
#                  parallelized over the cross-validation splits
# 	        None: 1
def FindBestAccruacy(X, y, scale_col, encode_col, scalers=None, encoders=None,
                      models=None, model_param=None, cv=None, n_jobs=None):

    # Set Encoder
    if encoders is None:
        encode = [OrdinalEncoder(), LabelEncoder()]
    else: encode = encoders

    # Set Scaler
    if scalers is None:
        scale = [StandardScaler(), MinMaxScaler(), MaxAbsScaler(), RobustScaler()]
    else: scale = scalers

    # Set Model
    if models is None:
        model = [
            LogisticRegression(),
            SVC(probability=True),
            GradientBoostingClassifier()
        ]
    else: model = models

    # Set Hyperparameter
    if model_param is None:
        parameter = [
                     # LogisticRegression()
                     {'penalty':['l2'], 'random_state':[0,1], 'C':[0.1, 1.0, 10.0],
                       'solver':["lbfgs", "sag", "saga"]},

                     # SVC()
                     {'random_state': [0,1], 'kernel': ['linear', 'rbf', 'sigmoid'],
                       'C': [0.1, 1.0, 10.0], 'gamma': ['scale', 'auto'],},

                     # GradientBoostingClassifier()
                     {'loss':['deviance','exponential'], 'learning_rate':[0.001, 0.1, 1],
                      'n_estimators':[1, 10, 100], 'subsample':[0.0001, 0.001, 0.1],
                      'min_samples_split':[10, 50, 100], 'min_samples_leaf':[5, 10, 15]}
                     ]

    else: parameter = model_param

    # Set CV(cross validation)
    if cv is None:
        setCV = 5
    else: setCV = cv

    # Set n_jobs
    if n_jobs is None:
        N_JOBS = -1
    else: N_JOBS = n_jobs

    best_score1 = 0; best_score2 = 0; best_score3 = 0
    best_combination1 = {}; best_combination2 = {}; best_combination3 = {}
    param1 = {}; param2 = {}; param3 = {}
    feat = str;

    print("Calculation is in progress....")
    ####################################################################
    # Iterate
    for i in scale:
        for j in encode:
            # Scaling
            df_scaled = pd.DataFrame(i.fit_transform(X[scale_col]))
            df_scaled.columns = scale_col
            # Encoding
            if encode_col is not None:
                # if encoder is LabelEncoder
                if type(j) == type(LabelEncoder()):
                    for i in range(len(encode_col)):
                        df_encoded[encode_col[i]] = j.fit_transform(X[encode_col[i]])
                    df_prepro = pd.concat([df_scaled, df_encoded], axis=1)
                else:
                    df_encoded = j.fit_transform(X[encode_col])
                    df_encoded = pd.DataFrame(df_encoded)
                    df_encoded.columns = encode_col
                    df_prepro = pd.concat([df_scaled, df_encoded], axis=1)
            else:
                 df_prepro = df_scaled[pd.get_dummies(y)]
            y = y.replace(['TRUE', 'FALSE'], [1, 0])
            df_prepro = pd.DataFrame(df_prepro)
            df_selected = pd.DataFrame(df_prepro)

            for n in range(0, 1):
                if n == 0:
                    # Feature Selection Using the Select KBest (K = 6)
                    selectK = SelectKBest(score_func=f_regression, k=6).fit(df_prepro, y.values.ravel())
                    cols = selectK.get_support(indices=True)
                    df_selected = df_prepro.iloc[:, cols]
                    df_selected = df_selected.fillna(method='ffill')

                else:
                    # Feature Selection Using the RFE(n = 6) RFE-Recursive feature elimination
                    rfe = RFE(estimator=DecisionTreeClassifier(), n_features_to_select=6, step=3).fit(df_prepro, y.values.ravel())
                    cols = rfe.support_
                    for m in cols:
                        if m is True:
                            df_selected = df_prepro.iloc[:, cols]
                            df_selected = df_selected.fillna(method='ffill')

                smote = SMOTE(sampling_strategy='minority', random_state=3)
                df_smote, y_smote = smote.fit_resample(df_selected, y)

                for z in model:
                    # Split train, testset
                    X_train, X_test, y_train, y_test = train_test_split(df_smote, y_smote)

                    # Set hyperparameter
                    if model_param is None:
                        if model[0] is z:
                            param = parameter[0]
                        elif model[1] is z:
                            param = parameter[1]
                        elif model[2] is z:
                            param = parameter[2]
                    else:
                        param = parameter
                    # Set scorer
                    grid_scorer = make_scorer(overall_average_score, greater_is_better=True)

                    # Modeling(Using the RandomSearchCV)
                    random_search = RandomizedSearchCV(estimator=z, param_distributions=param,
                                                       n_jobs=N_JOBS, cv=setCV)
                    # scoring = grid_scorer
                    random_search.fit(X_train, y_train.values.ravel())
                    score = random_search.score(X_test, y_test)
                    print("Score = {:0.6f}".format(score), "")

                    if n == 0:
                        feat = 'SelectKBest'
                    else:
                        feat = 'RFE'
                    # Get predictied value and values for confusion matrix.
                    pred = random_search.predict(X_test)
                    prob = random_search.predict_proba(X_test)
                    prob_positive = prob[:, 1]
                    fpr, tpr, threshold = roc_curve(y_test, prob_positive)

                    # Find Best Score
                    if (best_score1 == 0 or best_score1 < score) and z is model[0]:
                        best_score1 = score
                        best_combination1['scaler'] = i
                        best_combination1['encoder'] = j
                        best_combination1['model'] = z
                        best_combination1['feature'] = feat
                        best_combination1['parameter'] = random_search.best_params_
                        best_combination1['report'] = classification_report(y_test, pred)
                        best_combination1['confusion'] = pd.DataFrame(confusion_matrix(y_test, pred))
                        best_combination1['f1'] = f1_score(y_test, pred, average='binary')
                        best_combination1['fpr'] = fpr
                        best_combination1['tpr'] = tpr
                        best_combination1['threshold'] = threshold

                    # Find Best Score
                    if (best_score2 == 0 or best_score2 < score) and z is model[1]:
                        best_score2 = score
                        best_combination2['scaler'] = i
                        best_combination2['encoder'] = j
                        best_combination2['model'] = z
                        best_combination2['feature'] = feat
                        best_combination2['parameter'] = random_search.best_params_
                        best_combination2['report'] = classification_report(y_test, pred)
                        best_combination2['confusion'] = pd.DataFrame(confusion_matrix(y_test, pred))
                        best_combination2['f1'] = f1_score(y_test, pred, average='binary')
                        best_combination2['fpr'] = fpr
                        best_combination2['tpr'] = tpr
                        best_combination2['threshold'] = threshold

                    # Find Best Score
                    if (best_score3 == 0 or best_score3 < score) and z is model[2]:
                        best_score3 = score
                        best_combination3['scaler'] = i
                        best_combination3['encoder'] = j
                        best_combination3['model'] = z
                        best_combination3['feature'] = feat
                        best_combination3['parameter'] = random_search.best_params_
                        best_combination3['report'] = classification_report(y_test, pred)
                        best_combination3['confusion'] = pd.DataFrame(confusion_matrix(y_test, pred))
                        best_combination3['f1'] = f1_score(y_test, pred, average='binary')
                        best_combination3['fpr'] = fpr
                        best_combination3['tpr'] = tpr
                        best_combination3['threshold'] = threshold

    # Print them
    print("\nLogistic Regression")
    print("Best Score = {:0.6f}".format(best_score1), "")
    print("Best f1 score = {:0.6f}".format(best_combination1['f1']), "")
    print("Best Combination, Model {}, Encoder {}, Scaler {}, Feature Select {}".
          format(best_combination1['model'], best_combination1['encoder'],
                 best_combination1['scaler'], best_combination1['feature']))
    print("Hyperparameter {}".format(best_combination1['parameter']))
    print("### Classification Report ###")
    print(best_combination1['report'])
    print("### Confusion Matrix ###")
    ax = sns.heatmap(best_combination1['confusion'], annot=True, cmap='Blues')
    ax.set_title("Confusion matrix")
    ax.set_xlabel("\n Predicted values")
    ax.set_ylable("Actual values")

    ax.xaxis.set_ticklabels(['False','True'])
    ax.yaxis.set_ticklabels(['False','True'])
    plt.show()



    # Print SVC
    print("\nSVC")
    print("Best Score = {:0.6f}".format(best_score2), "")
    print("Best f1 score = {:0.6f}".format(best_combination2['f1']), "")
    print("Best Combination, Model {}, Encoder {}, Scaler {}, Feature Select {}".
          format(best_combination2['model'], best_combination2['encoder'],
                 best_combination2['scaler'], best_combination2['feature']))
    print("Hyperparameter {}".format(best_combination2['parameter']))
    print("### Classification Report ###")
    print(best_combination2['report'])
    print("### Confusion Matrix ###")
    ax = sns.heatmap(best_combination2['confusion'], annot=True, cmap='Blues')
    ax.set_title("Confusion matrix")
    ax.set_xlabel("\n Predicted values")
    ax.set_ylable("Actual values")

    ax.xaxis.set_ticklabels(['False', 'True'])
    ax.yaxis.set_ticklabels(['False', 'True'])
    plt.show()


    # Print GradientBossting Classifier
    print("\nGradientBoosting Classifier")
    print("Best Score = {:0.6f}".format(best_score3), "")
    print("Best f1 score = {:0.6f}".format(best_combination3['f1']), "")
    print("Best Combination, Model {}, Encoder {}, Scaler {}, Feature Select {}".
          format(best_combination3['model'], best_combination3['encoder'],
                 best_combination3['scaler'], best_combination3['feature']))
    print("Hyperparameter {}".format(best_combination3['parameter']))
    print("### Classification Report ###")
    print(best_combination3['report'])
    print("### Confusion Matrix ###")
    ax = sns.heatmap(best_combination3['confusion'], annot=True, cmap='Blues')
    ax.set_title("Confusion matrix")
    ax.set_xlabel("\n Predicted values")
    ax.set_ylable("Actual values")

    ax.xaxis.set_ticklabels(['False', 'True'])
    ax.yaxis.set_ticklabels(['False', 'True'])
    plt.show()
    # ROC curve
    plt.rcParams['figure.figsize'] = [10,8]
    plt.style.use("bmh")

    plt.title("ROC Curve", fontsize=15)
    plt.xlabel("False Positive Rate", fontsize=15)
    plt.ylabel("True Positive Rate", fontsize=15)

    plt.plot(best_combination1['fpr'], best_combination1['tpr'], color='red')
    plt.plot(best_combination2['fpr'], best_combination2['tpr'], color='blue')
    plt.plot(best_combination3['fpr'], best_combination3['tpr'], color='green')
    plt.gca().legend(['Logistic', 'SVC', 'GBC'], loc='lower right', frameon=True)

    plt.plot([0,1],[0,1], linestyle='--', color='black')
    plt.show()



    return


# Auto Find Best Accuracy
# print("Auto Find Best Accuracy")
FindBestAccruacy(X_data, y_data, scale_col=scale_col, encode_col=encode_col,
                 encoders=None, scalers=None, models=None, model_param=None)
