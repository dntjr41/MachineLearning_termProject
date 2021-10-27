# Import Class Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import purity
import seaborn as sns
from scipy.stats import stats
from sklearn.model_selection import RandomizedSearchCV, train_test_split, GridSearchCV
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, LabelEncoder
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler, MaxAbsScaler
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from imblearn.over_sampling import SMOTE
from sklearn.metrics import precision_recall_fscore_support, matthews_corrcoef, accuracy_score, make_scorer

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

df = pd.read_csv('online_shoppers_intention.csv')

print(list(df.columns.values))
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
print("\n***** Check null (Cleaned Data) *****")
print(df.isna().sum())

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

#Scoring function
def overall_average_score(actual,prediction):
    precision = precision_recall_fscore_support(actual, prediction, average = 'binary')[0]
    recall = precision_recall_fscore_support(actual, prediction, average = 'binary')[1]
    f1_score = precision_recall_fscore_support(actual, prediction, average = 'binary')[2]
    total_score = matthews_corrcoef(actual, prediction)+accuracy_score(actual, prediction)+precision+recall+f1_score
    return total_score/5
df.columns = df.columns.to_series().apply(lambda x: x.strip())
# Set X, y data
y_data = df.loc[:, target_label]
X_data = df.drop(target_label, axis=1)

def FindBestAccruacy(X, y, scale_col, encode_col, scalers=None, encoders=None,
                      models=None, model_param=None, cv=None, n_jobs=None):

    # Set Encoder
    if encoders is None:
        encode = [OrdinalEncoder(), OneHotEncoder(), LabelEncoder()]
    else: encode = encoders

    # Set Scaler
    if scalers is None:
        scale = [StandardScaler(), MinMaxScaler(), MaxAbsScaler(), RobustScaler()]
    else: scale = scalers

    # Set Model
    if models is None:
        model = [
        # LogisticRegression(),
        #        SVC(),
                 GradientBoostingClassifier()]
    else: model = models

    # Set Hyperparameter
    if model_param is None:
                    # LogisticRegression()
        parameter = [
                    # {'penalty':['none','l2'], 'random_state':[1, 2, 5, 10, 20], 'C':[0.01, 0.1, 1.0, 10.0, 100.0],
                    #   'solver':["lbfgs", "sag", "saga"], 'max_iter':[10, 50, 100]},
                    #  # SVC()
                    #  {'random_state': [1, 2, 5, 10, 20], 'kernel': ['linear', 'rbf', 'sigmoid'],
                    #   'C': [0.01, 0.1, 1.0, 10.0, 100.0], 'gamma': ['scale', 'auto']},
                    # GradientBoostingClassifier()
                     {'loss':['deviance','exponential'],
                      'learning_rate':[0.001, 0.1, 1],
                      'n_estimators':[1, 10,100,1000],
                      'subsample':[0.0001,0.001, 0.1],
                      'min_samples_split':[10,50, 100, 300],
                      'min_samples_leaf':[5, 10, 15,50]}
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

    best_score = 0
    best_combination = {}
    param = {}

    # SMOTE - Synthetic minority oversampling technique (Fixing the imbalanced data)
    target = y
    # smote = SMOTE(random_state=len(X))
    # X, y = smote.fit_resample(X, y)

    ####################################################################
    # Iterate
    for i in scale:
        for j in encode:
            # Scaling
            df_scaled = pd.DataFrame(i.fit_transform(X[scale_col]))
            df_scaled.columns = scale_col
            # Encoding
            if encode_col is not None:

                if type(j) == type(OrdinalEncoder()):
                    df_encoded = j.fit_transform(X[encode_col])
                    df_encoded = pd.DataFrame(df_encoded)
                    df_encoded.columns = encode_col
                    df_prepro = pd.concat([df_scaled, df_encoded], axis = 1)
                    #y=pd.DataFrame(j.fit_transform(y)) # todo
                else:
                    print("No")
                    dum = pd.DataFrame(pd.get_dummies(X[encode_col]))
                    df_prepro = pd.concat([df_scaled, dum], axis=1)
                    #y=pd.DataFrame(pd.get_dummies(y)) # todo
            else:
                df_prepro = df_scaled(pd.get_dummies(y))

            print(df_prepro.isna().sum())
            print(y.isna().sum())
            # smote = SMOTE(random_state=len(df_prepro))
            # df_prepro, y = smote.fit_resample(df_prepro, y)

            print(df_prepro.shape)
            print(y.shape)
            # Feature Selection Using the Select KBest (K = 6)
            selectK = SelectKBest(score_func=f_regression, k=6).fit(df_prepro, y.values.ravel())
            cols = selectK.get_support(indices=True)
            df_selected = df_prepro.iloc[:, cols]

            for z in model:
                print("model: ",z)
                print(z.get_params().keys())
                # Split train, testset
                X_train, X_test, y_train, y_test = train_test_split(df_selected, y)

                # Set hyperparameter
                if model_param is None:
                    if model[0] is z:
                        param = parameter[0]
                    elif model[1] is z:
                        param = parameter[1]
                    elif model[2] is z:
                        param = parameter[2]
                    elif model[3] is z:
                        param = parameter[3]

                else: param = parameter



                grid_scorer = make_scorer(overall_average_score, greater_is_better=True)

                # Modeling(Using the RandomSearchCV)
                random_search = RandomizedSearchCV(estimator=z, param_distributions=param, n_jobs=N_JOBS, scoring = grid_scorer, cv=setCV)
                result = random_search.fit(X_train, y_train.values.ravel())
                score = random_search.score(X_test, y_test)
                best_model = result.best_estimator_
                best_params = result.best_params_
                pred = best_model.predict(df_selected)


                # Find Best Score
                if best_score == 0 or best_score < score:
                    best_score = score
                    best_combination['scaler'] = i
                    best_combination['encoder'] = j
                    best_combination['model'] = z
                    best_combination['parameter'] = random_search.best_params_

    # Print them
    print("Best Score = {:0.6f}".format(best_score), "")
    print("Best Combination, Model {}, Encoder {}, Scaler {}".
          format(best_combination['model'], best_combination['encoder'], best_combination['scaler']))
    print("Hyperparameter {}".format(best_combination['parameter']))
    return


# Auto Find Best Accuracy
print("Auto Find Best Accuracy")
FindBestAccruacy(X_data, y_data, scale_col=scale_col, encode_col=encode_col,models=None, model_param=None )
