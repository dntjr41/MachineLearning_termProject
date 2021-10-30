# Import Class Libraries
import pandas as pd
import numpy as np
from scipy.stats import stats
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, LabelEncoder
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler, MaxAbsScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from imblearn.over_sampling import SMOTE
from sklearn.feature_selection import SelectKBest, f_classif, f_regression

##############################################################################################
# FindBestAccruacy (X, y, scale_col, encode_col, scalers = None, encoders = None,
#                   models = None, model_param = None, cv = None, n_jobs = None)
# Description = When parameters are put in, the highest accuracy and the best model are output
# Input = X: Data Feature
#         Y: Data Target
#         Scale_col: columns to scaled
#         Encode_col: columns to encode
#         Scalers: list of scalers
# 	         None: [StandardScaler(), RobustScaler(), MinMaxScaler(), MaxAbsScaler()]
#            If you want to scale other ways, then put the scaler in list.
#
#         Encoders: list of encoders
# 	         None: [OrdinalEncoder(), OneHotEncoder(), LabelEncoder()]
# 	         If you want to encode other ways, then put the encoder in list.
#
#         Models: list of models
#            None: [DecisionTreeClassifier(criterion=’entropy’), DecisionTreeClassifier(criterion=’gini’),
#                   LogisticRegression(), SVC()]
#            If you want to fit other ways, then put the (Classification)model in list.
#
#         Model_param: list of model’s hyperparameter
#         DecisionTreeClassifier(criterion=’entropy’)’s None: [random_state: None (int), max_depth: None (int),
#                                             Max_features: None (auto, sqrt, log2), max_leaf_node: None (int)]
#         DecisionTreeClassifier(criterion=’gini’)’s None: [random_state: None (int), max_depth: None (int),
#                                             Max_features: None (auto, sqrt, log2), max_leaf_node: None (int)]
#         LocisticRegression()’s None: [penalty: None (l2), random_state: None (int),
#                                     C: None (float), Solver: None (lbfgs, sag or saga), max_iter: None (int)]
#         SVC()’s None: [kernel: None (linear, rbf, sigmoid), random_state: None(int),
#                                                     C: None (float), gamma: None (int)]
#         If you want to set other ways, then put the hyperparameter in list.
#
#         Cv = K-Fold cross validation’s K
#            None: 5
#         N_jobs = number of jobs to run in parallel. Training the estimator and computing score are
#                  parallelized over the cross-validation splits
# 	        None: 1

# Output = Best Model, Best Accuracy

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
        model = [DecisionTreeClassifier(criterion='entropy'),
                 DecisionTreeClassifier(criterion='gini'),
                 LogisticRegression(), SVC()]
    else: model = models

    # Set Hyperparameter
    if model_param is None:
                    # DecisionTreeClassifier('Entropy')
        parameter = [{'criterion':['entropy'], 'random_state':[1, 2, 5, 10, 20], 'max_depth':[4, 6, 8, 10],
                      'max_features':["auto", "sqrt", "log2"], 'max_leaf_nodes':[2, 4, 6]},
                    # DecisionTreeClassifier('Gini')
                     {'criterion':['gini'], 'random_state':[1, 2, 5, 10, 20], 'max_depth':[4, 6, 8, 10],
                      'max_features':["auto", "sqrt", "log2"], 'max_leaf_nodes':[2, 4, 6]},
                    # LogisticRegression()
                     {'random_state':[1, 2, 5, 10, 20], 'penalty':['l2'], 'max_iter':[30000, 50000, 100000],
                      'C':[0.01, 0.1, 1.0, 10.0, 100.0], 'solver':["newton-cg", "lbfgs", "sag", "saga"]},
                    # SVC()
                     {'random_state':[1, 2, 5, 10, 20], 'kernel':['linear', 'rbf', 'sigmoid'],
                      'C':[0.01, 0.1, 1.0, 10.0, 100.0], 'gamma':['scale', 'auto']}]
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
    smote = SMOTE(random_state=len(X))
    X, y = smote.fit_resample(X, y)

    ####################################################################
    # Iterate
    for i in scale:
        for j in encode:

            # Scaling
            df_scaled = pd.DataFrame(i.fit_transform(X[scale_col]))

            # Encoding
            if encode_col is not None:
                if j == OrdinalEncoder():
                    df_encoded = j.fit_transform(X[encode_col])
                    df_prepro = pd.concat([df_scaled, df_encoded], axis=1)

                else:
                    dum = pd.DataFrame(pd.get_dummies(X[encode_col]))
                    df_prepro = pd.concat([df_scaled, dum], axis=1)

            else:
                df_prepro = df_scaled

            # Feature Selection Using the Select KBest (K = 6)
            selectK = SelectKBest(score_func=f_regression, k=6).fit(df_prepro, y.values.ravel())
            cols = selectK.get_support(indices=True)
            df_selected = df_prepro.iloc[:, cols]

            for z in model:

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

                # Modeling(Using the GridSearchCV)
                grid_search = GridSearchCV(estimator=z, param_grid=param, n_jobs=N_JOBS, cv=setCV)
                grid_search.fit(X_train, y_train.values.ravel())
                score = grid_search.score(X_test, y_test)

                # Find Best Score
                if best_score == 0 or best_score < score:
                    best_score = score
                    best_combination['scaler'] = i
                    best_combination['encoder'] = j
                    best_combination['model'] = z
                    best_combination['parameter'] = grid_search.best_params_

    # Print them
    print("Best Score = {:0.6f}".format(best_score), "")
    print("Best Combination, Model {}, Encoder {}, Scaler {}".
          format(best_combination['model'], best_combination['encoder'], best_combination['scaler']))
    print("Hyperparameter {}".format(best_combination['parameter']))
    return

#########################################################################################
# Read the dataset
# Dataset = The Wisconsin Cancer Dataset
# Feature = Sample code number, Clump Thickness, Uniformity of Cell Size,
#           Uniformity of Cell Shape, Marginal Adhension, Single Epithelial Cell Size,
#           Bare Nuclei, Bland Chromatin, Normal Nucleoli, Mitoses
# Target  = Class

# Number of Dataset = 699
# Numerical value = Sample code number, Clump Thickness, Uniformity of Cell Size,
#                   Uniformity of Cell Shape, Marginal Adhension, Single Epithelial Cell Size,
#                   Bare Nuclei, Bland Chromatin, Normal Nucleoli, Mitoses
# Categorical value = Class
feature_label = ['Sample_code_number', 'Clump_thickness', 'Uniformity_of_cell_size',
                 'Uniformity_of_cell_shape', 'Marginal_adhension', 'Single_epithelial_cell_size',
                 'Bare_nuclei', 'Bland_chromatin', 'Normal_nucleoli', 'Mitoses']
target_label = ['Class']

df = pd.read_csv("breast-cancer-wisconsin.data", header=None,
                 names=feature_label+target_label)

# Print Cancer data's information
# print("\n*************** Cancer ****************")
# print(df.head())

# print("\n************** Description ***************")
# print(df.describe())

# Check null value
# print("\n************** Check null ***************")
# print(df.isna().sum())

# Drop Sample code number(ID)
df = df.drop(["Sample_code_number"], axis=1)
feature_label.remove("Sample_code_number")

# Cleaning Dirty data
df = df.replace('?', np.NaN)
df = df.fillna(method='ffill')

# Casting type Bare_nuclei column
df['Bare_nuclei'] = pd.to_numeric(df['Bare_nuclei'])

# Cleaning Dirty Data, Remove outliers using Z score (>, < 3)
# Remove Outliers with z-score
# Description = Use the z-score to handle outlier over mean +- 3SD
# Input  = dataframe's column
# Output = index
def find_outliers(col):
    z = np.abs(stats.zscore(col))
    idx_outliers = np.where(z > 3, True, False)
    return pd.Series(idx_outliers, index=col.index)

for n in range(10):
    idx = None
    idx = find_outliers(df.iloc[:, n])
    df = df.loc[idx == False]


# print("\n******** Removed Outlier ******")
# print(df.describe())

# Set X, y data
y_data = df.loc[:, target_label]
X_data = df.drop(target_label, axis=1)

# Auto Find Best Accuracy
print("Auto Find Best Accuracy")
FindBestAccruacy(X_data, y_data, scale_col=feature_label, encode_col=None)

# Setting some values
print("\n\n\nSetting some values")
FindBestAccruacy(X_data, y_data, scale_col=feature_label, encode_col=None,
                 scalers=[StandardScaler(), MinMaxScaler()], encoders=[OneHotEncoder(), LabelEncoder()],
                 models=[DecisionTreeClassifier(criterion='entropy'), DecisionTreeClassifier(criterion='gini')],
                 model_param={'random_state':[1, 5, 10], 'max_depth':[4, 6, 8],
                      'max_features':["auto", "sqrt", "log2"], 'max_leaf_nodes':[2, 4, 6]})