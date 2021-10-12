# Import Class Libraries
import eyeball as eyeball
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import purity as purity
import seaborn as sns
import sklearn
import plotly.express as px
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from scipy.stats import stats
from pyclustering.cluster.clarans import clarans
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans, DBSCAN, OPTICS
from sklearn.decomposition import PCA
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.metrics import silhouette_score
sns.set()

##############################################################################
# AutoML (X, y = None, scale_col, encode_col, scalers = None, encoders = None,
#         feature_param = None, models = None, model_param = None,
#         scores = None, score_param = None)
#
# **************************************************************************************
# ******************************** Must Read *******************************************
# **************************************************************************************
# Description = When parameters are put in, the plot and scores are output
#               The method of producing results in AutoML function consists of three main steps
#
#           Step 1 = Feature Selection (PCA(), RandomSelect(), CustomSelect()) * model (KMeans(), GMM(), clarans(), DBSCAN(), OPTICS()) = 15,
#                    Find a combination with the best silhouette score in each combination
#
#           Step 2 = If there is a target value, Among the three Feature Selection (PCA(), RandomSelect(), CustomSelect()),
#                    check which model has the highest purity and return three results
#
#           Step 3 = Using the final three combinations (without a target value),
#                    we compare with the combinations (with a target value)
#               - The results are checked through the clustering plot and the silhouette score -
# ***************************************************************************************
# ***************************************************************************************
#
# Input = X: Data Feature
#         Y: Data Target (If you have a target value, enter it)
#         Scale_col: columns to scaled
#         Encode_col: columns to encode
#         Scalers: list of scalers
# 	         None: [StandardScaler(), RobustScaler(), MinMaxScaler()]
#            If you want to scale other ways, then put the scaler in list.
#         Encoders: list of encoders
# 	         None: [OrdinalEncoder(), LabelEncoder()]
# 	         If you want to encode other ways, then put the encoder in list.
#
#         Feature: list of features
#            None: [PCA(), RandomSelect(), CustomSelect()]
#            If you want to set other ways, then put specific feature in list
#
#         Feature_param: feature selection method's parameter
#            PCA()'s None: [n_components: None (int)]
#            RandomSelect()'s None: [number_of_features: None (int)]
#            CustomSelect()'s None: [combination_of_features: None (list)]
#
#         Models: list of models
#            None: [KMeans(), GMM(), clarans(), DBSCAN(), OPTICS()]
#            If you want to fit other ways, then put (Clustering)model in list.
#
#         Model_param: list of model's hyperparameter
#         KMeans()’s None: [n_clusters: None (int), init: None(k-means++, random),
#                           n_init: None (int), Random_state: None (int), max_iter: None (int)]
#         GMM()’s None: [n_components: None (int), covariance_type: None (spherical, tied, diag),
#                        n_init: None (int), Random_state: None (int),
#                        min_covar: None (float), tol: None (float)]
#         clarans()’s None: [number_clusters: None (int), numlocal_minima: None (int),
#                            max_neighbor: None (int)]
#         DBSCAN()’s None: [eps: None (float), min_samples: None (int), metric: None (str or callable),
#                           p: None (float), Algorithm: None (auto, ball_tree, kd_tree, brute)]
#         OPTICS()’s None: [eps: None (float), min_samples: None (int), p: None (int),
#                           cluster_method: None (xi, dbscan), algorithm: None (auto, ball_tree, kd_tree, brute)]
#         If you want to set other ways, then put the hyperparameter in list
#
#         Scores: list of score methods
#            None: [silhouette_score(), KelbowVisualizer(), purity(), eyeball()]
#            If you want to see other ways, then put the scoring model in list.
#
#         Score_param: list of score method's hyperparameter
#      		 Silhouette_score()’s None: [metric: None (str, callable), random_state: None (int)]
#            Purity()’s None: None
#            eyeball()'s None: None
#
# Output = some scores, plots

# Description = Calculate the silhouette score and return the value
# Input  = kind of model, Dataset
# Output = Silhouette score
def cv_silhouette_scorer(estimator, X):
    print("그리드 서치중 : ", estimator)

    # If GMM(EM) handle separately
    if type(estimator) is sklearn.mixture._gaussian_mixture.GaussianMixture:
        # print("it's GaussianMixture()")
        labels = estimator.fit_predict(X)
        return silhouette_score(X, labels, metric='euclidean')

    # Calculate and return Silhouette score
    else:
        estimator.fit(X)
        cluster_labels = estimator.labels_
        num_labels = len(set(cluster_labels))
        num_samples = len(X.index)
        if num_labels == 1 or num_labels == num_samples:
            return -1
        else:
            return silhouette_score(X, cluster_labels)


# purity를 구해주는 함수
def purity_score(y_true, y_pred):
    # compute contingency matrix
    contingency_matrix = metrics.cluster.contingency_matrix(y_true, y_pred)
    # return purity
    return np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix)


# Description = randomly determines features
# input  = Dataset, number of feature
# Output = (Random)Dataset
class RandomSelect:
    # number of feature (Default:4)
    n = 4

    # Accept N
    def set_params(self, n_components):
        self.n = n_components

    # Pick N and combination
    def fit_transform(self, data):
        choice = np.random.choice(data.columns, self.n)
        result = pd.DataFrame(data[choice[0]])

        for i in range(1, len(choice)):
            result = pd.concat([result, data[choice[i]]], axis=1)

        # Return Dataset
        return result


# Description = select specific features
# input  = Dataset, selected features
# Output = (Selected) Dataset
class CustomSelect:
    # Combination of selected features
    feature = None

    # Accept selected features
    def set_params(self, n_components):
        self.feature = n_components

    # Combine the selected features
    def fit_transform(self, data):
        result = pd.DataFrame(data[self.feature[0]])

        for i in range(1, len(self.feature)):
            result = pd.concat([result, data[self.feature[i]]], axis=1)

        # Return Dataset
        return result


# Description = It converts data according to each feature selection method
#               If PCA is reset column name
#               If RandomSelect is randomly determines features
#               If CustomSelect is select specific features
# Input  = Dataset, selected feature, number of feature
# Output = (Processed) Dataset
def makefeatureSubset(X, selection, n_feature):
    selection.set_params(n_components=n_feature)
    x_result = selection.fit_transform(X)
    x_result = pd.DataFrame(x_result)

    # Reset column name
    if type(selection) == type(PCA()):
        if n_feature == 3:
            x_result.columns = ["Principle-1", "Principle-2", "Principle-3"]

        elif n_feature == 4:
            x_result.columns = ["Principle-1", "Principle-2", "Principle-3", "Principle-4"]

        elif n_feature == 5:
            x_result.columns = ["Principle-1", "Principle-2", "Principle-3", "Principle-4", "Principle-5"]

    return x_result


def AutoML(X, y=None, scale_col=None, encode_col=None, scalers=None, encoders=None,
           features=None, feature_param=None, models=None, model_param=None,
           scores=None, score_param=None):

    # Set Encoder
    global df_score_and_encode, df_first_scaled, df_new_score_and_encode

    if encoders is None:
        encode = [OrdinalEncoder(), LabelEncoder()]
    else:
        encode = encoders

    # Set Scaler
    if scalers is None:
        scale = [StandardScaler(), MinMaxScaler(), RobustScaler()]
    else:
        scale = scalers

    # Set Feature
    # If it's None value, select all features, set PCA, selected features, random select
    if features is None:
        feature = [PCA(), RandomSelect(), CustomSelect()]
        customSelectParameter = [["longitude", "latitude"], ["total_rooms", "total_bedrooms"],
                                  ["longitude", "latitude", "total_rooms", "total_bedrooms"],
                                  ["total_rooms", "total_bedrooms", "population", "households", "median_income"]]
        feature_parameter = [[3, 4, 5], [3, 4, 5], customSelectParameter]

    else:
        feature = features
        feature_parameter = feature_param

    # Set Model
    if models is None:
        model = [KMeans(), GaussianMixture(), '''clarans()''', DBSCAN(), OPTICS()]
    else:
        model = models

    # Set Model parameter
    if model_param is None:
                           # KMeas Clustering
        model_parameter = [{'n_clusters': [2, 3, 4, 5], 'init': ["k-means++", "random"],
                            'n_init': [1, 10, 20], 'random_state': ["None", 0, 1],
                            'max_iter': [100, 200]},

                           # GMM(EM) Clustering
                           {'n_components': [2, 3, 4, 5], 'max_iter': [100, 200],
                            'covariance_type': ["spherical", "tied", "diag"],
                            'n_init': [1, 10, 20], 'random_state': ["None", 0, 1],
                            'min_covar': [1e-5, 1e-3], 'tol': [1e-5, 1e-3]},

                           # Clarans Clustering
                           # {'number_clusters': [2, 3, 4, 5], 'numlocal_minima': [2, 3, 4],
                           #  'max_neighbor': [2, 3, 4, 5]},

                           # DBSCAN Clustering
                           {'eps': [0.3, 0.4, 0.5], 'min_samples': [2, 3, 4, 5],
                            'metric': ["euclidean", "manhattan"], 'p': [1, 2, 3],
                            'algorithm': ["auto", "ball_tree", "kd_tree", "brute"]},

                           # Optics Clustering
                           {'eps': [0.3, 0.4, 0.5],
                            'min_samples': [2, 3, 4, 5], 'p': [1, 2, 3],
                            'cluster_method': ["xi", "dbscan"],
                            'algorithm': ["auto", "ball_tree", "kd_tree", "brute"]}]
    else:
        model_parameter = model_param

    # Set Score
    if scores is None:
        score = ['''silhouette_score(), purity(), eyeball()''']
    else:
        score = scores

    # Set Score parameter
    if score_param is None:
        score_parameter = [None]
    else:
        score_parameter = score_param

    # First Step's values (feature selection 3 * model 5) using silhouette score
    # [  PCA , Model1][  PCA , Model2]...[  PCA , Model5]
    # [Random, Model1][Random, Model2]...[Random, Model5]
    # [Custom, Model1][Custom, Model2]...[Custom, Model5]
    firstScore = [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]
    firstScoreScaler = [[None, None, None, None, None], [None, None, None, None, None], [None, None, None, None, None]]
    firstScoreEncoder = [[None, None, None, None, None], [None, None, None, None, None], [None, None, None, None, None]]
    firstScoreFeature = [[None, None, None, None, None], [None, None, None, None, None], [None, None, None, None, None]]
    firstScoreModel = [[None, None, None, None, None], [None, None, None, None, None], [None, None, None, None, None]]
    firstScoreParameter = [[None, None, None, None, None], [None, None, None, None, None],
                           [None, None, None, None, None]]

    cv = [(slice(None), slice(None))]

    # Second Step's values (feature selection 3) using purity and target value (If you have that)
    secondScore = [0, 0, 0]
    secondScoreScaler = [None, None, None]
    secondScoreEncoder = [None, None, None]
    secondScoreFeature = [None, None, None]
    secondScoreModel = [None, None, None]
    secondScoreParameter = [None, None, None]

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
                    df_prepro = pd.concat([df_scaled, df_encoded], axis=1)

                else:
                    print("NO")
                    dum = pd.DataFrame(pd.get_dummies(X[encode_col]))
                    df_prepro = pd.concat([df_scaled, dum], axis=1)

            else:
                df_prepro = df_scaled

            # feature selection (find feature subset : PCA, random select, custom select)
            featureIndex = 0
            for z, z_param in zip(feature, feature_parameter):
                modelIndex = 0
                for m in model:
                    for z_param_index in z_param:

                        # Step1 - Compare Silhouette score
                        # Feature Selection(PCA(), RandomSelection(), CustomSelect()) *
                        # model(KMeans(), GMM(), clarans(), DBSCAN(), OPTICS()) = 15
                        # Find a Combination with the best silhouette score in each combination

                        # If feature selection is PCA -> Iterate n_components 3,4,5
                        # If feature selection is RandomSelect -> Iterate n_components 3,4,5
                        # If feature selection is CustomSelect -> Iterate subset
                        # A feature subset that fits the selection and parameter came out
                        df_featureSubset = makefeatureSubset(df_prepro, z, z_param_index)

                        gridSearch = GridSearchCV(estimator=m, param_grid=model_parameter[modelIndex],
                                                  scoring=cv_silhouette_scorer, cv=cv)
                        # fit grid search
                        result = gridSearch.fit(df_featureSubset)
                        best_model = result.best_estimator_
                        best_params = result.best_params_
                        pred = best_model.fit_predict(df_featureSubset)
                        score = silhouette_score(df_featureSubset, pred)
                        print("현재 selection : ", z, "\n현재 모델 : ", m)
                        print(best_model)
                        print(best_params)
                        print("score: ", score)

                        if firstScore[featureIndex][modelIndex] == 0 or firstScore[featureIndex][modelIndex] < score:
                             print(featureIndex)
                             print(modelIndex)
                             print(i)
                             firstScore[featureIndex][modelIndex] = score
                             firstScoreScaler[featureIndex][modelIndex] = i
                             firstScoreEncoder[featureIndex][modelIndex] = j
                             firstScoreFeature[featureIndex][modelIndex] = df_featureSubset.columns
                             firstScoreModel[featureIndex][modelIndex] = best_model
                             firstScoreParameter[featureIndex][modelIndex] = best_params

                modelIndex += 1
                featureIndex += 1

    # Print step1's result
    for i in range(0, 3):
        for j in range(0, 5):
            print("최종 결과", i, " ", j)
            print(firstScoreScaler[i][j])
            print(firstScoreEncoder[i][j])
            print(firstScoreFeature[i][j])
            print(firstScoreModel[i][j])
            print(firstScoreParameter[i][j])
            print(firstScore)
            print(print())

    # Step 2 = If there is a target value, Among the three Feature Selection (PCA(), RandomSelect(), CustomSelect()),
    #          check which model has the highest purity and return three results
    for a in range(1, 3):
        for b in range(0, 3):

            # scale_col scaling => X[scale_col]
            if firstScoreScaler[a][b] is not None:  # If exist scaler
                df_first_scaled = pd.DataFrame(firstScoreScaler[a][b].fit_transform(X[scale_col]))
                df_first_scaled.columns = scale_col
            # else:   # If not exist scaler
            # df_first_scaled = X

            # encode_col encoding => X[encode_col]
            if firstScoreEncoder[a][b] is not None:  # If exist encoder
                df_first_encoded = pd.DataFrame(firstScoreEncoder[a][b].fit_transform(X[encode_col]))
                df_first_encoded.columns = encode_col
                # scaled + encoded
                df_score_and_encode = pd.concat([df_first_scaled, df_first_encoded], axis=1)
            # else:   # If not exist encoder
            # df_score_and_encode = df_first_scaled

            # print("**** Combination of Score and Encode ****\n")
            # print(df_score_and_encode)

            # Extract only features from feature_selection from scaling and encoded data frames.
            if firstScoreFeature[a][b] is not None:
                first_fture = []
                for k in firstScoreFeature[a][b]:
                    first_fture.append(k)

                df_new_score_and_encode = df_score_and_encode[first_fture]
                print("**** Apply feature selection ****\n")
                print(df_new_score_and_encode)

            df_values = df_new_score_and_encode.values

            # model fitting
            if firstScoreModel[a][b] is not None:
                pred_val = firstScoreModel[a][b].fit_predict(df_new_score_and_encode)
                # print("predict value shape: {}".format(pred_val.shape))
                # print("**** Predicted Value ****\n")
                # print(pred_val)

            min_y = np.min(y)
            max_y = np.max(y)
            gap = max_y - min_y
            gap /= len(np.unique(pred_val))

            labels = []
            for i in range(len(np.unique(pred_val))):
                labels.append(i)

            temp_df = pd.cut(y["median_house_value"], bins=len(np.unique(pred_val)), labels=labels, include_lowest=True)
            temp_df = temp_df.to_numpy()

            print("**** Purity Score ****")
            purityScore=purity_score(temp_df, pred_val)
            print(purityScore)

            if purityScore > secondScore[a]:
                secondScore[a]=purityScore
                secondScoreScaler[a] = firstScoreScaler[a][b]
                secondScoreEncoder[a] = firstScoreEncoder[a][b]
                secondScoreFeature[a]= firstScoreFeature[a][b]
                secondScoreModel[a] = firstScoreModel[a][b]
                secondScoreParameter[a] = firstScoreParameter[a][b]

    # Print step2's result
    print(secondScore)
    print(secondScoreScaler)
    print(secondScoreEncoder)
    print(secondScoreFeature)
    print(secondScoreModel)
    print(secondScoreParameter)

    # Step 3 = Using the final three combinations (without a target value),
    #          we compare with the combinations (with a target value)
    #        - The results are checked through the clustering plot and the silhouette score -

    for i in range(1,3):
        # scale_col scaling => X[scale_col]
        if secondScoreScaler[i] is not None:  # If exist scaler
            df_second_scaled = pd.DataFrame(secondScoreScaler[i].fit_transform(X[scale_col]))
            y_second_scaled=pd.DataFrame(secondScoreScaler[i].fit_transform(y))
            y_second_scaled.columns=y.columns
            df_second_scaled.columns = scale_col

        # encode_col encoding => X[encode_col]
        if secondScoreEncoder[i] is not None:  # If exist encoder
            df_second_encoded = pd.DataFrame(secondScoreEncoder[i].fit_transform(X[encode_col]))
            df_second_encoded.columns = encode_col
            # scaled + encoded
            df_score_and_encode = pd.concat([df_second_scaled, df_second_encoded], axis=1)

        # Extract only features from feature_selection from scaling and encoded data frames.
        if secondScoreFeature[i] is not None:
            second_fture = []
            for k in secondScoreFeature[i]:
                second_fture.append(k)
            df_new_score_and_encode = df_score_and_encode[second_fture]
            df_new_score_and_encode_y=pd.concat([df_new_score_and_encode,y_second_scaled],axis=1)

        # Using the plot and silhouette score
        # Compare the clustering results with medianHouseValue(target)
        # feature values in the original dataset

        # Without target value
        model = secondScoreModel[i]
        print(secondScoreScaler[i])
        print(secondScoreEncoder[i])
        print(secondScoreFeature[i])
        print(secondScoreModel[i])
        print(secondScoreParameter[i])
        cluster_no=secondScoreModel[i].fit(df_new_score_and_encode)
        label=cluster_no.labels_
        print(label)

        fig = px.scatter(df_new_score_and_encode, color=label)
        fig.show()

        pred_no = cluster_no.fit_predict(df_new_score_and_encode)
        score = silhouette_score(df_new_score_and_encode, pred_no)
        print("Silhouette score = ", score)

        # With target value
        cluster_yes = secondScoreModel[i].fit(df_new_score_and_encode_y)
        label=cluster_yes.labels_
        print(label)

        fig = px.scatter(df_new_score_and_encode_y, color=label)
        fig.show()

        pred_yes = cluster_yes.fit_predict(df_new_score_and_encode_y)
        score = silhouette_score(df_new_score_and_encode_y, pred_yes)
        print("Silhouette score = ", score)

##############################################################################
# Dataset = California housing price
# Feature = longitude, latitude, housing_median_age, total_rooms, total_bedrooms
#           population, households, median_income, ocean_proximity
# Target = median_house_value

# Number of dataset = 20640
# Numerical value = longitude, latitude, housing_median_age, total_rooms, total_bedrooms,
#                   population, households, median_income, median_house_value
# Categorical value = ocean_proximity

df = pd.read_csv("housing.csv")

feature_label = ['longitude', 'latitude', 'housing_median_age', 'total_rooms', 'total_bedrooms',
                 'population', 'households', 'median_income', 'ocean_proximity']
target_label = ['median_house_value']

# Print housing data's information
# print("\n***************** housing ****************")
# print(df.head())
#
# print("\n************** Description ***************")
# print(df.describe())
#
# print("\n************** Information ***************")
# print(df.info())

# Check null value
# print("\n************** Check null ***************")
# print(df.isna().sum())

# Fill null value
df['total_bedrooms'].fillna(df['total_bedrooms'].mean(), inplace=True)
# print("\n******** Check null (Cleaned data) ******")
# print(df.isna().sum())

# Remove Outliers with z-score
# Description = Use the z-score to handle outlier over mean +- 3SD
# Input  = dataframe's column
# Output = index
df_cate = df['ocean_proximity']

def find_outliers(col):
    z = np.abs(stats.zscore(col))
    idx_outliers = np.where(z > 3, True, False)
    return pd.Series(idx_outliers, index=col.index)


for n in range(len(feature_label)):
    idx = None
    idx = find_outliers(df.iloc[:, n])
    df = df.loc[idx == False]

# print("\n******** Removed Outlier ******")
# print(df.info())

# Set X, y data
y_data = df.loc[:, target_label]
X_data = df.drop(target_label, axis=1)

scale_col = ["longitude", "latitude", "housing_median_age", "total_rooms", "total_bedrooms", "population", "households",
             "median_income"]
end_col = ["ocean_proximity"]

AutoML(X_data,y_data, scale_col=scale_col, encode_col=end_col, models=None, model_param=None)