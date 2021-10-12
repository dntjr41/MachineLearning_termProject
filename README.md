# MachineLearning_termProject
Machine Learning term project

# Auto Clustering

############<br>
AutoML (X, y = None, scale_col, encode_col, scalers = None, encoders = None,
         feature_param = None, models = None, model_param = None,
         scores = None, score_param = None)

************************
******* Must Read ******
************************
Description = When parameters are put in, the plot and scores are output
              The method of producing results in AutoML function consists of three main steps

           Step 1 = Feature Selection (PCA(), RandomSelect(), CustomSelect()) * model (KMeans(), GMM(), clarans(), DBSCAN(), OPTICS()) = 15,
                    Find a combination with the best silhouette score in each combination

           Step 2 = If there is a target value, Among the three Feature Selection (PCA(), RandomSelect(), CustomSelect()),
                    check which model has the highest purity and return three results

           Step 3 = Using the final three combinations (without a target value),
                    we compare with the combinations (with a target value)
               - The results are checked through the clustering plot and the silhouette score -
**************************
**************************

Input = X: Data Feature
        Y: Data Target (If you have a target value, enter it)
        Scale_col: columns to scaled
        Encode_col: columns to encode
        Scalers: list of scalers
          None: [StandardScaler(), RobustScaler(), MinMaxScaler()]
          If you want to scale other ways, then put the scaler in list.
        Encoders: list of encoders
          None: [OrdinalEncoder(), LabelEncoder()]
          If you want to encode other ways, then put the encoder in list.

        Feature: list of features
          None: [PCA(), RandomSelect(), CustomSelect()]
          If you want to set other ways, then put specific feature in list

        Feature_param: feature selection method's parameter
          PCA()'s None: [n_components: None (int)]
          RandomSelect()'s None: [number_of_features: None (int)]
          CustomSelect()'s None: [combination_of_features: None (list)]

        Models: list of models
          None: [KMeans(), GMM(), clarans(), DBSCAN(), OPTICS()]
          If you want to fit other ways, then put (Clustering)model in list.

        Model_param: list of model's hyperparameter
          KMeans()’s None: [n_clusters: None (int), init: None(k-means++, random),
                           n_init: None (int), Random_state: None (int), max_iter: None (int)]
          GMM()’s None: [n_components: None (int), covariance_type: None (spherical, tied, diag),
                         n_init: None (int), Random_state: None (int),
                         min_covar: None (float), tol: None (float)]
          clarans()’s None: [number_clusters: None (int), numlocal_minima: None (int),
                             max_neighbor: None (int)]
          DBSCAN()’s None: [eps: None (float), min_samples: None (int), metric: None (str or callable),
                            p: None (float), Algorithm: None (auto, ball_tree, kd_tree, brute)]
          OPTICS()’s None: [eps: None (float), min_samples: None (int), p: None (int),
                            cluster_method: None (xi, dbscan), algorithm: None (auto, ball_tree, kd_tree, brute)]
          If you want to set other ways, then put the hyperparameter in list

          Scores: list of score methods
             None: [silhouette_score(), KelbowVisualizer(), purity(), eyeball()]
             If you want to see other ways, then put the scoring model in list.

          Score_param: list of score method's hyperparameter
        		 Silhouette_score()’s None: [metric: None (str, callable), random_state: None (int)]
             Purity()’s None: None
             eyeball()'s None: None

# Output = some scores, plots
