# Machine Learning Term Project <br> Team3 심우석, 이민서, 고효진

* Proposal - [Proposal_Team3 심우석, 이민서, 고효진.pptx](https://github.com/dntjr41/MachineLearning_termProject/files/7377451/Proposal_Team3.pptx)
* Final - [Final_Team3_심우석, 이민서, 고효진.pptx](https://github.com/dntjr41/MachineLearning_termProject/files/7569869/Final_Team3_.pptx)

***
# AutoML!
# Classification Manual
### Description = When parameters are put in, the highest score and the best model are printed <br> And Double check the parameter setting, add more parameters and compare the values.
***

### AutoML (X, y, scale_col, encode_col, scalers=None, encoders=None, <br> models=None, model_param=None, cv=None, n_jobs=None)
***

### Input = 

        X: Data Feature
        Y: Data Target

        Scale_col = columns to scaled
        Encode_col = columns to encoded

        Scalers: list of scalers
           None: [StandardScaler(), RobustScaler(), MinMaxScaler(), MaxAbsScaler()]

        Encoder: list of encoders
           None: [OrdinalEncoder(), LabelEncoder()]

        Models: list of models
           None: [LogisticRegression(), SVC(), GradientBoostingClassifier()]

        Model_param: list of model’s hyperparameter

        LogisticRegression()’s None: [penalty:(none,l2), random_state:(0,1), C:(0.01, 0.1, 1.0, 10.0, 100.0),
                                  solver:(lbfgs, sag, saga], max_iter:(10, 50, 100)]

        SVC()’s None: [random_state: (0,1), kernel: (linear, rbf, sigmoid),
                       C: (0.01, 0.1, 1.0, 10.0, 100.0), gamma: (scale, auto)]

        GradientBoostingClassifier()’s None: [loss:(deviance, exponential),learning_rate:(0.001, 0.1, 1),
                                 n_estimators:(1, 10,100,1000),  subsample: (0.0001,0.001, 0.1),
                                 min_samples_split:(10,50, 100, 300),min_samples_leaf:(5, 10, 15,50)]

        If you want to set other ways, then put the hyperparameter in list.

        Cv = cross validation’s K
           None: 5

        N_jobs = number of jobs to run in parallel. Training the estimator and computing score are
                 parallelized over the cross-validation splits
        None: 1

### Output = Best Accuracy, Combination, Plots, Final Result (Double Check AutoML)
***

# Clustering Manual
### Descripiton= When parameters are put in, the plot and scores are output <br> The method of producing results in AutoML function consists of four main steps

### Step 1 = Comparing using the silhouette score. <br> Feature selection(PCA(),RandomSelect(),CustomSelect()) * Model(KMeans(),GMM(),MeanShift()) = 9, <br> Find a combination with the best silhouette score in each combination

### Step 2 = Compare it once more using the purity score to select the best model. <br> Model(KMeans(),GMM(),MeanShift()) = 3

### Step 3 = Visualize using some plot for each model.

### Step 4 = Double check for the best model combination, add more parameters and compare the values.
***

### Input = 

        X: Data Feature
        Y(None): Data Target – If you have target value, put in the value (Not Requirements)

        Scale_col: columns to scales
        Encode_col: columns to encode
        Scalers: list of scalers
                  None:[StandardScaler(), RobustScaler()]
        Encoders: list of encoders
                  None:[OrdinalEncoder(),LabelEncoder()]

        Feature: list of features
                 None: [PCA(),RandomSelect(),CustomSelect()]

        Feature_param: feature selection method's parameter
                       PCA()'s None: [n_components: None(int)]
                       RandomSelect()'s None: [number_of_features: None(int)]
                       CustomSelect()'s None: [combination_of_features: None(list)]

        Models: list of models
                None:[KMeans(),GMM(),MeanShift()]

        Model_param: list of model's hyperparameter
                      KMeans()'s None:[n_clusters: None(int), init: None(k-means++, random),
                                 n_init: None(int), random_state:None(int)]
                      GMM()'s None:[n_components: None (int), covariance_type: None (spherical, tied, diag), 
                                 n_init: None (int), Random_state: None (int), tol: None (float)]
                      MeanShift()'s None:[bandwidth: None(int)]

        Scores: list of score methods
                None: [silhouette_score(), purity(), eyeball()]

        Score_param: list of score method's hyperparemeter
                      Silhouette_score()’s None: [metric: None (str, callable), random_state: None (int)]
                      Purity()’s None: None
                      eyeball()'s None: None

### Output = some scores, plots
***
