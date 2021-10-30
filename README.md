# MachineLearning PHW1 Auto Classification
AutoML - Classification Menual

# Code + Result file 
-> [PHW1_201636417 심우석.docx](https://github.com/dntjr41/MachineLearning_termProject/files/7446856/PHW1_201636417.docx)

# Dataset
-> https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/

# Auto Classification

<br>
FindBestAccruacy (X, y, scale_col, encode_col, scalers = None, encoders = None,
                   models = None, model_param = None, cv = None, n_jobs = None)

***************
Description = When parameters are put in, the highest accuracy and the best model are output
**************

Input = 

        X: Data Feature
        
        Y: Data Target
        
        Scale_col: columns to scaled
        
        Encode_col: columns to encode
        
        Scalers: list of scalers
          None: [StandardScaler(), RobustScaler(), MinMaxScaler(), MaxAbsScaler()]
           If you want to scale other ways, then put the scaler in list.

        Encoders: list of encoders
          None: [OrdinalEncoder(), OneHotEncoder(), LabelEncoder()]
          If you want to encode other ways, then put the encoder in list.

        Models: list of models
          None: [DecisionTreeClassifier(criterion=’entropy’), DecisionTreeClassifier(criterion=’gini’),
                 LogisticRegression(), SVC()]
           If you want to fit other ways, then put the (Classification)model in list.

        Model_param: list of model’s hyperparameter
        DecisionTreeClassifier(criterion=’entropy’)’s None: [random_state: None (int), max_depth: None (int),
                                              Max_features: None (auto, sqrt, log2), max_leaf_node: None (int)]
        DecisionTreeClassifier(criterion=’gini’)’s None: [random_state: None (int), max_depth: None (int),
                                              Max_features: None (auto, sqrt, log2), max_leaf_node: None (int)]
        LocisticRegression()’s None: [penalty: None (l2), random_state: None (int),
                                      C: None (float), Solver: None (lbfgs, sag or saga), max_iter: None (int)]
        SVC()’s None: [kernel: None (linear, rbf, sigmoid), random_state: None(int),
                                                    C: None (float), gamma: None (int)]
        If you want to set other ways, then put the hyperparameter in list.

        Cv = K-Fold cross validation’s K
             None: 5
             
        N_jobs = number of jobs to run in parallel. Training the estimator and computing score are
                 parallelized over the cross-validation splits
             None: 1

# Output = Best Model, Best Accuracy
