# Import Class Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import purity
import seaborn as sns
sns.set()
import sklearn
import plotly.express as px
from scipy.stats import stats
from pyclustering.cluster.clarans import clarans

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

feature_label = ['Administrative', 'Administrative_Duration', 'Informational',
                 'Informational_Duration', 'ProductRelated', 'ProductRelated_Duration',
                 'BounceRates', 'ExitRate', 'PageValue', 'SpecialDay', 'Browser', 'Region',
                 'TrafficType', 'VisitorType', 'Weekend', 'OperatingSystems', 'Month']
target_label = ['Revenue']

scale_col = ['Administrative', 'Administrative_Duration', 'Informational', 'Informational_Duration',
             'ProductRelated', 'ProductRelated_Duration', 'BounceRates',
             'ExitRates', 'PageValues', 'SpecialDay']
encode_col = ['Browser', 'Region', 'TrafficType', 'VisitorType',
              'Weekend', 'OperatingSystems', 'Month', 'Revenue']


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

# Set X, y data
y_data = df.loc[:, target_label]
X_data = df.drop(target_label, axis=1)