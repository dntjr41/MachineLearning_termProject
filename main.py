# Import Class Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import purity as purity
import seaborn as sns
sns.set()
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