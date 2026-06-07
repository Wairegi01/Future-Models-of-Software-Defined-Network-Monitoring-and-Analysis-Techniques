# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
import warnings
warnings.filterwarnings('ignore')

# Configure plots for notebook
# %matplotlib inline
plt.rcParams['figure.figsize'] = (14, 8)
sns.set_style("whitegrid")

print("✓ All libraries loaded successfully!")