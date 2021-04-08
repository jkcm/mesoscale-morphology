
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import plot_partial_dependence, permutation_importance
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import plot_confusion_matrix
from joblib import dump, load

data_dict = pickle.load(open('/home/disk/eos4/jkcm/Data/MEASURES/classified_data/PCA_data.pickle' ,'rb'))
# x_train = data_dict['x_train']
# y_train = data_dict['y_train']
# x_val = data_dict['x_val']
# y_val = data_dict['y_val']
x_data = data_dict['x_all']
y_data = data_dict['y_all']
labels={0: 'Closed-cellular MCC', 1: 'Clustered cumulus', 2: 'Disorganized MCC',
        3: 'Open-cellular MCC', 4: 'Solid Stratus', 5: 'Suppressed Cu'}

k = x_data.shape[1]

# doing a grid search
n_estimators = [int(x) for x in np.linspace(start = 32, stop = 512, num = 16)]
# Number of features to consider at every split
max_features = ['auto', k, k//2, k-2]
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]
class_weight=['balanced_subsample']
max_samples=[0.1, 0.5, 1]
# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap,
               'class_weight': class_weight,
               'max_samples': max_samples}

rf = RandomForestClassifier()
rf_random = RandomizedSearchCV(
    estimator = rf, param_distributions = random_grid, n_iter = 30, cv = 4, verbose=2, n_jobs = 12)

rf_random.fit(x_data, y_data)

sc = rf_random.best_score_
savename = f'/home/disk/eos4/jkcm/Data/MEASURES/models/random_CV_search_pca_{sc:0.3}_{hash(rf_random)}.pickle'
print(savename)
dump(rf_random, savename)
