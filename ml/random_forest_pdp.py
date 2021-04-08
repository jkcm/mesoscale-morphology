import pickle
import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
# from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
# from sklearn.preprocessing import StandardScaler
from sklearn.inspection import plot_partial_dependence, permutation_importance
# from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, ExtraTreesClassifier


model_save_dict =  pickle.load(open(f'/home/disk/eos4/jkcm/Data/MEASURES/models/simple_rf_pca_model.pickle', "rb" )) 

rf = model_save_dict['model']
rot_norm_pcs_test = model_save_dict['x_test']
rot_norm_pcs = model_save_dict['x_train']
loadings = model_save_dict['loadings']
pct_var = model_save_dict['pct_var']
pct_var_rot = model_save_dict['pct_var_rot']

for target in rf.classes_[1:2]:
#     break
#     pdp = plot_partial_dependence(rf, rot_norm_pcs_test, features=[0, 1, 2, 3, 4, 5, 6, 7], feature_names=[f'PC{i+1}' for i in range(loadings.shape[0])], 
#                             target=target, verbose=True, n_jobs=4)
    pdp_2d = plot_partial_dependence(rf, rot_norm_pcs_test, features=[(0,1), (0,4), (2,6)], feature_names=[f'PC{i+1}' for i in range(loadings.shape[0])], 
                            target=target, verbose=True, n_jobs=16)
    pickle.dump(pdp_2d, open(f'/home/disk/eos4/jkcm/Data/MEASURES/pdp/ver1_pdp_target_2d_{int(target)}.pickle', "wb" ))

# for target in rf.classes_