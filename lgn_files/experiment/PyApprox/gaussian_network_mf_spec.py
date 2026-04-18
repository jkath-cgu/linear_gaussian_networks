
# %%
import os
import sys

os.chdir('../MFdfn_v10')

# %%

sys.path.append("../../")
from emulator import preProcess
from emulator.utilsPyapprox import *

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

np.random.seed(2)
# # Imports
# %matplotlib inline
# %config InlineBackend.figure_format = 'svg'


# %%
def getGraphs(data_folder, quantile, fidelity, verbose):
    worker = preProcess(quantile=quantile, fidelity=fidelity, verbose=verbose)
    worker.build_graphs_from_dfn_data(data_folder)

    return worker

# %%
def getFeatures(worker, numShortestPath, flux_calc_pct, verbose):
    X, y = worker.extract_features_from_graphs(k_shortest_path=numShortestPath, flux_calc_pct=flux_calc_pct, verbose=verbose)

    return X, y


# %%
# build hi-fi graphs from data
# worker_hf = getGraphs('../../math-clinic-data/data_stream_1/', '50 percent', 'high', False)
# worker_hf = getGraphs('../../math-clinic-data/data_stream_3/', '50 percent', 'high', False)
# worker_hf = getGraphs('../../math-clinic-data/alpha_25/', '50 percent', 'high', False)

# %%
# build lo-fi graphs from data
# worker_lf = getGraphs('../../math-clinic-data/data_stream_1/', '50 percent', 'low', False)
# worker_lf = getGraphs('../../math-clinic-data/data_stream_3/', '50 percent', 'low', False)
# worker_lf = getGraphs('../../math-clinic-data/alpha_25/', '50 percent', 'low', False)


#%%
# gn mf cross fold validation (hi-fi)
nfolds = 50  # 10
swap = True  # True False
# poly degree
degree_cfv_l = 2     # 3 (2) 1
degree_cfv_h = 2     # 3 (2) 1
# poly sigma
sigma_cfv_l = 3.0    # (3.5) 3.4 3.6 3.4 [k=5] (4.3) 4.1/4.2 4.4/4.5 4.2/4.3 [k=100] (10fs 30fs 50fs)    [50th pct]
sigma_cfv_h = 3.0    # (3.0)             [k=5]                                                           [90th pct]
# glm precision
beta_cfv_l  = 25.5   # 4.5
alpha_cfv_l = 1      # 6 to 10
beta_cfv_h  = 6.5    # 8.5 (k=5) to 11.5 (k=100) (30fcvs) / 10.85 (5fcvs) 10.2 (10fcvs) 8.71 (50fcvs)   [50th pct]
alpha_cfv_h = 1      #                                       7.45 (5fcvs)               6.8  (50fcvs)   [90th pct]
# model correlation
model_corr = 0.65    # (0.655) 0.70 ds1 to 0.61 ds2
# training data samples (X) + values (y)
X_cv_l = X_lf.copy()
y_cv_l = y_log_lf.copy()
# # exclude first 100 lo-fi networks
# X_cv_l = X_lf[100:].copy()
# y_cv_l = y_log_lf[100:].copy()
X_cv_h = X_hf.copy()
y_cv_h = y_log_hf.copy()

y_pred, y_pred_var, y_test = gn_mf_cv(X_cv_l, y_cv_l, X_cv_h, y_cv_h, degree_l=degree_cfv_l, degree_h=degree_cfv_h, sigma_l=sigma_cfv_l, sigma_h=sigma_cfv_h, alpha_l=alpha_cfv_l, beta_l=beta_cfv_l, alpha_h=alpha_cfv_h, beta_h=beta_cfv_h, model_corr=model_corr, nfolds=nfolds, swap_folds=swap)

# calculate absolute raw and percent errors
y_pred_error = np.abs(np.exp(y_pred) - np.exp(y_test))
y_pred_percent_error = 100 * (y_pred_error / np.exp(y_test))

swap_s = '_sw' if swap else ''
pd.DataFrame(y_pred_percent_error, columns=['median breakthrough time hi-fi (testing only), '
                                           f'abs percent difference: |y_mfnet_{nfolds}fcv{swap_s} - y_dfn| / y_dfn']).describe()

#%%
# percent observations withing 95% confidence intervals
y = np.exp(y_test)
y_pred_std = np.sqrt(y_pred_var)
y_lower = np.exp(y_pred-1.96*y_pred_std)
y_upper = np.exp(y_pred+1.96*y_pred_std)
isInCI = np.logical_and(y >= y_lower,y <= y_upper)
print(f'number of observations in 95% CI: {np.count_nonzero(isInCI)} / {len(y_pred)}, {(100*np.count_nonzero(isInCI)/len(y_pred)):.2f}')
# y_pred_std
pd.DataFrame(np.exp(y_pred_std), columns=['median breakthrough time hi-fi (testing only), '
                                          'y_predicted standard deviation']).describe()


#%%
# poly degree
degree_h = 2 # 1 2 3
degree_l = 2 # 1 2 3
# poly sigma
sigma_l = 2.0
sigma_h = 2.0
# glm precision
beta_h  = 15       # 10
beta_l  = 15       # 4.5
alpha_h = 1        # 0.05
alpha_l = 1        # 0.005

#%%
# hi-fi training data samples (X) + values (y)
k = 100
X_h = X_hf_list[k-1].copy() # iperm
# single feature
feature_index = 0
X_h = X_h[:,feature_index].reshape(-1,1)
y_h = y_log_hf.copy()

#%%
# lo-fi training data samples (X) + values (y)
k = 100
X_l = X_lf_list[k-1].copy() # iperm
# single feature
feature_index = 0
X_l = X_l[:,feature_index].reshape(-1,1)
y_l = y_log_lf.copy()

#%%
# normalize X (relative)
# find max of common features
X_l_max = X_l.max(axis=0).reshape(1,-1)
X_h_max = X_h.max(axis=0).reshape(1,-1)
_, N_features_l = X_l_max.shape
_, N_features_h = X_h_max.shape
# normalize X_l & X_h depending of features
X_all = np.array([])
if N_features_l == N_features_h:
   X_all = np.max([X_l_max, X_h_max],axis=0)
   X_l = X_l / X_all
   X_h = X_h / X_all
elif N_features_l > N_features_h:
   X_all = np.max([X_l_max[0,0:N_features_h].reshape(1,-1), X_h_max],axis=0)
   X_l_max[0,0:N_features_h] = X_all
   X_l = X_l / X_l_max
   X_h = X_h / X_all
elif N_features_l < N_features_h:
   X_all = np.max([X_l_max, X_h_max[0,0:N_features_l].reshape(1,-1)],axis=0)
   X_h_max[0,0:N_features_l] = X_all
   X_l = X_l / X_all
   X_h = X_h / X_h_max

#%%
# set random seed for training data
rng = np.random.RandomState(1)

#%%
# choose training & testing data
X, y = [X_l.copy(), X_h.copy()], [y_l.copy(), y_h.copy()]

# choose all data for training
# X_train, y_train = X, y

# choose random samples / values for training
X_train, y_train, index_train = choose_sample_value_rnd(X, y, lf_pct=1.0, hf_pct=0.012, rng=rng)

# choose quantile samples / values for training
# X_train, y_train, index_train = choose_sample_value_pct(X, y, rng=rng)

#%%
# construct the feature matrix
Phi_train_h = pce_basis_matrix(X_train[1], degree=degree_h, dist_bounds=[-15, 15])
Phi_train_l = pce_basis_matrix(X_train[0], degree=degree_l, dist_bounds=[-15, 15])
Phi_test_h = pce_basis_matrix(X[1], degree=degree_h, dist_bounds=[-15, 15])
Phi_test_l = pce_basis_matrix(X[0], degree=degree_l, dist_bounds=[-15, 15])

#%%
# Design matrix of test observations
Phi_train_h = gn_expand(X_train[1], bf=gn_gaussian_basis_function, bf_deg_args=np.linspace(0, 1, degree_h), sigma=[sigma_h])
Phi_train_l = gn_expand(X_train[0], bf=gn_gaussian_basis_function, bf_deg_args=np.linspace(0, 1, degree_l), sigma=[sigma_l])
Phi_test_h = gn_expand(X[1], bf=gn_gaussian_basis_function, bf_deg_args=np.linspace(0, 1, degree_h), sigma=[sigma_h])
Phi_test_l = gn_expand(X[0], bf=gn_gaussian_basis_function, bf_deg_args=np.linspace(0, 1, degree_l), sigma=[sigma_l])

#%%
# Mean and covariance matrix of posterior
model_corr = 0.95 # (0.65)
post_mean_l, post_cov_l, post_mean_h, post_cov_h = gn_mf_posterior(Phi_train_l, y_train[0], Phi_train_h, y_train[1], alpha_l=alpha_l, beta_l=beta_l, alpha_h=alpha_h, beta_h=beta_h, model_corr=model_corr, post_l=True)
# post_mean_h, post_cov_h = gn_mf_posterior(Phi_train_l, y_train[0], Phi_train_h, y_train[1], alpha_l=alpha_l, beta_l=beta_l, alpha_h=alpha_h, beta_h=beta_h, model_corr=model_corr, post_l=False)

#%%
# Mean and variances of posterior predictive 
y_pred_h, y_pred_var_h = gn_posterior_predictive(Phi_test_h, post_mean_h, post_cov_h, beta=beta_h)
y_pred_l, y_pred_var_l = gn_posterior_predictive(Phi_test_l, post_mean_l, post_cov_l, beta=beta_l)

#%%
# MAPE on training only (samples, data)
values = y[1].reshape(-1,1)
post_mean_y_pred = y_pred_h

# calculate absolute raw and percent errors
post_mean_errors = np.abs(np.exp(post_mean_y_pred) - np.exp(values))
post_mean_percent_errors_train = 100 * (post_mean_errors / np.exp(values))

pd.DataFrame(post_mean_percent_errors_train, columns=['median breakthrough time hi-fi (testing and training), '
                                                      'abs percent difference: |y_mfnet - y_dfn| / y_dfn']).describe()

# #%%
# # y_pred_var
# post_mean_y_var = y_pred_var_h
# pd.DataFrame(np.exp(post_mean_y_var), columns=['median breakthrough time hi-fi (testing and training), '
#                                                'log y_predicted variance']).describe()

#%%
# plot mfnet prediction
degree, post_mean, post_cov, beta = degree_h, post_mean_h, post_cov_h, beta_h
samples_train, values_train = [X_train[0].T.copy(), X_train[1].T.copy()], [y_train[0], y_train[1]]
samples_test, values_test = [X[0].T.copy(), X[1].T.copy()], [y[0], y[1]]
y_ranges=[9.5, 14.5] # [8.5, 12.5] [6, 11.5] [6, 14.5] [np.exp(6), np.exp(11.5)] [np.exp(6), np.exp(14.5)]
x_ranges=[0, 1] # [0, 1] [0, 0.86] [0, 0.768]
y_log = True # True False

dim1_index = 0

xx = np.linspace(0, 1, 100)
fig, axs = plt.subplots(1, 1, figsize=(8, 8)) # (6, 6)
training_labels = [r'$\mathrm{{log\: y\: dfn}_{lo-fi}\: train}$', r'$\mathrm{{log\: y\: dfn}_{hi-fi}\: train}$', r'$\mathrm{{log\: y\: dfn}_{hi-fi}\: test}$']
# training_labels = [r'$\mathrm{{y\: dfn}_{lo-fi}\: train}$', r'$\mathrm{{y\: dfn}_{hi-fi}\: train}$', r'$\mathrm{{y\: dfn}_{hi-fi}\: test}$']
plot_nd_mf_lvn_approx(xx, axs, degree, post_mean, post_cov, beta,
                     samples_train, values_train, samples_test, values_test, training_labels,
                     dim1_index, x_ranges=x_ranges, y_ranges=y_ranges, y_log=y_log,
                     pct=[50], # [50]
                     # pct=[25, 75, 50],
                     # pct=[10, 90, 25, 75, 50],
                     colors=['grey', 'tab:orange', 'lightgrey', 'tab:orange'], # b c r k tab:orange grey lightgrey
                     colors_pct=['green'])
                     # colors_pct=['turquoise', 'paleturquoise', 'green'])
                     # colors_pct=['lavender', 'lightcyan', 'turquoise', 'paleturquoise', 'green'])
# axs.set_xlabel(r'$\mathrm{network\: iperm\: /\: max\: iperm}$')
# axs.set_xlabel(r'$\mathrm{network\: fracture\: length\: /\: max\: length}$')
# axs.set_xlabel(r'$\mathrm{network\: mass\: flux\: /\: max\: flux}$')
# axs.set_xlabel(r'$\mathrm{network\: travel\: time\: /\: max\: time}$')
axs.set_xlabel(r'$\mathrm{network\: path\: length\: /\: max\: length}$')
axs.set_ylabel(r'$\mathrm{median\: breakthrough\: time}$')
plt.show()


# %%
