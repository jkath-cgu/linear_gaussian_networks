
# %%
import os
import sys

os.chdir('../MFdfn_v10')

# %%
sys.path.append("../../")
from emulator import preProcess
from emulator.utilsPyapprox import *
from emulator.utilsBayesianLinearRegression import *

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

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
    X_list, y = worker.extract_features_from_graphs(k_shortest_path=numShortestPath, flux_calc_pct=flux_calc_pct, verbose=verbose)

    return X_list, y


# %%
# build hi-fi graphs from data
# worker_hf = getGraphs('../../math-clinic-data/gpr_data_stream_1/', '50 percent', 'high', False)
# worker_hf = getGraphs('../../math-clinic-data/gpr_data_stream_2/', '50 percent', 'high', False)
# worker_hf = getGraphs('../../math-clinic-data/ds00/', '50 percent', 'high', False)
worker_hf = getGraphs('../../math-clinic-data/ds01/', '50 percent', 'high', False)
# worker_hf = getGraphs('../../math-clinic-data/ds02/', '50 percent', 'high', False)
# worker_hf = getGraphs('../../math-clinic-data/ds01/', '90 percent', 'high', False)
# worker_hf = getGraphs('../../math-clinic-data/ds02/', '90 percent', 'high', False)
# worker_hf = getGraphs('../../math-clinic-data/ds1/', '90 percent', 'high', False)
# worker_hf = getGraphs('../../math-clinic-data/ds2/', '50 percent', 'high', False)
# worker_hf = getGraphs('../../math-clinic-data/ds3/', '50 percent', 'high', False)
# worker_hf = getGraphs('../../math-clinic-data/ds4/', '50 percent', 'high', False)
# worker_hf = getGraphs('../../math-clinic-data/ds5/', '50 percent', 'high', False)

# %%
# build lo-fi graphs from data
# worker_lf = getGraphs('../../math-clinic-data/gpr_data_stream_1/', '50 percent', 'low', False)
# worker_lf = getGraphs('../../math-clinic-data/gpr_data_stream_2/', '50 percent', 'low', False)
# worker_lf = getGraphs('../../math-clinic-data/ds00/', '50 percent', 'low', False)
worker_lf = getGraphs('../../math-clinic-data/ds01/', '50 percent', 'low', False)
# worker_lf = getGraphs('../../math-clinic-data/ds02/', '50 percent', 'low', False)
# worker_lf = getGraphs('../../math-clinic-data/ds01/', '90 percent', 'low', False)
# worker_lf = getGraphs('../../math-clinic-data/ds02/', '90 percent', 'low', False)
# worker_lf = getGraphs('../../math-clinic-data/ds1/', '50 percent', 'low', False)
# worker_lf = getGraphs('../../math-clinic-data/ds2/', '50 percent', 'low', False)
# worker_lf = getGraphs('../../math-clinic-data/ds3/', '50 percent', 'low', False)
# worker_lf = getGraphs('../../math-clinic-data/ds4/', '50 percent', 'low', False)
# worker_lf = getGraphs('../../math-clinic-data/ds5/', '50 percent', 'low', False)

# %%
# get hi-fi features from graphs
k = 100  # 1 3 (5*) 10 15 (25) 50 ... 100 1000
flux_calc_pct = 50      # (55) (59 59)[k=5] (44..57 57)[k=100] 55 .. 25 .. 50
X_hf_list, y_hf = getFeatures(worker_hf, k, flux_calc_pct, False)
# log y_dfn
y_log_hf = np.log(np.array(y_hf.reshape(-1, 1), dtype='float64'))

# %%
# get lo-fi features from graphs
k = 100 # 1000
flux_calc_pct = 50      # (10) (16 13)[k=5] 10 .. 25 .. 50
X_lf_list, y_lf = getFeatures(worker_lf, k, flux_calc_pct, False)
# log y_dfn
y_log_lf = np.log(np.array(y_lf.reshape(-1, 1), dtype='float64'))

# %%
# save copy of X
# X_hfm = X_hf.copy()
# X_hfm_list, X_lfm_list = X_hf_list.copy(), X_lf_list.copy()

# # %%
# k = 100
# X_hfm = X_hf_list[k-1].copy()
# X_lfm = X_lf_list[k-1].copy()

# %%
# use network features as needed
# feature_list = [path_len, flux_pct, iperm_sum, time_pct, length_sum]
# feature_list = [path_len 30,    flux_pct 31/30, iperm_sum 33/29, time_pct 42/41,  length_sum 48] / ds1 hi-fi k=1000 (alpha)
# feature_list = [path_len 26/27, flux_pct 28,    iperm_sum 28/27, time_pct 425/41, length_sum 43] / ds2 hi-fi k=1000 (beta)
# feature_list = [path_len 29,    flux_pct 31/30, iperm_sum 32/28, time_pct 44/41,  length_sum 45] / ds1 hi-fi k=100  (delta)
# feature_list = [path_len 26/27, flux_pct 27/26, iperm_sum 28,    time_pct 303/40, length_sum 42] / ds2 hi-fi k=100  (gamma)
# # single feature
# feature_index = 0
# # X_hf = X_hfm[:,feature_index].reshape(-1,1)
# X_hf, X_lf = X_hfm[:,feature_index].reshape(-1,1), X_lfm[:,feature_index].reshape(-1,1)
# X_hf, X_lf = np.hstack([X_hfm[:,feature_index].reshape(-1,1), X_hfm[:,feature_index+5].reshape(-1,1)]), np.hstack([X_lfm[:,feature_index].reshape(-1,1), X_lfm[:,feature_index+5].reshape(-1,1)])
# # multi feature
# # X_hf = np.hstack([X_hfm[:,0].reshape(-1,1), X_hfm[:,1].reshape(-1,1)])
# # X_hf = np.hstack([X_hfm[:,0].reshape(-1,1), X_hfm[:,1].reshape(-1,1), X_hfm[:,2].reshape(-1,1)])
# # # X_lf = np.hstack([X_lfm[:,1].reshape(-1,1), X_lfm[:,3].reshape(-1,1)])
# # X_lf = np.hstack([X_lfm[:,0].reshape(-1,1), X_lfm[:,1].reshape(-1,1), X_lfm[:,2].reshape(-1,1)])
# # all features
# # X_hf = X_hfm
# # X_lf = X_lfm


#%%
# set random seed for training data
rng = np.random.RandomState(1)

# %%
# choose random indices
X_index = np.sort(rng.choice(np.arange(len(y_log_hf)), size=int(0.03*len(y_log_hf)), replace=False))
# print(X_index)

# %%
# LSF line to y_graph (x-axis) vs y_dfn (y-axis)
reg = LinearRegression().fit(y_log_lf[0:len(y_log_hf)], y_log_hf)
# reg = LinearRegression().fit(y_log_hf, y_log_lf)
# reg = LinearRegression().fit(y_log_lf, y_log_hf)
# reg = LinearRegression().fit(y_log_lf[X_index], y_log_hf[X_index])
reg_m = reg.coef_[0][0]
reg_b = reg.intercept_[0]

print(f'score(bias): {reg.score(y_log_lf[0:len(y_log_hf)], y_log_hf):.4f}')
# print(f'score(bias): {reg.score(y_log_hf, y_log_lf):.4f}')
print(f'm(bias): {reg_m:.4f}')
print(f'b(bias): {reg_b:.4f}')

# remove bias (want y_dfn = y_graph)
y_log_lf_ub = reg_m * y_log_lf.copy() + reg_b
# y_log_lf_ub = (1/reg_m) * (y_log_lf - reg_b)

# y_log_hf_trans = reg_m * y_log_hf.copy() + reg_b

# check correction for bias
reg_ub = LinearRegression().fit(y_log_lf_ub[0:len(y_log_hf)], y_log_hf)
# reg_ub = LinearRegression().fit(y_log_hf, y_log_lf_ub)
reg_m_ub = reg_ub.coef_[0][0]
reg_b_ub = reg_ub.intercept_[0]

print(f'score(unbiased): {reg_ub.score(y_log_lf_ub[0:len(y_log_hf)], y_log_hf):.4f}')
# print(f'score(unbiased): {reg_ub.score(y_log_hf, y_log_lf_ub):.4f}')
print(f'm(unbiased): {reg_m_ub:.4f}')
print(f'b(unbiased): {reg_b_ub:.4f}')

# reg.predict(np.array([[3, 5]]))

# %%
# calculate absolute raw and percent errors
y_pred_error = np.abs(np.exp(y_log_lf_ub[0:len(y_log_hf)]) - np.exp(y_log_hf))
# y_pred_error = np.abs(np.exp(y_log_lf_ub) - np.exp(y_log_hf))
# y_pred_error = np.abs(np.exp(y_log_lf[0:len(y_log_hf)]) - np.exp(y_log_hf))
# y_pred_error = np.abs(np.exp(data_ub[0:len(y_log_hf)]) - np.exp(y_log_hf))
y_pred_percent_error = 100 * (y_pred_error / np.exp(y_log_hf))

pd.DataFrame(y_pred_percent_error, columns=['median breakthrough time hi-fi (all y_dfn vs y_graph(unbiased)), '
                                           f'abs percent difference: |y_graph_ub - y_dfn| / y_dfn']).describe()


# %%
# gn cross fold validation (hi-fi)
nfolds = 10  # 10
swap = False # True False

# poly degree
degree_cfv = 2      # 1 (2) 3
# poly sigma
sigma_cfv = 0.25     # 1.0 (3.0)
# sigma_cfv = sigma_h
# glm precision
# beta_cfv  = 15.5     # 20.5 6.5 7.5 8.5 (11.5) 12.5 (k=3) 13.9 (k=100)    [50th pct]
# alpha_cfv = 1       # 8 (k=3) 8.5 (k=100) | 7.5 (k=3) 7 (k=100)     [90th pct]
beta_cfv  = beta_h
alpha_cfv = alpha_h

# training data samples (X) + values (y)
k = 100
X_cv = X_hf_list[k-1].copy()
y_cv = y_log_hf.copy()

y_pred, y_pred_var, y_test = gn_cv(X_cv, y_cv, degree=degree_cfv, sigma=sigma_cfv, alpha=alpha_cfv, beta=beta_cfv, nfolds=nfolds, swap_folds=swap)

# calculate absolute raw and percent errors
y_pred_error = np.abs(np.exp(y_pred) - np.exp(y_test))
y_pred_percent_error = 100 * (y_pred_error / np.exp(y_test))

# print(f'pct: {ii}, hi-fi MAPE: {np.mean(y_pred_percent_error):.2f}')

swap_s = '_sw' if swap else ''
# pd.set_option("max_colwidth", 10)
pd.DataFrame(y_pred_percent_error, columns=['median breakthrough time hi-fi (testing only), '
                                           f'abs percent difference: |y_sfnet_{nfolds}fcv{swap_s} - y_dfn| / y_dfn']).describe()
# df.describe()

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
# gn cross fold validation (lo-fi)
nfolds = 10
swap = False # True False

# poly degree
degree_cfv = 2      # 1 2 (3)
# poly sigma
# sigma_cfv = 0.25     # 1.0 (3.0)
sigma_cfv = sigma_l
# glm precision
# beta_cfv  = 12.5    # 12.5 15.5
# alpha_cfv = 1       # 3.05 (k=3) 3.4 (k=100) | 3.73 (k=3) 3.93 (k=100)      [90th pct]
beta_cfv  = beta_l
alpha_cfv = alpha_l

# training data samples (X) + values (y)
k = 100
X_cv = X_lf_list[k-1].copy()
y_cv = y_log_lf.copy()
# y_cv = y_log_lf_ub.copy()

y_pred, y_pred_var, y_test = gn_cv(X_cv, y_cv, degree=degree_cfv, sigma=sigma_cfv, alpha=alpha_cfv, beta=beta_cfv, nfolds=nfolds, swap_folds=swap)

# calculate absolute raw and percent errors
y_pred_error = np.abs(np.exp(y_pred) - np.exp(y_test))
y_pred_percent_error = 100 * (y_pred_error / np.exp(y_test))

# print(f'pct: {ii}, lo-fi MAPE: {np.mean(y_pred_percent_error):.2f}')

swap_s = '_sw' if swap else ''
pd.DataFrame(y_pred_percent_error, columns=['median breakthrough time lo-fi (testing only), '
                                           f'abs percent difference: |y_sfnet_{nfolds}fcv{swap_s} - y_dfn| / y_dfn']).describe()

#%%
# percent observations withing 95% confidence intervals
y = np.exp(y_test)
y_pred_std = np.sqrt(y_pred_var)
y_lower = np.exp(y_pred-1.96*y_pred_std)
y_upper = np.exp(y_pred+1.96*y_pred_std)
isInCI = np.logical_and(y >= y_lower,y <= y_upper)
print(f'number of observations in 95% CI: {np.count_nonzero(isInCI)} / {len(y_pred)}, {(100*np.count_nonzero(isInCI)/len(y_pred)):.2f}')
# y_pred_std
pd.DataFrame(np.exp(y_pred_std), columns=['median breakthrough time lo-fi (testing only), '
                                          'y_predicted standard deviation']).describe()


#%%
# poly degree
degree_h = 2 # 1 2 3
degree_l = 2 # 1 2 3
# glm precision
beta_h  = 15    #  15.5 10.0 1.26 8 9
beta_l  = 15    #  15.5 6.5
alpha_h = 1       #  0.05
alpha_l = 1       #  0.005

#%%
# hi-fi training data samples (X) + values (y)
k = 100
X_h = X_hf_list[k-1].copy() # iperm
# single feature
# feature_index = 0
# X_h = X_h[:,feature_index].reshape(-1,1)
X_h = X_h / X_h.max(axis=0) # norm perm / iperm
y_h = y_log_hf.copy()

#%%
# lo-fi training data samples (X) + values (y)
k = 100
X_l = X_lf_list[k-1].copy() # iperm
# single feature
# feature_index = 0
# X_l = X_l[:,feature_index].reshape(-1,1)
X_l = X_l / X_l.max(axis=0) # norm perm / iperm
y_l = y_log_lf.copy()

#%%
# set random seed for training data
rng = np.random.RandomState(1)

#%%
# choose training & testing data
X, y = [X_l.copy(), X_h.copy()], [y_l.copy(), y_h.copy()]

# choose all data for training
# X_train, y_train = X, y

# choose random samples / values for training
X_train, y_train, index_train = choose_sample_value_rnd(X, y, lf_pct=1.0, hf_pct=0.012, rng=rng) # hf_pct=0.012

# choose quantile samples / values for training
# X_train, y_train, index_train = choose_sample_value_pct(X, y, rng=rng)

#%%
# construct the feature matrix
pce_bounds = [-15, 15] # [-15, 15] [-100, 100]
Phi_train_h = pce_basis_matrix(X_train[1], degree=degree_h, dist_bounds=pce_bounds)
Phi_train_l = pce_basis_matrix(X_train[0], degree=degree_l, dist_bounds=pce_bounds)
Phi_test_h = pce_basis_matrix(X[1], degree=degree_h, dist_bounds=pce_bounds)
Phi_test_l = pce_basis_matrix(X[0], degree=degree_l, dist_bounds=pce_bounds)

#%%
# Design matrix of test observations
Phi_train_h = gn_expand(X_train[1], bf=gn_polynomial_basis_function, bf_deg_args=np.linspace(0, 1, degree_h+1)[1:])
Phi_train_l = gn_expand(X_train[0], bf=gn_polynomial_basis_function, bf_deg_args=np.linspace(0, 1, degree_l+1)[1:])
Phi_test_h = gn_expand(X[1], bf=gn_polynomial_basis_function, bf_deg_args=np.linspace(0, 1, degree_h+1)[1:])
Phi_test_l = gn_expand(X[0], bf=gn_polynomial_basis_function, bf_deg_args=np.linspace(0, 1, degree_l+1)[1:])
# Phi_train_h = gn_expand(X_train[1], bf=gn_polynomial_basis_function, bf_deg_args=range(1, degree_h + 1))
# Phi_train_l = gn_expand(X_train[0], bf=gn_polynomial_basis_function, bf_deg_args=range(1, degree_l + 1))
# Phi_test_h = gn_expand(X[1], bf=gn_polynomial_basis_function, bf_deg_args=range(1, degree_h + 1))
# Phi_test_l = gn_expand(X[0], bf=gn_polynomial_basis_function, bf_deg_args=range(1, degree_l + 1))

#%%
# Design matrix of test observations
gbf_sigma = [0.25]
Phi_train_h = gn_expand(X_train[1], bf=gn_gaussian_basis_function, bf_deg_args=np.linspace(0, 1, degree_h), sigma=gbf_sigma)
Phi_train_l = gn_expand(X_train[0], bf=gn_gaussian_basis_function, bf_deg_args=np.linspace(0, 1, degree_l), sigma=gbf_sigma)
Phi_test_h = gn_expand(X[1], bf=gn_gaussian_basis_function, bf_deg_args=np.linspace(0, 1, degree_h), sigma=gbf_sigma)
Phi_test_l = gn_expand(X[0], bf=gn_gaussian_basis_function, bf_deg_args=np.linspace(0, 1, degree_l), sigma=gbf_sigma)

#%%
# Mean and covariance matrix of posterior
post_mean_h, post_cov_h = gn_posterior(Phi_train_h, y_train[1], alpha=alpha_h, beta=beta_h)
post_mean_l, post_cov_l = gn_posterior(Phi_train_l, y_train[0], alpha=alpha_l, beta=beta_l)

#%%
# Mean and variances of posterior predictive 
y_pred_h, y_pred_var_h = gn_posterior_predictive(Phi_test_h, post_mean_h, post_cov_h, beta=beta_h)
y_pred_l, y_pred_var_l = gn_posterior_predictive(Phi_test_l, post_mean_l, post_cov_l, beta=beta_l)

#%%
# MAPE on training only (samples, data)
values = y[1].reshape(-1,1)
post_mean_y_pred = y_pred_h
# values = y[0].reshape(-1,1)
# post_mean_y_pred = y_pred_l

# calculate absolute raw and percent errors
post_mean_errors = np.abs(np.exp(post_mean_y_pred) - np.exp(values))
post_mean_percent_errors_train = 100 * (post_mean_errors / np.exp(values))

pd.DataFrame(post_mean_percent_errors_train, columns=['median breakthrough time hi-fi (testing and training), '
                                                      'abs percent difference: |y_sfnet - y_dfn| / y_dfn']).describe()
# pd.DataFrame(post_mean_percent_errors_train, columns=['median breakthrough time lo-fi (testing and training), '
#                                                       'abs percent difference: |y_sfnet - y_dfn| / y_dfn']).describe()

# #%%
# # y_pred_var
# post_mean_y_var = y_pred_var_h
# pd.DataFrame(np.exp(post_mean_y_var), columns=['median breakthrough time hi-fi (testing and training), '
#                                                'log y_predicted variance']).describe()


#%%
# plot sfnet prediction
degree, post_mean, post_cov, beta = degree_h, post_mean_h, post_cov_h, beta_h
samples_train, values_train = [X_train[1].T.copy()], [y_train[1].copy()]
samples_test, values_test = [X[1].T.copy()], [y[1]]
y_ranges=[9.5, 14.5] # [8, 12.5] [6, 11.5] [np.exp(6), np.exp(11.5)]
x_ranges=[0, 1]
# degree, post_mean, post_cov, beta = degree_l, post_mean_l, post_cov_l, beta_l
# samples_train, values_train = [X_train[0].T.copy()], [y_train[0].copy()]
# samples_test, values_test = [X[0].T.copy()], [y[0]]
# y_ranges=[6, 14.5] # 
# x_ranges=[0, 1]
y_log = True # True False

dim1_index = 0

xx = np.linspace(0, 1, 100)
fig, axs = plt.subplots(1, 1, figsize=(8, 8)) # (6, 6)
training_labels = [r'$\mathrm{{log\: y\: dfn}_{hi-fi}\: train}$', r'$\mathrm{{log\: y\: dfn}_{hi-fi}\: test}$']
# training_labels = [r'$\mathrm{log\: y\: dfn}_{lo-fi}$']
# training_labels = [r'$\mathrm{y\: dfn}_{hi-fi}$']
# training_labels = [r'$\mathrm{y\: dfn}_{lo-fi}$']
plot_nd_sf_lvn_approx(xx, axs, degree, post_mean, post_cov, beta,
                   samples_train, values_train, samples_test, values_test, training_labels,
                   dim1_index, x_ranges=x_ranges, y_ranges=y_ranges, y_log=y_log,
                   pct=[50],
                  #  pct=[25, 75, 50],
                  #  pct=[10, 90, 25, 75, 50], # [25, 50, 75] [10, 25, 50, 75, 90]
                   colors=['tab:orange'], # b c r k tab:orange
                   colors_pct=['green'])
                  #  colors_pct=['turquoise', 'paleturquoise', 'green'])
                  #  colors_pct=['lavender', 'lightcyan', 'turquoise', 'paleturquoise', 'green'])
# axs.set_xlabel(r'$\mathrm{network\: iperm\: /\: max\: iperm}$')
# axs.set_xlabel(r'$\mathrm{network\: fracture\: length\: /\: max\: length}$')
# axs.set_xlabel(r'$\mathrm{network\: mass\: flux\: /\: max\: flux}$')
# axs.set_xlabel(r'$\mathrm{network\: travel\: time\: /\: max\: time}$')
axs.set_xlabel(r'$\mathrm{network\: path\: length\: /\: max\: length}$')
axs.set_ylabel(r'$\mathrm{median\: breakthrough\: time}$')
plt.show()


#%%
# evaluate the log marginal likelihood for all polynomials up to specified degree
# goal is to find poly degree which minimizes model error
mlls = []
degree_test = 5 # 2 .. 5
sigma = 2
degrees = range(degree_test + 1)
# X_dt, y_dt = X_h, y_h
X_dt, y_dt = X_l, y_l
# beta_test  = 11.11 # 5.0 1 / (0.3 ** 2) = 11.11 beta
# alpha_test = 0.005 # 2.0 0.005 alpha
# beta_test  = beta_h
# alpha_test = alpha_h
beta_test  = beta_l
alpha_test = alpha_l
# alpha_test = 1

for d in degrees:
    #  Phi_test = pce_basis_matrix(X_dt, degree=d, dist_bounds=[0, 1])
    # Phi_test = gn_expand(X_dt, bf=gn_gaussian_basis_function, bf_deg_args=np.linspace(0, 1, d))
    Phi_test = gn_expand(X_dt, bf=gn_gaussian_basis_function, bf_deg_args=np.linspace(0, 1, d), sigma=[sigma])
    mll = gn_log_marginal_likelihood(Phi_test, y_dt, alpha=alpha_test, beta=beta_test)
    #  mll = blr_log_marginal_likelihood(Phi_test, y_dt, alpha=alpha_test, beta=alpha_test)
    mlls.append(mll)

degree_max = np.argmax(mlls)
    
plt.plot(degrees, mlls)
plt.axvline(x=degree_max, ls='--', c='k', lw=1)
plt.xticks(range(0, degree_test+1))
plt.xlabel('Polynomial degree')
plt.ylabel('Log marginal likelihood');
plt.show()

#%%
# evaluate the log marginal likelihood for specified range of sigma values
# goal is to find poly sigma which minimizes model error
mlls = []
degree_test = 2
sigma_test = 5
sigmas = np.linspace(0.1, sigma_test, sigma_test*10)
# sigmas = range(1, sigma_test + 1)
# X_dt, y_dt = X_h, y_h
X_dt, y_dt = X_l, y_l
# beta_test  = beta_h
# alpha_test = alpha_h
beta_test  = beta_l
alpha_test = alpha_l
# alpha_test = 1

for s in sigmas:
    Phi_test = gn_expand(X_dt, bf=gn_gaussian_basis_function, bf_deg_args=np.linspace(0, 1, degree_test), sigma=[s])
    mll = gn_log_marginal_likelihood(Phi_test, y_dt, alpha=alpha_test, beta=beta_test)
    mlls.append(mll)

sigma_max = np.argmax(mlls)
# sigma_h = sigmas[sigma_max]
sigma_l = sigmas[sigma_max]

print(sigmas[sigma_max])
    
plt.plot(sigmas, mlls)
plt.axvline(x=sigmas[sigma_max], ls='--', c='k', lw=1)
plt.xticks(np.linspace(0.1, sigma_test, 5))
plt.xlabel('Sigma')
plt.ylabel('Log marginal likelihood');

#%%
# use fit to obtain the posterior over parameters 𝐰 and optimal values for 𝛼 and 𝛽
degree_test = 2 # 1 2 3
sigma = 2 # 1 2 sigma_l
X_dt, y_dt = X_h, y_h
# X_dt, y_dt = X_train[1], y_train[1]
# X_dt, y_dt = X_l, y_l
# X_dt, y_dt = X_train[0], y_train[0]
 
# Phi_test = pce_basis_matrix(X_dt, degree=degree_test, dist_bounds=[-15, 15]) # [0, 1] [-15, 15]
Phi_test = gn_expand(X_dt, bf=gn_gaussian_basis_function, bf_deg_args=np.linspace(0, 1, degree_test), sigma=[sigma])

alpha_h, beta_h, m_N, S_N = gn_fit(Phi_test, y_dt, rtol=1e-5, verbose=True)
# alpha_l, beta_l, m_N, S_N = gn_fit(Phi_test, y_dt, rtol=1e-5, verbose=True)

print(f'alpha_h: {alpha_h}')
print(f'beta_h: {beta_h}')
# print(f'alpha_l: {alpha_l}')
# print(f'beta_l: {beta_l}')

# %%
