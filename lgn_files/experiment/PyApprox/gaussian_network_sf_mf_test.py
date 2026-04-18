
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
# from IPython.display import display

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
# worker_hf = getGraphs('../../math-clinic-data/data_stream_1/', '50 percent', 'high', False)
# worker_hf = getGraphs('../../math-clinic-data/data_stream_3/', '50 percent', 'high', False)
# worker_hf = getGraphs('../../math-clinic-data/alpha_25/', '50 percent', 'high', False)

# %%
# build lo-fi graphs from data
# worker_lf = getGraphs('../../math-clinic-data/data_stream_1/', '50 percent', 'low', False)
# worker_lf = getGraphs('../../math-clinic-data/data_stream_3/', '50 percent', 'low', False)
# worker_lf = getGraphs('../../math-clinic-data/alpha_25/', '50 percent', 'low', False)


# -------- MFNets tests --------

#  feature_list = [path_len, flux_pct, iperm_sum, time_pct, length_min]
# path_len   corr = 0.95
# flux_pct   corr = 0.70 / 0.60 ( 0.85 / 0.65) ( 0.70 / 0.60)
# iperm_sum  corr = 0.95 / 0.90
# time_pct   corr = 0.90 / 0.80
# length_min corr = 0.80 / 0.60

# %%
# test corr coeff
corr_test = 0.95

k = 50
# get hi-fi features from graphs
flux_calc_pct = 55
X_hf, y_hf = getFeatures(worker_hf, k, flux_calc_pct, False)
# log y_dfn
y_log_hf = np.log(np.array(y_hf.reshape(-1, 1), dtype='float64'))

# get lo-fi features from graphs
flux_calc_pct = 55 # (10) 55
X_lf, y_lf = getFeatures(worker_lf, k, flux_calc_pct, False)
# log y_dfn
y_log_lf = np.log(np.array(y_lf.reshape(-1, 1), dtype='float64'))

# feature_index = 2

X_hfm = X_hf.copy()
if X_hf.shape[1] <= 5:
    # single feature
    X_hf = X_hfm[:,feature_index].reshape(-1,1)
    # multi feature
    # X_hf = np.hstack([X_hfm[:,0].reshape(-1,1), X_hfm[:,1].reshape(-1,1)])
    # X_hf = np.hstack([X_hfm[:,0].reshape(-1,1), X_hfm[:,2].reshape(-1,1)])
    # X_hf = np.hstack([X_hfm[:,0].reshape(-1,1), X_hfm[:,1].reshape(-1,1), X_hfm[:,2].reshape(-1,1)])
    # X_hf = np.hstack([X_hfm[:,0].reshape(-1,1), X_hfm[:,1].reshape(-1,1), X_hfm[:,4].reshape(-1,1)])
else:
    # single feature
    X_hf = np.hstack([X_hfm[:,feature_index].reshape(-1,1), X_hfm[:,feature_index+5].reshape(-1,1)])
    # multi feature
    # X_hf = np.hstack([X_hfm[:,0].reshape(-1,1), X_hfm[:,1].reshape(-1,1),X_hfm[:,0+5].reshape(-1,1), X_hfm[:,1+5].reshape(-1,1)])
    # X_hf = np.hstack([X_hfm[:,0].reshape(-1,1), X_hfm[:,2].reshape(-1,1),X_hfm[:,0+5].reshape(-1,1), X_hfm[:,2+5].reshape(-1,1)])
    # X_hf = np.hstack([X_hfm[:,0].reshape(-1,1), X_hfm[:,1].reshape(-1,1), X_hfm[:,2].reshape(-1,1),X_hfm[:,0+5].reshape(-1,1), X_hfm[:,1+5].reshape(-1,1), X_hfm[:,2+5].reshape(-1,1)])
    # X_hf = np.hstack([X_hfm[:,0].reshape(-1,1), X_hfm[:,1].reshape(-1,1), X_hfm[:,4].reshape(-1,1),X_hfm[:,0+5].reshape(-1,1), X_hfm[:,1+5].reshape(-1,1), X_hfm[:,4+5].reshape(-1,1)])

X_lfm = X_lf.copy()
if X_lf.shape[1] <= 5:
    # single feature
    X_lf = X_lfm[:,feature_index].reshape(-1,1)
    # multi feature
    # X_lf = np.hstack([X_lfm[:,0].reshape(-1,1), X_lfm[:,1].reshape(-1,1)])
    # X_lf = np.hstack([X_lfm[:,0].reshape(-1,1), X_lfm[:,2].reshape(-1,1)])
    # X_lf = np.hstack([X_lfm[:,0].reshape(-1,1), X_lfm[:,1].reshape(-1,1), X_lfm[:,2].reshape(-1,1)])
    # X_lf = np.hstack([X_lfm[:,0].reshape(-1,1), X_lfm[:,1].reshape(-1,1), X_lfm[:,4].reshape(-1,1)])
else:
    # single feature
    X_lf = np.hstack([X_lfm[:,feature_index].reshape(-1,1), X_lfm[:,feature_index+5].reshape(-1,1)])
    # multi feature
    # X_lf = np.hstack([X_lfm[:,0].reshape(-1,1), X_lfm[:,1].reshape(-1,1), X_lfm[:,0+5].reshape(-1,1), X_lfm[:,1+5].reshape(-1,1)])
    # X_lf = np.hstack([X_lfm[:,0].reshape(-1,1), X_lfm[:,2].reshape(-1,1), X_lfm[:,0+5].reshape(-1,1), X_lfm[:,2+5].reshape(-1,1)])
    # X_lf = np.hstack([X_lfm[:,0].reshape(-1,1), X_lfm[:,1].reshape(-1,1), X_lfm[:,2].reshape(-1,1),X_lfm[:,0+5].reshape(-1,1), X_lfm[:,1+5].reshape(-1,1), X_lfm[:,2+5].reshape(-1,1)])
    # X_lf = np.hstack([X_lfm[:,0].reshape(-1,1), X_lfm[:,1].reshape(-1,1), X_lfm[:,4].reshape(-1,1),X_lfm[:,0+5].reshape(-1,1), X_lfm[:,1+5].reshape(-1,1), X_lfm[:,4+5].reshape(-1,1)])

for c in np.linspace(0.25, corr_test, 15):
# for c in np.linspace(0.5, corr_test, 10):

    # gn mf cross fold validation (hi-fi)
    nfolds = 50   # 10
    swap = True  # True False
    # poly degree
    degree_cfv_l = 2    #
    degree_cfv_h = 2    #
    # poly sigma
    sigma_cfv_l = 1.0   #
    sigma_cfv_h = 1.0   #
    # glm precision
    beta_cfv_l  = 4.5   # 7
    alpha_cfv_l = 1
    beta_cfv_h  = 6.5   # 7
    alpha_cfv_h = 1
    # model correlation
    model_corr = c
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
    print(f'model_corr: {c:.2f}, mf hf-fi MAPE: {np.mean(y_pred_percent_error):.2f}')

    # percent observations withing 95% confidence intervals
    y = np.exp(y_test)
    y_pred_std = np.sqrt(y_pred_var)
    y_lower = np.exp(y_pred-1.96*y_pred_std)
    y_upper = np.exp(y_pred+1.96*y_pred_std)
    isInCI = np.logical_and(y >= y_lower,y <= y_upper)
    print(f'number of observations in 95% CI: {np.count_nonzero(isInCI)} / {len(y_pred)}, {(100*np.count_nonzero(isInCI)/len(y_pred)):.2f}')

# %%
# test sigma values
sigma_test = 3.5

k = 20
# get hi-fi features from graphs
flux_calc_pct = 55
X_hf, y_hf = getFeatures(worker_hf, k, flux_calc_pct, False)
# log y_dfn
y_log_hf = np.log(np.array(y_hf.reshape(-1, 1), dtype='float64'))

# get lo-fi features from graphs
flux_calc_pct = 55 # (10) 55
X_lf, y_lf = getFeatures(worker_lf, k, flux_calc_pct, False)
# log y_dfn
y_log_lf = np.log(np.array(y_lf.reshape(-1, 1), dtype='float64'))

for s in np.linspace(0.5, sigma_test, 16):

    # gn mf cross fold validation (hi-fi)
    nfolds = 50   # 10
    swap = True  # True False
    # poly degree
    degree_cfv_l = 2    #
    degree_cfv_h = 2    #
    # poly sigma
    sigma_cfv_l = s     #   3.4 3.6 3.4 [k=5] (4.3) 4.1/4.2 4.4/4.5 4.2/4.3 [k=100] (10fs 30fs 50fs)    [50th pct]
    sigma_cfv_h = s     #   3.5 (10fcvs) 3.0 (30fcvs) 2.8 (50fcvs) [k=5]                                [90th pct]
    # glm precision
    beta_cfv_l  = 4.5   # 7
    alpha_cfv_l = 1
    beta_cfv_h  = 6.5   # 7
    alpha_cfv_h = 1
    # model correlation
    model_corr = 0.65   #
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
    print(f'sigma: {s:.2f}, mf hf-fi MAPE: {np.mean(y_pred_percent_error):.2f}')

    # percent observations withing 95% confidence intervals
    y = np.exp(y_test)
    y_pred_std = np.sqrt(y_pred_var)
    y_lower = np.exp(y_pred-1.96*y_pred_std)
    y_upper = np.exp(y_pred+1.96*y_pred_std)
    isInCI = np.logical_and(y >= y_lower,y <= y_upper)
    print(f'number of observations in 95% CI: {np.count_nonzero(isInCI)} / {len(y_pred)}, {(100*np.count_nonzero(isInCI)/len(y_pred)):.2f}')

# -------- SFNets + MFNets tests --------
# %%
# get hi-fi features from graphs
k = 100  # 1000
flux_calc_pct = 50 # 10 (55)
X_hf_list, y_hf = getFeatures(worker_hf, k, flux_calc_pct, False)
# log y_dfn
y_log_hf = np.log(np.array(y_hf.reshape(-1, 1), dtype='float64'))
# %%
# get lo-fi features from graphs
k = 100  # 1000
flux_calc_pct = 50 # (10) 55
X_lf_list, y_lf = getFeatures(worker_lf, k, flux_calc_pct, False)
# log y_dfn
y_log_lf = np.log(np.array(y_lf.reshape(-1, 1), dtype='float64'))

# %%
%%time
# test fold values
# k_test = np.array([3])
# k_test = np.array([3, 10])
# k_test = np.array([20])
k_test = np.array([20])
# k_test = np.array([3, 10, 20, 100])
# k_test = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
# k_test = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
#                    20, 30, 40, 50, 60, 70, 80, 90, 100])
# k_test = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100,
                #    110, 120, 130, 140, 150, 160, 170, 180, 190, 200, 300, 400, 500])
                #    200, 300, 400, 500, 600, 700, 800, 900, 1000])

# print(f'k gpr_hi_fi_train_pts gpr_hi_fi_mape gpr_hi_fi_95pct_ci')

# print(f'k sf_hi_fi_train_pts sf_hi_fi_mape sf_hi_fi_95pct_ci')
# print(f'k sf_lo_fi_train_pts sf_lo_fi_mape sf_lo_fi_95pct_ci')
print(f'k mf_hi_fi_train_pts mf_hi_fi_mape mf_hi_fi_95pct_ci')
# print(f'k mf_hi_fi_train_pts mf_pce_hi_fi_mape mf_pce_hi_fi_95pct_ci')

for k in k_test:

    X_hf = X_hf_list[k-1].copy()
    X_lf = X_lf_list[k-1].copy()
    
    # single feature
    # feature_index = 0                           # path length
    # X_hf = X_hf[:,feature_index].reshape(-1,1)
    # X_lf = X_lf[:,feature_index].reshape(-1,1)

    # X_hfm = X_hf.copy()
    # feature_index = 0
    # if X_hf.shape[1] <= 5:
    #     # single feature
    #     X_hf = X_hfm[:,feature_index].reshape(-1,1) # / X_hfm[:,2].reshape(-1,1)
    #     # multi feature
    #     # X_hf = np.hstack([X_hfm[:,0].reshape(-1,1), X_hfm[:,1].reshape(-1,1)])
    #     # X_hf = np.hstack([X_hfm[:,0].reshape(-1,1), X_hfm[:,2].reshape(-1,1)])
    #     # X_hf = np.hstack([X_hfm[:,0].reshape(-1,1), X_hfm[:,1].reshape(-1,1), X_hfm[:,2].reshape(-1,1)])
    #     # X_hf = np.hstack([X_hfm[:,0].reshape(-1,1), X_hfm[:,1].reshape(-1,1), X_hfm[:,4].reshape(-1,1)])
    #     # X_hf = np.hstack([X_hfm[:,0].reshape(-1,1), X_hfm[:,1].reshape(-1,1), X_hfm[:,2].reshape(-1,1), X_hfm[:,3].reshape(-1,1)])
    # # else:
    #     # single feature
    #     # X_hf = np.hstack([X_hfm[:,feature_index].reshape(-1,1), X_hfm[:,feature_index+5].reshape(-1,1)])
    #     # multi feature
    #     # X_hf = np.hstack([X_hfm[:,0].reshape(-1,1), X_hfm[:,1].reshape(-1,1),X_hfm[:,0+5].reshape(-1,1), X_hfm[:,1+5].reshape(-1,1)])
    #     # X_hf = np.hstack([X_hfm[:,0].reshape(-1,1), X_hfm[:,2].reshape(-1,1),X_hfm[:,0+5].reshape(-1,1), X_hfm[:,2+5].reshape(-1,1)])
    #     # X_hf = np.hstack([X_hfm[:,0].reshape(-1,1), X_hfm[:,1].reshape(-1,1), X_hfm[:,2].reshape(-1,1),X_hfm[:,0+5].reshape(-1,1), X_hfm[:,1+5].reshape(-1,1), X_hfm[:,2+5].reshape(-1,1)])
    #     # X_hf = np.hstack([X_hfm[:,0].reshape(-1,1), X_hfm[:,1].reshape(-1,1), X_hfm[:,4].reshape(-1,1),X_hfm[:,0+5].reshape(-1,1), X_hfm[:,1+5].reshape(-1,1), X_hfm[:,4+5].reshape(-1,1)])

    # X_lfm = X_lf.copy()
    # feature_index = 0
    # if X_lf.shape[1] <= 5:
    #     # single feature
    #     X_lf = X_lfm[:,feature_index].reshape(-1,1) # / X_lfm[:,2].reshape(-1,1)
    #     # multi feature
    #     # X_lf = np.hstack([X_lfm[:,0].reshape(-1,1), X_lfm[:,1].reshape(-1,1)])
    #     # X_lf = np.hstack([X_lfm[:,0].reshape(-1,1), X_lfm[:,2].reshape(-1,1)])
    #     # X_lf = np.hstack([X_lfm[:,0].reshape(-1,1), X_lfm[:,1].reshape(-1,1), X_lfm[:,2].reshape(-1,1)])
    #     # X_lf = np.hstack([X_lfm[:,0].reshape(-1,1), X_lfm[:,1].reshape(-1,1), X_lfm[:,4].reshape(-1,1)])
    #     # X_lf = np.hstack([X_lfm[:,0].reshape(-1,1), X_lfm[:,1].reshape(-1,1), X_lfm[:,2].reshape(-1,1), X_lfm[:,3].reshape(-1,1)])
    # # else:
    #     # single feature
    #     # X_lf = np.hstack([X_lfm[:,feature_index].reshape(-1,1), X_lfm[:,feature_index+5].reshape(-1,1)])
    #     # multi feature
    #     # X_lf = np.hstack([X_lfm[:,0].reshape(-1,1), X_lfm[:,1].reshape(-1,1), X_lfm[:,0+5].reshape(-1,1), X_lfm[:,1+5].reshape(-1,1)])
    #     # X_lf = np.hstack([X_lfm[:,0].reshape(-1,1), X_lfm[:,2].reshape(-1,1), X_lfm[:,0+5].reshape(-1,1), X_lfm[:,2+5].reshape(-1,1)])
    #     # X_lf = np.hstack([X_lfm[:,0].reshape(-1,1), X_lfm[:,1].reshape(-1,1), X_lfm[:,2].reshape(-1,1),X_lfm[:,0+5].reshape(-1,1), X_lfm[:,1+5].reshape(-1,1), X_lfm[:,2+5].reshape(-1,1)])
    #     # X_lf = np.hstack([X_lfm[:,0].reshape(-1,1), X_lfm[:,1].reshape(-1,1), X_lfm[:,4].reshape(-1,1),X_lfm[:,0+5].reshape(-1,1), X_lfm[:,1+5].reshape(-1,1), X_lfm[:,4+5].reshape(-1,1)])

    # 100 = 2^2 x 5^2: 2 4 5 10 20 25 50
    # fold_test_1 = np.array([10, 5, 4])
    # fold_test_2 = np.array([2, 4, 5, 10, 20, 25, 50, 100, 250])
    # fold_test = np.concatenate([fold_test_1, fold_test_2])
    # fold_swap = np.array([False, False, False, True, True, True, True, True, True, True, True, True])
    # fold_test = np.array([10])
    # fold_swap = np.array([True])
    fold_test = np.array([10, 5, 2, 5, 10, 25, 50, 100, 125])
    fold_swap = np.array([False, False, False, True, True, True, True, True, True])
    # fold_test = np.array([10, 4, 2, 4, 5, 10, 20, 25, 50])
    # fold_swap = np.array([False, False, True, True, True, True, True, True, True])
    # fold_beta_h = np.linspace(12.5, 9.5, 9, True) # 12.5 to 9.5 7.5 5.5 3.5
    for f, s in zip(fold_test, fold_swap):
    # for f, s, b_h in zip(fold_test, fold_swap, fold_beta_h):
        # gn mf cross fold validation (hi-fi)
        nfolds = f  # 10
        swap = True if s else False # True False
        swap_s = '_sw' if swap else ''
        # print(f'{nfolds}fcv{swap_s}')
        # poly degree
        degree_cfv_l = 2
        degree_cfv_h = 2
        # poly sigma
        sigma_cfv_l = sigma_l   # sigma_l 4.0 2.0 (3.0) 3.5 1.0
        sigma_cfv_h = sigma_l   # sigma_l 4.0 2.0 (3.0) 3.5 1.0
        # glm precision
        beta_cfv_l = beta_l     # beta_l 12.5 15.5 25.5 (4.5) ... 9.5
        # beta_cfv_l = beta_l_ds2
        alpha_cfv_l = 1         # 1 alpha_l
        beta_cfv_h = beta_l     # beta_l 12.5 (15.5)
        # beta_cfv_h = beta_l_ds2
        # beta_cfv_h = b_h
        alpha_cfv_h = 1         # 1 alpha_h
        # model correlation
        model_corr = 0.60       # 0.999995 0.75 (0.65) 0.70 ds1 to 0.61 ds2
        # training data samples (X) + values (y)
        X_cv_l = X_lf.copy()
        y_cv_l = y_log_lf.copy()
        # y_cv_l = y_log_lf_ub.copy()
        # y_cv_l = y_log_lf_dist_ub.copy()
        X_cv_h = X_hf.copy()
        y_cv_h = y_log_hf.copy()
        # exclude first 100 lo-fi networks
        # X_cv_l = X_lf[100:].copy()
        # y_cv_l = y_log_lf[100:].copy()
        # split ds
        # X_cv_h = X_hf[0:49].copy()
        # y_cv_h = y_log_hf[0:49].copy()
        # X_cv_h = X_hf[49:].copy()
        # y_cv_h = y_log_hf[49:].copy()
        
        # args = {'kernel': Matern() + WhiteKernel(), 'alpha': 1e-5, 'normalize_y': True}
        # args = {'kernel': RBF() + WhiteKernel(), 'alpha': 1e-5, 'normalize_y': True}
        
        # y_pred, y_pred_std, y_test, errors, percent_errors = gpr_cv(X_cv_h, y_cv_h, nfolds, args, swap)
        # y_pred_var = y_pred_std**2

        # y_pred, y_pred_var, y_test = gn_cv(X_cv_h, y_cv_h, degree=degree_cfv_h, sigma=sigma_cfv_h, alpha=alpha_cfv_h, beta=beta_cfv_h, nfolds=nfolds, swap_folds=swap)
        # y_pred, y_pred_var, y_test = gn_cv(X_cv_l, y_cv_l, degree=degree_cfv_l, sigma=sigma_cfv_l, alpha=alpha_cfv_l, beta=beta_cfv_l, nfolds=nfolds, swap_folds=swap)
        y_pred, y_pred_var, y_test = gn_mf_cv(X_cv_l, y_cv_l, X_cv_h, y_cv_h, degree_l=degree_cfv_l, degree_h=degree_cfv_h, sigma_l=sigma_cfv_l, sigma_h=sigma_cfv_h, alpha_l=alpha_cfv_l, beta_l=beta_cfv_l, alpha_h=alpha_cfv_h, beta_h=beta_cfv_h, model_corr=model_corr, nfolds=nfolds, swap_folds=swap)
        # y_pred, y_pred_var, y_test = gn_mf_cv_pce(X_cv_l, y_cv_l, X_cv_h, y_cv_h, degree_l=degree_cfv_l, degree_h=degree_cfv_h, sigma_l=sigma_cfv_l, sigma_h=sigma_cfv_h, alpha_l=alpha_cfv_l, beta_l=beta_cfv_l, alpha_h=alpha_cfv_h, beta_h=beta_cfv_h, model_corr=model_corr, nfolds=nfolds, swap_folds=swap)

        # # calculate mean y_pred, mean y_pred_var (monte carlo estimates)
        # y_pred_mean = np.mean(y_pred)
        # y_pred_variance = np.std(y_pred)
        # y_pred_var_mean = np.mean(np.sqrt(y_pred_var))
        # # y_pred_var_mean = np.sqrt(np.sum(y_pred_var))/len(y_pred_var)
        # # y_pred_var_mean = 1.96*np.sqrt(np.mean(y_pred_var))
        # # compare MC estimates with ground truth
        # y_test_mean = np.mean(y_log_hf)
        # y_test_var = np.std(y_log_hf)

        # calculate absolute raw and percent errors
        y_pred_error = np.abs(np.exp(y_pred) - np.exp(y_test))
        y_pred_percent_error = 100 * (y_pred_error / np.exp(y_test))
        
        # calculate raw and percent errors
        y_pred_raw_error = np.exp(y_pred) - np.exp(y_test)
        y_pred_raw_percent_error = 100 * (y_pred_raw_error / np.exp(y_test))

        # percent observations withing 95% confidence intervals
        y = np.exp(y_test)
        y_pred_std = np.sqrt(y_pred_var)
        y_lower = np.exp(y_pred-1.96*y_pred_std)
        y_upper = np.exp(y_pred+1.96*y_pred_std)
        isInCI = np.logical_and(y >= y_lower,y <= y_upper)

        # print(f'{k:03d} {(len(y_cv_h)/nfolds) if swap else len(y_cv_h)-(len(y_cv_h)/nfolds):05.1f} {np.mean(y_pred_percent_error):.2f} {(100*np.count_nonzero(isInCI)/len(y_pred)):.2f} {y_pred_mean:.4f} {y_pred_variance:.4f} {y_pred_var_mean:.4f} {y_test_mean:.4f} {y_test_var:.4f}')
        print(f'{k:03d} {(len(y_cv_h)/nfolds) if swap else len(y_cv_h)-(len(y_cv_h)/nfolds):04.1f} {np.mean(y_pred_percent_error):.2f} {(100*np.count_nonzero(isInCI)/len(y_pred)):.2f}')
        # print(f'{nfolds:02d}fcv{swap_s} | k, sf hi-fi MAPE, 95% CI: {k:03d} {np.mean(y_pred_percent_error):.2f} {(100*np.count_nonzero(isInCI)/len(y_pred)):.2f}')


# -------- SFNets tests --------
# %%
# test k values hi-fi
k_test = np.array([3])
# k_test = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
# k_test = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
for k in k_test:

    # get hi-fi features from graphs
    flux_calc_pct = 55
    X_hf, y_hf = getFeatures(worker_hf, k, flux_calc_pct, False)
    # log y_dfn
    y_log_hf = np.log(np.array(y_hf.reshape(-1, 1), dtype='float64'))

    X_hfm = X_hf.copy()
    if X_hf.shape[1] <= 5:
        # single feature
        feature_index = 0
        X_hf = X_hfm[:,feature_index].reshape(-1,1)
        # multi feature
        # X_hf = np.hstack([X_hfm[:,0].reshape(-1,1), X_hfm[:,1].reshape(-1,1)])
        # X_hf = np.hstack([X_hfm[:,0].reshape(-1,1), X_hfm[:,2].reshape(-1,1)])
        # X_hf = np.hstack([X_hfm[:,0].reshape(-1,1), X_hfm[:,1].reshape(-1,1), X_hfm[:,2].reshape(-1,1)])
        # X_hf = np.hstack([X_hfm[:,0].reshape(-1,1), X_hfm[:,1].reshape(-1,1), X_hfm[:,3].reshape(-1,1)])
        # X_hf = np.hstack([X_hfm[:,0].reshape(-1,1), X_hfm[:,1].reshape(-1,1), X_hfm[:,4].reshape(-1,1)])
    else:
        # single feature
        feature_index = 0
        X_hf = np.hstack([X_hfm[:,feature_index].reshape(-1,1), X_hfm[:,feature_index+5].reshape(-1,1)])
        # multi feature
        # X_hf = np.hstack([X_hfm[:,0].reshape(-1,1), X_hfm[:,1].reshape(-1,1),X_hfm[:,0+5].reshape(-1,1), X_hfm[:,1+5].reshape(-1,1)])
        # X_hf = np.hstack([X_hfm[:,0].reshape(-1,1), X_hfm[:,2].reshape(-1,1),X_hfm[:,0+5].reshape(-1,1), X_hfm[:,2+5].reshape(-1,1)])
        # X_hf = np.hstack([X_hfm[:,0].reshape(-1,1), X_hfm[:,1].reshape(-1,1), X_hfm[:,2].reshape(-1,1),X_hfm[:,0+5].reshape(-1,1), X_hfm[:,1+5].reshape(-1,1), X_hfm[:,2+5].reshape(-1,1)])
        # X_hf = np.hstack([X_hfm[:,0].reshape(-1,1), X_hfm[:,1].reshape(-1,1), X_hfm[:,3].reshape(-1,1),X_hfm[:,0+5].reshape(-1,1), X_hfm[:,1+5].reshape(-1,1), X_hfm[:,3+5].reshape(-1,1)])
        # X_hf = np.hstack([X_hfm[:,0].reshape(-1,1), X_hfm[:,1].reshape(-1,1), X_hfm[:,4].reshape(-1,1),X_hfm[:,0+5].reshape(-1,1), X_hfm[:,1+5].reshape(-1,1), X_hfm[:,4+5].reshape(-1,1)])
        
    # print(X_hf.shape[1])

    # gn cross fold validation (hi-fi)
    nfolds = 10 # 10
    swap = False # True False

    # poly degree
    degree_cfv = 2   # (1) 2
    # glm precision
    beta_cfv  = 11.5 # 10.5 (11.5) 12.5
    # beta_cfv  = (13.9 * (k - 3) + 12.5 * (100 - k)) / 97           # 12.5 (k=3) 13.9 (k=100)                           [50th pct]
    # beta_cfv  = (8.5 * (k - 3) + 8 * (100 - k)) / 97 - 0.5         # 8 (k=3) 8.5 (k=100) | 7.5 (k=3) 7 (k=100)         [90th pct]
    alpha_cfv = 1

    # training data samples (X) + values (y)
    X_cv = X_hf.copy()
    y_cv = y_log_hf.copy()

    y_pred, y_pred_var, y_test = gn_cv(X_cv, y_cv, degree=degree_cfv, alpha=alpha_cfv, beta=beta_cfv, nfolds=nfolds, swap_folds=swap)

    # calculate absolute raw and percent errors
    y_pred_error = np.abs(np.exp(y_pred) - np.exp(y_test))
    y_pred_percent_error = 100 * (y_pred_error / np.exp(y_test))
    # print(f'k: {k}, sf hi-fi MAPE: {np.mean(y_pred_percent_error):.2f}')

    # percent observations withing 95% confidence intervals
    y = np.exp(y_test)
    y_pred_std = np.sqrt(y_pred_var)
    y_lower = np.exp(y_pred-1.96*y_pred_std)
    y_upper = np.exp(y_pred+1.96*y_pred_std)
    isInCI = np.logical_and(y >= y_lower,y <= y_upper)
    # print(f'number of observations in 95% CI: {np.count_nonzero(isInCI)} / {len(y_pred)}, {(100*np.count_nonzero(isInCI)/len(y_pred)):.2f}')

    print(f'k, sf hi-fi MAPE, 95% CI: {k:03d} {np.mean(y_pred_percent_error):.2f} {(100*np.count_nonzero(isInCI)/len(y_pred)):.2f}')

# %%
# test k values lo-fi
# k_test = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
k_test = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
for k in k_test:

    # get lo-fi features from graphs
    flux_calc_pct = 10 # (10) 55
    X_lf, y_lf = getFeatures(worker_lf, k, flux_calc_pct, False)
    # log y_dfn
    y_log_lf = np.log(np.array(y_lf.reshape(-1, 1), dtype='float64'))

    # X_lfm = X_lf.copy()
    # if X_lf.shape[1] <= 5:
    #     # single feature
    #     # feature_index = 0
    #     # X_lf = X_lfm[:,feature_index].reshape(-1,1)
    #     # multi feature
    #     X_lf = np.hstack([X_lfm[:,0].reshape(-1,1), X_lfm[:,1].reshape(-1,1)])
    #     # X_lf = np.hstack([X_lfm[:,0].reshape(-1,1), X_lfm[:,2].reshape(-1,1)])
    #     # X_lf = np.hstack([X_lfm[:,0].reshape(-1,1), X_lfm[:,1].reshape(-1,1), X_lfm[:,2].reshape(-1,1)])
    #     # X_lf = np.hstack([X_lfm[:,0].reshape(-1,1), X_lfm[:,1].reshape(-1,1), X_lfm[:,4].reshape(-1,1)])
    # else:
    #     # single feature
    #     # feature_index = 0
    #     # X_lf = np.hstack([X_lfm[:,feature_index].reshape(-1,1), X_lfm[:,feature_index+5].reshape(-1,1)])
    #     # multi feature
    #     X_lf = np.hstack([X_lfm[:,0].reshape(-1,1), X_lfm[:,1].reshape(-1,1), X_lfm[:,0+5].reshape(-1,1), X_lfm[:,1+5].reshape(-1,1)])
    #     # X_lf = np.hstack([X_lfm[:,0].reshape(-1,1), X_lfm[:,2].reshape(-1,1), X_lfm[:,0+5].reshape(-1,1), X_lfm[:,2+5].reshape(-1,1)])
    #     # X_lf = np.hstack([X_lfm[:,0].reshape(-1,1), X_lfm[:,1].reshape(-1,1), X_lfm[:,2].reshape(-1,1),X_lfm[:,0+5].reshape(-1,1), X_lfm[:,1+5].reshape(-1,1), X_lfm[:,2+5].reshape(-1,1)])
    #     # X_lf = np.hstack([X_lfm[:,0].reshape(-1,1), X_lfm[:,1].reshape(-1,1), X_lfm[:,4].reshape(-1,1),X_lfm[:,0+5].reshape(-1,1), X_lfm[:,1+5].reshape(-1,1), X_lfm[:,4+5].reshape(-1,1)])

    # print(X_lf.shape[1])

    # gn cross fold validation (lo-fi)
    nfolds = 10
    swap = False # True False

    # poly degree
    degree_cfv = 1  # (2) 3
    # glm precision
    beta_cfv  = 6.5 # 4.5 (5.5) 6.5
    # beta_cfv  = (8.5 * (k - 3) + 6.83 * (100 - k)) / 97            # 6.83 (k=3) 8.5 (k=100)                                [50th pct]
    # beta_cfv  = (3.4 * (k - 3) + 3.05 * (100 - k)) / 97 + 0.5      # 3.05 (k=3) 3.4 (k=100) | 3.73 (k=3) 3.93 (k=100)      [90th pct]
    alpha_cfv = 1

    # training data samples (X) + values (y)
    X_cv = X_lf.copy()
    y_cv = y_log_lf.copy()

    y_pred, y_pred_var, y_test = gn_cv(X_cv, y_cv, degree=degree_cfv, alpha=alpha_cfv, beta=beta_cfv, nfolds=nfolds, swap_folds=swap)

    # calculate absolute raw and percent errors
    y_pred_error = np.abs(np.exp(y_pred) - np.exp(y_test))
    y_pred_percent_error = 100 * (y_pred_error / np.exp(y_test))
    # print(f'k: {k}, sf lo-fi MAPE: {np.mean(y_pred_percent_error):.2f}')

    # percent observations withing 95% confidence intervals
    y = np.exp(y_test)
    y_pred_std = np.sqrt(y_pred_var)
    y_lower = np.exp(y_pred-1.96*y_pred_std)
    y_upper = np.exp(y_pred+1.96*y_pred_std)
    isInCI = np.logical_and(y >= y_lower,y <= y_upper)
    # print(f'number of observations in 95% CI: {np.count_nonzero(isInCI)} / {len(y_pred)}, {(100*np.count_nonzero(isInCI)/len(y_pred)):.2f}')

    print(f'k, sf lo-fi MAPE, 95% CI: {k:03d} {np.mean(y_pred_percent_error):.2f} {(100*np.count_nonzero(isInCI)/len(y_pred)):.2f}')

# %%
