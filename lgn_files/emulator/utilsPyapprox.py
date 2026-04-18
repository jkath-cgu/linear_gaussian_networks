
# %%
import numpy as np
from scipy import stats
import networkx as nx

from emulator.utilsBayesianLinearRegression import *

from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

from pyapprox.bayes.gaussian_network import *
# from pyapprox.gaussian_network import *

# # %%
# import os

# os.chdir('/Users/jjkath/Library/CloudStorage/GoogleDrive-jjkath@gmail.com/My Drive/Programs/Python/pyapprox')

# from pyapprox.gaussian_network import *

# os.chdir('/Users/jjkath/Library/CloudStorage/GoogleDrive-jjkath@gmail.com/My Drive/Programs/Python/math-clinic/MFdfn_v10')

# %%
def pce_basis_matrix(X, degree, dist_bounds=[0, 1]):
    """Build polynomial feature matrix."""

    # construct the polynomial by the given PCE Construct
    feature_dim = X.shape[1] # 1 ... 10
    ensemble_pce = [[stats.uniform(dist_bounds[0], dist_bounds[1])]*feature_dim]
    polys, _ = get_total_degree_polynomials(ensemble_pce, [degree])

    # construct the basis matrix fcns
    basis_matrix_funcs = [p.basis_matrix for p in polys]

    # construct the basis matrix
    Phi = [b(s) for b, s in zip(basis_matrix_funcs, [X.T])]

    return Phi[0]


# %%
def gn_posterior(Phi, y, alpha, beta):
    """Computes mean and covariance matrix of the gaussian network posterior distribution."""

    N, M = Phi.shape

    # build two node network for classical linear gaussian model inference
    nnodes = 1
    nparams = [M]
    graph = nx.DiGraph()
    prior_covs = [1 / alpha]
    prior_means = [0]
    node_labels = [f'Node_{ii}' for ii in range(nnodes)]
    cpd_mats = [None]
    ii = 0
    graph.add_node(
        ii, label=node_labels[ii], cpd_cov=prior_covs[ii]*np.eye(nparams[ii]),
        nparams=nparams[ii], cpd_mat=cpd_mats[ii],
        cpd_mean=prior_means[ii]*np.ones((nparams[ii], 1)))

    network = GaussianNetwork(graph)

    # To infer the uncertain coefficients we must add training data to the network.
    # noise
    nsamples = [N]
    noise_std = [1 / beta]*nnodes # [0.01]
    # network date ~ samples / phi = basis_matrix dot samples_train
    data_cpd_mats = [Phi]
    # WLOG assume coefficients (latent variables) have zero mean and 𝑏 = 0 (deterministic shift) 
    data_cpd_vecs = [np.zeros((nsamples[ii], 1)) for ii in range(nnodes)]
    # data_cpd_vecs = [np.ones((nsamples[ii], 1)) for ii in range(nnodes)]
    noise_covs = [np.eye(nsamples[ii])*noise_std[ii]**2
                for ii in range(nnodes)]

    # add data to network
    network.add_data_to_network(data_cpd_mats, data_cpd_vecs, noise_covs)

    # # visualize two node network network with data
    # fig, ax = plt.subplots(1, 1, figsize=(5, 6))
    # plot_hierarchical_network_network_with_data(network.graph, ax)
    # plt.show()

    # network date ~ values
    values_train = [y]
    # convert_to_compact_factors must be after add_data when doing inference
    network.convert_to_compact_factors()
    labels = ['Node_0']
    evidence, evidence_ids = network.assemble_evidence(values_train)
    factor_post = cond_prob_variable_elimination(
        network, labels, evidence_ids=evidence_ids, evidence=evidence)
    posterior = convert_gaussian_from_canonical_form(
        factor_post.precision_matrix, factor_post.shift)

    return posterior[0].reshape(-1,1), posterior[1]


# %%
def gn_mf_posterior(Phi_l, y_l, Phi_h, y_h, alpha_l, beta_l, alpha_h, beta_h, model_corr=0.7, post_l=False):
    """Computes mean and covariance matrix of the gaussian network posterior distribution."""

    N_l, M_l = Phi_l.shape
    N_h, M_h = Phi_h.shape

    # configure cov relationship between models
    s11, s22 = 1/alpha_l, 1/alpha_h # 1, 1
    a21 = model_corr # 0.0 ... 0.7 0.9 0.95 0.99 0.99999

    # setup lo-fi & hi-fi models with prior mean + covariance
    # build hierarchical network
    nnodes = 2
    nparams = [M_l, M_h]
    graph = nx.DiGraph()
    prior_means = [0, 0]
    prior_covs = [s11, s22]
    cpd_scales = [a21]
    # cpd_scales = [a21*np.ones((M_h, 1))]
    node_labels = [f'Node_{ii}' for ii in range(nnodes)]
    # cpd_mats = [None, np.array([[0.65],[0.95],[0.70],[0.95],[0.90],[0.80],[0.95],[0.70],[0.95],[0.90],[0.80]])*np.eye(nparams[1], nparams[0])]
    cpd_mats = [None, cpd_scales[0]*np.eye(nparams[1], nparams[0])]

    # print(f'nparams: {nparams}')

    for ii in range(nnodes-1):
        graph.add_node(
            ii, label=node_labels[ii],
            cpd_cov=prior_covs[ii]*np.eye(nparams[ii]),
            nparams=nparams[ii], cpd_mat=cpd_mats[ii],
            cpd_mean=prior_means[ii]*np.ones((nparams[ii], 1)))
    #WARNING Nodes have to be added in order their information appears in lists.
    #i.e. high-fidelity node must be added last.
    ii = nnodes-1

    # TODO: correlate model coefficients depending on individual features
    # # feature correlation vector

    # cov = np.eye(nparams[ii])*(prior_covs[ii]-np.dot(
    #     np.asarray(cpd_scales)**2, prior_covs[ii]))
    # cov = np.eye(11)*(1-np.dot(np.asarray(np.array([[0.65],[0.95],[0.70],[0.95],[0.90],[0.80],[0.95],[0.70],[0.95],[0.90],[0.80]]))**2, 1))
    cov = np.eye(nparams[ii])*(prior_covs[ii]-np.dot(
        np.asarray(cpd_scales)**2, prior_covs[:ii]))

    # print(f'cov: {cov}')

    graph.add_node(
        ii, label=node_labels[ii], cpd_cov=cov, nparams=nparams[ii],
        cpd_mat=cpd_mats[ii],
        # cpd_mean = 0 * np.ones((nparams[ii], 1)))
        cpd_mean=(prior_means[ii]-np.dot(cpd_scales[:ii], prior_means[:ii])) *
        np.ones((nparams[ii], 1)))

    graph.add_edges_from([(ii, nnodes-1) for ii in range(nnodes-1)])

    network = GaussianNetwork(graph)

    # To infer the uncertain coefficients we must add training data to the network.
    # noise
    nsamples = [N_l, N_h]
    noise_std = [1/beta_l, 1/beta_h] # [0.01,0.01]
    # network date ~ samples / phi = basis_matrix dot samples_train
    data_cpd_mats = [Phi_l, Phi_h]
    # WLOG assume coefficients (latent variables) have zero mean and 𝑏 = 0 (deterministic shift) 
    data_cpd_vecs = [np.zeros((nsamples[ii], 1)) for ii in range(nnodes)]
    # data_cpd_vecs = [np.ones((nsamples[ii], 1)) for ii in range(nnodes)]
    noise_covs = [np.eye(nsamples[ii])*noise_std[ii]**2
                for ii in range(nnodes)]

    # add data to network
    network.add_data_to_network(data_cpd_mats, data_cpd_vecs, noise_covs)

    # # visualize two node network network with data
    # fig, ax = plt.subplots(1, 1, figsize=(5, 6))
    # plot_hierarchical_network_network_with_data(network.graph, ax)
    # plt.show()

    # network date ~ values
    values_train = [y_l, y_h]

    if post_l is True:
        # convert_to_compact_factors must be after add_data when doing inference
        network.convert_to_compact_factors()
        labels = ['Node_0']
        evidence, evidence_ids = network.assemble_evidence(values_train)
        factor_post = cond_prob_variable_elimination(
            network, labels, evidence_ids=evidence_ids, evidence=evidence)
        posterior_l = convert_gaussian_from_canonical_form(
            factor_post.precision_matrix, factor_post.shift)

    # convert_to_compact_factors must be after add_data when doing inference
    network.convert_to_compact_factors()
    labels = ['Node_1']
    evidence, evidence_ids = network.assemble_evidence(values_train)
    factor_post = cond_prob_variable_elimination(
        network, labels, evidence_ids=evidence_ids, evidence=evidence)
    posterior_h = convert_gaussian_from_canonical_form(
        factor_post.precision_matrix, factor_post.shift)

    if post_l is False:
        return posterior_h[0].reshape(-1,1), posterior_h[1]
    else:
        return posterior_l[0].reshape(-1,1), posterior_l[1], posterior_h[0].reshape(-1,1), posterior_h[1]


# %%
def gn_posterior_predictive(Phi_test, m_N, S_N, beta):
    """Computes mean and variances of the posterior predictive distribution."""
    y = Phi_test.dot(m_N)
    # Only compute variances (diagonal elements of covariance matrix)
    y_var = 1 / beta + np.sum(Phi_test.dot(S_N) * Phi_test, axis=1)
    
    return y.reshape(-1,1), y_var.reshape(-1,1)


# %%
def gn_cv(X, y, degree, sigma, alpha, beta, nfolds, swap_folds=False):
    """
    :param X:
    :param y:
    :param nfolds:
    :param args:
    :return:
    """
    kf = KFold(n_splits=nfolds)

    y_preds = []
    y_preds_var = []
    y_tests = []

    # print(f'beta: {beta}')
    # print(f'alpha: {alpha}')

    # normalize X
    X = X / X.max(axis=0)

    for train, test in kf.split(y):
        if swap_folds is True:
            train, test = test, train

        # slice
        (X_train, X_test, y_train, y_test) = (X[train], X[test], y[train], y[test])

        # reshape
        y_train, y_test = y_train.reshape(-1, 1), y_test.reshape(-1, 1)

        # type coercion
        X_train, X_test, y_train, y_test = np.array(X_train, dtype='float64'), np.array(X_test, dtype='float64'),  np.array(y_train, dtype='float64'), np.array(y_test, dtype='float64')

        # construct the feature matrix
        # Phi_train = pce_basis_matrix(X_train, degree=degree, dist_bounds=[0, 1]) # [0, 1] [0, 5] [-5, 5] ([-15, 15]) [-20, 20]
        # Phi_test  = pce_basis_matrix(X_test, degree=degree, dist_bounds=[0, 1])

        # construct the feature matrix
        # Phi_train = gn_expand(X_train, bf=gn_polynomial_basis_function, bf_deg_args=np.linspace(0, 1, degree+1)[1:])
        # Phi_test = gn_expand(X_test, bf=gn_polynomial_basis_function, bf_deg_args=np.linspace(0, 1, degree+1)[1:])
        # Phi_train = gn_expand(X_train, bf=gn_polynomial_basis_function, bf_deg_args=range(1, degree + 1))
        # Phi_test = gn_expand(X_test, bf=gn_polynomial_basis_function, bf_deg_args=range(1, degree + 1))
        
        # construct the feature matrix
        Phi_train = gn_expand(X_train, bf=gn_gaussian_basis_function, bf_deg_args=np.linspace(0, 1, degree), sigma=[sigma]) # 1.0 3.0
        Phi_test = gn_expand(X_test, bf=gn_gaussian_basis_function, bf_deg_args=np.linspace(0, 1, degree), sigma=[sigma])

        # print(f'Phi_train shape: {Phi_train.shape}')
        # print(f'Phi_test shape: {Phi_test.shape}')

        # train the gn on the fold
        # Mean and covariance matrix of posterior
        post_mean, post_cov = gn_posterior(Phi_train, y_train, alpha=alpha, beta=beta)

        # gn predict on the fold
        # Mean and variances of posterior predictive 
        y_pred, y_pred_var = gn_posterior_predictive(Phi_test, post_mean, post_cov, beta=beta)

        y_preds.append(y_pred)
        y_preds_var.append(y_pred_var)
        y_tests.append(y_test)

    # clean up model predictions and results from cv folds
    y_preds_clean, y_preds_var_clean, y_tests_clean = [], [], []
    for i in range(len(y_preds)):
        for j in range(len(y_preds[i])):
            y_preds_clean.append(y_preds[i][j][0])
            y_preds_var_clean.append(y_preds_var[i][j][0])
            y_tests_clean.append(y_tests[i][j][0])

    # coercion again
    y_preds_clean, y_preds_var_clean, y_tests_clean = np.array(y_preds_clean, dtype='float64'), np.array(y_preds_var_clean, dtype='float64'), np.array(y_tests_clean, dtype='float64')

    return y_preds_clean, y_preds_var_clean, y_tests_clean


# %%
def gn_cv_pce(X, y, degree, sigma, alpha, beta, nfolds, swap_folds=False):
    """
    :param X:
    :param y:
    :param nfolds:
    :param args:
    :return:
    """
    kf = KFold(n_splits=nfolds)

    y_preds = []
    y_preds_var = []
    y_tests = []

    # print(f'beta: {beta}')
    # print(f'alpha: {alpha}')

    # normalize X
    X = X / X.max(axis=0)

    for train, test in kf.split(y):
        if swap_folds is True:
            train, test = test, train

        # slice
        (X_train, X_test, y_train, y_test) = (X[train], X[test], y[train], y[test])

        # reshape
        y_train, y_test = y_train.reshape(-1, 1), y_test.reshape(-1, 1)

        # type coercion
        X_train, X_test, y_train, y_test = np.array(X_train, dtype='float64'), np.array(X_test, dtype='float64'),  np.array(y_train, dtype='float64'), np.array(y_test, dtype='float64')

        # construct the feature matrix
        Phi_train = pce_basis_matrix(X_train, degree=degree, dist_bounds=[-100, 100]) # [0, 1] [0, 5] [-5, 5] ([-15, 15]) [-20, 20]
        Phi_test  = pce_basis_matrix(X_test, degree=degree, dist_bounds=[-100, 100])

        # # construct the feature matrix
        # Phi_train = gn_expand(X_train, bf=gn_polynomial_basis_function, bf_deg_args=np.linspace(0, 1, degree+1)[1:])
        # Phi_test = gn_expand(X_test, bf=gn_polynomial_basis_function, bf_deg_args=np.linspace(0, 1, degree+1)[1:])
        # Phi_train = gn_expand(X_train, bf=gn_polynomial_basis_function, bf_deg_args=range(1, degree + 1))
        # Phi_test = gn_expand(X_test, bf=gn_polynomial_basis_function, bf_deg_args=range(1, degree + 1))
        
        # # construct the feature matrix
        # Phi_train = gn_expand(X_train, bf=gn_gaussian_basis_function, bf_deg_args=np.linspace(0, 1, degree), sigma=[sigma]) # 1.0 3.0
        # Phi_test = gn_expand(X_test, bf=gn_gaussian_basis_function, bf_deg_args=np.linspace(0, 1, degree), sigma=[sigma])

        # print(f'Phi_train shape: {Phi_train.shape}')
        # print(f'Phi_test shape: {Phi_test.shape}')

        # train the gn on the fold
        # Mean and covariance matrix of posterior
        post_mean, post_cov = gn_posterior(Phi_train, y_train, alpha=alpha, beta=beta)

        # gn predict on the fold
        # Mean and variances of posterior predictive 
        y_pred, y_pred_var = gn_posterior_predictive(Phi_test, post_mean, post_cov, beta=beta)

        y_preds.append(y_pred)
        y_preds_var.append(y_pred_var)
        y_tests.append(y_test)

    # clean up model predictions and results from cv folds
    y_preds_clean, y_preds_var_clean, y_tests_clean = [], [], []
    for i in range(len(y_preds)):
        for j in range(len(y_preds[i])):
            y_preds_clean.append(y_preds[i][j][0])
            y_preds_var_clean.append(y_preds_var[i][j][0])
            y_tests_clean.append(y_tests[i][j][0])

    # coercion again
    y_preds_clean, y_preds_var_clean, y_tests_clean = np.array(y_preds_clean, dtype='float64'), np.array(y_preds_var_clean, dtype='float64'), np.array(y_tests_clean, dtype='float64')

    return y_preds_clean, y_preds_var_clean, y_tests_clean


# %%
def gn_mf_cv(X_l, y_l, X_h, y_h, degree_l, degree_h, sigma_l, sigma_h, alpha_l, beta_l, alpha_h, beta_h, model_corr=0.7, nfolds=10, swap_folds=False):
    """
    :param X:
    :param y:
    :param nfolds:
    :param args:
    :return:
    """
    kf = KFold(n_splits=nfolds)

    y_preds = []
    y_preds_var = []
    y_tests = []
    
    # assume X_l and X_h have the same features (N_features_l == N_features_h)
    # normalize both X_l and X_h relative to X_l to prevent data leakage
    # normalize X (relative)
    # find max of common features
    X_l_max = X_l.max(axis=0).reshape(1,-1)
    X_h_max = X_h.max(axis=0).reshape(1,-1)
    _, N_features_l = X_l_max.shape
    _, N_features_h = X_h_max.shape
    # normalize X_l & X_h depending of features
    X_all = np.array([])
    if N_features_l == N_features_h:
        X_l = X_l / X_l_max
        X_h = X_h / X_l_max
    else: # error X_l and X_h must have the same features
        return y_preds, y_preds_var, y_tests

    # # normalize X (relative)
    # # find max of common features
    # X_l_max = X_l.max(axis=0).reshape(1,-1)
    # X_h_max = X_h.max(axis=0).reshape(1,-1)
    # _, N_features_l = X_l_max.shape
    # _, N_features_h = X_h_max.shape
    # # normalize X_l & X_h depending of features
    # X_all = np.array([])
    # if N_features_l == N_features_h:
    #     X_all = np.max([X_l_max, X_h_max],axis=0)
    #     X_l = X_l / X_all
    #     X_h = X_h / X_all
    # elif N_features_l > N_features_h:
    #     X_all = np.max([X_l_max[0,0:N_features_h].reshape(1,-1), X_h_max],axis=0)
    #     X_l_max[0,0:N_features_h] = X_all
    #     X_l = X_l / X_l_max
    #     X_h = X_h / X_all
    # elif N_features_l < N_features_h:
    #     X_all = np.max([X_l_max, X_h_max[0,0:N_features_l].reshape(1,-1)],axis=0)
    #     X_h_max[0,0:N_features_l] = X_all
    #     X_l = X_l / X_all
    #     X_h = X_h / X_h_max

    # construct the feature matrix
    # Phi_train_l = pce_basis_matrix(X_l, degree=degree_l, dist_bounds=[-100, 100]) # [0, 1] [0, 5] [-5, 5] ([-15, 15]) [-20, 20]

    # construct the feature matrix
    # Phi_train_l = gn_expand(X_l, bf=gn_polynomial_basis_function, bf_deg_args=np.linspace(0, 1, degree_l+1)[1:])
    # Phi_train_l = gn_expand(X_l, bf=gn_polynomial_basis_function, bf_deg_args=range(1, degree_l+1))

    # construct the feature matrix
    Phi_train_l = gn_expand(X_l, bf=gn_gaussian_basis_function, bf_deg_args=np.linspace(0, 1, degree_l), sigma=[sigma_l])

    for train, test in kf.split(y_h):
        if swap_folds is True:
            train, test = test, train

        # slice
        (X_train, X_test, y_train, y_test) = (X_h[train], X_h[test], y_h[train], y_h[test])

        # reshape
        y_train, y_test = y_train.reshape(-1, 1), y_test.reshape(-1, 1)

        # type coercion
        X_train, X_test, y_train, y_test = np.array(X_train, dtype='float64'), np.array(X_test, dtype='float64'),  np.array(y_train, dtype='float64'), np.array(y_test, dtype='float64')

        # construct the feature matrix
        # Phi_train_h = pce_basis_matrix(X_train, degree=degree_h, dist_bounds=[-100, 100])
        # Phi_test_h  = pce_basis_matrix(X_test, degree=degree_h, dist_bounds=[-100, 100])

        # construct the feature matrix
        # Phi_train_h = gn_expand(X_train, bf=gn_polynomial_basis_function, bf_deg_args=np.linspace(0, 1, degree_h+1)[1:])
        # Phi_test_h = gn_expand(X_test, bf=gn_polynomial_basis_function, bf_deg_args=np.linspace(0, 1, degree_h+1)[1:])
        # Phi_train_h = gn_expand(X_train, bf=gn_polynomial_basis_function, bf_deg_args=range(1, degree_h+1))
        # Phi_test_h = gn_expand(X_test, bf=gn_polynomial_basis_function, bf_deg_args=range(1, degree_h+1))

        # construct the feature matrix
        Phi_train_h = gn_expand(X_train, bf=gn_gaussian_basis_function, bf_deg_args=np.linspace(0, 1, degree_h), sigma=[sigma_h])
        Phi_test_h = gn_expand(X_test, bf=gn_gaussian_basis_function, bf_deg_args=np.linspace(0, 1, degree_h), sigma=[sigma_h])

        # # correct graph model output for bias
        # if bias_correct:
        #     # LSF line to y_graph (x-axis) vs y_dfn (y-axis)
        #     reg = LinearRegression().fit(y_l[train], y_h[train])
        #     reg_m = reg.coef_[0][0]
        #     reg_b = reg.intercept_[0]
        #     # remove bias
        #     y_ub_l = reg_m * y_l.copy() + reg_b
        #     # check for fit
        #     y_pred_error_ub = np.abs(y_ub_l[0:len(y_h)] - y_h)
        #     y_pred_error = np.abs(y_l[0:len(y_h)] - y_h)
        #     if np.sum(y_pred_error_ub) < np.sum(y_pred_error): # throw out bad linear fit
        #         y_train_l = y_ub_l.copy()
        #     else:
        #         y_train_l = y_l.copy()
        # else:
        #     y_train_l = y_l.copy()
        
        y_train_l = y_l.copy()

        # train the gn on the fold
        # Mean and covariance matrix of posterior
        post_mean_h, post_cov_h = gn_mf_posterior(Phi_train_l, y_train_l, Phi_train_h, y_train, alpha_l=alpha_l, beta_l=beta_l, alpha_h=alpha_h, beta_h=beta_h, model_corr=model_corr)

        # gn predict on the fold
        # Mean and variances of posterior predictive 
        y_pred, y_pred_var = gn_posterior_predictive(Phi_test_h, post_mean_h, post_cov_h, beta=beta_h)

        y_preds.append(y_pred)
        y_preds_var.append(y_pred_var)
        y_tests.append(y_test)

    # clean up model predictions and results from cv folds
    y_preds_clean, y_preds_var_clean, y_tests_clean = [], [], []
    for i in range(len(y_preds)):
        for j in range(len(y_preds[i])):
            y_preds_clean.append(y_preds[i][j][0])
            y_preds_var_clean.append(y_preds_var[i][j][0])
            y_tests_clean.append(y_tests[i][j][0])

    # coercion again
    y_preds_clean, y_preds_var_clean, y_tests_clean = np.array(y_preds_clean, dtype='float64'), np.array(y_preds_var_clean, dtype='float64'), np.array(y_tests_clean, dtype='float64')

    return y_preds_clean, y_preds_var_clean, y_tests_clean


# %%
def gn_mf_cv_pce(X_l, y_l, X_h, y_h, degree_l, degree_h, sigma_l, sigma_h, alpha_l, beta_l, alpha_h, beta_h, model_corr=0.7, nfolds=10, swap_folds=False):
    """
    :param X:
    :param y:
    :param nfolds:
    :param args:
    :return:
    """
    kf = KFold(n_splits=nfolds)

    y_preds = []
    y_preds_var = []
    y_tests = []
    
    # assume X_l and X_h have the same features (N_features_l == N_features_h)
    # normalize both X_l and X_h relative to X_l to prevent data leakage
    # normalize X (relative)
    # find max of common features
    X_l_max = X_l.max(axis=0).reshape(1,-1)
    X_h_max = X_h.max(axis=0).reshape(1,-1)
    _, N_features_l = X_l_max.shape
    _, N_features_h = X_h_max.shape
    # normalize X_l & X_h depending of features
    X_all = np.array([])
    if N_features_l == N_features_h:
        X_l = X_l / X_l_max
        X_h = X_h / X_l_max
    else: # error X_l and X_h must have the same features
        return y_preds, y_preds_var, y_tests

    # # normalize X (relative)
    # # find max of common features
    # X_l_max = X_l.max(axis=0).reshape(1,-1)
    # X_h_max = X_h.max(axis=0).reshape(1,-1)
    # _, N_features_l = X_l_max.shape
    # _, N_features_h = X_h_max.shape
    # # normalize X_l & X_h depending of features
    # X_all = np.array([])
    # if N_features_l == N_features_h:
    #     X_all = np.max([X_l_max, X_h_max],axis=0)
    #     X_l = X_l / X_all
    #     X_h = X_h / X_all
    # elif N_features_l > N_features_h:
    #     X_all = np.max([X_l_max[0,0:N_features_h].reshape(1,-1), X_h_max],axis=0)
    #     X_l_max[0,0:N_features_h] = X_all
    #     X_l = X_l / X_l_max
    #     X_h = X_h / X_all
    # elif N_features_l < N_features_h:
    #     X_all = np.max([X_l_max, X_h_max[0,0:N_features_l].reshape(1,-1)],axis=0)
    #     X_h_max[0,0:N_features_l] = X_all
    #     X_l = X_l / X_all
    #     X_h = X_h / X_h_max

    # construct the feature matrix
    Phi_train_l = pce_basis_matrix(X_l, degree=degree_l, dist_bounds=[-100, 100]) # [0, 1] [0, 5] [-5, 5] ([-15, 15]) [-20, 20]

    # # construct the feature matrix
    # Phi_train_l = gn_expand(X_l, bf=gn_polynomial_basis_function, bf_deg_args=np.linspace(0, 1, degree_l+1)[1:])
    # Phi_train_l = gn_expand(X_l, bf=gn_polynomial_basis_function, bf_deg_args=range(1, degree_l+1))

    # # construct the feature matrix
    # Phi_train_l = gn_expand(X_l, bf=gn_gaussian_basis_function, bf_deg_args=np.linspace(0, 1, degree_l), sigma=[sigma_l])

    for train, test in kf.split(y_h):
        if swap_folds is True:
            train, test = test, train

        # slice
        (X_train, X_test, y_train, y_test) = (X_h[train], X_h[test], y_h[train], y_h[test])

        # reshape
        y_train, y_test = y_train.reshape(-1, 1), y_test.reshape(-1, 1)

        # type coercion
        X_train, X_test, y_train, y_test = np.array(X_train, dtype='float64'), np.array(X_test, dtype='float64'),  np.array(y_train, dtype='float64'), np.array(y_test, dtype='float64')

        # construct the feature matrix
        Phi_train_h = pce_basis_matrix(X_train, degree=degree_h, dist_bounds=[-100, 100])
        Phi_test_h  = pce_basis_matrix(X_test, degree=degree_h, dist_bounds=[-100, 100])

        # # construct the feature matrix
        # Phi_train_h = gn_expand(X_train, bf=gn_polynomial_basis_function, bf_deg_args=np.linspace(0, 1, degree_h+1)[1:])
        # Phi_test_h = gn_expand(X_test, bf=gn_polynomial_basis_function, bf_deg_args=np.linspace(0, 1, degree_h+1)[1:])
        # Phi_train_h = gn_expand(X_train, bf=gn_polynomial_basis_function, bf_deg_args=range(1, degree_h+1))
        # Phi_test_h = gn_expand(X_test, bf=gn_polynomial_basis_function, bf_deg_args=range(1, degree_h+1))

        # # construct the feature matrix
        # Phi_train_h = gn_expand(X_train, bf=gn_gaussian_basis_function, bf_deg_args=np.linspace(0, 1, degree_h), sigma=[sigma_h])
        # Phi_test_h = gn_expand(X_test, bf=gn_gaussian_basis_function, bf_deg_args=np.linspace(0, 1, degree_h), sigma=[sigma_h])

        # # correct graph model output for bias
        # if bias_correct:
        #     # LSF line to y_graph (x-axis) vs y_dfn (y-axis)
        #     reg = LinearRegression().fit(y_l[train], y_h[train])
        #     reg_m = reg.coef_[0][0]
        #     reg_b = reg.intercept_[0]
        #     # remove bias
        #     y_ub_l = reg_m * y_l.copy() + reg_b
        #     # check for fit
        #     y_pred_error_ub = np.abs(y_ub_l[0:len(y_h)] - y_h)
        #     y_pred_error = np.abs(y_l[0:len(y_h)] - y_h)
        #     if np.sum(y_pred_error_ub) < np.sum(y_pred_error): # throw out bad linear fit
        #         y_train_l = y_ub_l.copy()
        #     else:
        #         y_train_l = y_l.copy()
        # else:
        #     y_train_l = y_l.copy()
        
        y_train_l = y_l.copy()

        # train the gn on the fold
        # Mean and covariance matrix of posterior
        post_mean_h, post_cov_h = gn_mf_posterior(Phi_train_l, y_train_l, Phi_train_h, y_train, alpha_l=alpha_l, beta_l=beta_l, alpha_h=alpha_h, beta_h=beta_h, model_corr=model_corr)

        # gn predict on the fold
        # Mean and variances of posterior predictive 
        y_pred, y_pred_var = gn_posterior_predictive(Phi_test_h, post_mean_h, post_cov_h, beta=beta_h)

        y_preds.append(y_pred)
        y_preds_var.append(y_pred_var)
        y_tests.append(y_test)

    # clean up model predictions and results from cv folds
    y_preds_clean, y_preds_var_clean, y_tests_clean = [], [], []
    for i in range(len(y_preds)):
        for j in range(len(y_preds[i])):
            y_preds_clean.append(y_preds[i][j][0])
            y_preds_var_clean.append(y_preds_var[i][j][0])
            y_tests_clean.append(y_tests[i][j][0])

    # coercion again
    y_preds_clean, y_preds_var_clean, y_tests_clean = np.array(y_preds_clean, dtype='float64'), np.array(y_preds_var_clean, dtype='float64'), np.array(y_tests_clean, dtype='float64')

    return y_preds_clean, y_preds_var_clean, y_tests_clean


# %%
def gn_mf_run_model(X_lf, y_lf, X_hf, y_hf, lf_pct, hf_pct, degree_l, degree_h, sigma_l, sigma_h, alpha_l, beta_l, alpha_h, beta_h, model_corr=0.7, rng=np.random.RandomState(1)):

    # hi-fi training data samples (X) + values (y)
    X_h = X_hf.copy() # iperm
    y_h = y_hf.copy()

    # lo-fi training data samples (X) + values (y)
    X_l = X_lf.copy() # iperm
    y_l = y_lf.copy()

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

    # choose training & testing data
    X, y = [X_l.copy(), X_h.copy()], [y_l.copy(), y_h.copy()]

    # choose random samples / values for training
    X_train, y_train, index_train = choose_sample_value_rnd(X, y, lf_pct=lf_pct, hf_pct=hf_pct, rng=rng)

    # # construct the feature matrix
    # Phi_train_h = pce_basis_matrix(X_train[1], degree=degree_h, dist_bounds=[0, 1])
    # Phi_train_l = pce_basis_matrix(X_train[0], degree=degree_l, dist_bounds=[0, 1])

    # construct the feature matrix
    Phi_train_h = gn_expand(X_train[1], bf=gn_gaussian_basis_function, bf_deg_args=np.linspace(0, 1, degree_h), sigma=[sigma_h])
    Phi_train_l = gn_expand(X_train[0], bf=gn_gaussian_basis_function, bf_deg_args=np.linspace(0, 1, degree_l), sigma=[sigma_l])

    # Mean and covariance matrix of posterior
    post_mean_h, post_cov_h = gn_mf_posterior(Phi_train_l, y_train[0], Phi_train_h, y_train[1], alpha_l=alpha_l, beta_l=beta_l, alpha_h=alpha_h, beta_h=beta_h, model_corr=model_corr)

    # MAPE on testing only (samples, data)
    test_only_index = np.setdiff1d(np.arange(y[1].size),index_train[1])
    samples = X[1][test_only_index]
    y_test_test_only = y[1][test_only_index]

    # Phi_test_h = pce_basis_matrix(samples, degree=degree_h, dist_bounds=[0, 1])

    Phi_test_h = gn_expand(samples, bf=gn_gaussian_basis_function, bf_deg_args=np.linspace(0, 1, degree_h), sigma=[sigma_h])
    
    # print(samples.shape)
    # print(y_test_test_only.shape)
    # print(Phi_test_h.shape)

    # Mean and variances of posterior predictive 
    y_pred_test_only, y_pred_var_test_only = gn_posterior_predictive(Phi_test_h, post_mean_h, post_cov_h, beta=beta_h)

    # MAPE on training only (samples, data)
    samples = X_train[1]
    y_test_train_only = y_train[1]

    # print(samples.shape)
    # print(y_test_train_only.shape)
    # print(Phi_train_h.shape)

    # Mean and variances of posterior predictive 
    y_pred_train_only, y_pred_var_train_only = gn_posterior_predictive(Phi_train_h, post_mean_h, post_cov_h, beta=beta_h)

    # MAPE on training + testing (samples, data)
    samples = X[1]
    y_test_train_test = y[1]

    # Phi_train_test_h = pce_basis_matrix(samples, degree=degree_h, dist_bounds=[0, 1])

    Phi_train_test_h = gn_expand(samples, bf=gn_gaussian_basis_function, bf_deg_args=np.linspace(0, 1, degree_h), sigma=[sigma_h])

    # print(samples.shape)
    # print(y_test_train_test.shape)
    # print(Phi_train_test_h.shape)

    # Mean and variances of posterior predictive 
    y_pred_train_test, y_pred_var_train_test = gn_posterior_predictive(Phi_train_test_h, post_mean_h, post_cov_h, beta=beta_h)

    return y_pred_test_only, y_pred_var_test_only, y_test_test_only, y_pred_train_only, y_pred_var_train_only, y_test_train_only, y_pred_train_test, y_pred_var_train_test, y_test_train_test


# %%
def gn_mf_run_model_pce(X_lf, y_lf, X_hf, y_hf, lf_pct, hf_pct, degree_l, degree_h, sigma_l, sigma_h, alpha_l, beta_l, alpha_h, beta_h, model_corr=0.7, rng=np.random.RandomState(1)):

    # hi-fi training data samples (X) + values (y)
    X_h = X_hf.copy() # iperm
    y_h = y_hf.copy()

    # lo-fi training data samples (X) + values (y)
    X_l = X_lf.copy() # iperm
    y_l = y_lf.copy()

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

    # choose training & testing data
    X, y = [X_l.copy(), X_h.copy()], [y_l.copy(), y_h.copy()]

    # choose random samples / values for training
    X_train, y_train, index_train = choose_sample_value_rnd(X, y, lf_pct=lf_pct, hf_pct=hf_pct, rng=rng)

    # construct the feature matrix
    Phi_train_h = pce_basis_matrix(X_train[1], degree=degree_h, dist_bounds=[0, 1])
    Phi_train_l = pce_basis_matrix(X_train[0], degree=degree_l, dist_bounds=[0, 1])

    # # construct the feature matrix
    # Phi_train_h = gn_expand(X_train[1], bf=gn_gaussian_basis_function, bf_deg_args=np.linspace(0, 1, degree_h), sigma=[sigma_h])
    # Phi_train_l = gn_expand(X_train[0], bf=gn_gaussian_basis_function, bf_deg_args=np.linspace(0, 1, degree_l), sigma=[sigma_l])

    # Mean and covariance matrix of posterior
    post_mean_h, post_cov_h = gn_mf_posterior(Phi_train_l, y_train[0], Phi_train_h, y_train[1], alpha_l=alpha_l, beta_l=beta_l, alpha_h=alpha_h, beta_h=beta_h, model_corr=model_corr)

    # MAPE on testing only (samples, data)
    test_only_index = np.setdiff1d(np.arange(y[1].size),index_train[1])
    samples = X[1][test_only_index]
    y_test_test_only = y[1][test_only_index]

    Phi_test_h = pce_basis_matrix(samples, degree=degree_h, dist_bounds=[0, 1])

    # Phi_test_h = gn_expand(samples, bf=gn_gaussian_basis_function, bf_deg_args=np.linspace(0, 1, degree_h), sigma=[sigma_h])
    
    # print(samples.shape)
    # print(y_test_test_only.shape)
    # print(Phi_test_h.shape)

    # Mean and variances of posterior predictive 
    y_pred_test_only, y_pred_var_test_only = gn_posterior_predictive(Phi_test_h, post_mean_h, post_cov_h, beta=beta_h)

    # MAPE on training only (samples, data)
    samples = X_train[1]
    y_test_train_only = y_train[1]

    # print(samples.shape)
    # print(y_test_train_only.shape)
    # print(Phi_train_h.shape)

    # Mean and variances of posterior predictive 
    y_pred_train_only, y_pred_var_train_only = gn_posterior_predictive(Phi_train_h, post_mean_h, post_cov_h, beta=beta_h)

    # MAPE on training + testing (samples, data)
    samples = X[1]
    y_test_train_test = y[1]

    Phi_train_test_h = pce_basis_matrix(samples, degree=degree_h, dist_bounds=[0, 1])

    # Phi_train_test_h = gn_expand(samples, bf=gn_gaussian_basis_function, bf_deg_args=np.linspace(0, 1, degree_h), sigma=[sigma_h])

    # print(samples.shape)
    # print(y_test_train_test.shape)
    # print(Phi_train_test_h.shape)

    # Mean and variances of posterior predictive 
    y_pred_train_test, y_pred_var_train_test = gn_posterior_predictive(Phi_train_test_h, post_mean_h, post_cov_h, beta=beta_h)

    return y_pred_test_only, y_pred_var_test_only, y_test_test_only, y_pred_train_only, y_pred_var_train_only, y_test_train_only, y_pred_train_test, y_pred_var_train_test, y_test_train_test

# %%
def gn_gaussian_basis_function(x, mu, sigma=1.0): # 0.5 1.5 2.5 (3.5) / 0.1
    return np.exp(-0.5 * (x - mu) ** 2 / sigma ** 2)


# %%
def gn_polynomial_basis_function(x, power):
    return x ** power


# %%
def gn_expand(x, bf, bf_deg_args=None, sigma=None):
    if bf_deg_args is None:
        return np.concatenate([np.ones((x.shape[0], 1)), bf(x)], axis=1)
    elif sigma is None:
        return np.concatenate([np.ones((x.shape[0], 1))] + [bf(x, bf_deg_arg) for bf_deg_arg in bf_deg_args], axis=1)
    else:
        return np.concatenate([np.ones((x.shape[0], 1))] + [bf(x, bf_deg_arg, sigma[0]) for bf_deg_arg in bf_deg_args], axis=1)


# %%
def gn_log_marginal_likelihood(Phi, y, alpha, beta):
    """Computes the log of the marginal likelihood."""
    N, M = Phi.shape

    m_N, S_N = gn_posterior(Phi, y, alpha, beta)
    S_N_inv = np.linalg.inv(S_N)

    E_D = beta * np.sum((y - Phi.dot(m_N)) ** 2)
    E_W = alpha * np.sum(m_N ** 2)
    
    score = M * np.log(alpha) + \
            N * np.log(beta) - \
            E_D - \
            E_W - \
            np.log(np.linalg.det(S_N_inv)) - \
            N * np.log(2 * np.pi)

    return 0.5 * score


# %%
def gn_fit(Phi, y, alpha_0=1e-10, beta_0=1e-5, max_iter=200, rtol=1e-6, verbose=False):
# def gn_fit(Phi, y, alpha_0=1, beta_0=1, max_iter=200, rtol=1e-5, verbose=False):
    """
    Jointly infers the posterior sufficient statistics and optimal values 
    for alpha and beta by maximizing the log marginal likelihood.
    
    Args:
        Phi: Design matrix (N x M).
        t: Target value array (N x 1).
        alpha_0: Initial value for alpha.
        beta_0: Initial value for beta.
        max_iter: Maximum number of iterations.
        rtol: Convergence criterion.
        
    Returns:
        alpha, beta, posterior mean, posterior covariance.
    """
    
    N, M = Phi.shape

    eigenvalues_0 = np.linalg.eigvalsh(Phi.T.dot(Phi))

    beta = beta_0
    alpha = alpha_0 # 1

    for i in range(max_iter):
        beta_prev = beta
        alpha_prev = alpha # 1

        # print(f'beta: {beta}')
        # print(f'alpha: {alpha}')

        eigenvalues = eigenvalues_0 * beta

        m_N, S_N = gn_posterior(Phi, y, alpha, beta)

        gamma = np.sum(eigenvalues / (eigenvalues + alpha))
        alpha = gamma / np.sum(m_N ** 2)

        beta_inv = 1 / (N - gamma) * np.sum((y - Phi.dot(m_N)) ** 2)
        beta = 1 / beta_inv

        if np.isclose(alpha_prev, alpha, rtol=rtol) and np.isclose(beta_prev, beta, rtol=rtol):
            if verbose:
                print(f'Convergence after {i + 1} iterations.')
            return alpha, beta, m_N, S_N

    if verbose:
        print(f'Stopped after {max_iter} iterations.')
    return alpha, beta, m_N, S_N


# %%
def plot_nd_sf_lvn_approx(xx, axs, degree, post_mean, post_cov, beta,
                          samples_train, data_train, samples_test, data_test, labels,
                          dim1_index, x_ranges, y_ranges=None, y_log=True,
                          pct=None, colors=None, colors_pct=None, mean_label=r'$\mathrm{SFNet}$'):
    
    xx = (x_ranges[1]-x_ranges[0])*xx+x_ranges[0]

    if samples_train[0].shape[0] == 1:
        pct = [pct[len(pct)-1]]
        colors_pct = [colors_pct[len(colors_pct)-1]]

    for ii in range(len(pct)):
        if samples_train[0].shape[0] > 1:
            X_test = np.vstack([xx, [np.percentile(samples_train[0][jj], pct[ii]) * np.ones(xx.size) for jj in np.setdiff1d(np.arange(samples_train[0].shape[0]), dim1_index)]]).T
        else:
            X_test = np.vstack([xx]).T
        
        # Design matrix of test observations
        # Phi_test = pce_basis_matrix(X_test, degree=degree, dist_bounds=[-15, 15]) # [-100, 100] [-15, 15]

        # Design matrix of test observations
        # Phi_test = gn_expand(X_test, bf=gn_polynomial_basis_function, bf_deg_args=np.linspace(0, 1, degree+1)[1:])
        # Phi_test = gn_expand(X_test, bf=gn_polynomial_basis_function, bf_deg_args=range(1, degree + 1))

        # Design matrix of test observations
        Phi_test = gn_expand(X_test, bf=gn_gaussian_basis_function, bf_deg_args=np.linspace(0, 1, degree), sigma=[0.25])

        # print(Phi_test.shape)
        
        # Mean and variances of posterior predictive 
        approx_post_mean, approx_post_var = gn_posterior_predictive(Phi_test, post_mean, post_cov, beta=beta)

        # print(approx_post_mean.shape)
        # print(approx_post_std.shape)

        y = approx_post_mean
        approx_post_std = np.sqrt(approx_post_var)
        y_lower = approx_post_mean-2*approx_post_std
        y_upper = approx_post_mean+2*approx_post_std
        if (ii == len(pct)-1):
            approx_sigma_std = np.sqrt(1/beta)
            y_sigma_lower = approx_post_mean-2*approx_sigma_std
            y_sigma_upper = approx_post_mean+2*approx_sigma_std
        if y_log is False:
            y = np.exp(y)
            y_lower = np.exp(y_lower)
            y_upper = np.exp(y_upper)
            if (ii == len(pct)-1):
                y_sigma_lower = np.exp(y_sigma_lower)
                y_sigma_upper = np.exp(y_sigma_upper)

        axs.plot(xx, y, '--', color=colors_pct[ii], label=mean_label)
        # axs.plot(xx, y, '--', color=colors_pct[ii], label=mean_label+'$\mathrm{''}_{'+str(pct[ii])+'\%}$')
        axs.fill_between(xx, y_lower.squeeze(), y_upper.squeeze(), color=colors_pct[ii], alpha=0.5)
        # if (ii == len(pct)-1):
        #     axs.fill_between(xx, y_sigma_lower.squeeze(), y_sigma_upper.squeeze(), color=colors_pct[ii], alpha=0.5) # alpha=0.25

    # training (samples,data)
    if labels is None:
        labels = [None]*len(samples_train)
    for ii in range(len(samples_train)):
        samples_train[ii] = np.atleast_2d(samples_train[ii])
        if colors is not None:
            data_ii = data_train[ii]
            if y_log is False:
                data_ii = np.exp(data_ii)
            axs.plot(samples_train[ii][dim1_index, :], data_ii, 'o', label=labels[ii],
                     c=colors[ii])
        else:
            data_ii = data_train[ii]
            if y_log is False:
                data_ii = np.exp(data_ii)
            axs.plot(samples_train[ii][dim1_index, :], data_ii, 'o', label=labels[ii])

    # testing (samples,data)
    if labels is None:
        labels = [None]*len(samples_test)
    for ii in range(len(samples_test)):
        samples_test[ii] = np.atleast_2d(samples_test[ii])
        if colors is not None:
            data_ii = data_test[ii]
            if y_log is False:
                data_ii = np.exp(data_ii)
            axs.plot(samples_test[ii][dim1_index, :], data_ii, 'o', label=labels[ii+1],
                     markerfacecolor='none', c=colors[ii], alpha=0.50)
        else:
            data_ii = data_test[ii]
            if y_log is False:
                data_ii = np.exp(data_ii)
            axs.plot(samples_test[ii][dim1_index, :], data_ii, 'o', label=labels[ii+1],
                     markerfacecolor='none', alpha=1.0)

    axs.set_xlim([xx.min(), xx.max()])
    if y_ranges is not None:
        axs.set_ylim([y_ranges[0], y_ranges[1]])
    # get handles and labels
    handles, labels = axs.get_legend_handles_labels()
    # specify order of items in legend
    if len(labels) == 3:
        order = [0, 1, 2]
    elif len(labels) == 5:
        order = [0, 2, 1, 3, 4]
    elif len(labels) == 7:
        order = [0, 2, 4, 3, 1, 5, 6]
    # add legend to plot
    axs.legend([handles[idx] for idx in order], [labels[idx] for idx in order])         


# %%
def plot_nd_mf_lvn_approx(xx, axs, degree, post_mean, post_cov, beta,
                          samples_train, data_train, samples_test, data_test, labels,
                          dim1_index, x_ranges, y_ranges=None, y_log=True,
                          pct=None, colors=None, colors_pct=None, mean_label=r'$\mathrm{MFNet}$'):
    
    xx = (x_ranges[1]-x_ranges[0])*xx+x_ranges[0]

    if samples_train[0].shape[0] == 1:
        pct = [pct[len(pct)-1]]
        colors_pct = [colors_pct[len(colors_pct)-1]]

    for ii in range(len(pct)):
        # hi-fi post mean and cov prediction
        if samples_train[1].shape[0] > 1:
            X_test = np.vstack([xx, [np.percentile(samples_train[1][jj], pct[ii]) * np.ones(xx.size) for jj in np.setdiff1d(np.arange(samples_train[1].shape[0]), dim1_index)]]).T
        else:
            X_test = np.vstack([xx]).T

        # Design matrix of test observations
        # Phi_test = pce_basis_matrix(X_test, degree=degree, dist_bounds=[-15, 15]) # [-100, 100] [-15, 15]

        # Design matrix of test observations
        Phi_test = gn_expand(X_test, bf=gn_gaussian_basis_function, bf_deg_args=np.linspace(0, 1, degree), sigma=[0.25])

        # print(Phi_test.shape)
        
        # Mean and variances of posterior predictive 
        approx_post_mean, approx_post_var = gn_posterior_predictive(Phi_test, post_mean, post_cov, beta=beta)

        # print(approx_post_mean.shape)
        # print(approx_post_std.shape)
        
        y = approx_post_mean
        approx_post_std = np.sqrt(approx_post_var)
        y_lower = approx_post_mean-2*approx_post_std
        y_upper = approx_post_mean+2*approx_post_std
        if (ii == len(pct)-1):
            approx_sigma_std = np.sqrt(1/beta)
            y_sigma_lower = approx_post_mean-2*approx_sigma_std
            y_sigma_upper = approx_post_mean+2*approx_sigma_std
        if y_log is False:
            y = np.exp(y)
            y_lower = np.exp(y_lower)
            y_upper = np.exp(y_upper)
            if (ii == len(pct)-1):
                y_sigma_lower = np.exp(y_sigma_lower)
                y_sigma_upper = np.exp(y_sigma_upper)

        axs.plot(xx, y, '--', color=colors_pct[ii], label=mean_label)
        # axs.plot(xx, y, '--', color=colors_pct[ii], label=mean_label+'$\mathrm{''}_{'+str(pct[ii])+'\%}$')
        axs.fill_between(xx, y_lower.squeeze(), y_upper.squeeze(), color=colors_pct[ii], alpha=0.5) # alpha=0.5
        # if (ii == len(pct)-1):
        #     axs.fill_between(xx, y_sigma_lower.squeeze(), y_sigma_upper.squeeze(), color=colors_pct[ii], alpha=0.25) # alpha=0.25

    # training (samples,data)
    if labels is None:
        labels = [None]*len(samples_train)
    for ii in range(len(samples_train)):
        samples_train[ii] = np.atleast_2d(samples_train[ii])
        if ii == 0:
            alpha=1.0 # alpha=0.50
            # markersize=6
            markerfacecolor=colors[ii] # markerfacecolor='none'
        else:
            alpha=1.0
            # markersize=6
            markerfacecolor=colors[ii]
        if colors is not None:
            data_ii = data_train[ii]
            if y_log is False:
                data_ii = np.exp(data_ii)
            axs.plot(samples_train[ii][dim1_index, :], data_ii, 'o', label=labels[ii],
                     c=colors[ii], markerfacecolor=markerfacecolor, alpha=alpha) # markersize=markersize, 
        else:
            data_ii = data_train[ii]
            if y_log is False:
                data_ii = np.exp(data_ii)
            axs.plot(samples_train[ii][dim1_index, :], data_ii, 'o', label=labels[ii],
                     markerfacecolor=markerfacecolor, alpha=alpha) # markersize=markersize, 

    # testing (samples,data)
    if labels is None:
        labels = [None]*len(samples_test)
    # for ii in range(len(samples_test)):
    for ii in range(1, len(samples_test)):
        samples_test[ii] = np.atleast_2d(samples_test[ii])
        if colors is not None:
            data_ii = data_test[ii]
            if y_log is False:
                data_ii = np.exp(data_ii)
            axs.plot(samples_test[ii][dim1_index, :], data_ii, 'o', label=labels[ii+1],
                     markerfacecolor='none', c=colors[ii], alpha=0.50)
        else:
            data_ii = data_test[ii]
            if y_log is False:
                data_ii = np.exp(data_ii)
            axs.plot(samples_test[ii][dim1_index, :], data_ii, 'o', label=labels[ii+1],
                     markerfacecolor='none', alpha=0.50)

    axs.set_xlim([xx.min(), xx.max()])
    if y_ranges is not None:
        axs.set_ylim([y_ranges[0], y_ranges[1]])
    # get handles and labels
    handles, labels = axs.get_legend_handles_labels()
    # specify order of items in legend
    if len(labels) == 4:
        order = [0, 2, 3, 1]
    elif len(labels) == 6:
        order = [0, 2, 1, 4, 5, 3]
    elif len(labels) == 8:
        order = [0, 2, 4, 3, 1, 6, 7, 5]
    # add legend to plot
    axs.legend([handles[idx] for idx in order], [labels[idx] for idx in order])         


# %%
def choose_sample_value_rnd(X, y, lf_pct, hf_pct, rng=np.random.RandomState(1)):
    '''
    :param
    :return:
    '''

    # choose training data
    # choose random indices
    X_l_index = np.sort(rng.choice(np.arange(y[0].size), size=int(lf_pct*y[0].size), replace=False))
    # X_l_index = np.sort(rng.choice(np.arange(100,y[0].size), size=int(lf_pct*(y[0].size-100)), replace=False)) # exclude first 100 lo-fi networks
    X_h_index = np.sort(rng.choice(np.arange(y[1].size), size=int(hf_pct*y[1].size), replace=False))
    # X    
    X_l = X[0].copy()
    X_h = X[1].copy()
    samples_train = [X_l[X_l_index], X_h[X_h_index]]
    # y
    y_l = y[0].copy()
    y_h = y[1].copy()
    values_train = [y_l[X_l_index], y_h[X_h_index]]

    index_train = [X_l_index, X_h_index]

    return samples_train, values_train, index_train


# %%
def choose_sample_value_rnd_sf(X, y, hf_pct, rng=np.random.RandomState(1)):
    '''
    :param
    :return:
    '''

    # choose training data
    # choose random indices
    X_h_index = np.sort(rng.choice(np.arange(y[0].size), size=int(hf_pct*y[0].size), replace=False))
    # X    
    X_h = X[0].copy()
    samples_train = [X_h[X_h_index]]
    # y
    y_h = y[0].copy()
    values_train = [y_h[X_h_index]]

    index_train = [X_h_index]

    return samples_train, values_train, index_train


# %%
# def choose_sample_value_pct(X, y, rng=np.random.RandomState(1)):
#     '''
#     :param
#     :return:
#     '''

#     # choose training data
#     # X    
#     X_l = X[0].copy()
#     X_h = X[1].copy()
#     # choose X_hf networks from 33 66 and 100 iperm quantiles
#     quant_num_draws = 1
#     X_hf_perm_norm = X_h[:,2] / X_h[:,2].max() # mean iperm
#     quant_33 = np.quantile(X_hf_perm_norm,0.33)
#     quant_66 = np.quantile(X_hf_perm_norm,0.66)
#     X_hf_perm_quant_33 = np.flatnonzero(X_hf_perm_norm < quant_33)
#     X_hf_perm_quant_33_66 = np.flatnonzero((quant_33 <= X_hf_perm_norm) & (X_hf_perm_norm < quant_66))
#     X_hf_perm_quant_66_100 = np.flatnonzero(quant_66 <= X_hf_perm_norm)
#     # rng = np.random.RandomState(1)
#     X_hf_perm_index = np.array([rng.choice(X_hf_perm_quant_33, size=quant_num_draws, replace=False),
#                                 rng.choice(X_hf_perm_quant_33_66, size=quant_num_draws, replace=False),
#                                 rng.choice(X_hf_perm_quant_66_100, size=quant_num_draws, replace=False)]).squeeze()
#     # quant_10 = np.quantile(X_hf_perm_norm,0.10)
#     # quant_20 = np.quantile(X_hf_perm_norm,0.20)
#     # quant_30 = np.quantile(X_hf_perm_norm,0.30)
#     # quant_40 = np.quantile(X_hf_perm_norm,0.40)
#     # quant_50 = np.quantile(X_hf_perm_norm,0.50)
#     # quant_60 = np.quantile(X_hf_perm_norm,0.60)
#     # quant_70 = np.quantile(X_hf_perm_norm,0.70)
#     # quant_80 = np.quantile(X_hf_perm_norm,0.80)
#     # quant_90 = np.quantile(X_hf_perm_norm,0.90)
#     # X_hf_perm_quant_10 = np.flatnonzero(X_hf_perm_norm < quant_10)
#     # X_hf_perm_quant_10_20 = np.flatnonzero((quant_10 <= X_hf_perm_norm) & (X_hf_perm_norm < quant_20))
#     # X_hf_perm_quant_20_30 = np.flatnonzero((quant_20 <= X_hf_perm_norm) & (X_hf_perm_norm < quant_30))
#     # X_hf_perm_quant_30_40 = np.flatnonzero((quant_30 <= X_hf_perm_norm) & (X_hf_perm_norm < quant_40))
#     # X_hf_perm_quant_40_50 = np.flatnonzero((quant_40 <= X_hf_perm_norm) & (X_hf_perm_norm < quant_50))
#     # X_hf_perm_quant_50_60 = np.flatnonzero((quant_50 <= X_hf_perm_norm) & (X_hf_perm_norm < quant_60))
#     # X_hf_perm_quant_60_70 = np.flatnonzero((quant_60 <= X_hf_perm_norm) & (X_hf_perm_norm < quant_70))
#     # X_hf_perm_quant_70_80 = np.flatnonzero((quant_70 <= X_hf_perm_norm) & (X_hf_perm_norm < quant_80))
#     # X_hf_perm_quant_80_90 = np.flatnonzero((quant_80 <= X_hf_perm_norm) & (X_hf_perm_norm < quant_90))
#     # X_hf_perm_quant_90_100 = np.flatnonzero(quant_90 <= X_hf_perm_norm)
#     # X_hf_perm_index = np.array([rng.choice(X_hf_perm_quant_10, size=quant_num_draws, replace=False),
#     #                             rng.choice(X_hf_perm_quant_10_20, size=quant_num_draws, replace=False),
#     #                             rng.choice(X_hf_perm_quant_20_30, size=quant_num_draws, replace=False),
#     #                             rng.choice(X_hf_perm_quant_30_40, size=quant_num_draws, replace=False),
#     #                             rng.choice(X_hf_perm_quant_40_50, size=quant_num_draws, replace=False),
#     #                             rng.choice(X_hf_perm_quant_50_60, size=quant_num_draws, replace=False),
#     #                             rng.choice(X_hf_perm_quant_60_70, size=quant_num_draws, replace=False),
#     #                             rng.choice(X_hf_perm_quant_70_80, size=quant_num_draws, replace=False),
#     #                             rng.choice(X_hf_perm_quant_80_90, size=quant_num_draws, replace=False),
#     #                             rng.choice(X_hf_perm_quant_90_100, size=quant_num_draws, replace=False)]).squeeze()
#     X_h_index = X_hf_perm_index.reshape(1,-1).squeeze()
#     # choose random indices
#     # rng = np.random.RandomState(1)
#     X_l_index = np.sort(rng.choice(np.arange(y[0].size), size=int(1.0*y[0].size), replace=False))
#     # X_l_index = np.sort(rng.choice(np.arange(100,y[0].size), size=int(1.0*(y[0].size-100)), replace=False))
#     samples_train = [X_l[X_l_index], X_h[X_h_index]]
#     # y
#     y_l = y[0].copy()
#     y_h = y[1].copy()
#     values_train = [y_l[X_l_index], y_h[X_h_index]]

#     index_train = [X_l_index, X_h_index]

#     return samples_train, values_train, index_train


# %%
