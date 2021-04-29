#! usr/bin/env python

"""
Base module for POD-RBF NIROM
"""

import numpy as np
import pickle

import pod as pod
import rbf as rbf
import greedy as gdy
import rom as rom
import plotting as plo


class PODRBFBase(object):
    """
    Base class for Non-intrusive Reduced Order Modeling (NIROM)
    with Proper Orthogonal Decomposition (POD) for the selection
    of an optimal linear reduced basis space and Radial Basis
    Function (RBF) interpolation for the approximation of the
    dynamics in the space spanned by the POD modes.

    """
    @staticmethod
    def save_to_disk(filename, ROM, **options):
        """
        Save the instance in ROM to disk using options
        Start with absolutely simplest approach using pickle
        """
        outfile = open(filename, 'wb')
        protocol = options.get('protocol', pickle.HIGHEST_PROTOCOL)
        try:
            pickle.dump(ROM, outfile, protocol=protocol)
        finally:
            outfile.close()
        return ROM

    @staticmethod
    def read_from_disk(filename, **options):
        """
        Read the instance in ROM to disk using options
        Start with absolutely simplest approach using pickle
        """
        infile = open(filename, 'rb')
        encoding = options.get('encoding', 'latin1')
        ROM = None
        try:
            ROM = pickle.load(infile, encoding=encoding)
        except TypeError:
            ROM = pickle.load(infile)
        finally:
            infile.close()
        return ROM

    def __init__(self, kernel='matern', **options):
        # self._trunc  = trunc
        self._kernel = kernel
        print("Using {0} kernel for RBF approximation".format(kernel))
        if 'rank' in options:
            # rank is a tuple, one entry per component
            self._rank = options['rank']
        elif 'trunc' in options:
            self._trunc = options['trunc']
        else:
            self._trunc = 0.99999
            self._rank = None
            print('Setting POD truncation level at %f' % self._trunc)

    @property
    def trunc(self):
        """
        Truncation level of POD basis
        """
        return self._trunc

    @property
    def rank(self):
        """
        User specified rank of truncated
        POD basis, Output as an ordered
        tuple corresponding to system
        components
        """
        return self._rank

    @property
    def n_snap(self):
        """
        Return number of snapshots used in training
        """
        return self._n_snap_train

    @property
    def n_fine(self):
        """
        return fine grid dimension
        """
        return self._Phi.shape[0]

    @property
    def n_pod(self):
        """
        Truncated rank of POD basis,
        Output as a dictionary with
        system components as keys
        """
        return self._n_pod

    @property
    def t0_fit(self):
        """
        initial time value for POD-RBF calculation
        """
        return self._t0_fit

    @property
    def rbf_matrix(self):
        """
        RBF kernel interpolation matrix
        """
        return self._A_rbf

    @property
    def basis(self):
        """
        POD basis or left singular vectors
        (Truncated)
        """
        return self._Phi

    @property
    def pod_singular_values(self):
        """
        Singular values of snapshot Matrix
        (Non-truncated)
        """
        return self._Sigma

    def compute_pod_basis(self, S, times_offline):
        """
        Compute POD basis for each system
        component of high-fidelity snapshots

        Input:
        S --  Dict of N x M snapshots for each solution component
        times_offline -- Discretized time points corresponding to Snapshots

        Output:
        Phi -- Dict of N x m_i POD bases for each solution component
        Sigma -- Dict of m_i x m_i diagonal matrices of singular values
                        for each solution component
        """
        self._S = S
        self._comp_keys = S.keys()
        self._times_offline = times_offline
        self._n_snap_train = times_offline.shape[0]
        assert self._n_snap_train == self._S[list(S.keys())[0]].shape[1]
        # Compute SVD of snapshot matrix
        print("----Computing POD basis of snapshots----")
        S_fit, S_mean, Phi, Sigma, W = pod.compute_pod_multicomponent(
            S, subtract_mean=True)
        self._Sigma = Sigma
        self._S_fit = S_fit
        self._S_mean = S_mean

        # Compute truncated POD basis
        n_basis, Phi_r = pod.compute_trunc_basis(
            Sigma, Phi, eng_cap=self._trunc)
        self._Phi = Phi_r
        self._n_pod = n_basis

        # Compute modal coefficients in POD space
        Z_fit = pod.project_onto_basis(self._S, self._Phi, self._S_mean)
        self._Z_train = Z_fit

        dZdt = rbf.build_dFdt_multistep(Z_fit, times_offline,
                                        self._n_pod, flag=None)
        self._dzdt_train = dZdt

        return self._Phi, self._Sigma, self._Z_train

    def fit_rbf(self, Z_fit, times_fit, kernel='matern', eps=0.05, method=None, msg=False):
        """
        Compute RBF approximation of snapshots
        projected to POD basis space

        Input:
        Z_fit -- modal coefficients of projected snapshots
                in POD basis space
        times_fit -- Discretized time points corr to Z_fit
        kernel -- RBF kernel function used for approximation
                    Options are - 'matern', 'matern1',
                        'gaussian', 'MQ', 'inverseMQ'.
                        Default is 'matern'
        eps -- User specified maximum scaling factor
                for RBF interpolant
        method -- Time discretization method used for
                    representing dzdt. Options are -
                    1st order: 'None' (default method)
                    2nd order: 'LF', 'AB2', 'BDF-EP2'
                    3rd order: 'AB3', 'NY3', 'BDF-EP3'

        Output:
        A_rbf -- (M-1) x (M-1) RBF interpolation Matrix
        rbf_centers -- m x (M-1) array of RBF centers
                    where m = \sum_i (m_i) is total number
                    of POD modes for all components
        rbf_coeff -- m_i x (M-1) array of RBF coefficients
        Z_fit -- m_i x M array of modal coefficients for
                snapshots projected on to POD basis space
        dZdt -- m_i x (M-1) array of time derivatives of
                modal coefficients
        """
        self._time_disc_method = method

        # Compute RBF system
        Z_center, A, epsilon = rbf.compute_rbf(Z_fit, times_fit, ep=eps, msg=msg)
        self._scale = epsilon
        self._rbf_centers_train = Z_center
        self._A_train = A
        weights, dZdt = rbf.build_dFdt_weights_multistep(Z_fit, times_fit,
                                                         self._n_pod, self._A_train, flag=method)
        self._rbf_coeff_train = weights
        self._dzdt_greedy = dZdt

        return self._A_train, self._rbf_centers_train, self._rbf_coeff_train

    def predict_time(self, times_online, use_greedy=True, eps=0.05, **options):
        """
        Evaluate reduced order POD-RBF model
        at queried online time points

        Input:
        times_online -- array of time points to be queried
                    for evaluating ROM
        use_greedy -- Boolean variable for toggling greedy algorithms
        eng_cap -- Total energy fraction captured in mode list L (Optional)
        greedy_alg -- Greedy algorithm to be used (Optional)
                     Available options - 'p', 'f', 'psr' (default)
        """
        self._times_online = times_online
        if 't0_ind' in options:
            t0_ind = options['t0_ind']
        else:
            t0_ind = 0
        self._scale_user = eps
        if use_greedy:
            try:        # User-specified greedy algorithm
                self._greedy_alg = options['greedy_alg']
            except:
                self._greedy_alg = 'psr'
            try:        # User-specified energy threshold
                eng_cap = options['eng_cap']
            except:
                eng_cap = 0.91
            try:
                self._tau = options['tau']
            except:
                self._tau = None
            try:
                num_greedy_centers = options['num_cent']
            except:
                num_greedy_centers = 500
            try:
                greedy_output = options['greedy_output']
            except:
                greedy_output = False
            try:
                self._time_disc_method = options['method']
            except:
                self._time_disc_method = None

            Z_greedy, times_greedy = self.construct_greedy_rbf(eng_cap, num_greedy_centers,
                                                               greedy_output=greedy_output)
            print(
                "\n---- Computing RBF system using {0} greedy centers----\n".format(self._greedy_alg))
            A_greedy, centers_greedy, coeff_greedy = self.fit_rbf(Z_greedy, times_greedy,
                                                                  eps=self._scale_user,
                                                                  method=self._time_disc_method,
                                                                  msg = options['msg'])
            self._rbf_centers = centers_greedy
            self._rbf_coeff = coeff_greedy

        else:
            self._rbf_centers = self._rbf_centers_train
            self._rbf_coeff = self._rbf_coeff_train
            # self._ind_greedy = None

        print("\n---- Computing RBF NIROM solution ----\n")
        S_pred, Z_pred = rom.rbf_online(self._rbf_centers, self._rbf_coeff, self._S,
                                        self._times_online, self._scale_user, self._Phi, self._S_mean,
                                        self._kernel, time_disc=self._time_disc_method, beta=2.5,
                                        t0_ind=t0_ind)
        self._S_pred = S_pred
        self._Z_pred = Z_pred

        if use_greedy:
            return S_pred, Z_pred, self._ind_greedy
        else:
            return S_pred, Z_pred

    def construct_greedy_rbf(self, eng_cap, num_greedy_centers, **options):
        """
        Compute RBF system for greedy implementation
        """
        try:
            greedy_output = options['greedy_output']
        except:
            greedy_output = False
        if self._greedy_alg in ['psr','f-modal']:
            print("\n---- Computing list of modes L ----\n")
            mode_list = gdy.greedy_modelist(self._dzdt_train, self._comp_keys,
                                        eng_cap=eng_cap, msg=greedy_output)
            L_modes = gdy.combine_L(mode_list, self._n_pod)
            self._L = L_modes
        else:
            self._L = None

        print("\n---- Finding %s-greedy centers ----\n" % self._greedy_alg)
        A_tmp, Z_center_tmp = rbf.compute_interp_matrix(self._Z_train,
                                                        self._n_snap_train-1, rbf_kernel=self._kernel,
                                                        epsilon=self._scale_user)
        ind_greedy = gdy.greedy(self._dzdt_train, Z_center_tmp,
                                self._times_offline, self._kernel, self._scale_user, self._L,
                                alg=self._greedy_alg, msg=greedy_output, tau=self._tau)
        self._ind_greedy = np.sort(ind_greedy[:num_greedy_centers+1])

        S_greedy = {}
        for key in self._comp_keys:
            S_greedy[key] = self._S[key][:, self._ind_greedy]
        Z_greedy = pod.project_onto_basis(S_greedy, self._Phi, self._S_mean)
        times_greedy = self._times_offline[self._ind_greedy]

        return Z_greedy, times_greedy
