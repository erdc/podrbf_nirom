#! /usr/bin/env python

import numpy as np
import scipy

import rbf as rbf
import greedy as gdy
import rom as rom
import plotting as plo

"""
Simple Utilities for Managing Snapshots and computing Proper Orthogonal Decompositions
"""


def compute_pod_multicomponent(S_pod, subtract_mean=True, subtract_initial=False, full_matrices=False):
    """
    Compute standard SVD [Phi,Sigma,W] for all variables stored in dictionary S_til
     where S_til[key] = Phi . Sigma . W is an M[key] by N[key] array
    Input:
    :param: S_pod -- dictionary of snapshots
    :param: subtract_mean -- remove mean or not
    :param: full_matrices -- return Phi and W as (M,M) and (N,N) [True] or (M,min(M,N)) and (min(M,N),N)

    Returns:
    S      : perturbed snapshots if requested, otherwise shallow copy of S_pod
    S_mean : mean of the snapshots
    Phi : left basis vector array
    sigma : singular values
    W   : right basis vectors

    """
    S_mean, S = {}, {}
    Phi, sigma, W = {}, {}, {}

    for key in S_pod.keys():
        if subtract_mean:
            S_mean[key] = np.mean(S_pod[key], 1)
            S[key] = S_pod[key].copy()
            S[key] -= np.tile(S_mean[key], (S_pod[key].shape[1], 1)).T
            Phi[key], sigma[key], W[key] = scipy.linalg.svd(
                S[key][:, 1:], full_matrices=full_matrices)
        elif subtract_initial:
            S_mean[key] = S_pod[key][:, 0]
            S[key] = S_pod[key].copy()
            S[key] -= np.tile(S_mean[key], (S_pod[key].shape[1], 1)).T
            Phi[key], sigma[key], W[key] = scipy.linalg.svd(
                S[key][:, :], full_matrices=full_matrices)
        else:
            S_mean[key] = np.mean(S_pod[key], 1)
            S[key] = S_pod[key]
            Phi[key], sigma[key], W[key] = scipy.linalg.svd(
                S[key][:, :], full_matrices=full_matrices)

    return S, S_mean, Phi, sigma, W


def compute_trunc_basis(D, U, eng_cap=0.999999):
    """
    Compute the number of modes and truncated basis to use based on getting 99.9999% of the 'energy'
    Input:
    D -- dictionary of singular values for each system component
    U -- dictionary of left singular basis vector arrays
    eng_cap -- fraction of energy to be captured by truncation
    Output:
    nw -- list of number of truncated modes for each component
    U_r -- truncated left basis vector array as a list (indexed in order of dictionary keys in D)
    """

    nw = {}
    for key in D.keys():
        nw[key] = 0
        total_energy = (D[key]**2).sum()
        assert total_energy > 0.
        energy = 0.
        while energy/total_energy < eng_cap and nw[key] < D[key].shape[0]-2:
            nw[key] += 1
            energy = (D[key][:nw[key]]**2).sum()
        print('{3} truncation level for {4}% = {0}, \sigma_{1} = {2}'.format(nw[key], nw[key]+1,
                                                                             D[key][nw[key]+1], key, eng_cap*100))

    U_r = {}
    for key in D.keys():
        U_r[key] = U[key][:, :nw[key]]

    return nw, U_r


def project_onto_basis(S, Phi, S_mean, msg=False):
    """
    Convenience function for computing projection of values in high-dimensional space onto
    Orthonormal basis stored in Phi.
    Only projects entries that are in both. Assumes these have compatible dimensions

    Input:
    S -- Dict of High-dimensional snapshots for each component
    Phi -- Dict of POD basis vectors for each component
    S_mean -- Dict of temporal mean for each component
    Output:
    Z -- Dict of modal coefficients for POD-projected snapshots
    """
    soln_names = S.keys()
    S_normalized = {}
    Z = {}
    for key in soln_names:
        S_normalized[key] = S[key].copy()
        S_normalized[key] -= np.outer(S_mean[key], np.ones(S[key].shape[1]))
        Z[key] = np.dot(Phi[key].T, S_normalized[key])
        if msg:
            print('{0} projected snapshot matrix size: {1}'.format(key, Z[key].shape))

    return Z


def reconstruct_from_rom(Z,Phi,S_mean,nw,msg=False):
    """
    Convenience function for computing projection of values in high-dimensional space onto
    Orthonormal basis stored in Phi.
    Only projects entries that are in both. Assumes these have compatible dimensions

    Input:
    S -- Dict of High-dimensional snapshots for each component
    Phi -- Dict of POD basis vectors for each component
    S_mean -- Dict of temporal mean for each component
    Output:
    Z -- Dict of modal coefficients for POD-projected snapshots
    """
    soln_names = nw.keys()
    S = {}
    ctr= 0
    for key in soln_names:
        S[key] = np.dot(Phi[key],Z[key]) + np.outer(S_mean[key],np.ones(Z[key].shape[1]))
        ctr = nw[key]

    return S
