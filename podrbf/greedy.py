#! /usr/bin/env python

"""
Module for greedy algorithms to select optimal
RBF interpolation centers
"""

import numpy as np
import scipy
from scipy.spatial.distance import cdist
from numpy.lib.scimath import sqrt as csqrt
from scipy import interpolate
from scipy.integrate import simps

import pod as pod
import rbf as rbf
import rom as rom


def greedy_modelist(dZdata, comp_names, eng_cap=0.91, msg=False):
    """
    Module for selecting list of modes for
    each of which greedy iterations are
    performed

    Input:
    dZdata -- Dict containing m_i x (M-1) arrays of time
                derivatives of modal coefficients
    """
    dZdata_int = {}
    mode_lst = {}
    for ky in comp_names:
        dZdata_int[ky] = np.zeros(dZdata[ky].shape[0])
        dZdata_int[ky] = simps(np.abs(dZdata[ky])**2, axis=1)

        if msg:
            print('The {2} modal energy content = {0}% \n for modes {1}\n'.format(
                np.sort(dZdata_int[ky])[::-1][:10]/dZdata_int[ky].sum()*100,
                np.argsort(dZdata_int[ky])[::-1][:10], ky))
        total_energy = dZdata_int[ky].sum()
        assert total_energy > 0.
        energy = np.zeros((1,))
        mode_lst[ky] = []
        while energy/total_energy < eng_cap and len(mode_lst[ky]) < dZdata_int[ky].shape[0]-2:
            energy = np.sort(dZdata_int[ky])[::-1][:len(mode_lst[ky])+1].sum()
            mode_lst[ky] = np.argsort(dZdata_int[ky])[
                ::-1][:len(mode_lst[ky])+1]

        print('For {0}% of {2} energy, no. of modes required = {1}.'.format(
            eng_cap*100, len(mode_lst[ky]), ky))
        if msg:
            print('The selected modes are {0}\n'.format(mode_lst[ky]))

    return mode_lst  # , dZdata_int


def combine_L(mode_lst, npod):
    """
    Combine the modes selected for all components
    into a list of modes L by cycling over each
    component

    Input:
    mode_lst -- Dict of modes with components as keys
    Output:
    L -- List of combined modes for sequential greedy
        iterations
    """
    L_modes = np.array([], dtype='int64')
    soln_names = list(npod.keys())

    lcm = np.amin([len(mode_lst[soln_names[0]]), len(
        mode_lst[soln_names[1]]), len(mode_lst[soln_names[2]])])
    lcm2 = np.amin([len(mode_lst[soln_names[1]]),
                    len(mode_lst[soln_names[2]])])

    for ii in np.arange(lcm):
        for ky in [soln_names[1], soln_names[2], soln_names[0]]:
            if ky == soln_names[1]:
                L_modes = np.append(
                    L_modes, npod[soln_names[0]]+mode_lst[ky][ii])
            elif ky == soln_names[2]:
                L_modes = np.append(
                    L_modes, npod[soln_names[0]] + npod[soln_names[1]] + mode_lst[ky][ii])
            else:
                L_modes = np.append(L_modes, mode_lst[ky][ii])
    for jj in np.arange(lcm, lcm2):
        for ky in [soln_names[1], soln_names[2]]:
            if ky == soln_names[1]:
                L_modes = np.append(
                    L_modes, npod[soln_names[0]] + mode_lst[ky][jj])
            else:
                L_modes = np.append(
                    L_modes, npod[soln_names[0]] + npod[soln_names[1]] + mode_lst[ky][jj])

    for ky in soln_names:
        if len(mode_lst[ky] > jj):
            if ky == soln_names[0]:
                L_modes = np.append(L_modes, mode_lst[ky][jj+1:])
            elif ky == soln_names[1]:
                L_modes = np.append(
                    L_modes, npod[soln_names[0]] + mode_lst[ky][jj+1:])
            else:
                L_modes = np.append(
                    L_modes, npod[soln_names[0]] + npod[soln_names[1]] + mode_lst[ky][jj+1:])

    print("Total number of modes selected : {0}".format(len(L_modes)))
    print("List of selected modes : {0}".format(L_modes))

    return L_modes


def greedy(dZdt, Zcenter, times_offline, kernel, eps, L_modes, beta=2.5,
           alg='psr', max_iter=4000, msg=True, **kwargs):
    """
    Matrix-free implementation of greedy algorithms using Newton basis formulation
    """
    phi0 = rbf.compute_kernel(0, rbf_kernel=kernel, epsilon=eps)
    Nt = times_offline.size
    pod_mask = np.arange(0, Nt)
    max_iter = np.minimum(max_iter, len(pod_mask)-1)  # Max basis size
    Ind = np.zeros(max_iter, dtype='i')
    # Initializing set of centers not yet selected
    notInd = np.arange(len(pod_mask)-1)

    Nc = Zcenter.shape[1]  # Number of available centers
    Nm = Zcenter.shape[0]  # Total number of modes for all components
    # Nc x (# iter) matrix for evaluation of selected Newton basis functions
    Nn_eval = np.zeros((Nc, max_iter))
    pmax = np.zeros(max_iter)
    fmax = pmax.copy()
    psrmax = pmax.copy()

    # Compute the data function (time derivative) values
    dZdata = np.zeros((Nm, Nc))
    dZdata_norm = np.zeros((Nm, Nc))
    assert Nc == dZdt[list(dZdt.keys())[0]].shape[1]
    offset = 0
    for key in dZdt.keys():
        dZdata[offset:offset+dZdt[key].shape[0], :] = dZdt[key]
        # Normalizing the data for the f-greedy algorithm
        dZdata_norm[offset:offset+dZdt[key].shape[0], :] = dZdt[key] / np.linalg.norm(dZdt[key],axis=0)
        offset += dZdt[key].shape[0]

    p = phi0*np.ones(Nc)  # For translationally invariant kernel
    f = dZdata.copy()
    c = np.zeros((Nm, max_iter))
    # fnorm = np.linalg.norm(dZdata,axis=0);  ## f-greedy without normalization
    fnorm = np.linalg.norm(dZdata_norm,axis=0);  ## f-greedy with normalization
    cnorm = np.zeros((max_iter))
    psr = np.zeros((Nm, Nc))

    for kk in np.arange(Nm):
        psr[kk, :] = np.multiply(p, f[kk, :])
    iter = 0
    iter_prev = -1

    if alg == 'p':
        if kwargs['tau'] == None:
            tau_p = 1e-1
        else:
            tau_p = kwargs['tau']
        while iter < max_iter-1:
            pmax[iter] = np.amax(p[notInd])
            imax = np.argmax(p[notInd])
            if msg:
                print("Greedy iteration # {0}, Index selected {1}".format(
                    iter, notInd[imax]))
                print("Maximum of power function {0}".format(p[notInd][imax]))
            Ind[iter] = pod_mask[notInd[imax]]
            if pmax[iter] >= tau_p**2:
                p = compute_p(Zcenter, Nn_eval, p, f, notInd,
                              imax, iter, kernel, eps, beta)
                notInd = np.delete(notInd, imax)
                iter += 1
            else:
                print("Greedy tolerance reached")
                break

    elif alg == 'f-modal':
        if kwargs['tau'] == None:
            tau_f = 1e-5
        else:
            tau_f = kwargs['tau']
        for q in L_modes:
            fmax_q = np.amax(np.abs(f[q, notInd]))
            while iter < max_iter-1:
                fmax[iter] = np.amax(np.abs(f[q, notInd]))
                if iter > iter_prev:
                    imax = np.argmax(np.abs(f[q, notInd]))
                    if msg:
                        print("Greedy iteration # {0}, Index selected = {1}, mode = {2}".format(
                            iter, notInd[imax], q))
                        print(
                            "Maximum of function residual {0}".format(fmax[iter]))
                    Ind[iter] = pod_mask[notInd[imax]]
                    iter_prev = iter

                if fmax[iter]/fmax_q >= tau_f:
                    f, c, p = compute_psr(
                        Zcenter, Nn_eval, p, f, c, q, notInd, imax, iter, kernel, eps, beta)
                    notInd = np.delete(notInd, imax)
                    iter += 1
                else:
                    print(
                        "Greedy tolerance reached. No new center selected for mode {0}".format(q))
                    break

    elif alg == 'f':
        if 'tau' in kwargs:
            tau_f = kwargs['tau']
        else:
            tau_f = 1e-5
        print("tau_f = ",tau_f)
        fmax_l2 = np.amax(np.abs(fnorm[notInd]))
        while iter < max_iter-1:
            fmax[iter] = np.amax(np.abs(fnorm[notInd]))
            if iter > iter_prev:
                imax = np.argmax(np.abs(fnorm[notInd]))
                Ind[iter] = pod_mask[notInd[imax]]
                if msg:
                    print("Greedy iteration # {0}, Index selected = {1}".format(iter,
                    notInd[imax]))
                    print("Maximum of function residual {0}".format(fmax[iter]))
                    if Ind[iter] == Ind.any():
                        print("\n**********Duplicate indices*********\n")
                iter_prev = iter

            if fmax[iter]/fmax_l2 >= tau_f:
                fnorm, cnorm, p = compute_psr_l2(
                    Zcenter, Nn_eval, p, fnorm, cnorm, notInd, imax, iter, kernel, eps, beta)
                notInd = np.delete(notInd, imax)
                iter +=1
            else:
                print("Greedy tolerance reached.")
                break

    elif alg == 'psr':
        if kwargs['tau'] == None:
            tau_psr = 1e-5
        else:
            tau_psr = kwargs['tau']

        for q in L_modes:
            psrmax_q = np.amax(
                np.abs(np.multiply(f[q, notInd], np.sqrt(p[notInd]))))
            while iter < max_iter-1:
                psr[q, notInd] = np.abs(np.multiply(
                    np.abs(f[q, notInd]), np.sqrt(p[notInd])))
                psrmax[iter] = np.amax(psr[q, notInd])
                if iter > iter_prev:
                    imax = np.argmax(psr[q, notInd])
                    if msg:
                        print("Greedy iteration # {0}: Index selected = {1}, mode = {2}".format(
                            iter, notInd[imax], q))
                        print("Maximum of power scaled function residual {0:1.4g}".format(
                            psrmax[iter]))
                    Ind[iter] = pod_mask[notInd[imax]]
                    iter_prev = iter

                if psrmax[iter]/psrmax_q >= tau_psr:
                    f, c, p = compute_psr(
                        Zcenter, Nn_eval, p, f, c, q, notInd, imax, iter, kernel, eps, beta)
                    notInd = np.delete(notInd, imax)
                    iter += 1
                else:
                    print(
                        "Greedy tolerance reached. No new center selected for mode {0}".format(q))
                    break
    else:
        print("Not a valid greedy algorithm")

    Ind = np.delete(Ind, np.arange(iter+1, max_iter))
    # NOTE: Ind is not sorted.

    return Ind


def compute_p(Zcenter, Nn_eval, p, f, notInd, imax, iter, kernel, eps, beta):
    """
    Computes the recursive Newton basis functions and power functions for p-greedy algorithm
    """
    EM = rbf.rbf_norms(Zcenter[:, notInd],
                       Zcenter[:, notInd[imax]][:, np.newaxis])
    Nn_eval[notInd, iter] = np.squeeze(rbf.compute_kernel(
        EM, kernel, eps)) - np.dot(Nn_eval[notInd, :iter], Nn_eval[notInd[imax], :iter].T)
    Nn_eval[notInd, iter] /= np.sqrt(p[notInd[imax]])
    p[notInd] -= Nn_eval[notInd, iter]**2

    assert not np.isnan(p[notInd]).any()  # Check for funky NaNs
    return p


def compute_psr(Zcenter, Nn_eval, p, f, c, q, notInd, imax, iter, kernel, eps, beta):
    """
    Computes the recursive Newton basis functions and residuals for modal f-greedy and psr-greedy algorithms
    """
    EM = rbf.rbf_norms(Zcenter[:, notInd],
                       Zcenter[:, notInd[imax]][:, np.newaxis])
    Nn_eval[notInd, iter] = np.squeeze(rbf.compute_kernel(
        EM, kernel, eps)) - np.dot(Nn_eval[notInd, :iter], Nn_eval[notInd[imax], :iter].T)
    Nn_eval[notInd, iter] /= np.sqrt(p[notInd[imax]])
    c[:, iter] = f[:, notInd[imax]]/np.sqrt(p[notInd[imax]])
    p[notInd] -= Nn_eval[notInd, iter]**2
    f[q, notInd] -= c[q, iter]*Nn_eval[notInd, iter]

    assert not np.isnan(p[notInd]).any()  # Check for funky NaNs
    assert not np.isnan(f[q, notInd].any())
    return f, c, p


def compute_psr_l2(Zcenter, Nn_eval, p, f, c, notInd, imax, itern, kernel, eps, beta):
    """
    Computes the recursive Newton basis functions and residuals for f-l2-greedy and psr-l2-greedy algorithms
    """
    EM = rbf.rbf_norms(Zcenter[:, notInd],
                        Zcenter[:, notInd[imax]][:, np.newaxis])
    Nn_eval[notInd, itern]  = np.squeeze(rbf.compute_kernel(
        EM, kernel, eps)) - np.dot(Nn_eval[notInd, :itern], Nn_eval[notInd[imax], :itern].T)
    Nn_eval[notInd, itern] /= np.sqrt(p[notInd[imax]])
    c[itern] = f[notInd[imax]]/np.sqrt(p[notInd[imax]])
    p[notInd] -= Nn_eval[notInd, itern]**2
    f[notInd] -= c[itern]*Nn_eval[notInd, itern]

    assert not np.isnan(p[notInd]).any()   ## Check for funky NaNs
    assert not np.isnan(f[notInd].any())
    return f, c, p
