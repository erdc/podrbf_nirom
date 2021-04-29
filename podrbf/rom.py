#! /usr/bin/env python

"""
Module for evaluating the POD-RBF
reduced order model at new time-query
points during the online stage
"""

import numpy as np
import scipy
from scipy.spatial.distance import cdist
from numpy.lib.scimath import sqrt as csqrt
from scipy import interpolate

import pod as pod
import rbf as rbf
import greedy as gdy
import plotting as plo


# ---- Save initial conditions from snapshots
def init_cond_onl(snap_data, nt_online, n_pod, U, Smean, t0_ind=0):
    soln_names = n_pod.keys()
    uh0, zh0 = {}, {}
    uh_save, zh_save = {}, {}
    for ii, key in enumerate(n_pod.keys()):
        uh0[key] = snap_data[key][:, t0_ind].copy()
        zh0[key] = U[ii].T.dot(uh0[key]-Smean[key])

    for key in soln_names:
        zh_save[key] = np.zeros((n_pod[key], nt_online), 'd')
        uh_save[key] = np.zeros((snap_data[key].shape[0], nt_online), 'd')

    return uh_save, zh_save, uh0, zh0


def rbf_online(Zcent, wts, snap_data, times_online, epsn, U_trunc, Smn, kernel,
               time_disc='None', beta=2.5, t0_ind=0):
    """
    Compute RBF ROM solutions at online evaluation time points
    """
    soln_names = wts.keys()
    n_pod = {}
    nw = np.zeros(len(wts.keys()), 'i')
    U_incr = []
    for ii, key in enumerate(wts.keys()):
        n_pod[key] = wts[key].shape[0]
        nw[ii] = n_pod[key]
        U_incr.append(U_trunc[key])

    nt_online = times_online.shape[0]
    if t0_ind == 0:
        uh_save, zh_save, uh0, zh0 = init_cond_onl(snap_data, nt_online,
                                                   n_pod, U_incr, Smn)
    else:
        uh_save, zh_save, uh0, zh0 = init_cond_onl(snap_data, nt_online,
                                                   n_pod, U_incr, Smn, t0_ind=t0_ind)
    uh, zh, uhn, zhn = {}, {}, {}, {}
    for key in soln_names:
        uh[key] = uh0[key].copy()
        zh[key] = zh0[key].copy()
        uhn[key] = uh[key].copy()
        zhn[key] = zh[key].copy()
        zh_save[key][:, 0] = zh[key].copy()
        uh_save[key][:, 0] = uh[key].copy()

    #
    zeval = np.zeros((nw.sum(),), 'd')
    for istep, tn in enumerate(times_online[:-1]):
        if istep % 500 == 0:
            print("Computing solutions for time step %d" % (istep+1))
        t = times_online[istep+1]
        dt = t-tn
        offset = 0
        for key in soln_names:
            zeval[offset:offset+n_pod[key]] = zhn[key]
            offset += n_pod[key]

        for ii, key in enumerate(soln_names):
            # FOR HIGHER ORDER TIME STEPPING METHODS
            zh[key] = time_march_multistep(zeval, Zcent, wts, zh_save, key, istep,
                                           times_online, kernel, epsn, beta, time_disc)

            # ## FOR ORDER 1 TIME STEPPING METHOD
            # dtmp = rbf.rbf_evaluate(zeval[:,np.newaxis],Zcent,wts[key],
            #           epsilon=epsn, kernel=kernel,beta=beta)
            # dtmp = np.squeeze(dtmp)
            # zh[key] = zhn[key]+dt*dtmp

        # update u = \bar{u} + \Phiw . w
        for ii, key in enumerate(soln_names):
            uh[key][:] = Smn[key]
            uh[key][:] += np.dot(U_incr[ii], zh[key])
        # update reduced solution (ie generalized coordinates)
        for key in soln_names:
            # update zh and uh history
            zhn[key][:] = zh[key][:]
            uhn[key][:] = uh[key][:]
        # save for evaluation later
        for key in soln_names:
            zh_save[key][:, istep+1] = zh[key].copy()
            uh_save[key][:, istep+1] = uh[key].copy()

    return uh_save, zh_save


def time_march_multistep(zeval, Zcent, wts, zh_save, key, istep, times_online, kernel,
                         epsn=0.01, beta=2.5, time_disc='None'):
    """
    Compute modal time steps for a chosen solution component
    using a selected multistep method
    Available routines:
    1) Explicit midpoint or LeapFrog scheme,
    2) 2nd & 3rd order Adams Bashforth
    3) Explicit 3rd order Nystrom method
    4) 2nd and 3rd order extrapolated BDF methods
    ======
    Input-
    zeval:        Snapshot vector at time "n" (vertically stacked for all components)
    Zcent:        Matrix of RBF centers (vertically stacked), size = [nw_total x Nt_pod-1]
    wts:          Dictionary of computed weights for RBF interpolation
    zh_save:      Dictionary of all computed snapshots until time step "n"
    istep:        index of current time point
    times_online: Array of normalized time points for online computation of snapshots
    nw:           Dictionary of number of POD modes per component
    time_disc:    Denotes the selected time discretization scheme
    ======
    Output-
    zh:           Computed snapshot vector at time "n+1" (vertically stacked for all components)
    """

    n_pod = {}
    nw_total = 0
    soln_names = wts.keys()
    dt = times_online[istep+1] - times_online[istep]
    for ky in wts.keys():
        n_pod[ky] = wts[ky].shape[0]
        nw_total += n_pod[ky]

    if time_disc == 'LF':
        dtmp = rbf.rbf_evaluate_modal(
            zeval[:, np.newaxis], Zcent, wts[key], epsn, kernel)
        dtmp = np.squeeze(dtmp)
        if istep == 0:
            zh = zh_save[key][:, istep]+dt*dtmp
        else:
            zh = zh_save[key][:, istep-1] + \
                (times_online[istep+1]-times_online[istep-1])*dtmp

    elif time_disc == 'AB2':
        dtmp_1 = rbf.rbf_evaluate_modal(
            zeval[:, np.newaxis], Zcent, wts[key], epsn, kernel)
        dtmp_1 = np.squeeze(dtmp_1)
        if istep == 0:
            dtmp_2 = dtmp_1
        else:
            offset = 0
            zeval2 = np.zeros((nw_total,), 'd')
            for ky in soln_names:
                zeval2[offset:offset+n_pod[ky]] = zh_save[ky][:, istep-1]
                offset += n_pod[ky]

            dtmp_2 = rbf.rbf_evaluate_modal(
                zeval2[:, np.newaxis], Zcent, wts[key], epsn, kernel)
            dtmp_2 = np.squeeze(dtmp_2)
        zh = zh_save[key][:, istep] + dt*(1.5*dtmp_1 - 0.5*dtmp_2)

    elif time_disc == 'AB3':
        dtmp_1 = rbf.rbf_evaluate_modal(
            zeval[:, np.newaxis], Zcent, wts[key], epsn, kernel)
        dtmp_1 = np.squeeze(dtmp_1)
        if istep <= 1:
            dtmp_2 = dtmp_1
            dtmp_3 = dtmp_1
        else:
            offset = 0
            zeval2 = np.zeros((nw_total,), 'd')
            zeval3 = np.zeros((nw_total,), 'd')
            for ky in soln_names:
                zeval2[offset:offset+n_pod[ky]] = zh_save[ky][:, istep-1]
                zeval3[offset:offset+n_pod[ky]] = zh_save[ky][:, istep-2]
                offset += n_pod[ky]

            dtmp_2 = rbf.rbf_evaluate_modal(
                zeval2[:, np.newaxis], Zcent, wts[key], epsn, kernel)
            dtmp_3 = rbf.rbf_evaluate_modal(
                zeval3[:, np.newaxis], Zcent, wts[key], epsn, kernel)
            dtmp_2 = np.squeeze(dtmp_2)
            dtmp_3 = np.squeeze(dtmp_3)

        zh = zh_save[key][:, istep] + dt * \
            ((23./12.)*dtmp_1 - (4./3.)*dtmp_2 + (5./12.)*dtmp_3)

    elif time_disc == 'NY3':
        dtmp_1 = rbf.rbf_evaluate_modal(
            zeval[:, np.newaxis], Zcent, wts[key], epsn, kernel)
        dtmp_1 = np.squeeze(dtmp_1)
        if istep <= 1:
            zh = zh_save[key][:, istep]+dt*dtmp_1
        else:
            offset = 0
            zeval2 = np.zeros((nw_total,), 'd')
            zeval3 = np.zeros((nw_total,), 'd')
            for ky in soln_names:
                zeval2[offset:offset+n_pod[ky]] = zh_save[ky][:, istep-1]
                zeval3[offset:offset+n_pod[ky]] = zh_save[ky][:, istep-2]
                offset += n_pod[ky]

            dtmp_2 = rbf.rbf_evaluate_modal(
                zeval2[:, np.newaxis], Zcent, wts[key], epsn, kernel)
            dtmp_3 = rbf.rbf_evaluate_modal(
                zeval3[:, np.newaxis], Zcent, wts[key], epsn, kernel)
            dtmp_2 = np.squeeze(dtmp_2)
            dtmp_3 = np.squeeze(dtmp_3)
            zh = zh_save[key][:, istep-1] + dt * \
                (7.*dtmp_1 - 2.*dtmp_2 + dtmp_3)/3.

    elif time_disc == 'BDF-EP2':
        dtmp_1 = rbf.rbf_evaluate_modal(
            zeval[:, np.newaxis], Zcent, wts[key], epsn, kernel)
        dtmp_1 = np.squeeze(dtmp_1)
        if istep == 0:
            zh = zh_save[key][:, istep]+dt*dtmp_1
        else:
            offset = 0
            zeval2 = np.zeros((nw_total,), 'd')
            for ky in soln_names:
                zeval2[offset:offset+n_pod[ky]] = zh_save[ky][:, istep-1]
                offset += n_pod[ky]

            dtmp_2 = rbf.rbf_evaluate_modal(
                zeval2[:, np.newaxis], Zcent, wts[key], epsn, kernel)
            dtmp_2 = np.squeeze(dtmp_2)
            zh = 4. * zh_save[key][:, istep]/3. - zh_save[key][:, istep-1]/3. \
                + dt*((4./3.)*dtmp_1 - (2./3.)*dtmp_2)

    elif time_disc == 'BDF-EP3':
        dtmp_1 = rbf.rbf_evaluate_modal(
            zeval[:, np.newaxis], Zcent, wts[key], epsn, kernel)
        dtmp_1 = np.squeeze(dtmp_1)
        if istep <= 1:
            dtmp_2 = dtmp_1
            dtmp_3 = dtmp_1
        else:
            offset = 0
            zeval2 = np.zeros((nw_total,), 'd')
            zeval3 = np.zeros((nw_total,), 'd')
            for ky in soln_names:
                zeval2[offset:offset+n_pod[ky]] = zh_save[ky][:, istep-1]
                zeval3[offset:offset+n_pod[ky]] = zh_save[ky][:, istep-2]
                offset += n_pod[ky]

            dtmp_2 = rbf.rbf_evaluate_modal(
                zeval2[:, np.newaxis], Zcent, wts[key], epsn, kernel)
            dtmp_3 = rbf.rbf_evaluate_modal(
                zeval3[:, np.newaxis], Zcent, wts[key], epsn, kernel)
            dtmp_2 = np.squeeze(dtmp_2)
            dtmp_3 = np.squeeze(dtmp_3)

        zh = 18.*zh_save[key][:, istep]/11. - 9.*zh_save[key][:, istep-1]/11. + 2.*zh_save[key][:, istep-2]/11. \
            + dt*(3.*dtmp_1 - 3.*dtmp_2 + dtmp_3)

    else:
        dtmp = rbf.rbf_evaluate_modal(
            zeval[:, np.newaxis], Zcent, wts[key], epsn, kernel)
        dtmp = np.squeeze(dtmp)
        zh = zh_save[key][:, istep]+dt*dtmp

    return zh
