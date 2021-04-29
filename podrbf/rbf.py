#! /usr/bin/env python

"""
Module for generating an RBF approximation
of temporal dynamics in POD basis space
"""

import numpy as np
import scipy
from scipy.spatial.distance import cdist
from numpy.lib.scimath import sqrt as csqrt
from scipy import interpolate

import pod as pod
import greedy as gdy
import rom as rom
import plotting as plo


def compute_rbf(Zsnap_rbf, time_rbf, ep=0.05, beta=2.5, rbf_kernel='matern', msg=False):
    """
    Compute the rbf system that includes the rbf interpolation matrix,
    the weights for the rbf interpolation, and the optimal scaling factor

    Input::
            Zsnap_rbf: 	Dictionary of snapshots containing projected
                                    snapshots of every state variable
            time_rbf:  	Array of time points for the snapshots
            ep:			Pre-specified minimum threshold of scaling factor
            beta:		Secondary parameter for some IMQ rbf kernels
    Output::
            Zcenter_gr:	Interpolation Matrix
            weights_gr:	RBF interpolation coefficients
            epsilon_gr: Optimal scaling factor based on fill distances
    """
    # --- Recompute RBF system with optimized RBF centers
    soln_names = Zsnap_rbf.keys()
    rij = compute_radial_distances(Zsnap_rbf)
    epsilon = np.minimum(ep, estimate_epsilon_fill(rij))
    A, Zcenter = compute_interp_matrix(Zsnap_rbf, Zsnap_rbf[list(Zsnap_rbf.keys())[0]].shape[1]-1,
                                       rbf_kernel=rbf_kernel, epsilon=epsilon, beta=beta)

    if msg:
        print("Epsilon specified = {0}, epsilon computed = {1}".format(
            ep, estimate_epsilon_fill(rij)))
        print("Epsilon used = {0}".format(epsilon))

        print('Condition number of A: {0}'.format(np.linalg.cond(A)))

    return Zcenter, A, epsilon


def compute_interp_matrix(Zsnap, Nt, rbf_kernel='matern', epsilon=0.05, beta=2.5):
    """
    Build a radial basis function (RBF) interpolant using the entries in Zsnap to form the centers

    Zsnap is input on a solution component basis, e.g., Zsnap['h'] is a N_w['h'] x N_snap array of POD modes
      for the 'h' variable.

    For now assumes Nt is equal to N_snap-1

    Input:
    :param: Zsnap -- dictionary of modes for all snapshots
    :param: Nt -- Total number of snapshots - 1
    :param: component_keys -- which entries to use in building the components
    :param: rbf_kernel_flag -- type of kernel to use.
                               1 Gaussian
                                 Multiquadric otherwise
    Returns:
    Zcenter -- Composite [nw_total,Nt] array of evaluation points for RBF interpolant
    A  -- [Nt,Nt] array containing the rbf kernel evaluations on the Zcenter vectors
    rij -- [Nt,Nt] array containing the euclidean distances between all paris of vectors in Zcenter
    """

    component_keys = Zsnap.keys()

    # compute centers
    nw_sizes = [Zsnap[key].shape[0] for key in component_keys]
    nw_total = sum(nw_sizes)

    Zcenter = np.zeros((nw_total, Nt), 'd')
    offset = 0
    for ii, key in enumerate(component_keys):
        # evaluation points are (t_0,t_1,...,t_{snap-1})
        Zcenter[offset:offset+Zsnap[key].shape[0], :] = Zsnap[key][:, 0:-1]
        offset += Zsnap[key].shape[0]

    # distances between all of the evaluation points
    rij = rbf_norms(Zcenter, Zcenter)
    A = compute_kernel(rij, rbf_kernel, epsilon)

    return A, Zcenter


def compute_kernel(rij, rbf_kernel='matern', epsilon=0.05):
    """
    Compute Nc x Nc RBF kernel matrix,
    A = Phi(r,r) where Nc = # of RBF centers
    """

    if rbf_kernel == 'gaussian':
        A = rbf_gaussian(rij, epsilon)
    elif rbf_kernel == 'inverseMQ':
        A = rbf_inverse_multiquadric(rij, epsilon, beta)
    elif rbf_kernel == 'matern':
        A = rbf_matern(rij, epsilon)
    elif rbf_kernel == 'matern1':
        A = rbf_matern1(rij, epsilon)
    elif rbf_kernel == 'MQ':
        A = rbf_multiquadric(rij, epsilon)

    return A


def compute_radial_distances(Zsnap):
    """
    Routine to compute the distance between data points
    """
    component_keys = Zsnap.keys()
    nw_sizes = [Zsnap[key].shape[0] for key in component_keys]
    nw_total = sum(nw_sizes)
    Nt = Zsnap[list(Zsnap.keys())[0]].shape[1]-1

    Zcenter = np.zeros((nw_total, Nt), 'd')
    offset = 0
    for ii, key in enumerate(component_keys):
        # evaluation points are (t_0,t_1,...,t_{snap-1})
        Zcenter[offset:offset+Zsnap[key].shape[0], :] = Zsnap[key][:, 0:-1]
        offset += Zsnap[key].shape[0]

    # distances between all of the evaluation points
    rij = rbf_norms(Zcenter, Zcenter)

    return rij


def build_dFdt_multistep(Z_pod, times_pod, nw, flag=None):
    """
    Compute RBF weights for different high order time
    discretization methods
    Available routines:
    1) Explicit midpoint or LeapFrog scheme,
    2) 2nd & 3rd order Adams Bashforth
    3) Explicit 3rd order Nystrom method
    4) 2nd and 3rd order extrapolated BDF methods
    ======
    Input-
    Z_pod:     dictionary of projected snapshots per component
    times_pod: array of normalized time points corresponding to snapshots
    nw:        dictionary of number of POD modes per component
    flag:      Denotes the selected time discretization scheme
    ======
    Output-
    dZdata:    Dictionary of time derivative of modal coefficients,
                                size = [ nw[key] x Nt_pod-1 ]
    """

    soln_names = nw.keys()

    dt_pod = times_pod[1:]-times_pod[0:-1]
    dZdata = {}
    for key in soln_names:
        dZdata[key] = np.zeros((nw[key], times_pod.size-1), 'd')
        for mode in range(nw[key]):
            if flag == 'LF':
                dZdata[key][mode, 0] = Z_pod[key][mode, 1]-Z_pod[key][mode, 0]
                dZdata[key][mode, 0] /= dt_pod[0]
                dZdata[key][mode, 1:] = Z_pod[key][mode, 2:] - \
                    Z_pod[key][mode, 0:-2]
                dZdata[key][mode, 1:] /= (dt_pod[1:]+dt_pod[0:-1])
            elif flag == 'AB2':     # Adams Bashforth Order 2
                dZdata[key][mode, 0] = Z_pod[key][mode, 1]-Z_pod[key][mode, 0]
                dZdata[key][mode, 0] /= dt_pod[0]
                for inx in range(1, times_pod.size-1):
                    dZdata[key][mode, inx] = 2. * \
                        (Z_pod[key][mode, inx+1]-Z_pod[key]
                         [mode, inx])/(3.*dt_pod[inx])
                    dZdata[key][mode, inx] += dZdata[key][mode, inx-1]/3.
            elif flag == 'AB3':     # Adams Bashforth Order 3
                dZdata[key][mode, 0] = Z_pod[key][mode, 1]-Z_pod[key][mode, 0]
                dZdata[key][mode, 0] /= dt_pod[0]
                dZdata[key][mode, 1] = 2. * \
                    (Z_pod[key][mode, 2]-Z_pod[key][mode, 1])/(3.*dt_pod[1])
                dZdata[key][mode, 1] += dZdata[key][mode, 0]/3.
                for inx in range(2, times_pod.size-1):
                    dZdata[key][mode, inx] = 12 * \
                        (Z_pod[key][mode, inx+1]-Z_pod[key]
                         [mode, inx])/(23.*dt_pod[inx])
                    dZdata[key][mode, inx] += 16.*dZdata[key][mode,
                                                              inx-1]/23. - 5.*dZdata[key][mode, inx-2]/23.
            elif flag == 'NY3':    # Explicit Nystrom (k=3)
                dZdata[key][mode, 0:2] = Z_pod[key][mode, 1:3] - \
                    Z_pod[key][mode, 0:2]
                dZdata[key][mode, 0:2] /= dt_pod[0:2]
                for inx in range(2, times_pod.size-1):
                    dZdata[key][mode, inx] = 3. * \
                        (Z_pod[key][mode, inx+1] - Z_pod[key]
                         [mode, inx-1]) / (7.*dt_pod[inx])
                    dZdata[key][mode, inx] += (2.*dZdata[key]
                                               [mode, inx-1] - dZdata[key][mode, inx-2])/7.
            elif flag == 'BDF-EP2':   # Extrapolated BDF order 2
                dZdata[key][mode, 0] = Z_pod[key][mode, 1]-Z_pod[key][mode, 0]
                dZdata[key][mode, 0] /= dt_pod[0]
                for inx in range(1, times_pod.size-1):
                    dZdata[key][mode, inx] = .75*Z_pod[key][mode, inx+1] - Z_pod[key][mode, inx] \
                        + 0.25*Z_pod[key][mode, inx]
                    dZdata[key][mode, inx] /= (dt_pod[inx])
                    dZdata[key][mode, inx] += 0.5*dZdata[key][mode, inx-1]
            elif flag == 'BDF-EP3':   # Extrapolated BDF Order 3
                dZdata[key][mode, 0:2] = Z_pod[key][mode, 1:3] - \
                    Z_pod[key][mode, 0:2]
                dZdata[key][mode, 0:2] /= dt_pod[0:2]
                # dZdata[key][mode,1] = 2.*(Z_pod[key][mode,2]-Z_pod[key][mode,1])/(3.*dt_pod[1]);
                # dZdata[key][mode,1] += dZdata[key][mode,0]/3.
                for inx in range(2, times_pod.size-1):
                    dZdata[key][mode, inx] = 11.*Z_pod[key][mode, inx+1]/18. - Z_pod[key][mode, inx] \
                        + 0.5*Z_pod[key][mode, inx-1] - \
                        Z_pod[key][mode, inx-2]/9.
                    dZdata[key][mode, inx] /= dt_pod[inx]
                    dZdata[key][mode, inx] += dZdata[key][mode,
                                                          inx-1] - dZdata[key][mode, inx-2]/3.
            else:
                dZdata[key][mode, :] = Z_pod[key][mode, 1:] - \
                    Z_pod[key][mode, 0:-1]
                dZdata[key][mode, :] /= dt_pod

    return dZdata


def build_dFdt_weights_multistep(Z_pod, times_pod, nw, A, flag=None):
    """
    Compute RBF weights for different high order time
    discretization methods
    Available routines:
    1) Explicit midpoint or LeapFrog scheme,
    2) 2nd & 3rd order Adams Bashforth
    3) Explicit 3rd order Nystrom method
    4) 2nd and 3rd order extrapolated BDF methods
    ======
    Input-
    Z_pod:     dictionary of projected snapshots per component
    times_pod: array of normalized time points corresponding to snapshots
    nw:        dictionary of number of POD modes per component
    A:         RBF interpolation matrix
    flag:      Denotes the selected time discretization scheme
    ======
    Output-
    W_p:       dictionary of RBF interpolation coefficients, size = [ nw[key] x Nt_pod-1 ]
    """

    W_p = {}
    soln_names = Z_pod.keys()
    dZdata = build_dFdt_multistep(Z_pod, times_pod, nw, flag=flag)

    for key in soln_names:
        W_p[key] = np.zeros((nw[key], times_pod.size-1), 'd')
        for mode in range(nw[key]):
            W_p[key][mode, :] = np.linalg.solve(A, dZdata[key][mode, :])

    return W_p, dZdata


def rbf_multiquadric(r, epsilon=1.0, beta=2.5):
    """
    multiquadric
    """
    return np.sqrt((epsilon*r)**2 + 1.0)
#    return np.sqrt((1.0/epsilon*r)**2 + 1.0)


def rbf_inverse_multiquadric(r, epsilon=1.0, beta=2.5):
    """
    inverse multiquadric
    """
    return np.power((epsilon*r)**2 + 1, -beta)
    # return np.power(1.0 + (1.0/epsilon*r)**2,-beta)


def rbf_gaussian(r, epsilon=1.0, beta=2.5):
    """
    gaussian
    """
    return np.exp(-(epsilon*r)**2)


def rbf_matern(r, epsilon=1.0, beta=2.5):
    """
    matern kernel, order 0
    """
    return np.exp(-epsilon*r)


def rbf_matern1(r, epsilon=1.0, beta=2.5):
    """
    matern kernel, order 1
    """
    return np.exp(-epsilon*r)*(1 + epsilon*r)


def rbf_norms(x1, x2, kernel=None):
    """
    Computes the distance matrix for vector arguments x1, x2

    x1 : N x M_A matrix, M_A vectors in N-space
    x2 : N x M_B matrix, M_B vectors in N-space
    If kernel is None, returns euclidean distance,
    else returns 1D distance matrix for Exp Sin Sqd kernel
    """
    if kernel is None:
        return scipy.spatial.distance.cdist(x1.T, x2.T, 'euclidean')
    else:
        assert x1.shape[1] == x2.shape[1], 'x1 and x2 dimensions donot match'
        DM = np.empty((x1.shape[0], x2.shape[0]), dtype=np.double)
        for i in np.arange(0, x1.shape[0]):
            for j in np.arange(0, x2.shape[0]):
                DM[i, j] = (x1[i] - x2[j])
        return DM


def rbf_evaluate(x, centers, weights, epsilon=0.05, kernel='matern', beta=2.5):
    """
    Evaluates an RBF interpolant at unseen point(s)

    Input:
    x -- N x M matrix, M unseen center points in N-space
    centers -- N x P matrix, P centers in N-space used to
             define RBF interpolant
    weights -- P vector of weights defining RBF interpolant
    epsilon -- scale factor used to define RBF interpolant
    kernel 	-- RBF kernel function

    Output:
                    lin. comb. of weights and basis functions
    """
    r = rbf_norms(x, centers)
    phi = compute_kernel(r, kernel, epsilon)
    # phi = kernel(r,epsilon,beta)
    return weights.dot(phi.T)


def rbf_evaluate_modal(x, centers, wts, epsilon=0.05, kernel='matern', beta=2.5):
    """
    Evaluates an RBF interpolant at unseen point(s)
    x : N vector, unseen center point in N-space
    centers: N x P matrix, P centers in N-space used to
             define RBF interpolant
    wts: P vector of weights defining RBF interpolant
    epsilon: scale factor used to define RBF interpolant
    kernel: RBF kernel function

    returns lin. comb. of weights and basis functions
    """
    r_online = rbf_norms(x, centers)
    phi_online = compute_kernel(r_online, kernel, epsilon)
    # phi_online = kernel(r_online,epsilon,beta)
    dzdt = np.zeros((wts.shape[0],))
    for J in range(wts.shape[0]):
        dzdt[J] = wts[J, :].dot(phi_online.T)

    return dzdt


def estimate_epsilon(centers):
    """
    Estimates scaling factor using
    the method outlined in scikit-learn
    """
    Nb = centers.shape[1]
    ximax2 = np.amax(centers, axis=1)
    ximin2 = np.amin(centers, axis=1)
    edges2 = ximax2-ximin2
    edges2 = edges2[np.nonzero(edges2)]
    epsilon2 = np.power(np.prod(edges2)/Nb, 1.0/edges2.size)
    return epsilon2


def estimate_epsilon_fill(rij):
    """
    Estimates scaling factor as the elementwise
    minimum of the distance matrix, rij
    """
    rij[rij == 0.] = 999.  # mask out the zero values
    # compute the minimum distance between any two centers
    fill_dist = np.amin(np.amin(rij, axis=0))

    # technically the fill distance is the largest of such distances
    # fill_dist = np.amax(np.amin(R,axis=0))

    return fill_dist


def err_comp(uh, snap, times_offline, times_online):
    """
    Computes the absolute l2 error norm and the rms error
    norm between the true solution and the nirom solution projected
    on to the full dimensional space
    """
    err = {}
    w_rms = {}
    soln_names = uh.keys()
    # ky = list(uh.keys())[0]
    N = snap[list(uh.keys())[0]].shape[0]

    tstep = np.searchsorted(times_offline, times_online)
    for key in soln_names:
        interp = uh[key]
        true = snap[key][:, tstep]
        err[key] = np.linalg.norm(true - interp, axis=0)
        w_rms[key] = err[key]/(np.sqrt(N))

    return w_rms
