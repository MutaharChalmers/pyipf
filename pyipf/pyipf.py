#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from tqdm.auto import tqdm


def ipf(Z0, marginals, tol_convg=1e-4, convg='relative', max_itr=1000, pbar=False):
    """Iterative Proportional Fitting (IPF) matrix balancing.

    IPF is also known as RAS or biproportional fitting. It is the operation of
    adjusting an initial N-dimensional matrix such that its marginal sums match
    N target matrices of dimension N-1. In the 2-D case, the matrix is adjusted
    until the row and column sums match two target vectors.

    The function assumes an N-dimensional input matrix will have N-1 dimension
    marginals passed as arguments.

    Parameters
    ----------
        Z0 : ndarray
            N-dimensional numpy array to be adjusted.
        marginals : list of ndarrays
            Collection of N numpy arrays of dimension N-1 to which Z0's marginal
            sums will be adjusted.
        tol_convg : float, optional
            Convergence tolerance of marginals of adjusted matrix relative to 
            target marginals.
        convg : str, optional
            Specify whether convergence tolerance is relative (default) or 
            absolute. 
        max_itr : int, optional
            Maximum number of iterations.
        pbar : bool, optional
            Show progress bar or not.

    Returns
    -------
        Z : ndarray
            N-dimensional numpy array to be adjusted.
    """
    
    Z = Z0*1
    tol = np.inf

    # Expand dimensions of marginals to be consistent with initial matrix
    marginals = [np.expand_dims(marginal, axis=i) 
                 for i, marginal in enumerate(marginals)]

    # Check that all marginals are consistent - works up to 3D (2D marginals)
    ndims = len(Z.shape)
    dims = set(range(ndims))

    # Loop over dimensions, marginalise marginals and check consistency
    check_sums = {}
    for dim in dims:
        for marg_dim in dims-{dim}:
            res_dims = tuple(dims - {dim, marg_dim})
            marg_dim_sum = marginals[dim].sum(axis=res_dims).ravel()
            check_sums.setdefault(marg_dim, []).append(marg_dim_sum)
    check_sum = [np.all(np.isclose(marg_dim_sums, marg_dim_sums[0]))
                 for dim, marg_dim_sums in check_sums.items()]
    if not all(check_sum):
        print('Warning: all marginals must be consistent for convergence')

    # Main loop
    if pbar:
        itr = tqdm(range(max_itr))
    else:
        itr = range(max_itr)
        
    for i in itr:
        # Loop over each marginal array
        for j, marginal in enumerate(marginals):
            # Calculate and apply scaling factors, filling nans and infs with 1
            sc = marginal/Z.sum(axis=j, keepdims=True)
            sc[(np.isnan(sc)) | (np.isinf(sc))] = 1
            Z = np.multiply(Z, sc)

        # Check convergence
        if 'rel' in convg.lower():
            # Relative convergence criterion
            tol = min(tol, max([abs(Z.sum(axis=k, keepdims=True)/marginal - 1).max() 
                                for k, marginal in enumerate(marginals)]))
        else:
            # Assume absolute convergence criterion
            tol = min(tol, max([abs(Z.sum(axis=k, keepdims=True)-marginal).max()
                                for k, marginal in enumerate(marginals)]))
        if pbar:
            itr.set_description(f'Convergence: {tol:.2e}')
        if tol <= tol_convg:
            break

    # Warning if not converged after max_itr iterations
    if tol > tol_convg:
        print(f'Warning: convergence threshold not met within max_itr={max_itr} iterations: {tol:.3e}')

    return Z
