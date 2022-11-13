import numpy as np
import scipy.stats as stats
import scipy.spatial as scpspatial


def weighted_distance_merit(num_pts, surrogate, X, cand, weights, Xpend=None, dtol=1e-3):
    dim = X.shape[1]
    if Xpend is None or len(Xpend) == 0:
        Xpend = np.empty([0, dim])
    dists = scpspatial.distance.cdist(cand, np.vstack((X, Xpend)))
    dmerit = np.amin(dists, axis=1, keepdims=True)

    fvals = surrogate.predict(cand)
    fvals = unit_rescale(fvals)

    new_points = []
    for i in range(num_pts):
        w = weights[i]
        merit = w * fvals + (1.0 - w) * (1.0 - unit_rescale(np.copy(dmerit)))

        merit[dmerit < dtol] = np.inf
        if np.min(merit) < np.inf:
            jj = np.argmin(merit)
            fvals[jj] = np.inf
            new_points.append(cand[jj, :].copy())

            ds = scpspatial.distance.cdist(
                cand, np.atleast_2d(cand[jj, :].copy()))
            dmerit = np.minimum(dmerit, ds)

    return new_points


def candidate_dycors(num_pts, opt_prob, surrogate, xbest, X, weights, init_evals, max_evals, evals, sampling_radius, Xpend=None, subset=None, dtol=1e-3):
    num_cand = 100*opt_prob.dim

    if subset is None:
        subset = np.arange(0, opt_prob.dim)

    scalefactors = sampling_radius * (opt_prob.ub - opt_prob.lb)
    ind = np.intersect1d(opt_prob.int_var, subset)
    if len(ind) > 0:
        scalefactors[ind] = np.maximum(scalefactors[ind], 1.0)

    # prob_perturb = min([20.0 / opt_prob.dim, 1.0]) * (1.0 - (
    #     np.log(evals - init_evals + 1.0) / np.log(
    #     max_evals - init_evals + 1)))
    # prob_perturb = max(prob_perturb, min(1.0, 1.0 / opt_prob.dim))
    prob_perturb = 1.0

    if len(subset) == 1:
        ar = np.ones((num_cand, 1))
    else:
        ar = (np.random.rand(num_cand, len(subset)) < prob_perturb)
        ind = np.where(np.sum(ar, axis=1) == 0)[0]
        ar[ind, np.random.randint(0, len(subset) - 1, size=len(ind))] = 1

    cand = np.multiply(np.ones((num_cand, opt_prob.dim)), xbest)
    for (i, j) in zip(subset, list(range(len(subset)))):
        lower, upper, sigma = opt_prob.lb[i], opt_prob.ub[i], scalefactors[i]
        ind = np.where(ar[:, j] == 1)[0]
        cand[ind, i] = stats.norm.rvs(loc=xbest[i], scale=sigma, size=len(ind))
        cand[:, i] = np.minimum(upper, np.maximum(lower, cand[:, i]))

    for i in range(len(cand)):
        for j in opt_prob.int_var:
            cand[i, j] = max([min([round(cand[i, j]), opt_prob.ub[j]]), opt_prob.lb[j]])

    return weighted_distance_merit(
        num_pts=num_pts, surrogate=surrogate, X=X, Xpend=Xpend, cand=cand, dtol=dtol, weights=weights
    )


def candidate_uniform(num_pts, opt_prob, surrogate, X, weights, Xpend=None, subset=None, dtol=1e-3):
    # Fix default values
    num_cand = 100*opt_prob.dim

    if subset is None:
        subset = list(range(opt_prob.dim))

    cand = np.ones((num_cand, opt_prob.dim))
    cand[:, subset] = np.random.uniform(opt_prob.lb[subset], opt_prob.ub[subset], (num_cand, len(subset)))

    for i in range(len(cand)):
        for j in opt_prob.int_var:
            cand[i, j] = max([min([round(cand[i, j]), opt_prob.ub[j]]), opt_prob.lb[j]])

    return weighted_distance_merit(
        num_pts=num_pts, surrogate=surrogate, X=X, Xpend=Xpend, cand=cand, dtol=dtol, weights=weights
    )


def unit_rescale(x):
    x_max = x.max()
    x_min = x.min()
    if x_max == x_min:
        return np.ones(x.shape)
    else:
        return (x - x_min)/(x_max - x_min)
