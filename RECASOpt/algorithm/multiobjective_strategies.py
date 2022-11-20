import abc
import time
from copy import deepcopy
import numpy as np
import scipy.spatial as scp
import scipy.stats as stats
from poap.strategy import BaseStrategy, Proposal
from ..utils.multiobjective_archives import Record, FrontArchive, SimpleFrontArchive
from ..utils.multiobjective_utilities import uniform_points, radius_rule
from sklearn.cluster import KMeans

INF = float('inf')


class RECAS(BaseStrategy):
    __metaclass__ = abc.ABCMeta

    def __init__(
        self,
        max_evals,
        opt_prob,
        exp_design,
        surrogate,
        batch_size = None,
        extra_points = None,
        extra_vals = None,
        asynchronous = True,
        interactive2D = False,
    ):
        self.asynchronous = asynchronous
        self.interactive2D = interactive2D
        self.batch_size = batch_size

        # Save the objects
        self.opt_prob = opt_prob
        self.opt_prob.records = []
        self.exp_design = exp_design
        self.aggregated_surrogate = deepcopy(surrogate)

        # Sampler state
        self.proposal_counter = 0
        self.terminate = False
        self.accepted_count = 0
        self.rejected_count = 0

        # Initial design info
        self.extra_points = extra_points
        self.extra_vals = extra_vals
        self.batch_queue = []        # Unassigned points in initial experiment
        self.init_pending = 0        # Number of outstanding initial fevals
        self.phase = 1               # 1 for initial, 2 for adaptive

        # Budgeting state
        self.num_evals = 0             # Number of completed fevals
        self.max_evals = max_evals     # Remaining feval budget
        self.pending_evals = 0         # Number of outstanding fevals
        self.budget = None

        # Completed evaluations
        self.X = np.empty([0, opt_prob.dim])
        self.fX = np.empty([0, opt_prob.nobj])
        self.Xpend = np.empty([0, opt_prob.dim])

        # Auxiliary variables
        self.archive = FrontArchive(1000)
        self.improvement_archive = SimpleFrontArchive()

        self.centers = []
        self.pairs = []
        self.newpoints = []

        self.num_RVs = 55
        self.fixed_RVs = None
        self.adapt_RVs = None
        self.alpha = 10.0

        self.kmeans = None
        self.cluster_centers = None
        self.active_RVs = None

        # Start with first experimental design
        self.start_time = time.time()
        self.end_time = None
        self.sample_initial()

    def transform(self, x):
        x_max = x.max()
        x_min = x.min()
        if x_max == x_min:
            return np.ones(x.shape)
        else:
            return (x - x_min) / (x_max - x_min)

    def angle(self, rA, rB):
        temp = rA.dot(rB) / np.sqrt(rA.dot(rA)) * np.sqrt(rB.dot(rB))
        return np.arccos(min(max(1e-6, temp), 1.0 - 1e-6))

    def construct_RVs(self):
        self.fixed_RVs = uniform_points(exp_npoints=self.num_RVs, nobj=self.opt_prob.nobj)
        self.num_RVs = len(self.fixed_RVs)
        for i in range(self.num_RVs):
            self.fixed_RVs[i] = np.divide(self.fixed_RVs[i], np.sqrt(np.sum([x ** 2 for x in self.fixed_RVs[i]])))

        self.adapt_RVs = np.copy(self.fixed_RVs)

    def propose_action(self):
        if self.terminate:  # Check if termination has been triggered
            if self.pending_evals == 0:
                return Proposal('terminate')
        elif self.num_evals + self.pending_evals >= self.max_evals or \
                self.terminate:
            if self.pending_evals == 0:  # Only terminate if nothing is pending
                self.end_time = time.time()
                print('Normal Termination')
                return Proposal('terminate')
        elif self.batch_queue:  # Propose point from the batch_queue
            if self.phase == 1:
                return self.init_proposal()
            else:
                return self.adapt_proposal()
        else:  # Make new proposal in the adaptive phase
            self.phase = 2
            if self.asynchronous:  # Always make proposal with asynchrony
                self.generate_evals(num_pts=1)
            elif self.pending_evals == 0:  # Make sure the entire batch is done
                self.generate_evals(num_pts=self.batch_size)

            if self.terminate:  # Check if termination has been triggered
                if self.pending_evals == 0:
                    return Proposal('terminate')

            # Launch evaluation (the others will be triggered later)
            return self.adapt_proposal()

    def make_proposal(self, x):
        """Create proposal and update counters and budgets."""
        proposal = Proposal('eval', x)
        self.pending_evals += 1
        self.Xpend = np.vstack((self.Xpend, np.copy(x)))
        return proposal

    def remove_pending(self, x):
        """Delete a pending point from self.Xpend."""
        idx = np.where((self.Xpend == x).all(axis=1))[0]
        self.Xpend = np.delete(self.Xpend, idx, axis=0)

    def sample_initial(self):
        """Initialization Phase."""
        self.construct_RVs()

        start_sample = self.exp_design.generate_points(
            lb=self.opt_prob.lb,
            ub=self.opt_prob.ub,
            int_var=self.opt_prob.int_var,
        )
        assert start_sample.shape[1] == self.opt_prob.dim, \
            "Dimension mismatch between problem and experimental design"

        for j in range(self.exp_design.num_pts):
            self.batch_queue.append(start_sample[j, :])

        if self.extra_points is not None:
            for i in range(self.extra_points.shape[0]):
                if self.extra_vals is None or \
                        np.all(np.isnan(self.extra_vals[i])):  # Unknown value
                    self.batch_queue.append(self.extra_points[i, :])
                else:  # Known value, save point and add to surrogate model
                    x = np.copy(self.extra_points[i, :])
                    self.X = np.vstack((self.X, x))
                    self.fX = np.vstack((self.fX, self.extra_vals[i]))

    def init_proposal(self):
        """Propose a point from the initial experimental design."""
        proposal = self.make_proposal(self.batch_queue.pop())
        proposal.add_callback(self.on_initial_proposal)
        return proposal

    def on_initial_proposal(self, proposal):
        """Handle accept/reject of proposal from initial design."""
        if proposal.accepted:
            self.on_initial_accepted(proposal)
        else:
            self.on_initial_rejected(proposal)

    def on_initial_accepted(self, proposal):
        """Handle proposal accept from initial design."""
        self.accepted_count += 1
        proposal.record.add_callback(self.on_initial_update)

    def on_initial_rejected(self, proposal):
        """Handle proposal rejection from initial design."""
        self.rejected_count += 1
        self.pending_evals -= 1
        xx = proposal.args[0]
        self.batch_queue.append(xx)  # Add back to queue
        self.Xpend = np.vstack((self.Xpend, np.copy(xx)))
        self.remove_pending(xx)

    def on_initial_update(self, record):
        """Handle update of feval from initial design."""
        if record.status == 'completed':
            self.on_initial_completed(record)
        elif record.is_done:
            self.on_initial_aborted(record)

    def on_initial_completed(self, record):
        """Handle successful completion of feval from initial design."""
        self.num_evals += 1
        self.pending_evals -= 1

        xx, fx = np.copy(record.params[0]), np.copy(record.value)
        self.X = np.vstack((self.X, np.copy(np.atleast_2d(xx))))
        self.fX = np.vstack((self.fX, np.copy(fx)))
        self.remove_pending(xx)

        rec = Record(x=np.copy(xx), fx=np.copy(fx))
        self.opt_prob.records.append(rec)

    def on_initial_aborted(self, record):
        """Handle aborted feval from initial design."""
        self.pending_evals -= 1
        xx = record.params[0]
        self.batch_queue.append(xx)
        self.remove_pending(xx)

    def candidate_generation_and_selection(self):
        self.assign_to_reference_vectors(self.opt_prob.records, self.cluster_centers)
        self.centers = []
        self.pairs = []
        self.newpoints = []

        for i in range(len(self.cluster_centers)):
            gamma = INF
            for j in range(len(self.cluster_centers)):
                if i != j:
                    theta = self.angle(rA=self.cluster_centers[i], rB=self.cluster_centers[j])
                    gamma = min(theta, gamma)

            local_max = -INF
            APD = np.asarray([INF] * len(self.opt_prob.records))
            for j in range(len(self.opt_prob.records)):
                theta = self.opt_prob.records[j].associate_theta if self.opt_prob.records[j].associate == i else \
                    self.angle(rA=self.opt_prob.records[j].bar_fx, rB=self.cluster_centers[i])
                APD[j] = (1.0 + float(self.opt_prob.nobj) * (theta / gamma) * np.power(
                    float(self.num_evals) / float(self.max_evals), self.alpha)) * np.sqrt(
                    self.opt_prob.records[j].bar_fx.dot(self.opt_prob.records[j].bar_fx))
                local_max = max(local_max, APD[j])

            APD_zip = [[APD[j], j] if self.opt_prob.records[j].associate == i else [APD[j] + local_max, j] for j in range(len(self.opt_prob.records))]
            APD_zip.sort(key=lambda x: x[0], reverse=False)
            for j in range(len(self.opt_prob.records)):
                if radius_rule(self.opt_prob.records[APD_zip[j][1]], self.centers):
                    self.centers.append(self.opt_prob.records[APD_zip[j][1]])
                    break

            # Fit local surrogates
            distance_zip = [[scp.distance.euclidean(self.opt_prob.records[j].x[:], self.centers[-1].x[:]), j] for j in range(len(self.opt_prob.records))]
            distance_zip.sort(key=lambda x: x[0], reverse=False)

            self.aggregated_surrogate.reset()
            for j in range(min([len(self.opt_prob.records), 400])):
                self.aggregated_surrogate.add_points(xx=self.opt_prob.records[distance_zip[j][1]].x[:], fx=APD[distance_zip[j][1]])

            # Generate candidates and select one for evaluation
            self.candidate(num_pts=1, xbest=self.centers[-1].x[:], weights=[0.95], sampling_radius=self.centers[-1].radius)
            uniqueness = True
            for xx in self.X:
                if scp.distance.euclidean(xx, self.new_cand[0]) < 1e-6:
                    uniqueness = False
                    break
            for _, xx in self.pairs:
                if scp.distance.euclidean(xx, self.new_cand[0]) < 1e-6:
                    uniqueness = False
                    break
            if uniqueness: self.pairs.append([self.centers[-1], self.new_cand[0][:]])

        if len(self.pairs) > 0:
            for center, cand in self.pairs:
                self.batch_queue.append(cand)
        else:
            self.generate_evals(self.batch_size)

    def update_parameters(self):
        self.archive.reset()
        for record in self.opt_prob.records:
            self.archive.add(record=record)
        self.opt_prob.transform()

        # ==============================================================================
        # todo: Update adaptive reference vectors
        self.adapt_RVs = []
        for RV in self.fixed_RVs:
            self.adapt_RVs.append(np.copy(np.multiply(RV, self.opt_prob.maxpt - self.opt_prob.minpt)))
        for RV in self.adapt_RVs:
            RV[:] = np.divide(RV, np.sqrt(RV.dot(RV)))

    def cluster_center_selection(self):
        # Construct base population to determine active vectors
        population = []
        for i in range(self.num_evals - 1, max([self.num_evals - self.exp_design.num_pts, -1]), -1):
            population.append(self.opt_prob.records[i])

        # Assign evaluated points in population to adaptive reference vectors
        active_index, inactive_index = self.assign_to_reference_vectors(population, self.adapt_RVs)

        # Determine the center reference vectors
        if len(inactive_index) > len(active_index):
            nclusters = min([self.batch_size, len(inactive_index)])
            self.active_RVs = np.copy([self.adapt_RVs[i][:] for i in inactive_index])
        else:
            nclusters = min([self.batch_size, len(active_index)])
            self.active_RVs = np.copy([self.adapt_RVs[i][:] for i in active_index])

        self.cluster_centers = []
        self.kmeans = KMeans(n_clusters=nclusters, max_iter=10000).fit(np.copy(self.active_RVs))

        dis = np.asarray([0.0] * len(self.adapt_RVs))
        index = []
        for i in range(nclusters):
            for j in range(len(self.adapt_RVs)):
                dis[j] = INF if j in index else scp.distance.euclidean(np.copy(self.adapt_RVs[j]), np.copy(self.kmeans.cluster_centers_[i]))
            index.append(int(np.argmin(dis)))
            self.cluster_centers.append(np.copy(self.adapt_RVs[index[-1]]))

    def plot_progress(self):
        if len(self.fX[0]) == 2:
            try:
                from matplotlib import pyplot as plt
                fig = plt.figure()
                ax = fig.add_subplot(111)

                ax.scatter(self.fX[:, 0], self.fX[:, 1], c='black', s=2.0)

                if self.newpoints:
                    for rec in self.newpoints:
                        ax.scatter(rec.fx[0], rec.fx[1], c='red', s=10.0)

                if self.archive.fronts:
                    for rec in self.archive.fronts[0]:
                        ax.scatter(rec.fx[0], rec.fx[1], c='blue', s=5.0)

                for record in self.centers:
                    ax.scatter(record.fx[0], record.fx[1], c='green', s=10.0)

                ax.set_title('Number of Evals Completed: ' + str(self.num_evals))
                plt.show()
            except ImportError:
                print("matplotlib package is required for 2D interactive plotting!")

    def weighted_distance_merit(self, num_pts, cand, weights, Xpend=None, dtol=1e-3):
        dim = self.X.shape[1]
        if Xpend is None or len(Xpend) == 0:
            Xpend = np.empty([0, dim])
        dists = scp.distance.cdist(cand, np.vstack((self.X, Xpend)))
        dmerit = np.amin(dists, axis=1, keepdims=True)

        fvals = self.transform(self.aggregated_surrogate.predict(cand))

        self.new_cand = []
        for i in range(num_pts):
            w = weights[i]
            merit = w * fvals + (1.0 - w) * (1.0 - self.transform(np.copy(dmerit)))

            merit[dmerit < dtol] = np.inf
            if np.min(merit) < np.inf:
                jj = np.argmin(merit)
                fvals[jj] = np.inf
                self.new_cand.append(cand[jj, :].copy())

                ds = scp.distance.cdist(
                    cand, np.atleast_2d(cand[jj, :].copy()))
                dmerit = np.minimum(dmerit, ds)

    def candidate(self, num_pts, xbest, weights, sampling_radius, Xpend=None, subset=None, dtol=1e-3):
        num_cand = 100 * self.opt_prob.dim

        if subset is None:
            subset = np.arange(0, self.opt_prob.dim)

        scalefactors = sampling_radius * (self.opt_prob.ub - self.opt_prob.lb)
        ind = np.intersect1d(self.opt_prob.int_var, subset)
        if len(ind) > 0:
            scalefactors[ind] = np.maximum(scalefactors[ind], 1.0)

        prob_perturb = 1.0
        # Note:
        # Dynamically Dimensioned Search (DDS) strategy with a decreasing perturbation probability (Tolson and Shoemaker,
        # 2007) is recommended when using RECAS to solve problems with high-dimensional decision space.
        #
        # Reference: Tolson, B. A., & Shoemaker, C. A. (2007). Dynamically dimensioned search algorithm for computationally
        # efficient watershed model calibration. Water Resources Research, 43(1).
        #
        # prob_perturb = min([20.0 / opt_prob.dim, 1.0]) * (1.0 - (
        #     np.log(evals - init_evals + 1.0) / np.log(
        #     max_evals - init_evals + 1)))
        # prob_perturb = max(prob_perturb, 1.0 / opt_prob.dim)

        if len(subset) == 1:
            ar = np.ones((num_cand, 1))
        else:
            ar = (np.random.rand(num_cand, len(subset)) < prob_perturb)
            ind = np.where(np.sum(ar, axis=1) == 0)[0]
            ar[ind, np.random.randint(0, len(subset) - 1, size=len(ind))] = 1

        cand = np.multiply(np.ones((num_cand, self.opt_prob.dim)), xbest)
        for (i, j) in zip(subset, list(range(len(subset)))):
            lower, upper, sigma = self.opt_prob.lb[i], self.opt_prob.ub[i], scalefactors[i]
            ind = np.where(ar[:, j] == 1)[0]
            cand[ind, i] = stats.norm.rvs(loc=xbest[i], scale=sigma, size=len(ind))
            cand[:, i] = np.minimum(upper, np.maximum(lower, cand[:, i]))

        for i in range(len(cand)):
            for j in self.opt_prob.int_var:
                cand[i, j] = max([min([round(cand[i, j]), self.opt_prob.ub[j]]), self.opt_prob.lb[j]])

        self.weighted_distance_merit(num_pts=num_pts, Xpend=Xpend, cand=cand, dtol=dtol, weights=weights)

    def assign_to_reference_vectors(self, population, RVs):
        status = [0] * len(RVs)
        for record in population:
            theta_list = []
            for i in range(len(RVs)):
                theta = self.angle(rA=record.bar_fx, rB=RVs[i])
                theta_list.append(theta)
            record.associate_theta = np.min(theta_list)
            record.associate = int(np.argmin(theta_list))
            status[int(np.argmin(theta_list))] = 1

        active_index = np.asarray([i for i in range(len(RVs)) if status[i] == 1])
        inactive_index = np.asarray([i for i in range(len(RVs)) if status[i] == 0])

        return active_index, inactive_index

    def save_to_file(self, fpath):
        solutions = np.zeros((self.max_evals, self.opt_prob.dim + self.opt_prob.nobj))
        solutions[:, 0:self.opt_prob.dim] = self.X[:self.max_evals, :]
        solutions[:, self.opt_prob.dim:self.opt_prob.dim + self.opt_prob.nobj] = self.fX[:self.max_evals, :]
        np.savetxt(fpath, solutions)

    def generate_evals(self, num_pts):
        print('Number of evaluations completed = {}'.format(self.num_evals))

        self.update_parameters()
        self.cluster_center_selection()
        if self.interactive2D:
            self.plot_progress()
        self.candidate_generation_and_selection()

    def adapt_proposal(self):
        """Propose a point from the batch_queue."""
        if self.batch_queue:
            proposal = self.make_proposal(self.batch_queue.pop())
            proposal.add_callback(self.on_adapt_proposal)
            return proposal

    def on_adapt_proposal(self, proposal):
        """Handle accept/reject of proposal from sampling phase."""
        if proposal.accepted:
            self.on_adapt_accept(proposal)
        else:
            self.on_adapt_reject(proposal)

    def on_adapt_accept(self, proposal):
        """Handle accepted proposal from sampling phase."""
        self.accepted_count += 1
        proposal.record.add_callback(self.on_adapt_update)

    def on_adapt_reject(self, proposal):
        """Handle rejected proposal from sampling phase."""
        self.rejected_count += 1
        self.pending_evals -= 1
        xx = np.copy(proposal.args[0])
        self.remove_pending(xx)
        if not self.asynchronous:  # Add back to the queue in synchronous case
            self.batch_queue.append(xx)

    def on_adapt_update(self, record):
        """Handle update of feval from sampling phase."""
        if record.status == 'completed':
            self.on_adapt_completed(record)
        elif record.is_done:
            self.on_adapt_aborted(record)

    def on_adapt_completed(self, record):
        """Handle completion of feval from sampling phase."""
        self.num_evals += 1
        self.pending_evals -= 1

        xx, fx = np.copy(record.params[0]), np.copy(record.value)
        self.X = np.vstack((self.X, np.copy(np.atleast_2d(xx))))
        self.fX = np.vstack((self.fX, np.copy(fx)))
        self.remove_pending(xx)

        rec = Record(x=np.copy(xx), fx=np.copy(fx))
        self.opt_prob.records.append(rec)

        self.improvement_archive.reset()
        for record in self.archive.fronts[0]:
            self.improvement_archive.add(record)
        improvement = self.improvement_archive.add(Record(x=np.copy(xx), fx=np.copy(fx)))

        for center, cand in self.pairs:
            if scp.distance.euclidean(xx, cand) < 1e-6:
                if not improvement:
                    center.radius = max([center.radius / 2.0, center.minradius])
                break

        self.newpoints.append(rec)

    def on_adapt_aborted(self, record):
        """Handle aborted feval from sampling phase."""
        self.pending_evals -= 1
        xx = np.copy(record.params[0])
        self.remove_pending(xx)
