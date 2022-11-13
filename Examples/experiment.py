from RECASOpt.problems.multiobjective_problems import DTLZ2
from RECASOpt.optimize.optimization import RECASoptimize

# Create a DTLZ2 problem with 2 objectives and 10 decision variables
OPT_PROB = DTLZ2(nobj=2, dim=10)

# Experiment Setup
N_TRIALS = 10
INIT_EVALS = 11 * OPT_PROB.dim - 1
MAX_EVALS = 300
BATCH_SIZE = 5

# Run multiple independent trials for RECAS
for trial in range(1, N_TRIALS + 1):
    RECASoptimize(
        trial_number=trial,         # int: Current trial number
        opt_prob=OPT_PROB,          # pySOT.OptimizationProblem: multi-objective test problem
        exp_design=None,            # pySOT.ExperimentalDesign: Default method is Latin Hypercube Sampling
        surrogate=None,             # pySOT.Surrogate: Default model is RBF with cubic kernel and linear tail
        init_evals=INIT_EVALS,      # int: Initial number of evaluations for experimental design
        max_evals=MAX_EVALS,        # int: Maximum number of evaluations
        batch_size=BATCH_SIZE,      # int: The size of each batch
    )
