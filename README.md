# RECAS: An RBF Surrogate-Based Optimizer for Expensive Many-Objective Optimization Problems

## Introduction

Multi-objective Optimization Problems (MOPs), which aim to find optimal trade-off solutions regarding to multiple conflicting and equally important objectives, commonly exist in real-world applications. Without loss of generality, an MOP is mathematically defined by

$$
\min_{\mathbf{x} \in \mathcal{D}} \mathbf{f}(\mathbf{x}) = \[f_1(\mathbf{x}), \dots, f_k(\mathbf{x})\]^T
$$

where $\mathbf{x}$ denotes a vector composed of $d$ decision variables. The search space $\mathcal{D}$ denotes a hypercube in $\mathbb{R}^d$ and it is bounded by a lower bound $\mathbf{l}$ and a upper bound $\mathbf{u} \in \mathbb{R}^d$. $\mathbf{f}$ is composed of $k$ objective functions with $f_i : \mathbb{R}^d \rightarrow \mathbb{R}$ representing the $i$-th objective to be optimized, $i = 1, \dots, k$. In the literature, MOPs with more than three objectives are also known as Many-objective Optimization Problems (MaOPs).

**RECAS** is an effective Radial Basis Function (RBF) surrogate-based algorithm for computationally expensive
multi- and many-objective optimization problem where each objective is assumed to be black-box and expensive-to-evaluate.
The proposed algorithm iteratively determines new sample points for expensive evaluation through multiple candidate searches 
with the assistance of a set of adapative reference vectors. Meanwhile, it constructs the surrogate model in an aggregated manner 
to approximate the quality assessment indicator of each point rather than a certain objective function. 
Under some mild assumptions, RECAS converges almost surely to the Pareto-optimal front. 

## Installation

The Python version of RECAS is implemented upon a surrogate optimization toolbox, pySOT, which provides various types of surrogate models, 
experimental desings, acquisition functions, and test problems. To find out more about pySOT, please visit 
its [toolbox documentation](http://pysot.readthedocs.io/) or refer to the corresponding paper [David Eriksson, David Bindel, Christine A. Shoemaker. pySOT and POAP: An event-driven asynchronous framework for surrogate optimization. arXiv preprint arXiv:1908.00420, 2019](https://doi.org/10.48550/arXiv.1908.00420). 
In a virtual environment with Python 3.4 or newer, pySOT package can be installed by
```
pip install pySOT
```

## Using RECAS

The example below shows how to run RECAS on a multi-objective optimization problem with predefined experiment setups.

### Test Problem Setup

The following codes create an instance for DTLZ2 test problem with 2 objectives and 10 decision variables.

```python
from RECASOpt.problems.multiobjective_problems import DTLZ2

OPT_PROB = DTLZ2(nobj=2, dim=10)
```

*Note: The RECASOpt package have already implemented some problems for testing. You can also easily design your own test problem classes by inheriting
from the OptimizationProblem class defined in pySOT.*

### Run Experiment

The following codes run the RECAS for 10 independent trials on the DTLZ2 test problem instantiated above. Each trial starts with $11d-1$ (where 
$d$ denotes the number of decison variables) sample points and ends after 300 objective evaluations. 

```python
from RECASOpt.optimize.optimization import RECASoptimize

# Experiment Setup
N_TRIALS = 10
INIT_EVALS = 11 * OPT_PROB.dim - 1
MAX_EVALS = 300
BATCH_SIZE = 5

# Run multiple independent trials for RECAS
for trial in range(1, N_TRIALS + 1):
    RECASoptimize(
        trial_number=trial,     # int: Current trial number
        opt_prob=OPT_PROB,      # pySOT.OptimizationProblem: multi-objective test problem
        exp_design=None,        # pySOT.ExperimentalDesign: Default method is Latin Hypercube Sampling
        surrogate=None,         # pySOT.Surrogate: Default model is RBF with cubic kernel and linear tail
        init_evals=INIT_EVALS,  # int: Initial number of evaluations for experimental design
        max_evals=MAX_EVALS,    # int: Maximum number of evaluations
        batch_size=BATCH_SIZE,  # int: The size of each batch
    )
```

*Note: Other experimental design methods (exp_design) and surrogate models (surrogate) are available in pySOT. Like the customization for test
problem (opt_prob), you can also program and use your own exp_design and surrogate classes in RECAS, but they must inherit from the parent classes,
pySOT.ExperimentalDesign and pySOT.Surrogate, respectively.*

**Support**
For support in using RECAS, please contact the developer by [email](mailto:wangwenyu0928@gmail.com)