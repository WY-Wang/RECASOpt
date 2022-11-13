from poap.controller import BasicWorkerThread, ThreadController
from pySOT.surrogate import RBFInterpolant, CubicKernel, LinearTail
from pySOT.experimental_design import LatinHypercube
from ..algorithm.multiobjective_strategies import RECAS


def RECASoptimize(
    trial_number,
    opt_prob,
    exp_design,
    surrogate,
    init_evals,
    max_evals,
    batch_size,
):
    print(f"Trial Number: {trial_number}")

    if init_evals is None:
        init_evals = 11 * opt_prob.dim - 1
    if exp_design is None:
        exp_design = LatinHypercube(dim=opt_prob.dim, num_pts=init_evals)
    if surrogate is None:
        surrogate = RBFInterpolant(
            dim=opt_prob.dim,
            lb=opt_prob.lb,
            ub=opt_prob.ub,
            kernel=CubicKernel(),
            tail=LinearTail(opt_prob.dim),
        )

    controller = ThreadController()
    controller.strategy = RECAS(
        max_evals=max_evals,
        opt_prob=opt_prob,
        exp_design=exp_design,
        surrogate=surrogate,
        batch_size=batch_size,
        asynchronous=False,
    )

    # Launch the threads and give them access to the objective function
    for _ in range(batch_size):
        worker = BasicWorkerThread(controller, opt_prob.eval)
        controller.launch_worker(worker)

    # Run the optimization strategy
    def merit(r):
        return r.value[0]
    controller.run(merit=merit)

    fpath = "_".join(['RECAS', opt_prob.name, str(opt_prob.nobj), str(opt_prob.dim), str(trial_number)]) + ".txt"
    controller.strategy.save_to_file(fpath)


if __name__ == '__main__':
    pass
