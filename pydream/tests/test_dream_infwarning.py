from collections import OrderedDict
import numpy as np
from scipy.stats import truncnorm as truncated_normal
from pydream.core import run_dream as pd_run
from pydream.parameters import SampledParam as pd_param

np.seterr(invalid='raise')

def infmodel(parvals):
    return - np.inf

def test_neginf_model():
    """
    """
    def trpd(my_mean, my_std, lb, ub):
        a, b = (lb - my_mean) / my_std, (ub - my_mean) / my_std
        return pd_param(truncated_normal, a=a, b=b, scale=my_std, loc=my_mean)

    prior_args = OrderedDict({
        "p1" : [7, 20, 0, 100],
        "p2" : [6, 10, 0, 30],
        "p3" :  [4, 5, 0, 30]
    })
    priors = OrderedDict()
    for p in prior_args.keys():
        priors[p] = trpd(*prior_args[p])

    sampled_params, log_ps = pd_run(list(priors.values()), infmodel, nchains=3, niterations=10, restart=False, verbose=False, model_name="infmodel")