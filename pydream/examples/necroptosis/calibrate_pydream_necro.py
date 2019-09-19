'''
Generated by pydream_it
PyDREAM run script for necro.py 
'''
from pydream.core import run_dream
from pysb.simulator import ScipyOdeSimulator
import numpy as np
from pydream.parameters import SampledParam
from pydream.convergence import Gelman_Rubin
from scipy.stats import norm,uniform
from necro_uncal_new import model
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
import random 
random.seed(0)

# DREAM Settings
# Number of chains - should be at least 3.
nchains = 5
# Number of iterations
niterations = 50000

#Initialize PySB solver object for running simulations.  Simulation timespan should match experimental data.
# tspan = np.linspace(0,1440, num=100)
# solver = ScipyOdeSimulator(model, tspan=tspan)
# parameters_idxs = [14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50]
# rates_mask = [False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True]
# param_values = np.array([p.value for p in model.parameters])
#
# # USER must add commands to import/load any experimental data for use in the likelihood function!
# experiments_avg = np.load()
# experiments_sd = np.load()
# like_data = norm(loc=experiments_avg, scale=experiments_sd)
# # USER must define a likelihood function!
# def likelihood(position):
#     Y=np.copy(position)
#     param_values[rates_mask] = 10 ** Y
#     sim = solver.run(param_values=param_values).all
#     logp_data = np.sum(like_data.logpdf(sim['observable']))
#     if np.isnan(logp_data):
#         logp_data = -np.inf
#     return logp_data

obs_names = ['MLKLa_obs']

# Defining a few helper functions to use
def normalize(trajectories):
    """even though this is not really needed, if the data is already between 1 and 0!"""
    """Rescale a matrix of model trajectories to 0-1"""
    ymin = trajectories.min(0)
    ymax = trajectories.max(0)
    return (trajectories - ymin) / (ymax - ymin)

t = np.array([0., 30,  60,   120,  180, 270,  480,  960, 1440])
# newt = np.array([0., 60, 240,  480, 600, 720, 840, 960, 1080, 1200])
data100 = np.array([0., 0., 0., 0., 0.01, 0.05, 0.5, 0.99, 1.])
# # stdev100 =
# data10 = np.array([0.0096, 0.048, 0.178, 0.287, 0.497, 0.547, 0.770, 0.808, 0.953, 1.0])
# stdev10 = np.array([.05, .02, .08, .11, .11, .12, .16, .09, .06, .01])

# x100 = np.array([0., .5, 1.5, 4.5, 8, 10,  12, 16])
# y100 = np.array([0.,
# 0.0088569170874609,0.0161886154261265,
# 0.0373005242261882,
# 0.2798939020159581, 0.517425,
# 0.639729406776,
# 1])
#
# x10 = np.array([0., .5, 1.5, 4.5, 8, 12, 16])
# y10 = np.array([0., 0.0106013664572332,
# 0.00519576571714913,
# 0.02967443048221,
# 0.050022163974868,
# 0.198128107774737,
# 0.56055140114867])

x100 = np.array([30, 90, 270, 480, 600, 720, 840, 960])
y100 = np.array([
0.00885691708746097,0.0161886154261265,
0.0373005242261882,
0.2798939020159581,0.51,
0.7797294067, 0.95,
1])

# x10 = np.array([.5, 1.5, 4.5, 8, 10, 12, 14, 16])
y10 = np.array([0.0106013664572332,
0.00519576571714913,
0.02967443048221,
0.050022163974868,
0.108128107774737, 0.25,
0.56055140114867, 0.77])
solver = ScipyOdeSimulator(model, tspan=x100) #, rtol=1e-6, # rtol : float or sequence relative tolerance for solution
                            #atol=1e-6) #atol : float or sequence absolute tolerance for solution

rate_params = model.parameters_rules() # these are only the parameters involved in the rules
param_values = np.array([p.value for p in model.parameters]) # these are all the parameters
rate_mask = np.array([p in rate_params for p in model.parameters])  # this picks the element of intersection

def likelihood(position):
    # params_tmp = np.copy(position)
    # rate_params = 10 ** params_tmp #don't need to change
    # param_values[rate_mask] = 10 ** params_tmp  # don't need to change
    # #make a new parameter value set for each of the KD
    # # x1_params = np.copy(param_values)
    # # x1_params[0] = 233
    # # ko_pars = [x1_params, param_values]
    #
    # result = solver.run(param_values=param_values)
    #
    # ysim_array11 = result.observables[0]['MLKLa_obs']
    # # ysim_array22 = result.observables[1]['MLKLa_obs']
    #
    # # ysim_array = extract_records(solver.yobs, obs_names)
    # ysim_norm11 = normalize(ysim_array11)
    # # ysim_norm22 = normalize(ysim_array22)
    #
    # # mlkl_10 = np.array([0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10])
    # # mlkl_1 = np.array([0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10])
    # # mlkl_10 = np.array([0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05])
    # # mlkl_1 = np.array([0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05])
    #
    # # e1 = np.sum((y10 - ysim_norm11) ** 2 / (mlkl_10))
    # # e2 = np.sum((y1 - ysim_norm22) ** 2 / (mlkl_1))
    # #
    # e1 = np.sum((data100 - ysim_norm11) ** 2)
    # # e2 = np.sum((y10 - ysim_norm22) ** 2)
    #
    # error = e1
    # return error,


    params_tmp = np.copy(position)  # here you pass the parameter vector; the point of making a copy of it is in order not to modify it
    param_values[rate_mask] = 10 ** params_tmp  # see comment above *

    params_10 = np.copy(param_values)
    params_10[0] = 233
    pars = [param_values, params_10]
    result = solver.run(param_values=pars)

    ysim_norm100 = normalize(result.observables[0]['MLKLa_obs'])
    ysim_norm10 = normalize(result.observables[1]['MLKLa_obs'])
    # result = solver.run(param_values=param_values)
    # ysim_norm = normalize(result.observables['MLKLa_obs'])
    # error = np.sum(((y100 - ysim_norm) ** 2))
    e1 = np.sum(((y100 - ysim_norm100) ** 2))
    e2 = np.sum(((y10 - ysim_norm10) ** 2))
    error = e1 + e2

    return -error


# def likelihood(position):
#     params_tmp1 = np.copy(position[:37])  # here you pass the parameter vector; the point of making a copy of it is in order not to modify it
#     params_tmp2 = np.copy(position[37:74])  # here you pass the parameter vector; the point of making a copy of it is in order not to modify it
#     params_tmp3 = np.copy(position[74:110])  # here you pass the parameter vector; the point of making a copy of it is in order not to modify it
#
#     param_values[rate_mask] = 10 ** params_tmp1  # see comment above *
#     param_values[rate_mask] = 10 ** params_tmp2 # see comment above *
#     param_values[rate_mask] = 10 ** params_tmp3 # see comment above *
#
#     result1 = solver.run(param_values=param_values1)
#     result2 = solver.run(param_values=param_values2)
#     result3 = solver.run(param_values=param_values3)
#
#     ysim_norm1 = normalize(result1.observables['MLKLa_obs'])
#     ysim_norm2 = normalize(result2.observables['MLKLa_obs'])
#     ysim_norm3 = normalize(result3.observables['MLKLa_obs'])
#
#     error = np.sum((data1 - ysim_norm1) ** 2)/(0.02) + np.sum((data1 - ysim_norm1) ** 2)/(0.02)  # measurement error
#     error += chi_squared(variance(params_tmp1, params_tmp2, params_tmp3))  # biological variability
#     return -error

sampled_params_list = list()
sp_p1f = SampledParam(norm, loc=np.log10(3.304257e-05), scale=2.0)
sampled_params_list.append(sp_p1f)
sp_p1r = SampledParam(norm, loc=np.log10(0.009791216), scale=2.0)
sampled_params_list.append(sp_p1r)
sp_p2f = SampledParam(norm, loc=np.log10(0.006110069), scale=2.0)
sampled_params_list.append(sp_p2f)
sp_p3f = SampledParam(norm, loc=np.log10(4.319219e-05), scale=2.0)
sampled_params_list.append(sp_p3f)
sp_p3r = SampledParam(norm, loc=np.log10(0.004212645), scale=2.0)
sampled_params_list.append(sp_p3r)
sp_p4f = SampledParam(norm, loc=np.log10(1.164332e-05), scale=2.0)
sampled_params_list.append(sp_p4f)
sp_p4r = SampledParam(norm, loc=np.log10(0.02404257), scale=2.0)
sampled_params_list.append(sp_p4r)
sp_p5f = SampledParam(norm, loc=np.log10(3.311086e-05), scale=2.0)
sampled_params_list.append(sp_p5f)
sp_p5r = SampledParam(norm, loc=np.log10(0.04280399), scale=2.0)
sampled_params_list.append(sp_p5r)
sp_p6f = SampledParam(norm, loc=np.log10(2.645815e-05), scale=2.0)
sampled_params_list.append(sp_p6f)
sp_p6r = SampledParam(norm, loc=np.log10(0.01437707), scale=2.0)
sampled_params_list.append(sp_p6r)
sp_p7f = SampledParam(norm, loc=np.log10(0.2303744), scale=2.0)
sampled_params_list.append(sp_p7f)
sp_p8f = SampledParam(norm, loc=np.log10(2.980688e-05), scale=2.0)
sampled_params_list.append(sp_p8f)
sp_p8r = SampledParam(norm, loc=np.log10(0.04879773), scale=2.0)
sampled_params_list.append(sp_p8r)
sp_p9f = SampledParam(norm, loc=np.log10(1.121503e-05), scale=2.0)
sampled_params_list.append(sp_p9f)
sp_p9r = SampledParam(norm, loc=np.log10(0.001866713), scale=2.0)
sampled_params_list.append(sp_p9r)
sp_p10f = SampledParam(norm, loc=np.log10(0.7572178), scale=2.0)
sampled_params_list.append(sp_p10f)
sp_p11f = SampledParam(norm, loc=np.log10(1.591283e-05), scale=2.0)
sampled_params_list.append(sp_p11f)
sp_p11r = SampledParam(norm, loc=np.log10(0.03897146), scale=2.0)
sampled_params_list.append(sp_p11r)
sp_p12f = SampledParam(norm, loc=np.log10(3.076363), scale=2.0)
sampled_params_list.append(sp_p12f)
sp_p13f = SampledParam(norm, loc=np.log10(3.73486), scale=2.0)
sampled_params_list.append(sp_p13f)
sp_p13r = SampledParam(norm, loc=np.log10(3.2162e-06), scale=2.0)
sampled_params_list.append(sp_p13r)
sp_p14f = SampledParam(norm, loc=np.log10(8.78243e-05), scale=2.0)
sampled_params_list.append(sp_p14f)
sp_p14r = SampledParam(norm, loc=np.log10(0.02906341), scale=2.0)
sampled_params_list.append(sp_p14r)
sp_p15f = SampledParam(norm, loc=np.log10(5.663104e-05), scale=2.0)
sampled_params_list.append(sp_p15f)
sp_p15r = SampledParam(norm, loc=np.log10(0.02110469), scale=2.0)
sampled_params_list.append(sp_p15r)
sp_p16f = SampledParam(norm, loc=np.log10(0.1294086), scale=2.0)
sampled_params_list.append(sp_p16f)
sp_p16r = SampledParam(norm, loc=np.log10(0.3127598), scale=2.0)
sampled_params_list.append(sp_p16r)
sp_p17f = SampledParam(norm, loc=np.log10(0.429849), scale=2.0)
sampled_params_list.append(sp_p17f)
sp_p18f = SampledParam(norm, loc=np.log10(2.33291e-06), scale=2.0)
sampled_params_list.append(sp_p18f)
sp_p18r = SampledParam(norm, loc=np.log10(0.007077505), scale=2.0)
sampled_params_list.append(sp_p18r)
sp_p19f = SampledParam(norm, loc=np.log10(0.6294062), scale=2.0)
sampled_params_list.append(sp_p19f)
sp_p20f = SampledParam(norm, loc=np.log10(0.06419313), scale=2.0)
sampled_params_list.append(sp_p20f)
sp_p21f = SampledParam(norm, loc=np.log10(0.008584654), scale=2.0)
sampled_params_list.append(sp_p21f)
sp_p22f = SampledParam(norm, loc=np.log10(8.160445e-05), scale=2.0)
sampled_params_list.append(sp_p22f)
sp_p22r = SampledParam(norm, loc=np.log10(4.354384e-03), scale=2.0)
sampled_params_list.append(sp_p22r)
sp_p23f = SampledParam(norm, loc=np.log10(0.008584654), scale=2.0)
sampled_params_list.append(sp_p23f)
sp_p24f = SampledParam(norm, loc=np.log10(8.160445e-05), scale=2.0)
sampled_params_list.append(sp_p24f)
sp_p24r = SampledParam(norm, loc=np.log10(4.354384e-06), scale=2.0)
sampled_params_list.append(sp_p24r)
sp_p25f = SampledParam(norm, loc=np.log10(4.278903), scale=2.0)
sampled_params_list.append(sp_p25f)
# sampled_params_list = list()
# sp_p1f = SampledParam(norm, loc=np.log10(3.304257e-05), scale=3.0)
# sampled_params_list.append(sp_p1f)
# sp_p1r = SampledParam(norm, loc=np.log10(0.009791216), scale=3.0)
# sampled_params_list.append(sp_p1r)
# sp_p2f = SampledParam(norm, loc=np.log10(0.006110069), scale=3.0)
# sampled_params_list.append(sp_p2f)
# sp_p3f = SampledParam(norm, loc=np.log10(4.319219e-05), scale=3.0)
# sampled_params_list.append(sp_p3f)
# sp_p3r = SampledParam(norm, loc=np.log10(0.004212645), scale=3.0)
# sampled_params_list.append(sp_p3r)
# sp_p4f = SampledParam(norm, loc=np.log10(1.164332e-05), scale=3.0)
# sampled_params_list.append(sp_p4f)
# sp_p4r = SampledParam(norm, loc=np.log10(0.02404257), scale=3.0)
# sampled_params_list.append(sp_p4r)
# sp_p5f = SampledParam(norm, loc=np.log10(3.311086e-05), scale=3.0)
# sampled_params_list.append(sp_p5f)
# sp_p5r = SampledParam(norm, loc=np.log10(0.04280399), scale=3.0)
# sampled_params_list.append(sp_p5r)
# sp_p6f = SampledParam(norm, loc=np.log10(2.645815e-05), scale=3.0)
# sampled_params_list.append(sp_p6f)
# sp_p6r = SampledParam(norm, loc=np.log10(0.01437707), scale=3.0)
# sampled_params_list.append(sp_p6r)
# sp_p7f = SampledParam(norm, loc=np.log10(0.2303744), scale=3.0)
# sampled_params_list.append(sp_p7f)
# sp_p8f = SampledParam(norm, loc=np.log10(2.980688e-05), scale=3.0)
# sampled_params_list.append(sp_p8f)
# sp_p8r = SampledParam(norm, loc=np.log10(0.04879773), scale=3.0)
# sampled_params_list.append(sp_p8r)
# sp_p9f = SampledParam(norm, loc=np.log10(1.121503e-05), scale=3.0)
# sampled_params_list.append(sp_p9f)
# sp_p9r = SampledParam(norm, loc=np.log10(0.001866713), scale=3.0)
# sampled_params_list.append(sp_p9r)
# sp_p10f = SampledParam(norm, loc=np.log10(0.7572178), scale=3.0)
# sampled_params_list.append(sp_p10f)
# sp_p11f = SampledParam(norm, loc=np.log10(1.591283e-05), scale=3.0)
# sampled_params_list.append(sp_p11f)
# sp_p11r = SampledParam(norm, loc=np.log10(0.03897146), scale=3.0)
# sampled_params_list.append(sp_p11r)
# sp_p12f = SampledParam(norm, loc=np.log10(3.076363), scale=3.0)
# sampled_params_list.append(sp_p12f)
# sp_p13f = SampledParam(norm, loc=np.log10(3.73486), scale=3.0)
# sampled_params_list.append(sp_p13f)
# sp_p13r = SampledParam(norm, loc=np.log10(3.2162e-06), scale=3.0)
# sampled_params_list.append(sp_p13r)
# sp_p14f = SampledParam(norm, loc=np.log10(8.78243e-05), scale=3.0)
# sampled_params_list.append(sp_p14f)
# sp_p14r = SampledParam(norm, loc=np.log10(0.02906341), scale=3.0)
# sampled_params_list.append(sp_p14r)
# sp_p15f = SampledParam(norm, loc=np.log10(5.663104e-05), scale=3.0)
# sampled_params_list.append(sp_p15f)
# sp_p15r = SampledParam(norm, loc=np.log10(0.02110469), scale=3.0)
# sampled_params_list.append(sp_p15r)
# sp_p16f = SampledParam(norm, loc=np.log10(0.1294086), scale=3.0)
# sampled_params_list.append(sp_p16f)
# sp_p16r = SampledParam(norm, loc=np.log10(0.3127598), scale=3.0)
# sampled_params_list.append(sp_p16r)
# sp_p17f = SampledParam(norm, loc=np.log10(0.429849), scale=3.0)
# sampled_params_list.append(sp_p17f)
# sp_p18f = SampledParam(norm, loc=np.log10(2.33291e-06), scale=3.0)
# sampled_params_list.append(sp_p18f)
# sp_p18r = SampledParam(norm, loc=np.log10(0.007077505), scale=3.0)
# sampled_params_list.append(sp_p18r)
# sp_p19f = SampledParam(norm, loc=np.log10(0.6294062), scale=3.0)
# sampled_params_list.append(sp_p19f)
# sp_p20f = SampledParam(norm, loc=np.log10(0.06419313), scale=3.0)
# sampled_params_list.append(sp_p20f)
# sp_p21f = SampledParam(norm, loc=np.log10(0.0008584654), scale=3.0)
# sampled_params_list.append(sp_p21f)
# sp_p22f = SampledParam(norm, loc=np.log10(8.160445e-05), scale=3.0)
# sampled_params_list.append(sp_p22f)
# sp_p22r = SampledParam(norm, loc=np.log10(4.354384e-06), scale=3.0)
# sampled_params_list.append(sp_p22r)
# sp_p23f = SampledParam(norm, loc=np.log10(4.278903), scale=3.0)
# sampled_params_list.append(sp_p23f)

# plt.figure()
# sns.distplot(sp_p1f, fit=norm, kde=False)
# plt.show()
# quit()

# sampled_params_list = sampled_params_list

converged = False
sampled_params, log_ps = run_dream(parameters=sampled_params_list,
                                   likelihood=likelihood,
                                   niterations=niterations,
                                   nchains=nchains,
                                   multitry=False,
                                   gamma_levels=4,
                                   adapt_gamma=True,
                                   history_thin=1,
                                   model_name='dreamzs_5chain',
                                   verbose=True)

total_iterations = niterations
# Save sampling output (sampled parameter values and their corresponding logps).
for chain in range(len(sampled_params)):
    np.save('dreamzs_5chain_sampled_params_chain_919_' + str(chain)+'_'+str(total_iterations), sampled_params[chain])
    np.save('dreamzs_5chain_logps_chain_919_' + str(chain)+'_'+str(total_iterations), log_ps[chain])
GR = Gelman_Rubin(sampled_params)
print('At iteration: ',total_iterations,' GR = ',GR)
np.savetxt('dreamzs_5chain_GelmanRubin_iteration_919_'+str(total_iterations)+'.txt', GR)
old_samples = sampled_params
if np.any(GR>1.2):
    starts = [sampled_params[chain][-1, :] for chain in range(nchains)]
    while not converged:
        total_iterations += niterations
        sampled_params, log_ps = run_dream(parameters=sampled_params_list,
                                           likelihood=likelihood,
                                           niterations=niterations,
                                           nchains=nchains,
                                           start=starts,
                                           multitry=False,
                                           gamma_levels=4,
                                           adapt_gamma=True,
                                           history_thin=1,
                                           model_name='dreamzs_5chain',
                                           verbose=False,
                                           restart=True)
        for chain in range(len(sampled_params)):
            np.save('dreamzs_5chain_sampled_params_chain_919_' + str(chain)+'_'+str(total_iterations), sampled_params[chain])
            np.save('dreamzs_5chain_logps_chain_919_' + str(chain)+'_'+str(total_iterations), log_ps[chain])
        old_samples = [np.concatenate((old_samples[chain], sampled_params[chain])) for chain in range(nchains)]
        GR = Gelman_Rubin(old_samples)
        print('At iteration: ',total_iterations,' GR = ',GR)
        np.savetxt('dreamzs_5chain_GelmanRubin_iteration_919_' + str(total_iterations)+'.txt', GR)
        if np.all(GR<1.2):
            converged = True
try:
    #Plot output
    import seaborn as sns
    from matplotlib import pyplot as plt
    total_iterations = len(old_samples[0])
    burnin = total_iterations/2
    #samples = np.concatenate((old_samples[0][burnin:, :], old_samples[1][burnin:, :], old_samples[2][burnin:, :],old_samples[3][burnin:, :], old_samples[4][burnin:, :]))
    samples = np.concatenate(tuple([old_samples[i][int(burnin):, :] for i in range(nchains)]))
    ndims = len(sampled_params_list)
    colors = sns.color_palette(n_colors=ndims)
    for dim in range(ndims):
        fig = plt.figure()
        sns.distplot(samples[:, dim], color=colors[dim], norm_hist=True)
    fig.savefig('fig_PyDREAM_dimension_9_19_'+str(dim))
except ImportError:
    pass
