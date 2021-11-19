"""
inspired by https://github.com/MasoudMo/cpsc532w_hw/blob/master/HW6/smc.py

"""

from evaluator import evaluate
import torch
from torch import tensor
import numpy as np
import json


def run_until_observe_or_end(res):
    cont, args, sigma = res
    res = cont(*args)
    while type(res) is tuple:
        if res[2]['type'] == 'observe':
            return res
        cont, args, sigma = res
        res = cont(*args)

    res = (res, None, {'done' : True}) #wrap it back up in a tuple, that has "done" in the sigma map
    return res


def resample_particles(particles, log_weights):
    """
    Eq. 4.24 in course textbook (https://arxiv.org/abs/1809.10756v2, pp. 122)
    See Algorithm 15 in the course textbook, section 6.7 Sequantial Monte Carlo, p. 176
    """
    log_weights = tensor(log_weights)
    n_particles = log_weights.size().numel()
    
    unnormalized_particle_weights = torch.exp(log_weights).detach().numpy()

    particle_idxs = np.random.choice(
        a=range(n_particles),
        size=n_particles,
        p=unnormalized_particle_weights/unnormalized_particle_weights.sum(),
        replace=True,
        )
    #print('particle_idxs',particle_idxs)

    new_particles = []
    for idx in range(n_particles):
        new_particles.append(particles[particle_idxs[idx]]) # TODO: copy?

    log_Z = np.log(np.sum(unnormalized_particle_weights)/n_particles)
    return log_Z, new_particles


def SMC(n_particles, exp,do_log=False):

    particles = []
    weights = []
    logZs = []
    output = lambda x: x

    for i in range(n_particles):

        res = evaluate(exp, env=None)('addr_start', output)
        logW = 0.


        particles.append(res)
        weights.append(logW)

    #can't be done after the first step, under the address transform, so this should be fine:
    done = False
    smc_cnter = 0
    while not done:
        if do_log: print('In SMC step {}, Zs: '.format(smc_cnter), logZs)
        for i in range(n_particles): #Even though this can be parallelized, we run it serially
            res = run_until_observe_or_end(particles[i]) # particle i at next breakbpoint
            if 'done' in res[2]: #this checks if the calculation is done
                particles[i] = res[0]
                if i == 0:
                    done = True  #and enforces everything to be the same as the first particle
                    address = ''
                else:
                    if not done: # triggered when i=0 was not done and i>0 was done
                        raise RuntimeError('Failed SMC, finished one calculation before the other')
            else:
                #TODO: check particle addresses, and get weights and continuations
                particles[i] = res
                cont, args, sigma = res
                assert 'observe' == sigma['type']
                weights[i] = sigma['distribution'].log_prob(sigma['observed_constant'])

                 # check particle addresses
                if i == 0:
                    break_point_address = sigma['address']
                else:
                    if sigma['address'] != break_point_address:
                        assert False, 'particles at different break points'



        if not done:
            #resample and keep track of logZs
            logZn, particles = resample_particles(particles, weights)
            logZs.append(logZn)
        smc_cnter += 1
    logZ = sum(logZs)
    return logZ, particles

