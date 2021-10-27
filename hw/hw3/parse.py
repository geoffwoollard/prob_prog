import numpy as np
from evaluation_based_sampling import evaluate_program


def take_samples(num_samples,ast):  
    samples = []
    sigmas = np.zeros(num_samples)
    for idx in range(num_samples):
        sample, sigmas[idx] = evaluate_program(ast)
        samples.append(sample)
    return samples, sigmas

