from evaluator import evaluate, sample_from_prior
import torch
import numpy as np
import json
import sys



def get_IS_sample(exp):
    #init calc:
    output = lambda x: x
    res =  evaluate(exp, env=None)('addr_start', output)
    #TODO : hint, "get_sample_from_prior" as a basis for your solution

    logW = 0.0

    while type(res) is tuple:
        cont, args, sigma = res
        if sigma['type'] == 'observe':
            distribution = sigma['distribution']
            observed_constant = sigma['observed_constant']
            logW += distribution.log_prob(sigma['observed_constant']).item()
        res = cont(*args)
        
    return logW, res

if __name__ == '__main__':

    for i in range(1,5):
        with open('programs/{}.json'.format(i),'r') as f:
            exp = json.load(f)
        print('\n\n\nSample of prior of program {}: {}'.format(i,sample_from_prior(exp)))
        log_weights = []
        values = []
        for i in range(10000):
            logW, sample = get_IS_sample(exp)
            log_weights.append(logW)
            values.append(sample)

        log_weights = torch.tensor(log_weights)

        values = torch.stack(values)
        values = values.reshape((values.shape[0],values.size().numel()//values.shape[0]))

        # the product sum of the probs in log space. ie joint prob of iid pieces
        # logsumexp converts log_probs to probs, then sums, then logs https://pytorch.org/docs/stable/generated/torch.logsumexp.html
        log_Z = torch.logsumexp(log_weights,0) - torch.log(torch.tensor(log_weights.shape[0],dtype=float)) # second piece normalizes to num_samples, instead of one (1)?

        log_norm_weights = log_weights - log_Z
        weights = torch.exp(log_norm_weights).detach().numpy()
        weighted_samples = (torch.exp(log_norm_weights).reshape((-1,1))*values.float()).detach().numpy()
    
        print('covariance: ', np.cov(values.float().detach().numpy(),rowvar=False, aweights=weights))    
        print('posterior mean:', weighted_samples.mean(axis=0))

