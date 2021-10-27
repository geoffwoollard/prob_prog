import numpy as np

def weighted_average(samples,sigmas,reshape_probs=None,func=None,axis=None):
    """sum because prob sums to 1 and properly normalized. 
    doing sum of expectation integral
    reshape_probs : tupple to match samples shape
    """
     
    probs = np.exp(sigmas)
    probs /= probs.sum()
    if func is not None:
    	samples = func(samples)
    if reshape_probs is not None:
    	probs = probs.reshape(reshape_probs)
    return (probs*samples).sum(axis=axis), probs 