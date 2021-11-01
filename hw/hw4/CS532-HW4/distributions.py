import torch
import torch.distributions as dist

class Normal(dist.Normal):
    
    def __init__(self, loc, scale):
        
        if scale > 20.:
            self.optim_scale = scale.clone().detach().requires_grad_()
        else:
            self.optim_scale = torch.log(torch.exp(scale) - 1).clone().detach().requires_grad_()
        
        
        super().__init__(loc, torch.nn.functional.softplus(self.optim_scale))
    
    def Parameters(self):
        """Return a list of parameters for the distribution"""
        return [self.loc, self.optim_scale]
        
    def make_copy_with_grads(self):
        """
        Return a copy  of the distribution, with parameters that require_grad
        """
        
        ps = [p.clone().detach().requires_grad_() for p in self.Parameters()]
         
        return Normal(*ps)
    
    def log_prob(self, x):
        
        self.scale = torch.nn.functional.softplus(self.optim_scale)
        
        return super().log_prob(x)
        
class Bernoulli(dist.Bernoulli):
    
    def __init__(self, probs=None, logits=None):
        if logits is None and probs is None:
            raise ValueError('set probs or logits')
        elif logits is None:
            if type(probs) is float:
                probs = torch.tensor(probs)
            logits = torch.log(probs/(1-probs)) ##will fail if probs = 0
        #
        super().__init__(logits = logits)
    
    def Parameters(self):
        """Return a list of parameters for the distribution"""
        return [self.logits]
        
    def make_copy_with_grads(self):
        """
        Return a copy  of the distribution, with parameters that require_grad
        """
        
        logits = [p.clone().detach().requires_grad_() for p in self.Parameters()][0]
        
        return Bernoulli(logits = logits)
    
class Categorical(dist.Categorical):
    
    def __init__(self, probs=None, logits=None, validate_args=None):
        
        if (probs is None) == (logits is None):
            raise ValueError("Either `probs` or `logits` must be specified, but not both.")
        if probs is not None:
            if probs.dim() < 1:
                raise ValueError("`probs` parameter must be at least one-dimensional.")
            probs = probs / probs.sum(-1, keepdim=True)
            logits = dist.utils.probs_to_logits(probs)

        else:
            if logits.dim() < 1:
                raise ValueError("`logits` parameter must be at least one-dimensional.")
            # Normalize
            logits = logits - logits.logsumexp(dim=-1, keepdim=True)
        super().__init__(logits = logits)

        self.logits = logits.clone().detach().requires_grad_()
        self._param = self.logits
    
    def Parameters(self):
        """Return a list of parameters for the distribution"""
        return [self.logits]
        
    def make_copy_with_grads(self):
        """
        Return a copy  of the distribution, with parameters that require_grad
        """
        
        logits = [p.clone().detach().requires_grad_() for p in self.Parameters()][0]
        
        return Categorical(logits = logits)    

class Dirichlet(dist.Dirichlet):
    
    def __init__(self, concentration):
        #NOTE: logits automatically get added
        super().__init__(concentration)
    
    def Parameters(self):
        """Return a list of parameters for the distribution"""
        return [self.concentration]
        
    def make_copy_with_grads(self):
        """
        Return a copy  of the distribution, with parameters that require_grad
        """
        
        concentration = [p.clone().detach().requires_grad_() for p in self.Parameters()][0]
        
        return Dirichlet(concentration)

class Gamma(dist.Gamma):
    
    def __init__(self, concentration, rate):
        if rate > 20.:
            self.optim_rate = rate.clone().detach().requires_grad_()
        else:
            self.optim_rate = torch.log(torch.exp(rate) - 1).clone().detach().requires_grad_()
        
        
        super().__init__(concentration, torch.nn.functional.softplus(self.optim_rate))
    
    def Parameters(self):
        """Return a list of parameters for the distribution"""
        return [self.concentration, self.optim_rate]
        
    def make_copy_with_grads(self):
        """
        Return a copy  of the distribution, with parameters that require_grad
        """
        
        concentration,rate = [p.clone().detach().requires_grad_() for p in self.Parameters()]
        
        return Gamma(concentration, rate)

    def log_prob(self, x):
        
        self.rate = torch.nn.functional.softplus(self.optim_rate)
        
        return super().log_prob(x)




if __name__ == '__main__':
    
    ##how to use this, 
    #given some input tensors that don't necessarily have gradients enables
    scale = torch.tensor(1.)
    loc = torch.tensor(0.)
    
    #and some data
    data = torch.tensor(2.)
    
    #construct a distribution
    d = Normal(loc, scale)
    
    #now you can make a copy, that has gradients enabled
    dg = d.make_copy_with_grads()
    
    #the function .Parameters() returns a list of parameters that you can pass to an optimizer
    optimizer = torch.optim.Adam(dg.Parameters(), lr=1e-2)
    
    #do the optimization. Here we're maximizing the log_prob of some data at 2.0
    #the scale should move to 2.0 as well,
    #furthermore, the scale should be constrained to the positive reals,
    #this last thing is taken care of by the new distributions defined above
    for i in range(1000):
        nlp = -dg.log_prob(data)
        nlp.backward()
        optimizer.step()
        optimizer.zero_grad()
    
    #check the result is correct:
    print(dg.Parameters())
    
    
    #note: Parameters() returns a list of tensors that parametrize the distributions
    # gradients can be taken with respect to these parameters, and you can assume gradient updates are "safe"
    # i.e., under the hood, parameters constrained to the positive reals are transformed so that they can be optimized
    # over without worrying about the constraints
    