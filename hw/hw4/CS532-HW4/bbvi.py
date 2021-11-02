import logging

import numpy as np
import torch
from torch import tensor

from primitives import primitives_d, distributions_d, number, distribution_types
import distributions # for unconstrained optimization


number = (int,float)

logging.basicConfig(format='%(levelname)s:%(message)s')
logger = logging.getLogger('simple_example')
logger.setLevel(logging.DEBUG)


def eval_algo11(e,sigma,local_env={},defn_d={},do_log=False,logger_string='',vertex=None):
    # remember to return evaluate (recursive)
        # everytime we call evaluate, we have to use local_env, otherwise it gets overwritten with the default {}
    if do_log: logger.info('ls {}'.format(logger_string))
    if do_log: logger.info('e {}, local_env {}, sigma {}'.format(e, local_env, sigma))

    # get first expression out of list or list of one
    if not isinstance(e,list) or len(e) == 1:
        if isinstance(e,list):
            e = e[0]
        if isinstance(e,bool):
            if do_log: logger.info('match case number: e {}, sigma {}'.format(e, sigma))
            return torch.tensor(e), sigma
        if isinstance(e, number):
            if do_log: logger.info('match case number: e {}, sigma {}'.format(e, sigma))
            return torch.tensor(float(e)), sigma
        elif isinstance(e,list):
            if do_log: logger.info('match case list: e {}, sigma {}'.format(e, sigma))
            return e, sigma
        elif e in list(primitives_d.keys()):
            if do_log: logger.info('match case primitives_d: e {}, sigma {}'.format(e, sigma))
            return e, sigma
        elif e in list(distributions_d.keys()):
            if do_log: logger.info('match case distributions_d: e {}, sigma {}'.format(e, sigma))
            return e, sigma
        elif torch.is_tensor(e):
            if do_log: logger.info('match case is_tensor: e {}, sigma {}'.format(e, sigma))
            return e, sigma
        elif e in local_env.keys():
            if do_log: logger.info('match case local_env: e {}, sigma {}'.format(e, sigma))
            if do_log: logger.info('match case local_env: local_env[e] {}'.format(local_env[e]))
            return local_env[e], sigma 
        elif e in list(defn_d.keys()):
            if do_log: logger.info('match case defn_d: e {}, sigma {}'.format(e, sigma))
            return e, sigma
        elif isinstance(e,distribution_types):
            if do_log: logger.info('match case distribution: e {}, sigma {}'.format(e,sigma))
            return e, sigma
        else:
            assert False, 'case not matched'
    elif e[0] == 'sample':
        if do_log: logger.info('match case sample: e {}, sigma {}'.format(e,sigma))
        distribution, sigma = eval_algo11(e[1],sigma,local_env,defn_d,do_log=do_log)
        # TODO: initialize proposal using prior
        if vertex not in sigma['Q'].keys():
            if do_log: logger.info('match case sample: using prior for vertex {}'.format(vertex))
            if isinstance(distribution,torch.distributions.normal.Normal):
                p = local_env['prior_dist'][vertex]
                loc, scale = p.loc, p.scale
                q = distributions.Normal(loc,scale)
                q = q.make_copy_with_grads()
            elif isinstance(distribution,torch.distributions.bernoulli.Bernoulli):
                q = distributions.Bernoulli
            elif isinstance(distribution,torch.distributions.Gamma): 
                q = distributions.Gamma
            elif isinstance(distribution,torch.distributions.Categorical): 
                q = distributions.Categorical
            elif isinstance(distribution,torch.distributions.Dirichlet): 
                q = distributions.Dirichlet
            else:
                assert False, 'no suitable proposal distribution'
            sigma['Q'][vertex] = q

        q = sigma['Q'][vertex]
        constant = q.sample()
        G_v = grad_log_prob(q,constant)
        sigma['grad'][vertex] = G_v
        log_wv = score(distribution,constant) - score(q,constant)
        sigma['logW'] += log_wv
        if do_log: logger.info('match case sample: q {}, constant {}, G_v {}, log_wv {}, logW {}'.format(q, constant, G_v,log_wv, sigma['logW']))
        return constant, sigma # match shape in number base case
    elif e[0] == 'observe':
        if do_log: logger.info('match case observe: e {}, sigma {}'.format(e,sigma))
        e1, e2 = e[1:]
        d1, sigma = eval_algo11(e1,sigma,local_env,defn_d,do_log=do_log)
        c2, sigma = eval_algo11(e2,sigma,local_env,defn_d,do_log=do_log)
        log_w =score(d1,c2)
        if do_log: logger.info('match case observe: d1 {}, c2 {}, log_w {}, sigma {}'.format(e,d1, c2, log_w, sigma))
        sigma['logW'] += log_w
        return c2, sigma
    elif e[0] == 'let': 
        if do_log: logger.info('match case let: e {}, sigma {}'.format(e, sigma))
        # let [v1 e1] e0
        # here 
            # e[0] : "let"
            # e[1] : [v1, e1]
            # e[2] : e0
        # evaluates e1 to c1 and binds this value to e0
        # this means we update the context with old context plus {v1:c1}
        c1, sigma = eval_algo11(e[1][1],sigma,local_env,defn_d,do_log=do_log) # evaluates e1 to c1
        v1 = e[1][0]
        return eval_algo11(e[2], sigma, local_env = {**local_env,v1:c1},defn_d=defn_d,do_log=do_log)
    elif e[0] == 'if': # if e0 e1 e2
        if do_log: logger.info('match case if: e {}, sigma {}'.format(e, sigma))
        e1 = e[1]
        e2 = e[2]
        e3 = e[3]
        e1_prime, sigma = eval_algo11(e1,sigma,local_env,defn_d,do_log=do_log)
        if e1_prime:
            return eval_algo11(e2,sigma,local_env,defn_d,do_log=do_log)
        else:
            return eval_algo11(e3,sigma,local_env,defn_d,do_log=do_log) 

    else:
        cs = []
        for ei in e:
            if do_log: logger.info('cycling through expressions: ei {}, sigma {}'.format(ei,sigma))
            c, sigma = eval_algo11(ei,sigma,local_env,defn_d,do_log=do_log)
            cs.append(c)
        if cs[0] in primitives_d:
            if do_log: logger.info('do case primitives_d: cs0 {}'.format(cs[0]))
            if do_log: logger.info('do case primitives_d: cs1 {}'.format(cs[1:]))
            if do_log: logger.info('do case primitives_d: primitives_d[cs[0]] {}'.format(primitives_d[cs[0]]))
            return primitives_d[cs[0]](cs[1:]), sigma
        elif cs[0] in distributions_d:
            if do_log: logger.info('do case distributions_d: cs0 {}'.format(cs[0]))
            return distributions_d[cs[0]](cs[1:]), sigma
        elif cs[0] in defn_d:
            if do_log: logger.info('do case defn: cs0  {}'.format(cs[0]))
            defn_function_li = defn_d[cs[0]]
            defn_function_args, defn_function_body = defn_function_li
            local_env_update = {key:value for key,value in zip(defn_function_args, cs[1:])}
            if do_log: logger.info('do case defn: update to local_env from defn_d {}'.format(local_env_update))
            return eval_algo11(defn_function_body,sigma,local_env = {**local_env, **local_env_update},defn_d=defn_d,do_log=do_log)
        else:
            assert False, 'not implemented'

    

def grad_log_prob(distribution_unconst_optim,c):
    """TODO: derive these analytically for normal and verify same results
    """
    neg_log_prob = -distribution_unconst_optim.log_prob(c)
    neg_log_prob.backward()
    lambda_v = distribution_unconst_optim.Parameters()
    D_v = len(lambda_v)
    G_v = torch.zeros(D_v)
    for d in range(D_v):
        lambda_v_d = lambda_v[d]
        G_v[d] = lambda_v_d.grad
    return G_v


def elbo_gradients(G,logW,union_G_keys):
    for v in union_G_keys:
        #F_v = []
        G_v = []
        g_hat = {}
        for l in range(L):
            G_l = G[l]
            if v in G_l.keys():
                G_l_v = G_l[v].tolist()
                G_v.append(G_l_v)
                D_v = len(G_l_v)
                #F_v_l = G_l_v*logW[t][l]
                #F_v.append(F_v_l) # data specific tolist might not always work
            else:
                assert False, 'zero not implemented'
        G_v = np.array(G_v)
        F_v = G_v*logW.reshape(-1,1)

        # cov and var to compute b_v
        assert G_v.ndim == 2
        D_v = G_v.shape[1]
        b_v = np.zeros(D_v)
        for d in range(D_v):
            F_v_d = F_v[:,d]
            G_v_d = G_v[:,d]
            cov_F_G = np.cov(F_v_d,G_v_d)
            b_v[d] = cov_F_G[0,1]/cov_F_G[1,1]
        g_hat_v = (F_v - G_v*b_v).sum(0)  # sum over samples
        g_hat[v] = g_hat_v
    return g_hat


def optimizer_step(Q,g_hat):
    """
    no return of Q since modifies in place, and can't deep copy Q, and copy Q still accumulates
    """
    for v in g_hat.keys():
        lambda_v = Q[v].Parameters()
        optimizer = torch.optim.Adam(lambda_v, lr=1e-2)
        D_v = len(lambda_v)

        for idx in range(D_v):
            param = lambda_v[idx]
        #     param.requires_grad = True # TODO: include???

            param.grad = tensor(g_hat['sample2'][idx],dtype=torch.float32)
        optimizer.step() # moves lambda_v
        optimizer.zero_grad() # TODO: need this? 


def bbvi_algo12(T,L):
    r, G = [], []
    logW = np.zeros((T,L))

    E, sampled_graph = sample_from_joint(graph,do_log=False)

    sigma={'logW':tensor(0.),'Q':{},'grad':{}}
    e = ['sample',['normal',1,1.1]]

    for t in range(T):
        G = []
        r_t=[]
        union_G_t_keys = set()
        for l in range(L):
            # loop through vertex and evaluate linker functions as e
            r_t_l, sigma = eval_algo11(e,sigma=sigma,local_env = sampled_graph, vertex='sample2',do_log=False)
            logW[t,l] = sigma['logW'].item()
            G_l = (sigma['grad']).copy()
            union_G_keys.update(set(G_l.keys()))
            G.append(G_l)
            r_t.append(r_t_l)
        g_hat = elbo_gradients(G,logW[t],union_G_keys) 
        Q = sigma['Q']
        optimizer_step(Q,g_hat) # in place modification of Q

        r.append(r_t)
    return r, logW