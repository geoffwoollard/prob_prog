import logging

import numpy as np
import torch
from torch import tensor

from primitives import primitives_d, distributions_d, number, distribution_types
import distributions # for unconstrained optimization
from graph_based_sampling import sample_from_joint, score
from distributions import Normal

number = (int,float)

logging.basicConfig(format='%(levelname)s:%(message)s')
logger = logging.getLogger('simple_example')
logger.setLevel(logging.DEBUG)

logging.basicConfig(format='%(levelname)s:%(message)s')
logger_graph = logging.getLogger('simple_example')
logger_graph.setLevel(logging.DEBUG)

def eval_algo11_deterministic(e,sigma,local_env={},defn_d={},do_log=False,logger_string='',vertex=None):
    """
    do not handle sample or observe.
    done in higher level parser of linker function.
    just eval distribution object that gets sampled or observed
    """
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
        assert False
        if do_log: logger.info('match case sample: e {}, sigma {}'.format(e,sigma))
        distribution, sigma = eval_algo11_deterministic(e[1],sigma,local_env,defn_d,do_log=do_log)
        # TODO: initialize proposal using prior
        if vertex not in sigma['Q'].keys():
            if do_log: logger.info('match case sample: using prior for vertex {}'.format(vertex))
            # TODO: change primitives to distributions, so don't need to do this
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
        # TODO: double check do not have to do this again
        #q = q.make_copy_with_grads()
        constant = q.sample()
        #print('constant sampled in eval',constant)
        G_v = grad_log_prob(q,constant)
        sigma['G'][vertex] = G_v
        log_wv = score(distribution,constant) - score(q,constant)
        sigma['logW'] += log_wv
        if do_log: logger.info('match case sample: q {}, constant {}, G_v {}, log_wv {}, logW {}'.format(q, constant, G_v,log_wv, sigma['logW']))
        return constant, sigma # match shape in number base case
    elif e[0] == 'observe':
        assert False
        if do_log: logger.info('match case observe: e {}, sigma {}'.format(e,sigma))
        e1, e2 = e[1:]
        d1, sigma = eval_algo11_deterministic(e1,sigma,local_env,defn_d,do_log=do_log)
        c2, sigma = eval_algo11_deterministic(e2,sigma,local_env,defn_d,do_log=do_log)
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
        c1, sigma = eval_algo11_deterministic(e[1][1],sigma,local_env,defn_d,do_log=do_log) # evaluates e1 to c1
        v1 = e[1][0]
        return eval_algo11_deterministic(e[2], sigma, local_env = {**local_env,v1:c1},defn_d=defn_d,do_log=do_log)
    elif e[0] == 'if': # if e0 e1 e2
        if do_log: logger.info('match case if: e {}, sigma {}'.format(e, sigma))
        e1 = e[1]
        e2 = e[2]
        e3 = e[3]
        e1_prime, sigma = eval_algo11_deterministic(e1,sigma,local_env,defn_d,do_log=do_log)
        if e1_prime:
            return eval_algo11_deterministic(e2,sigma,local_env,defn_d,do_log=do_log)
        else:
            return eval_algo11_deterministic(e3,sigma,local_env,defn_d,do_log=do_log) 

    else:
        cs = []
        for ei in e:
            if do_log: logger.info('cycling through expressions: ei {}, sigma {}'.format(ei,sigma))
            c, sigma = eval_algo11_deterministic(ei,sigma,local_env,defn_d,do_log=do_log)
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
            return eval_algo11_deterministic(defn_function_body,sigma,local_env = {**local_env, **local_env_update},defn_d=defn_d,do_log=do_log)
        else:
            assert False, 'not implemented'


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
            # TODO: change primitives to distributions, so don't need to do this
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
        # TODO: double check do not have to do this again
        #q = q.make_copy_with_grads()
        constant = q.sample()
        #print('constant sampled in eval',constant)
        G_v = grad_log_prob(q,constant)
        sigma['G'][vertex] = G_v
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



def evaluate_link_function_algo11(P,verteces_topsorted,sigma,local_env,do_log):
    """
    evaluates all linking functions in P, using ancestral sampling
    """
    for vertex in verteces_topsorted:
        link_function = P[vertex]
        if link_function[0] == 'sample*':
            if do_log: logger_graph.info('match case sample*: link_function {}'.format(link_function))
            assert len(link_function) == 2
            e = link_function[1]
            # because e evaluates to distribution in linking function
            # no sample or observe in eval_algo11
            distribution, sigma = eval_algo11_deterministic(e,sigma,local_env = local_env, do_log=do_log) 
            
            # bbvi algo 11
            q = sigma['Q'][vertex]
            constant = q.sample()
            G_v = grad_log_prob(q,constant)
            sigma['G'][vertex] = G_v
            log_wv = score(distribution,constant) - score(q,constant)
            sigma['logW'] += log_wv
            if do_log: logger.info('match case sample: q {}, constant {}, G_v {}, log_wv {}, logW {}'.format(q, constant, G_v,log_wv, sigma['logW']))
            update_local_env = {vertex:constant}
            local_env.update(update_local_env)

        elif link_function[0] == 'observe*':
            if do_log: logger_graph.info('match case observe*: link_function {} sigma {}'.format(link_function, sigma))
            e1, e2 = link_function[1:]
            d1, sigma = eval_algo11_deterministic(e1,sigma,local_env,do_log=do_log)
            c2, sigma = eval_algo11_deterministic(e2,sigma,local_env,do_log=do_log)
            log_w = score(d1,c2)
            sigma['logW'] += log_w
            if do_log: logger_graph.info('match case observe*: d1 {}, c2 {}, log_w {}, sigma {}'.format(d1, c2, log_w, sigma))
    
        else:
            assert False

    return local_env, sigma


def grad_log_prob(distribution_unconst_optim,c):
    """TODO: derive these analytically for normal and verify same results
    """
    log_prob = distribution_unconst_optim.log_prob(c)
    log_prob.backward()
    lambda_v = distribution_unconst_optim.Parameters()
    D_v = len(lambda_v)
    G_v = torch.zeros(D_v)
    for d in range(D_v):
        lambda_v_d = lambda_v[d]
        G_v[d] = lambda_v_d.grad.clone().detach() # seem not to need to do clone().detach(), but keep just in case
        lambda_v_d.grad = None
        # these grads need to be added to lambda_v to make log_prob maximal 
        # (backwards because log_prob.backward() assumes log_prob is a loss to be mimimized)
        # we will flip later when we convert these raw per sample gradients to g_hat (over L mini match with b chosen to lower variance)
    return G_v


def elbo_gradients(G,logW,union_G_keys):
    """
    conversion of per sample gradients in mini match of size L to average gradient g_hat
    b chosen to minimize variance of g_hat
    """
    for v in union_G_keys:
        #F_v = []
        G_v = []
        g_hat = {}
        L = len(logW)
        for l in range(L):
            G_l = G[l]
            if v in G_l.keys():
                G_l_v = G_l[v].tolist()
                G_v.append(G_l_v)
                D_v = len(G_l_v)
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
        g_hat_v = (F_v - G_v*b_v).mean(0)  # sum over samples divided by L
        g_hat[v] = g_hat_v
    return g_hat


def optimizer_step(Q,g_hat,**kwargs):
    """
    no return of Q since modifies in place, and can't deep copy Q, and copy Q still accumulates
    """
    for v in g_hat.keys():
        lambda_v = Q[v].Parameters()
        optimizer = torch.optim.Adam(lambda_v, **kwargs)
        D_v = len(lambda_v)

        # TODO: although params already has grad from grad_log_prob, this is not the b adjusted g_hat
        for idx in range(D_v):
            param = lambda_v[idx]
            # param.requires_grad = True # TODO: include???
            param.grad = tensor(-g_hat[v][idx],dtype=torch.float32) # TODO: check sign. maximizing
            # Optimizers subtract the gradient of all passed parameters using their .grad attribute as seen here 182. 
            # Thus you would minimize the loss using gradient descent.
            # https://discuss.pytorch.org/t/do-optimizers-minimize-or-maximize/69062

        optimizer.step() # moves lambda_v
        optimizer.zero_grad() # TODO: need this? 
    return Q


def bbvi_algo12(graph,T,L,do_log=False,**kwargs):
    r, G = [], []
    logW = np.zeros((T,L))

    E, sampled_graph = sample_from_joint(graph,do_log=do_log)
    print('sampled_graph',sampled_graph)

    sigma={'logW':tensor(0.),'Q':{},'G':{}}
    e = ['sample',torch.distributions.normal.Normal(tensor(-2.),tensor(10.))]

    for t in range(T):
        G = []
        r_t=[]
        union_G_keys = set()
        # sigma['logW']:tensor(0.) # TODO: re-zero? does it make a difference to grads?
        # sigma['G'] = {}
        
        for l in range(L):
            sigma={'logW':tensor(0.),'Q':sigma['Q'],'G':{}}
            # loop through vertex and evaluate linker functions as e
            r_t_l, sigma = graph_eval_algo11(e,sigma=sigma,local_env = sampled_graph, vertex='sample2',do_log=do_log)
            logW[t,l] = sigma['logW'].item()
            G_l = (sigma['G']).copy()
            union_G_keys.update(set(G_l.keys()))
            G.append(G_l)
            r_t.append(r_t_l)
            #print('l {}, sigma {}'.format(l,sigma))
        #print('sigma',sigma)
        g_hat = elbo_gradients(G,logW[t],union_G_keys) 
        #print('g_hat',g_hat)
        Q = sigma['Q']
        #print('Q before step',Q)
        Q = optimizer_step(Q,g_hat,**kwargs) # in place modification of Q
        print('Q after step',Q)

        r.append(r_t)
    return r, logW

def graph_eval_algo11(graph,sigma={},do_log=False,verteces_topsorted=None):
    """This function does ancestral sampling starting from the prior.
    And then ancestral sampling from a learned proposal with bbvi
    """
    G = graph[1]
    verteces = G['V']
    arcs = G['A']
    if verteces_topsorted is None:
        verteces_topsorted = topsort(verteces, arcs)
    else:
        assert set(verteces) == set(verteces_topsorted)
    P = G['P']
    Y = G['Y']
    
    E, sampled_graph = sample_from_joint(graph,do_log=do_log) 
    # now returns Berens' distributions primitives in sampled_graph['prior_dist']
    # print('sampled_graph',sampled_graph)
        
    # initialize once
    d_prior = distributions.Normal(tensor(0.),tensor(1.))
    d_prior = d_prior.make_copy_with_grads()
    sigma = {'G':{},'logW':tensor(0.),'Q':{}}
    for vertex in sampled_graph['prior_dist'].keys():
        d_prior = sampled_graph['prior_dist'][vertex]
        d_prior_withgrads = d_prior.make_copy_with_grads() 
            # only do this once!
            # no check cases or prior init needed within evaluate_link_function_algo11 etc.
        sigma['Q'][vertex] = d_prior_withgrads
    # print('sigma',sigma)
    
    local_env, sigma = evaluate_link_function_algo11(P,verteces_topsorted,sigma,local_env={},do_log=do_log)

    sampled_graph = local_env
    return_of_graph = graph[2] # meaning of program, but need to evaluate
    # if do_log: print('sample_from_joint local_env',local_env)
    # if do_log: print('sample_from_joint sampled_graph',sampled_graph)
    return_of_graph_E, sigma = eval_algo11_deterministic(return_of_graph,sigma, local_env = sampled_graph, do_log=do_log)
    return return_of_graph_E, sigma # can return sampled_graph if needed