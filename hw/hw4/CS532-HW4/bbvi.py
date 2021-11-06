import logging

import numpy as np
import torch
from torch import tensor

from primitives import primitives_d, distributions_d, number, distribution_types
import distributions # for unconstrained optimization
from graph_based_sampling import sample_from_joint, score, topsort
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
            assert False, 'case not matched {}'.format(e)
    
    elif e[0] == 'sample':
        assert False, 'deterministic evaluator'

    elif e[0] == 'observe':
        assert False, 'deterministic evaluator'

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
            assert False, 'not implemented {}'.format(cs)


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
            
            # bbvi evaluator algo 11
            # get proposal from sigma
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
    # if D_v==1:
    #     lambda_vd = lambda_v[0]
    #     if 
    #     else:
    #         assert False, 'not implemented for lambda_v {}'.format(lambda_v)

    for d in range(D_v):
        lambda_v_d = lambda_v[d]
        if lambda_v_d.ndim > 0: # e.g. Categorical concentration
            assert D_v == 1, 'only implemented for multi dimentional lambda_vd for one component, incase different sizes  {}'.format(lambda_v)
            # TODO: generlaize with dict to lambda_vd component, with different sizes
            G_v_d = torch.zeros_like(lambda_v_d)
            G_v_d = lambda_v_d.grad.clone().detach()
            G_v = G_v_d.reshape(1,-1)
            lambda_v_d.grad = None

        else:
        
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
    g_hat = {}
    for v in union_G_keys:
        #F_v = []
        G_v = []
        
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
        # assert G_v.ndim == 2, 'G_v {}, type {}'.format(G_v, type(G_v))


        # cov and var to compute b_v
        if G_v.ndim == 2:
            F_v = G_v*logW.reshape(-1,1)
            D_v = G_v.shape[1]
            b_v = np.zeros(D_v)
            for d in range(D_v):
                F_v_d = F_v[:,d]
                G_v_d = G_v[:,d]
                cov_F_G = np.cov(F_v_d,G_v_d)
                b_v[d] = cov_F_G[0,1]/cov_F_G[1,1]
            g_hat_v = (F_v - G_v*b_v).mean(0)  # sum over samples divided by L
            g_hat[v] = g_hat_v

        elif G_v.ndim == 3:
            F_v = G_v*logW.reshape(-1,1,1)
            L, D_v, n_D_v = G_v.shape
            assert D_v == 1
            d=0
            b_v_d1 = np.zeros(D_v)
            cov_sum_d1, var_sum_d1 = 0, 0
            for j in range(n_D_v):
                G_v_1_j = G_v[:,d,j]
                F_v_1_j = F_v[:,d,j]
                cov_F_G_j = np.cov(F_v[:,d,j],G_v[:,d,j])
                cov_sum_d1 += cov_F_G_j[0,1]
                var_sum_d1 += cov_F_G_j[1,1]
            b_v_d1 = cov_sum_d1 / var_sum_d1
            if np.isnan(b_v_d1):
                b_v_d1 = 0 # or put a small number in the demonenator
            g_hat_v_d1 = (F_v - G_v*b_v_d1).mean(0)
            g_hat[v] = g_hat_v_d1
            # print('G_v {}, cov_sum_d1 {}, var_sum_d1 {}, g_hat_v_d1 {}'.format(G_v,cov_sum_d1,var_sum_d1,g_hat_v_d1))

        else:
            assert False, 'G_v {}, type {}'.format(G_v, type(G_v))

    return g_hat


def optimizer_step(Q,global_optimizers,g_hat,**kwargs):
    """
    no return of Q since modifies in place, and can't deep copy Q, and copy Q still accumulates
    """
    for vertex in g_hat.keys():

        lambda_v = Q[vertex].Parameters()
        optimizer = global_optimizers[vertex]
        D_v = len(lambda_v)

        # TODO: although params already has grad from grad_log_prob, this is not the b adjusted g_hat
        for idx in range(D_v):
            param = lambda_v[idx]
            # param.requires_grad = True # TODO: include???
            param.grad = tensor(-g_hat[vertex][idx],dtype=torch.float32) # TODO: check sign. maximizing
            # Optimizers subtract the gradient of all passed parameters using their .grad attribute as seen here 182. 
            # Thus you would minimize the loss using gradient descent.
            # https://discuss.pytorch.org/t/do-optimizers-minimize-or-maximize/69062

        optimizer.step() # moves lambda_v
        optimizer.zero_grad() # TODO: need this? 
    return Q



def graph_bbvi_algo12(graph,T,L,sigma=None,init_local_env={},do_log=False,custom_proposals=None,**kwargs):
    """This function does ancestral sampling starting from the prior.
    And then ancestral sampling from a learned proposal with bbvi
    TODO: fails when L=1. shapes? averaging?
    """
    r, G = [], []
    logW = np.zeros((T,L))

    G = graph[1]
    return_of_graph = graph[2] # meaning of program, but need to evaluate
    verteces = G['V']
    arcs = G['A']
    verteces_topsorted = topsort(verteces, arcs)
    P = G['P']
    Y = G['Y']

    if (sigma is not None) and ('Q' in sigma) and ('global_optimizers' in sigma):
        pass

    else:
    
        # local_env={}
        # if custom_proposals is not None:
        #     for vertex in custom_proposals.keys():
        #         local_env[vertex] = custom_proposals[vertex]

        E, sampled_graph = sample_from_joint(graph,local_env=init_local_env,do_log=do_log)
        # now returns Berens' distributions primitives in sampled_graph['prior_dist']
        # print('sampled_graph',sampled_graph)
            
        # initialize once
        sigma={'logW':tensor(0.),'Q':{},'G':{},'global_optimizers':{}}
        for vertex in sampled_graph['prior_dist'].keys():
            if custom_proposals is not None and vertex in custom_proposals.keys():
                d_prior = custom_proposals[vertex]
            elif vertex not in sigma['Q']:
                d_prior = sampled_graph['prior_dist'][vertex]

            d_prior_withgrads = d_prior.make_copy_with_grads() 
                # only do this once!
                # no check cases or prior init needed within evaluate_link_function_algo11 etc.
            
            if do_log: logger_graph.info('sigma {}'.format(sigma))
            if do_log: logger_graph.info('custom_proposals {}'.format(custom_proposals))

            sigma['Q'][vertex] = d_prior_withgrads


            # global optimizer
            Q = sigma['Q']
            lambda_v = Q[vertex].Parameters()
            optimizer = torch.optim.Adam(lambda_v, **kwargs)
            sigma['global_optimizers'][vertex] = optimizer
            

    if do_log: logger_graph.info('sigma {}'.format(sigma))


    logW_best = -np.inf
    elbo_best = -np.inf
    sigma['Q_best_t'] = {}
    rand_str = str(np.random.randint(low=0,high=10000000))
    for t in range(T):
        G = []
        r_t=[]
        union_G_keys = set()

        for l in range(L):
            sigma={
                'logW':tensor(0.),
                'Q':sigma['Q'],
                'G':{},
                'global_optimizers':sigma['global_optimizers'],
                'Q_best_t' : sigma['Q_best_t']
            } # re init gradients

            # graph eval algo 11
            local_env, sigma = evaluate_link_function_algo11(P,verteces_topsorted,sigma,local_env={},do_log=do_log)
            sampled_graph = local_env
            r_t_l, sigma = eval_algo11_deterministic(return_of_graph,sigma,local_env=sampled_graph,do_log=do_log)

            logW[t,l] = sigma['logW'].item()
            # if logW[t,l] > logW_best:
            #     logW_best = logW[t,l]
            #     sigma['Q_best_t_l'] = sigma['Q']
            G_l = (sigma['G']).copy() # warning for underlying gradients to still tie back to the same objects
            union_G_keys.update(set(G_l.keys()))
            G.append(G_l)
            r_t.append(r_t_l)
            if do_log: logger_graph.info('t {}, l {}, sigma {}, local_env {}'.format(t, l,sigma,local_env))

        if do_log: logger_graph.info('sigma {}'.format(sigma))
        g_hat = elbo_gradients(G,logW[t],union_G_keys) 
        if do_log: logger_graph.info('g_hat {}, union_G_keys {}'.format(g_hat,union_G_keys))
        Q = sigma['Q']
        if do_log: logger_graph.info('Q before step',Q)
        global_optimizers = sigma['global_optimizers']
        Q = optimizer_step(Q,global_optimizers,g_hat,**kwargs) # in place modification of Q, so Q same as sigma['Q']
        if T <= 10 or t % (T // 10) == 0:
            print('t={}, Q after step={}'.format(t,Q))

        r.append(r_t)

        elbo = logW[t].mean()
        if elbo > elbo_best:
            elbo_best = elbo
            sigma['Q_best_t'] = torch.save(sigma['Q'],rand_str+'tmp') # cant use copy or deep copy, so just save and load

    sigma['Q_best_t'] = torch.load(rand_str+'tmp')

    return r, logW, sigma

