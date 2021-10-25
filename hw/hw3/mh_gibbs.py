import torch

from evaluation_based_sampling import evaluate, score
from graph_based_sampling import sample_from_joint
from graph_based_sampling import sample_from_joint_precompute

def accept(vertex,local_env,local_env_prime,A,P,do_log):
    link_function = P[vertex]
    e = link_function[1]
    d_q, _ = evaluate(e,local_env=local_env,do_log=do_log)
    d_q_prime, _ = evaluate(e,local_env=local_env_prime,do_log=do_log)
    log_a = d_q_prime.log_prob(local_env[vertex]) - d_q.log_prob(local_env_prime[vertex])
    V_x = A[vertex] + [vertex] 
    for observed_vertex in V_x:
        d_p_prime = evaluate(P[observed_vertex][1],local_env = local_env_prime,do_log=do_log)[0]
        if do_log: evaluate(1,do_log=do_log,logger_string='d_p_prime {}'.format(d_p_prime))
        
        d_p = evaluate(P[observed_vertex][1],local_env = local_env,do_log=do_log)[0]
        if do_log: evaluate(1,do_log=do_log,logger_string='d_p {}'.format(d_p))
        if do_log: evaluate(1,do_log=do_log,logger_string='local_env_prime {}, observed_vertex {}, local_env_prime[observed_vertex] {}'.format(local_env_prime,observed_vertex,local_env_prime[observed_vertex]))
        log_a += score(d_p_prime,local_env_prime[observed_vertex])
        # log_a += d_p_prime.log_prob(local_env_prime[observed_vertex])
        log_a -= score(d_p,local_env[observed_vertex])
        # log_a -= d_p.log_prob(local_env[observed_vertex])

    return torch.exp(log_a)

def gibbs_step(local_env,P,A,X_sample_vertices,do_log):
    for vertex in X_sample_vertices:
        link_function = P[vertex]
        e = link_function[1]

        distribution, sigma = evaluate(e,local_env=local_env,do_log=do_log)
        local_env_prime = local_env.copy()
        local_env_prime[vertex] = distribution.sample()

        alpha = accept(vertex,local_env,local_env_prime,A,P,do_log=do_log)
        u = torch.rand(1)
        if u < alpha:
            local_env = local_env_prime
    return local_env

def gibbs(num_steps,local_env,P,A,X,do_log):
    local_env_list = []
    for step in range(num_steps):
        local_env = gibbs_step(local_env,P,A,X,do_log=do_log)
        local_env_list.append(local_env)
    return local_env_list


def mh_gibbs_wrapper(graph,num_steps,do_log):
    G = graph[1]
    verteces = G['V']
    A = G['A']
    P = G['P']
    X = set(verteces) - set(G['Y'].keys())
    Y = G['Y']
    Y = {key:evaluate([Y[key]], do_log=do_log)[0] for key in Y.keys()}


    verteces_topsorted = sample_from_joint_precompute(graph)
    _, local_env = sample_from_joint(graph,verteces_topsorted=verteces_topsorted)
    local_env = {**local_env,**Y}
    local_env_list0 = [local_env]

    local_env_list = gibbs(num_steps,local_env,P,A,X,do_log=do_log)

    local_env_list = local_env_list0 + local_env_list
    return local_env_list