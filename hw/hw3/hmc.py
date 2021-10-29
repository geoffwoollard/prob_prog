import copy

import torch
from torch import tensor
from evaluation_based_sampling import evaluate
from graph_based_sampling import sample_from_joint, score


def hmc_wrapper(graph,num_samples,T=10,epsilon=0.1,M=tensor(1.)):
    #set up X, Y list of verteces
    G = graph[1]
    verteces = ['V']
    Y = G['Y']
    P = G['P']
    
    # evaluate to constants
    Y = {key:evaluate([value])[0] for key,value in Y.items()}
    
    #X = set(vertices) - set(Y.keys())
    #X = sample_from_joint(graph)
    _, X0 = sample_from_joint(graph) # does not include observes
    
    # initialize in dict
    X_vertex_names_to_idx_d = {key:idx for idx,key in enumerate(X0.keys())}

    # set up autograd on tensors
    turn_on_autodiff(X0)
    turn_on_autodiff(Y) # TODO: why do we need this?

    
    # run HMC algorithm 20 from book
        # inside use leapfrog algorithm 19 from book
        # include kinetic and potential energy functions
        # MC acceptance criteria
    samples_whole_graph = hmc_algo20(X0,num_samples,T,epsilon,M,Y,P,X_vertex_names_to_idx_d)
    
    # evaluate samples (on whatever function, here the return of the program) as needed
    e = graph[2]
    # TODO suggest daphne put return as program, so return is ['sample2'] not 'sample2'
    if isinstance(e,str):
        e = [e]
    return_list = []
    for X_s in samples_whole_graph:
        return_s, _ = evaluate(e,local_env = X_s) # TODO: handle defns
        return_list.append(return_s)

    return return_list, samples_whole_graph


def hmc_algo20(X0,num_samples,T,epsilon,M,Y,P,X_vertex_names_to_idx_d):
    X_s = X0
    samples = []
    size = len(X0.keys())
    normal_R_reuse = torch.distributions.Normal(torch.zeros(size),M)
    
    for s in range(num_samples):
        R_s = normal_R_reuse.sample()
        R_p, X_p = leapfrog(copy.deepcopy(X_s),copy.deepcopy(R_s),T,epsilon,Y,P,X_vertex_names_to_idx_d)
        # X_p, R_p = leapfrog(X_s,R_s,T,epsilon,X_vertex_names_to_idx_d)
            # copy X_s?
        u = torch.rand(1)
        delta_H = compute_H(X_p,R_p,M,Y,P) - compute_H(X_p,R_p,M,Y,P)
        boltzmann_ratio = torch.exp(-delta_H)
        if u < boltzmann_ratio:
            X_s = X_p
        #no need to update X_s because should stay the same for next round. 
        #X_s turns into X_s_minus from algo 20 by indexing
        samples.append(X_s)
    return samples

    
def leapfrog(X0,R0,T,epsilon,Y,P,X_vertex_names_to_idx_d):
    """
    leapfrog as in algo 19 of book
    Y and P needed for grad calc
    # TODO: bug for mass matrix on Xt update equations. M_inverse

    """
    
    epsilon_2 = epsilon/2
    R_t = R0 - epsilon_2 * grad_U(X0,Y,P,X_vertex_names_to_idx_d) 
    
    X_t = X0
    for t in range(T-1):
        # TODO: save all in loop instead of overwriting to visualize
        X_t = add_dict_to_tensor(X_t,epsilon*R_t,X_vertex_names_to_idx_d)
        R_t = R_t - epsilon*grad_U(X_t,Y,P,X_vertex_names_to_idx_d)
    X_T = add_dict_to_tensor(X_t,epsilon*R_t,X_vertex_names_to_idx_d)
    R_T = R_t - epsilon_2*grad_U(X_T,Y,P,X_vertex_names_to_idx_d)
    return R_T, X_T


def add_dict_to_tensor(X,R,X_vertex_names_to_idx_d):
    """
    X+R using mapping from X_vertex_names_to_idx_d
    TODO: avoid detach?
    R must be dimension 1, not 0 D
    similar to using with torch.no_grad() as in https://github.com/MasoudMo/cpsc532w_hw/blob/master/HW3/graph_based_sampling.py#L275
    """
    assert R.dim() >= 1
    X_new = {}
    for vertex in X.keys():
        idx = X_vertex_names_to_idx_d[vertex]
        # overwriting the value, and only want to autograd accumulated gradient to depend on the final value
        # TODO: would this problem go away if we stored a vector over all leapfrog time steps?
        X_new[vertex] = X[vertex].detach() + R[idx]
        X_new[vertex].requires_grad = True
    return X_new


def grad_U(X,Y,P,X_vertex_names_to_idx_d):
    """
    call autodiff backward pass, with constant indexing given by X_vertex_names_to_idx_d
    return vector of gradients
    """
    energy_U = compute_U(X,Y,P)
    
    # Zero the gradients. 
    # without this running grad_U back to back accumulates the grad (not what we want!)
    for key in X.keys():
        if X[key].grad is not None:
            X[key].grad.zero_()
            
    energy_U.backward()
    grads = torch.zeros(len(X.keys()))
    for key in X.keys():
#         print('key',key)
        idx = X_vertex_names_to_idx_d[key]
#         print('idx',idx)
#         print('X[key]',X[key])
#         print('X[key].grad',X[key].grad)
        grads[idx] = X[key].grad 
            # Need to have been referenced in linking function.
            # So key connected to evaluation of energy.
            # Otherwise key not a part of the computational graph of energy_U, 
                # and grad remains none when run energy_U.backwards()
    return grads


def turn_on_autodiff(dictionary_of_tensors):
    """
    cant be integers, ie long tensors, but floats or complex
    """
    for x in dictionary_of_tensors.values():
        if torch.is_tensor(x):
            x.requires_grad = True


def compute_H(X,R,M,Y,P):
    """Compute Hamiltonian.
    """
    energy_U = compute_U(X,Y,P)
    energy_K = compute_K(R,M)
    energy_H = energy_U + energy_K
    return energy_H


def compute_log_joint_prob(X,Y,P):
    """
    call link functions under context and score.
    TODO: remove Y dependence
    TODO: add in user defns https://github.com/MasoudMo/cpsc532w_hw/blob/master/HW3/graph_based_sampling.py#L208
    need to parse link functions like 
        # 'sample2': ['sample*', ['normal', 1, ['sqrt', 5]]]
        # 'observe3': ['observe*', ['normal', 'sample2', ['sqrt', 2]], 8],
    """
    log_prob = tensor(0.0)
    for X_vertex in X.keys():
        e = P[X_vertex][1]
        distribution = evaluate(e,local_env=X)[0]
        log_prob += score(distribution,X[X_vertex])
    for Y_vertex in Y.keys():
        e = P[Y_vertex][1]
        distribution = evaluate(e,local_env=X)[0]
        log_prob += score(distribution,Y[Y_vertex])
    
    return log_prob


def compute_U(X,Y,P):
    energy_U = -compute_log_joint_prob(X,Y,P)
    return energy_U 
        

def compute_K(R,M):
    R_over_2M = R/(2*M) # TODO: generalize for non scalar M, e.g. diagonal M
    if R.dim() == 0:
        energy_K = R*R_over_2M
    elif R.dim() >= 1:
        energy_K = torch.matmul(R,R_over_2M)
    else:
        assert False
    return energy_K
    
    
    