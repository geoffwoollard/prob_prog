import torch
from torch import tensor
from evaluation_based_sampling import evaluate
from graph_based_sampling import sample_from_joint, sample_from_joint_precompute


def hmc_wrapper(graph,num_samples):
    #set up X, Y list of verteces
    # initialize in dict
    # X_vertex_names_to_idx_d
    # set up autograd on X tensors
    
    # set up hyper params, T, epsilon, M
    
    # run HMC algorithm 20 from book
        # inside use leapfrog algorithm 19 from book
        # include kinetic and potential energy functions
        # MC acceptance criteria
    # evaluate samples as needed
    # return list of samples, num_samples long
    samples = [1]*num_samples
    return samples


def hmc_algo20(X0,num_samples,T,epsilon,M,X_vertex_names_to_idx_d):
    X_s = X0
    samples = [1]*num_samples
    for s in range(num_samples):
        pass
        # sample R_s from Normal(0,M). tensor of length X
        # X_p, R_p = leapfrog(X_s,R_s,T,epsilon,X_vertex_names_to_idx_d)
            # copy X_s?
        # u = torch.rand(1)
        # delta_H = - hamiltonian(X_p,R_p,M) + hamiltonian(X_p,R_p,M)
        # boltzmann_ratio = torch.exp(delta_H)
        # if u < boltzmann_ratio:
        # X_s = X_p
        # else:
            #pass # no need to update X_s because should stay the same for next round. 
                    #turns into X_s_minus from algo 20 by indexing
        #samples.append(X_s)
    return samples

    
def leapfrog(X0,R0,T,epsilon,Y,P,X_vertex_names_to_idx_d):
    """
    leapfrog as in algo 19 of book
    Y and P needed for grad calc
    """
    epsilon_2 = epsilon/2
    R_t = R0 - epsilon_2 * grad_U(X0,Y,P)
    X_t = X0
    for t in range(T-1):
        # TODO: save all in loop instead of overwriting to visualize
        X_t = add_dict_to_tensor(X_t,epsilon*R_t,X_vertex_names_to_idx_d)
        R_t = R - epsilon*grad_U(X_t,Y,P)
    X_T = add_dict_to_tensor(X_t,epsilon*R_t,X_vertex_names_to_idx_d)
    R_T = R_t - epsilon_2*grad_U(X_T,Y,P)
    return R_T, X_T


def add_dict_to_tensor(X,R,X_vertex_names_to_idx_d):
    # TODO: X+R using mapping from X_vertex_names_to_idx_d
    return X


def grad_U(X,Y,P,X_vertex_names_to_idx_d):
    U = compute_U(X,Y,P)
    U.backward()
    # TODO: return vector of gradients
    grads = torch.zeros(len(X.keys()))
    for key in X_keys:
        idx = X_vertex_names_to_idx_d[key]
        grads[idx] = X[key].grad
    return grads


def compute_U(X,Y,P):
    log_prob = 0
    # TODO: call link functions under context and score
    U = -log_prob
    return U


def compute_H(X,R,M,Y,P):
    U = compute_U(X,Y,P)
    K = compute_K(R,M)
    H = U + K
    return H
        

def compute_K(R,M):
    R_over_2M = R/(2*M) # TODO: generalize for non scalar M, e.g. diagonal M
    K = torch.matmul(R,R_over_2M)
    return K
    
    
    