from collections import defaultdict
import os
import json
import logging

import torch
from torch import tensor
import torch.distributions as dist

from daphne import daphne

from evaluation_based_sampling import evaluate, score

logging.basicConfig(format='%(levelname)s:%(message)s')
logger_graph = logging.getLogger('simple_example')
logger_graph.setLevel(logging.DEBUG)


def deterministic_eval(exp):
    return evaluate(exp)


class Graph:
    # https://favtutor.com/blogs/topological-sort-python
    # Topological Sort Algorithm | Graph Theory | https://youtu.be/eL-KzMXSXXI

    def __init__(self,n):

        self.graph = defaultdict(list)

        self.N = n

    def addEdge(self,m,n):

        self.graph[m].append(n)

    def sortUtil(self,n,visited,stack):

        visited[n] = True

        for element in self.graph[n]:

            if visited[element] == False:

                self.sortUtil(element,visited,stack)

        stack.insert(0,n)

    def topologicalSort(self):

        visited = [False]*self.N

        stack =[]

        for element in range(self.N):

            if visited[element] == False:

                self.sortUtil(element,visited,stack)

        self.topsorted = stack


def topsort(verteces,arcs):
    """Toplogical sort Daphne Graph.

    G = {A,V,P,Y}
    """
    
    v_to_int = {key:val for val, key in enumerate(verteces)}
    int_to_v = {val:key for key, val in v_to_int.items()}

    # make adjecency from parsed format
    g = Graph(len(verteces))
    for key_u, val_v_list in arcs.items():
        if key_u in verteces:
            for v in val_v_list:
                g.addEdge(v_to_int[key_u],v_to_int[v])
    # topsort
    g.topologicalSort()
    # map back to vertex string labels
    verteces_topsorted = [int_to_v[i] for i in g.topsorted]
    return verteces_topsorted


def sample_from_joint_precompute(graph):
    """factor out topological sort. 
    sufficiently quivalent (although not unique) for the same graph, so no need to recompute."""
    G = graph[1]
    verteces = G['V']
    arcs = G['A']
    verteces_topsorted = topsort(verteces, arcs)
    return verteces_topsorted


def evaluate_link_function(P,verteces_topsorted,sigma,local_env,do_log):
    if 'prior_dist' not in local_env.keys(): local_env['prior_dist']={}
    for vertex in verteces_topsorted:
        link_function = P[vertex]
        if link_function[0] == 'sample*':
            if do_log: logger_graph.info('match case sample*: link_function {}'.format(link_function))
            assert len(link_function) == 2
            e = link_function[1]

            # overwrite
            if 'prior_dist' in local_env.keys() and vertex in local_env['prior_dist'].keys():
                distribution = local_env['prior_dist'][vertex]
            else:
                distribution, sigma = evaluate(e,sigma,local_env = local_env, do_log=do_log)
            if do_log: logger_graph.info('match case sample*: distribution {}, sigma {}'.format(sigma, distribution))
            E = distribution.sample() # now have concrete value. need to pass it as var to evaluate

            update_local_env = {vertex:E}#{vertex:E, vertex+'_dist':distribution}
            

            local_env.update(update_local_env)
            local_env['prior_dist'][vertex] = distribution
        elif link_function[0] == 'observe*':
            if do_log: logger_graph.info('match case observe*: link_function {} sigma {}'.format(link_function, sigma))
            e1, e2 = link_function[1:]
            d1, sigma = evaluate(e1,sigma,local_env,do_log=do_log)
            c2, sigma = evaluate(e2,sigma,local_env,do_log=do_log)
            log_w = score(d1,c2)
            sigma  += log_w
            if do_log: logger_graph.info('match case observe*: d1 {}, c2 {}, log_w {}, sigma {}'.format(d1, c2, log_w, sigma))
        else:
            assert False

    return local_env, sigma


def sample_from_joint(graph,sigma=tensor(0.),local_env={'prior_dist':{}},do_log=False,verteces_topsorted=None):
    """This function does ancestral sampling starting from the prior.

    graph output from `daphne graph -i sugared.daphne`
    * list of length 3
      * first entry is defn dict
        * {"string-defn-function-name":["fn", ["var_1", ..., "var_n"], e_function_body], ...}
      * second entry is graph: {V,A,P,Y}
        * "V","A","P","Y" are keys in dict
        * "V" : ["string_name_vertex_1", ..., "string_name_vertex_n"] # list of string names of vertices
        * "A" : {"string_name_vertex_1" : [..., "string_name_vertex_i", ...] # dict of arc pairs (u,v) with u a string key in the dict, and the value a list of string names of the vertices. note that the keys can be things like "uniform" and don't have to be vetex name strings
        * "P" : "string_name_vertex_i" : ["sample*", e_i] # dict. keys vertex name strings and value a rested list with a linking function in it. typically e_i is a distribution object. 
        * "Y" : observes
      * third entry is return
        * name of return rv, or constant

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
    sampled_graph = {}
    local_env, sigma = evaluate_link_function(P,verteces_topsorted,sigma,local_env=local_env,do_log=do_log)

    sampled_graph = local_env
    return_of_graph = graph[2] # meaning of program, but need to evaluate
    # if do_log: print('sample_from_joint local_env',local_env)
    # if do_log: print('sample_from_joint sampled_graph',sampled_graph)
    E = evaluate(return_of_graph,sigma, local_env = sampled_graph, do_log=do_log)
    return E, sampled_graph


def evaluate_program_return_from_samples_whole_graph(graph,samples_whole_graph):
    # evaluate samples (on whatever function, here the return of the program) as needed
    e = graph[2]
    # TODO suggest daphne put return as program, so return is ['sample2'] not 'sample2'
    if isinstance(e,str):
        e = [e]
    return_list = []
    for X_s in samples_whole_graph:
        return_s, _ = evaluate(e,local_env = X_s) # TODO: handle defns
        return_list.append(return_s)
    return return_list
        
          

    