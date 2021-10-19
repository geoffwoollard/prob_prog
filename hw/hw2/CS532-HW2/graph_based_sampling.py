from collections import defaultdict
import os
import json

import torch
import torch.distributions as dist

from daphne import daphne

# from primitives import funcprimitives #TODO
from tests import is_tol, run_prob_test,load_truth
from evaluation_based_sampling import evaluate

# Put all function mappings from the deterministic language environment to your
# Python evaluation context here:
env = {'normal': dist.Normal,
       'sqrt': torch.sqrt}


def deterministic_eval(exp):
    return evaluate(exp)
#     "Evaluation function for the deterministic target language of the graph based representation."
#     if type(exp) is list:
#         op = exp[0]
#         args = exp[1:]
#         return env[op](*map(deterministic_eval, args))
#     elif type(exp) is int or type(exp) is float:
#         # We use torch for all numerical objects in our evaluator
#         return torch.tensor(float(exp))
#     else:
#         raise("Expression type unknown.", exp)


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

def topsort(G):
    """Toplogical sort Daphne Graph.

    G = {A,V,P,Y}
    """
    
    v_to_int = {key:val for val, key in enumerate(G['V'])}
    int_to_v = {val:key for key, val in v_to_int.items()}

    # make adjecency from parsed format
    g = Graph(len(G['V']))
    for key_u, val_v_list in G['A'].items():
        if key_u in G['V']:
            for v in val_v_list:
                g.addEdge(v_to_int[key_u],v_to_int[v])
    # topsort
    g.topologicalSort()
    # map back to vertex string labels
    V_topsorted = [int_to_v[i] for i in g.topsorted]
    return V_topsorted


def sample_from_joint(graph,do_log=False):
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
    V_topsorted = topsort(G)
    P = G['P']
    Y = G['Y']
    sampled_graph = {}
    local_env = {}
    for vertex in V_topsorted:
        link_function = P[vertex]

        if link_function[0] == 'sample*':
            assert len(link_function) == 2
            e = link_function[1]
    #         print('e in as',e)
            distribution = evaluate(e,local_env = local_env, do_log=do_log)
            E = distribution.sample() # now have concrete value. need to pass it as var to evaluate
            update_local_env = {vertex:E}
            local_env.update(update_local_env)
        elif link_function[0] == 'observe*':
            assert len(link_function) == 3
            E = Y[vertex]
        else:
            assert False
        sampled_graph[vertex] = E
    return_of_graph = graph[2] # meaning of program, but need to evaluate
    # if do_log: print('sample_from_joint local_env',local_env)
    # if do_log: print('sample_from_joint sampled_graph',sampled_graph)
    return evaluate(return_of_graph,local_env = sampled_graph, do_log=do_log)


def get_stream(graph):
    """Return a stream of prior samples
    Args: 
        graph: json graph as loaded by daphne wrapper
    Returns: a python iterator with an infinite stream of samples
        """
    while True:
        yield sample_from_joint(graph).item()


#Testing:

def run_deterministic_tests():
    
    for i in [1,2,3,4,5,7,9,10,12]: # TODO: vector returns
        os.chdir('/Users/gw/repos/prob_prog/hw/hw2/CS532-HW2/')

        #note: this path should be with respect to the daphne path!
        sugared_fname = '../prob_prog/hw/hw2/CS532-HW2/programs/tests/deterministic/test_{}.daphne'.format(i)
        
        graph_json_fname = '/Users/gw/repos/prob_prog/' + sugared_fname.replace('.daphne','_graph.json')
        if os.path.isfile(graph_json_fname):
            with open(graph_json_fname) as f:
                graph = json.load(f)
        else:
            #note: the sugared path that goes into daphne desugar should be with respect to the daphne path!
            graph = daphne(['graph', '-i', sugared_fname]) 
            with open(graph_json_fname, 'w') as f:
                json.dump(graph, f)

        
        truth = load_truth('programs/tests/deterministic/test_{}.truth'.format(i))
        ret = deterministic_eval(graph[-1])
        try:
            assert(is_tol(ret, truth))
        except AssertionError:
            raise AssertionError('return value {} is not equal to truth {} for graph {}'.format(ret,truth,graph))
        
        print('Test passed')
        
    print('All deterministic tests passed')
    


def run_probabilistic_tests():
    
    #TODO: 
    num_samples=1e4
    max_p_value = 1e-4
    
    for i in range(1,7):
        os.chdir('/Users/gw/repos/prob_prog/hw/hw2/CS532-HW2/')
        sugared_fname = '../prob_prog/hw/hw2/CS532-HW2/programs/tests/probabilistic/test_{}.daphne'.format(i)
        graph_json_fname = '/Users/gw/repos/prob_prog/' + sugared_fname.replace('.daphne','_graph.json')
        if os.path.isfile(graph_json_fname):
            with open(graph_json_fname) as f:
                graph = json.load(f)
        else:
            #note: the sugared path that goes into daphne desugar should be with respect to the daphne path!
            graph = daphne(['graph', '-i', sugared_fname]) 
            with open(graph_json_fname, 'w') as f:
                json.dump(graph, f)
        
        truth = load_truth('programs/tests/probabilistic/test_{}.truth'.format(i)) 
        stream = get_stream(graph)
        
        p_val = run_prob_test(stream, truth, num_samples)
        
        print('p value', p_val)
        assert(p_val > max_p_value)
    
    print('All probabilistic tests passed')    


        
        
if __name__ == '__main__':
    

    run_deterministic_tests()
    run_probabilistic_tests()




    for i in range(1,5):
        # graph = daphne(['graph','-i','../CS532-HW2/programs/{}.daphne'.format(i)])
        os.chdir('/Users/gw/repos/prob_prog/hw/hw2/CS532-HW2/')
        sugared_fname = '../prob_prog/hw/hw2/CS532-HW2/programs/{}.daphne'.format(i)
        graph_json_fname = '/Users/gw/repos/prob_prog/' + sugared_fname.replace('.daphne','_graph.json')
        if os.path.isfile(graph_json_fname):
            with open(graph_json_fname) as f:
                graph = json.load(f)
        else:
            #note: the sugared path that goes into daphne desugar should be with respect to the daphne path!
            graph = daphne(['graph', '-i', sugared_fname]) 
            with open(graph_json_fname, 'w') as f:
                json.dump(graph, f)
        print('\n\n\nSample of prior of program {}:'.format(i))
        print(sample_from_joint(graph))    

    