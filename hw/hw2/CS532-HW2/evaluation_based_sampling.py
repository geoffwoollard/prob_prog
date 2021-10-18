import os
import json
import logging

import torch

from daphne import daphne
from tests import is_tol, run_prob_test,load_truth
from primitives import primitives_d, distributions_d


logging.basicConfig(format='%(levelname)s:%(message)s')
logger = logging.getLogger('simple_example')
logger.setLevel(logging.DEBUG)

number = (int,float)
        

def evaluate_program(ast,sig=None):
    """Evaluate a program as desugared by daphne, generate a sample from the prior
    Args:
        ast: json FOPPL program
    Returns: sample from the prior of ast
    """
    defn_d ={}
    ast0 = ast[0]
    
    if ast0[0] == 'defn':
        defn_function_name = ast0[1]
        defn_function_args = ast0[2]
        defn_function_body = ast0[3]
        defn_d[defn_function_name] = [defn_function_args,defn_function_body]
        ast1 = ast[1]
        res = evaluate(ast1,defn_d=defn_d)
    elif len(ast) == 1:
        res = evaluate(ast0,defn_d=defn_d)
    else:
        assert False
    return res, sig


number = (int,float)
def evaluate(e,local_env={},defn_d={}):
    # remember to return evaluate (recursive)
    # everytime we call evaluate, we have to use local_env, otherwise it gets overwritten with the default {}
    if not isinstance(e,list) or len(e) == 1:
        if isinstance(e,list):
            e = e[0]

        if isinstance(e, number):
            return torch.tensor([float(e)])
        elif e in list(primitives_d.keys()):
            return e
        elif e in list(distributions_d.keys()):
            return e
        elif torch.is_tensor(e):
            return e
        elif e in local_env.keys():
            return local_env[e]
        elif e in list(defn_d.keys()):
            return e
        else:
            assert False
    elif e[0] == 'sample':
        distribution = evaluate(e[1],local_env,defn_d)
        return distribution.sample()
    elif e[0] == 'let': 
        # let [v1 e1] e0
        # here 
            # e[0] : "let"
            # e[1] : [v1, e1]
            # e[2] : e0
        # evaluates e1 to c1 and binds this value to e0
        # this means we update the context with old context plus {v1:c1}
        c1 = evaluate(e[1][1],local_env,defn_d) # evaluates e1 to c1
        v1 = e[1][0]
        return evaluate(e[2], local_env = {**local_env,v1:c1},defn_d=defn_d)
    elif e[0] == 'if': # if e0 e1 e2
        e0 = e[1]
        e1 = e[2]
        e2 = e[3]
        if evaluate(e0,local_env,defn_d):
            return evaluate(e1,local_env,defn_d)
        else:
            return evaluate(e2,local_env,defn_d) 

    else:
        cs = []
        for ei in e:
            c = evaluate(ei,local_env,defn_d)
            cs.append(c)
        if cs[0] in primitives_d:
            return primitives_d[cs[0]](cs[1:])
        elif cs[0] in distributions_d:
            return distributions_d[cs[0]](cs[1:])
        elif cs[0] in defn_d:
            defn_function_li = defn_d[cs[0]]
            defn_function_args, defn_function_body = defn_function_li
            local_env_update = {key:value for key,value in zip(defn_function_args, cs[1:])}
            return evaluate(defn_function_body,local_env = {**local_env, **local_env_update},defn_d=defn_d)
        else:
            assert False

def get_stream(ast):
    """Return a stream of prior samples"""
    while True:
        yield evaluate_program(ast)[0].item() # just return value, not context


def run_deterministic_tests():
    
    for i in range(1,14):
        os.chdir('/Users/gw/repos/prob_prog/hw/hw2/CS532-HW2/')
        sugared_fname = '../prob_prog/hw/hw2/CS532-HW2/programs/tests/deterministic/test_{}.daphne'.format(i)
        desugared_ast_json_fname = '/Users/gw/repos/prob_prog/' + sugared_fname.replace('.daphne','.json')
        if os.path.isfile(desugared_ast_json_fname):
            with open(desugared_ast_json_fname) as f:
                ast = json.load(f)
        else:
            assert False
            #note: the sugared path that goes into daphne desugar should be with respect to the daphne path!
            ast = daphne(['desugar', '-i', sugared_fname])
        truth = load_truth('programs/tests/deterministic/test_{}.truth'.format(i))
        ret, sig = evaluate_program(ast)
        try:
            assert(is_tol(ret, truth))
        except AssertionError:
            raise AssertionError('return value {} is not equal to truth {} for exp {}'.format(ret,truth,ast))
        
        print('Test %i passed'%i)
        
    print('All deterministic tests passed')
    


def run_probabilistic_tests():
    
    num_samples=1e4
    max_p_value = 1e-4
    
    for i in range(1,7):
        os.chdir('/Users/gw/repos/prob_prog/hw/hw2/CS532-HW2/')
        sugared_fname = '../prob_prog/hw/hw2/CS532-HW2/programs/tests/probabilistic/test_{}.daphne'.format(i)
        desugared_ast_json_fname = '/Users/gw/repos/prob_prog/' + sugared_fname.replace('.daphne','.json')
        if os.path.isfile(desugared_ast_json_fname):
            with open(desugared_ast_json_fname) as f:
                ast = json.load(f)
        else:
            assert False
            #note: the sugared path that goes into daphne desugar should be with respect to the daphne path!
            ast = daphne(['desugar', '-i', sugared_fname])  
        truth = load_truth('programs/tests/probabilistic/test_{}.truth'.format(i))
        
        stream = get_stream(ast)
        
        p_val = run_prob_test(stream, truth, num_samples)
        
        print('p value', p_val)
        assert(p_val > max_p_value)
    
    print('All probabilistic tests passed')    

        
if __name__ == '__main__':

    run_deterministic_tests()
    
    run_probabilistic_tests()


    for i in range(1,5):
        ast = daphne(['desugar', '-i', '../CS532-HW2/programs/{}.daphne'.format(i)])
        print('\n\n\nSample of prior of program {}:'.format(i))
        print(evaluate_program(ast)[0])