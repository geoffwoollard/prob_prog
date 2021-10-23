""" evaluate algorithm 6 in 
van de Meent, J.-W., Paige, B., Yang, H., & Wood, F. (2018). 
An Introduction to Probabilistic Programming, XX(Xx), 1–221. 
http://doi.org/10.1561/XXXXXXXXXX

Acknowledgements to Yuan T https://github.com/yuant95/CPSC532W/blob/master/CS532-HW2/evaluation_based_sampling.py
and 
Masoud Mokhtari https://github.com/MasoudMo/CPSC-532W/blob/master/HW2/evaluation_based_sampling.py
"""

import os
import json
import logging

import torch

from daphne import daphne
from tests import is_tol, run_prob_test,load_truth
from primitives import primitives_d, distributions_d, number, distribution_types


logging.basicConfig(format='%(levelname)s:%(message)s')
logger = logging.getLogger('simple_example')
logger.setLevel(logging.DEBUG)
        

def evaluate_program(ast,sigma=0,do_log=False):
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
        res, sigma = evaluate(ast1,sigma,defn_d=defn_d,do_log=do_log)
    elif len(ast) == 1:
        res, sigma = evaluate(ast0,sigma,defn_d=defn_d,do_log=do_log)
    else:
        assert False
    return res, sigma

# def log_prob(distribution, value_of_rv):
#     return distribution.log_prob(value_of_rv)


number = (int,float)
def evaluate(e,sigma=0,local_env={},defn_d={},do_log=False):
    # TODO: get local_env to evaluate values to tensors, not regular floats
    # remember to return evaluate (recursive)
    # everytime we call evaluate, we have to use local_env, otherwise it gets overwritten with the default {}
    if do_log: logger.info('local_env {}, sigma {}'.format(local_env, sigma))
    if do_log: logger.info('e {}, sigma {}'.format(e, sigma))

    # get first expression out of list or list of one
    if not isinstance(e,list) or len(e) == 1:
        if isinstance(e,list):
            e = e[0]
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
        elif torch.is_tensor(e) and not len(list(e.shape)) == 0:
            if do_log: logger.info('match case is_tensor: e {}, sigma {}'.format(e, sigma))
            return e, sigma
        elif e in local_env.keys():
            if do_log: logger.info('match case local_env: e {}, sigma {}'.format(e, sigma))
            if do_log: logger.info('match case local_env: local_env[e] {}'.format(local_env[e]))
            return local_env[e] # TODO return evaluate?
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
        distribution, sigma = evaluate(e[1],sigma,local_env,defn_d,do_log=do_log)
        return distribution.sample(), sigma # match shape in number base case
    elif e[0] == 'observe':
        e1, e2 = e[1:]
        d1, sigma = evaluate(e1,sigma,local_env,defn_d,do_log=do_log)
        c2, sigma = evaluate(e2,sigma,local_env,defn_d,do_log=do_log)
        log_w = d1.log_prob(c2)
        if do_log: logger.info('match case observe: e {}, sigma {}, log_w {}'.format(e,sigma,log_w))
        sigma  = sigma + log_w
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
        c1, sigma = evaluate(e[1][1],sigma,local_env,defn_d,do_log=do_log) # evaluates e1 to c1
        v1 = e[1][0]
        return evaluate(e[2], sigma, local_env = {**local_env,v1:c1},defn_d=defn_d,do_log=do_log), sigma
    elif e[0] == 'if': # if e0 e1 e2
        if do_log: logger.info('match case if: e {}, sigma {}'.format(e, sigma))
        e1 = e[1]
        e2 = e[2]
        e3 = e[3]
        e1_prime, sigma = evaluate(e1,sigma,local_env,defn_d,do_log=do_log)
        if e1_prime:
            return evaluate(e2,sigma,local_env,defn_d,do_log=do_log)
        else:
            return evaluate(e3,sigma,local_env,defn_d,do_log=do_log) 

    else:
        cs = []
        for ei in e:
            if do_log: logger.info('cycling through expressions: ei {}, sigma {}'.format(ei,sigma))
            c, sigma = evaluate(ei,sigma,local_env,defn_d,do_log=do_log)
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
            return evaluate(defn_function_body,sigma,local_env = {**local_env, **local_env_update},defn_d=defn_d,do_log=do_log)
        else:
            assert False, 'not implemented'

def get_stream(ast):
    """Return a stream of prior samples"""
    while True:
        yield evaluate_program(ast,do_log=False)[0].item() # just return value, not context


def run_deterministic_tests():
    
    tot=0
    for i in range(1,14):
        os.chdir('/Users/gw/repos/prob_prog/hw/hw2/CS532-HW2/')
        sugared_fname = '../prob_prog/hw/hw2/CS532-HW2/programs/tests/deterministic/test_{}.daphne'.format(i)
        desugared_ast_json_fname = '/Users/gw/repos/prob_prog/' + sugared_fname.replace('.daphne','.json')
        if os.path.isfile(desugared_ast_json_fname):
            with open(desugared_ast_json_fname) as f:
                ast = json.load(f)
        else:
            #note: the sugared path that goes into daphne desugar should be with respect to the daphne path!
            ast = daphne(['desugar', '-i', sugared_fname])
            with open(desugared_ast_json_fname, 'w') as f:
                json.dump(ast, f) 
        truth = load_truth('programs/tests/deterministic/test_{}.truth'.format(i))
        ret, sig = evaluate_program(ast)
        try:
            assert(is_tol(ret, truth))
            tot += 1
        except AssertionError:
            raise AssertionError('return value {} is not equal to truth {} for exp {}'.format(ret,truth,ast))
        
        print('Test %i passed'%i)
        
    print('All {} deterministic tests passed'.format(i))
    


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
            # TODO: put in write json
            #note: the sugared path that goes into daphne desugar should be with respect to the daphne path!
            ast = daphne(['desugar', '-i', sugared_fname]) 
            with open(desugared_ast_json_fname, 'w') as f:
                json.dump(ast, f) 
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
        os.chdir('/Users/gw/repos/prob_prog/hw/hw2/CS532-HW2/')
        sugared_fname = '../prob_prog/hw/hw2/CS532-HW2/programs/{}.daphne'.format(i)
        desugared_ast_json_fname = '/Users/gw/repos/prob_prog/' + sugared_fname.replace('.daphne','.json')
        if os.path.isfile(desugared_ast_json_fname):
            with open(desugared_ast_json_fname) as f:
                ast = json.load(f)
        else:
            #note: the sugared path that goes into daphne desugar should be with respect to the daphne path!
            ast = daphne(['desugar', '-i', sugared_fname]) 
            with open(desugared_ast_json_fname, 'w') as f:
                json.dump(ast, f)

        print('\n\n\nSample of prior of program {}:'.format(i))
        print(evaluate_program(ast)[0])