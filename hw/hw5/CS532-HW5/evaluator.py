from pyrsistent import pmap,plist
import copy
import os
import torch
import json

from daphne import daphne
from tests import is_tol, run_prob_test,load_truth
from primitives import penv, number




# def standard_env():
#     "An environment with some Scheme standard procedures."
#     env = pmap(penv)
#     env = env.update({'alpha' : ''}) 

#     return env



def evaluate(exp, env=None,do_log=False): #TODO: add sigma, or something

    if env is None:
        env = standard_env()

    fn, _ = eval_hoppl(exp,env,sigma=None, do_log=do_log)
    ret, sigma = fn("")
    if do_log: print('return',ret)
    return ret


class Env():
    "An environment: a dict of {'var': val} pairs, with an outer Env."
    def __init__(self, parms=(), args=(), outer=None):
        self.data = pmap(zip(parms, args))
        self.outer = outer
        if outer is None:
            self.level = 0
        else:
            self.level = outer.level+1
    def __getitem__(self, item):
        return self.data[item]
    def find(self, var):
        "Find the innermost Env where var appears."
        if (var in self.data):
            return self
        else:
            if self.outer is not None:
                return self.outer.find(var)
            else:
                raise RuntimeError('var "{}" not found in outermost scope'.format(var))
    
    def print_env(self, print_lowest=False):
        print_limit = 1 if print_lowest == False else 0
        outer = self
        while outer is not None:
            if outer.level >= print_limit:
                print('Scope on level ', outer.level)
                if 'f' in outer:
                    print('Found f, ')
                    print(outer['f'].body)
                    print(outer['f'].parms)
                    print(outer['f'].env)
                print(outer,'\n')
            outer = outer.outer

class Procedure(object):
    "A user-defined Scheme procedure."
    def __init__(self, parms, body, env,do_log):
        self.parms, self.body, self.env = parms, body, env
        self.do_log = do_log
    def __call__(self, *args): 
        new_env = copy.deepcopy(self.env)
        return eval_hoppl(self.body, Env(self.parms, args, new_env),do_log=self.do_log) # [0]



def standard_env():
    "An environment with some Scheme standard procedures."
    env = Env(penv.keys(), penv.values())
    return env


def eval_hoppl(x,env=standard_env(),sigma=None,do_log=False):
    # TODO: remove default env=standard_env()

    # base cases
    if do_log: print('x',x)
    if isinstance(x,str):
        return env.find(x)[x], sigma
    elif not isinstance(x,list):
        return torch.tensor(x),sigma

    op, param, *args = x
    if 'op' == 'hash-map': assert False, 'bug'
    
    if op == 'if':
        assert len(x) == 4
        test, conseq, alt = x[1:4]
        if do_log: print('case if: x',x)
        exp = (conseq if eval_hoppl(test, env, sigma,do_log=do_log)[0] else alt) # be careful to get return in [0] and not sigma!!!
        if do_log: print('case if: exp',exp)
        return eval_hoppl(exp, env, sigma,do_log=do_log)

    if op == 'sample':
        if do_log: print('case sample: x',x)
        _, address, exp_distribution = x
        distribution, sigma = eval_hoppl(exp_distribution, env, sigma,do_log=do_log)
        if do_log: print('case sample: distribution',distribution)
        evaluated_sample = distribution.sample()
        if do_log: print('case sample: evaluated_sample',evaluated_sample)
        return evaluated_sample, sigma

    elif op == 'observe':
        if do_log: print('case observe: (pass)')
        return 'observed', sigma
    elif op == 'push-address':
        return '', sigma
    elif op == 'fn':
        if do_log: print('case fn: args',args)
#         param, body = args
        body = args[0]
        return Procedure(param, body, env, do_log=do_log), sigma # has eval in it
        # param ['alpha', 'x']
        # body ['*', ['push-address', 'alpha', 'addr2'], 'x', 'x']
        # env
    
    else:
        if do_log: print('case else: x',x)
        proc, _ = eval_hoppl(op, env, sigma,do_log=do_log)
        vals = ['']
        if do_log: print('case else: args',args)
        vals.extend([eval_hoppl(arg, env, sigma,do_log=do_log)[0] for arg in args])
        if do_log: print('case else: vals',vals)
        if do_log: print('case Procedure:', proc, vals)

        if isinstance(proc, Procedure): # lambdas, not primitives
            r, _ = proc(*vals)
            if do_log: print('case Procedure: r', r)
        else:
            if do_log: print('case primitives: vals[1:]', vals[1:])
            r = proc(vals[1:]) # primitives
            if do_log: print('case primitives: r', r)
            
        return r, sigma


def get_stream(exp):
    while True:
        ret = evaluate(exp)
        yield ret


def run_deterministic_tests():
    
    for i in range(1,14):

        exp = ast_helper(
            fname='test_{}.daphne'.format(i),
            directory='programs/tests/deterministic')
        truth = load_truth('programs/tests/deterministic/test_{}.truth'.format(i))
        ret = evaluate(exp)
        try:
            assert(is_tol(ret, truth))
        except:
            raise AssertionError('return value {} is not equal to truth {} for exp {}'.format(ret,truth,exp))
        
        print('FOPPL test {} passed'.format(i))
    
        
    for i in range(1,13):

        exp = ast_helper(
            fname='test_{}.daphne'.format(i),
            directory='programs/tests/hoppl-deterministic')
        truth = load_truth('programs/tests/hoppl-deterministic/test_{}.truth'.format(i))
        try:
            ret = evaluate(exp)

            try:
                assert(is_tol(ret, truth))
                print('HOPPL test {} passed'.format(i))
            except:
                print('HOPPL test {} failed tolerance'.format(i))
        except:
            print('HOPPL test {} failed to evaluate'.format(i))



            #raise AssertionError('return value {} is not equal to truth {} for exp {}'.format(ret,truth,exp))
        
        
        
    print('All deterministic tests passed')
    


def run_probabilistic_tests():
    
    num_samples=1e4
    max_p_value = 1e-2
    
    for i in range(1,7):

        exp = ast_helper(
            fname='test_{}.daphne'.format(i),
            directory='programs/tests/probabilistic')
        truth = load_truth('programs/tests/probabilistic/test_{}.truth'.format(i))
        
        stream = get_stream(exp)
        
        p_val = run_prob_test(stream, truth, num_samples)
        
        print('p value', p_val)
        assert(p_val > max_p_value)
    
    print('All probabilistic tests passed')    


def ast_helper(fname,directory):
    sugared_fname = '../prob_prog/hw/hw5/CS532-HW5/{}/{}'.format(directory,fname)
    desugared_ast_json_fname = '/Users/gw/repos/prob_prog/' + sugared_fname.replace('.daphne','.daphne.json')
    if os.path.isfile(desugared_ast_json_fname):
        with open(desugared_ast_json_fname) as f:
            ast = json.load(f)
    else:
        #note: the sugared path that goes into daphne desugar should be with respect to the daphne path!
        ast = daphne(['desugar-hoppl', '-i', sugared_fname]) 

        with open(desugared_ast_json_fname, 'w') as f:
            json.dump(ast, f)
    return ast

if __name__ == '__main__':
    
    run_deterministic_tests()
    run_probabilistic_tests()
    

    for i in range(1,4):
        print(i)
        exp = daphne(['desugar-hoppl', '-i', '../../HW5/programs/{}.daphne'.format(i)])
        print('\n\n\nSample of prior of program {}:'.format(i))
        print(evaluate(exp))        
