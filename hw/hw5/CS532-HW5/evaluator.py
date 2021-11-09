from primitives import env as penv
from daphne import daphne
from tests import is_tol, run_prob_test,load_truth
from pyrsistent import pmap,plist

# def standard_env():
#     "An environment with some Scheme standard procedures."
#     env = pmap(penv)
#     env = env.update({'alpha' : ''}) 

#     return env



# def evaluate(exp, env=None): #TODO: add sigma, or something

#     if env is None:
#         env = standard_env()

#     #TODO:
#     return    


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
    def __init__(self, parms, body, env):
        self.parms, self.body, self.env = parms, body, env
    def __call__(self, *args): 
        new_env = copy.deepcopy(self.env)
        return eval(self.body, Env(self.parms, args, new_env)) # [0]

penv = {
    '+': torch.add,
    '*':torch.mul,
#     'push-address':push_addr
}


def standard_env():
    "An environment with some Scheme standard procedures."
    env = Env(penv.keys(), penv.values())
    return env


def eval(x,env=standard_env(),sigma=None):

    print('x',x)
    if isinstance(x,str):
        return env.find(x)[x], sigma
    elif not isinstance(x,list):
        return torch.tensor(x),sigma

    
    op, param, *args = x
    
    if False:
        pass
    
    elif op == 'push-address':
        return '', sigma
    elif op == 'fn':
        print('args',args)
        body = args[0]
        return Procedure(param, body, env), sigma # has eval in it
        # param ['alpha', 'x']
        # body ['*', ['push-address', 'alpha', 'addr2'], 'x', 'x']
        # env
    
    else:
        print('in else. x',x)
        proc, _ = eval(op,env,sigma)
        vals = ['']
        vals.extend([eval(arg, env, sigma)[0] for arg in args])
        print('vals',vals)

        if isinstance(proc, Procedure): # lambdas, not primitives
            print('in Procedure',proc, vals)
            r = proc(*vals)
        else:
            r = proc(*vals[1:]) # primitives
            print('in primitives',proc, vals)
            
        return r


def get_stream(exp):
    while True:
        yield evaluate(exp)


def run_deterministic_tests():
    
    for i in range(1,14):

        exp = daphne(['desugar-hoppl', '-i', '../../HW5/programs/tests/deterministic/test_{}.daphne'.format(i)])
        truth = load_truth('programs/tests/deterministic/test_{}.truth'.format(i))
        ret = evaluate(exp)
        try:
            assert(is_tol(ret, truth))
        except:
            raise AssertionError('return value {} is not equal to truth {} for exp {}'.format(ret,truth,exp))
        
        print('FOPPL Tests passed')
        
    for i in range(1,13):

        exp = daphne(['desugar-hoppl', '-i', '../../HW5/programs/tests/hoppl-deterministic/test_{}.daphne'.format(i)])
        truth = load_truth('programs/tests/hoppl-deterministic/test_{}.truth'.format(i))
        ret = evaluate(exp)
        try:
            assert(is_tol(ret, truth))
        except:
            raise AssertionError('return value {} is not equal to truth {} for exp {}'.format(ret,truth,exp))
        
        print('Test passed')
        
    print('All deterministic tests passed')
    


def run_probabilistic_tests():
    
    num_samples=1e4
    max_p_value = 1e-2
    
    for i in range(1,7):
        exp = daphne(['desugar-hoppl', '-i', '../../HW5/programs/tests/probabilistic/test_{}.daphne'.format(i)])
        truth = load_truth('programs/tests/probabilistic/test_{}.truth'.format(i))
        
        stream = get_stream(exp)
        
        p_val = run_prob_test(stream, truth, num_samples)
        
        print('p value', p_val)
        assert(p_val > max_p_value)
    
    print('All probabilistic tests passed')    



if __name__ == '__main__':
    
    run_deterministic_tests()
    run_probabilistic_tests()
    

    for i in range(1,4):
        print(i)
        exp = daphne(['desugar-hoppl', '-i', '../../HW5/programs/{}.daphne'.format(i)])
        print('\n\n\nSample of prior of program {}:'.format(i))
        print(evaluate(exp))        
