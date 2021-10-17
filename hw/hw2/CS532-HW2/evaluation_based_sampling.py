import os
from daphne import daphne
from tests import is_tol, run_prob_test,load_truth
import torch
from primitives import primitives_d, distributions_d

number = (int,float)
        

def evaluate_program(ast,sig=None):
    """Evaluate a program as desugared by daphne, generate a sample from the prior
    Args:
        ast: json FOPPL program
    Returns: sample from the prior of ast
    """
    ast0 = ast[0]
    res = evaluate(ast0)
    return res, sig


def evaluate(e):
    if type(e) != list or len(e) == 1:
        if type(e) == list:
            e = e[0]
        if isinstance(e, number):
            return torch.tensor([float(e)])
        elif e in list(primitives_d.keys()):
            return e
        elif e in list(distributions_d.keys()):
            return e
        elif torch.is_tensor(e):
            return e
        else:
            assert False
    elif e[0] == 'sample':
        dist = evaluate(e[1])
        return dist.sample()
        
    else:
        cs = []
        for ei in e:
            c = evaluate(ei)
            cs.append(c)
        if cs[0] in primitives_d:
            return primitives_d[cs[0]](cs[1:])
        elif cs[0] in distributions_d:
            return distributions_d[cs[0]](cs[1:])
        else:
            assert False



def get_stream(ast):
    """Return a stream of prior samples"""
    while True:
        yield evaluate_program(ast)[0].item() # just return value, not context


def run_deterministic_tests():
    
    for i in range(1,14):
        #note: this path should be with respect to the daphne path!
        os.chdir('/Users/gw/repos/prob_prog/hw/hw2/CS532-HW2/')
        ast = daphne(['desugar', '-i', '../prob_prog/hw/hw2/CS532-HW2/programs/tests/deterministic/test_{}.daphne'.format(i)])
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
    
    for i in range(1,2):
        #note: this path should be with respect to the daphne path!        
        ast = daphne(['desugar', '-i', '../prob_prog/hw/hw2/CS532-HW2/programs/tests/probabilistic/test_{}.daphne'.format(i)])
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