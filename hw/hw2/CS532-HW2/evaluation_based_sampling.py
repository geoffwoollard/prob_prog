from daphne import daphne
from tests import is_tol, run_prob_test,load_truth
import torch

two_arg_primitives = {
    '+':torch.add,
    '-':torch.subtract,
    '*':torch.multiply,
    '/':torch.divide
    }
    
one_arg_primitives = {
    'sqrt':torch.sqrt
}
        
def evaluate_program(ast):
    """Evaluate a program as desugared by daphne, generate a sample from the prior
    Args:
        ast: json FOPPL program
    Returns: sample from the prior of ast
    """

    # constants

    s,l = {},{}
    c = one_arg_primitives.keys() + two_arg_primitives.keys()
    e = ast[0]

    if e[0] in c:
        E,s = base_case(e,s,l)
        return E,s

    return None


def make_tensor(v):
    return torch.tensor(v)


def base_case(e,s,l):
    if e[0] in two_arg_primitives.keys():
        e0,arg1,arg2 = e
        E = two_arg_primitives[e0](make_tensor(arg1),make_tensor(arg2))
    elif e[0] in one_arg_primitives.keys():
        e0,arg1 = e
        E = one_arg_primitives[e0](make_tensor(arg1))
    return E,s


def get_stream(ast):
    """Return a stream of prior samples"""
    while True:
        yield evaluate_program(ast)


def run_deterministic_tests():
    
    for i in range(1,14):
        #note: this path should be with respect to the daphne path!
        ast = daphne(['desugar', '-i', '../prob_prog/hw/hw2/CS532-HW2/programs/tests/deterministic/test_{}.daphne'.format(i)])
        truth = load_truth('../prob_prog/hw/hw2/CS532-HW2/programs/tests/deterministic/test_{}.truth'.format(i))
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
        #note: this path should be with respect to the daphne path!        
        ast = daphne(['desugar', '-i', '../CS532-HW2/programs/tests/probabilistic/test_{}.daphne'.format(i)])
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