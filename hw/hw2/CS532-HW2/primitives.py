import torch
import numpy as np

number = (float, int)
distribution_types = (
    torch.distributions.Normal,
    torch.distributions.Beta,
    torch.distributions.Uniform,
    torch.distributions.Exponential,
    torch.distributions.Categorical)


def two_arg_op_primitive(op,arg1_arg2):
    arg1, arg2 = arg1_arg2
    return op(arg1, arg2)


def add_primitive(arg1_arg2):
    return two_arg_op_primitive(torch.add,arg1_arg2)


def subtract_primitive(arg1_arg2):
    return two_arg_op_primitive(torch.subtract,arg1_arg2)


def multiply_primitive(arg1_arg2):
    return two_arg_op_primitive(torch.multiply,arg1_arg2)


def divide_primitive(arg1_arg2):
    return two_arg_op_primitive(torch.divide,arg1_arg2)


def one_arg_op_primitive(op,arg):
    arg0 = arg[0] # because list of len one passed, i.e. [arg0]
    return op(arg0)


def sqrt_primitive(arg):
    return one_arg_op_primitive(torch.sqrt,arg)


def get_primitive(vector_and_index):
    vector, index = vector_and_index
    if isinstance(vector,dict):
        return vector[index.item()]
    elif torch.is_tensor(vector):
        return vector[index.long()]
    elif isinstance(vector,list):
        index_int = int(index)
        assert np.isclose(index_int,index) # TODO: use native pytorch
        return vector[index]
    else:
        assert False,  'vector type {} case not implemented'.format(type(vector))


def put_primitive(vector_index_overwritevalue):
    vector, index, overwritevalue = vector_index_overwritevalue
    if isinstance(vector,dict):
        vector[index.item()] = overwritevalue
    elif torch.is_tensor(vector):
        vector[index.long()] = overwritevalue
    else:
        assert False,  'vector type {} case not implemented'.format(type(vector))
    return vector


def return_idx_primitive(vector,idx_i,idx_f):
    return vector[0][idx_i:idx_f]


def first_primitive(vector):
    return return_idx_primitive(vector,idx_i=0,idx_f=1)


def second_primitive(vector):
    return return_idx_primitive(vector,idx_i=1,idx_f=2)


def last_primitive(vector):
    return return_idx_primitive(vector,idx_i=-1,idx_f=None)


def nth_primitive(vector_nth):
    vector, nth = vector_nth
    return return_idx_primitive(vector,idx_i=nth,idx_f=nth+1)


def hash_map_primitive(hash_pairs):
    keys = hash_pairs[::2]
    # dict keys as tensors problematic. can make but lookup fails on fresh but equivalent tensor (bc memory look up?) 
    keys = [tensor_key.item() for tensor_key in keys] 
    vals = hash_pairs[1::2]
    return dict(zip(keys, vals))


def gt_primitive(consequent_alternative):
    return two_arg_op_primitive(torch.gt,consequent_alternative)


def lt_primitive(consequent_alternative):
    return two_arg_op_primitive(torch.lt,consequent_alternative)


def ge_primitive(consequent_alternative):
    return two_arg_op_primitive(torch.ge,consequent_alternative)


def le_primitive(consequent_alternative):
    return two_arg_op_primitive(torch.le,consequent_alternative)


def eq_primitive(consequent_alternative):
    return two_arg_op_primitive(torch.eq,consequent_alternative)


def rest_primative(vector):
    return vector[0][1:]


def freshvar_primitive(arg):
    return None


def vector_primitive(vector):
    ret = list()
    for e in vector:
        try:
            ret.append(e.tolist())
        except:
            ret.append(e)
    try:
        return torch.FloatTensor(ret)
    except:
        return ret

def conj_primitive(args):
    # TODO: write test
    base = args[0]
    rest = args[1:]
    for i in rest:
        base = torch.cat(base, i)

            


# NB: these functions take a list [c0] or [c0, c1, ..., cn]
# rely on user to not write something non-sensitcal that will fail (e.g. ["+",1,2,3])
primitives_d = {
    '+': add_primitive,
    '-': subtract_primitive,
    '/': divide_primitive,
    '*': multiply_primitive,
    'sqrt': sqrt_primitive,
    'vector': vector_primitive,
    'get' : get_primitive,
    'put' : put_primitive,
    'first' : first_primitive,
    'second' : second_primitive,
    'last' : last_primitive,
    'nth' : nth_primitive,
    'append' : torch.cat,
    'hash-map' : hash_map_primitive,
    '>':gt_primitive,
    '<':lt_primitive,
    '>=':ge_primitive,
    '<=':le_primitive,
    '==':eq_primitive,
    'rest' : rest_primative,
    # '_':freshvar_primitive,
    'mat-transpose': lambda a: a[0].T,
    'mat-tanh': torch.tanh,
    'mat-mul': lambda a: torch.matmul(a[0],a[1]),
    'mat-add': add_primitive,
    'mat-repmat': lambda a: a[0].repeat((int(a[1].item()), int(a[2].item()))),
}


def normal(mean_std):
    return two_arg_op_primitive(torch.distributions.Normal,mean_std)


def beta(alpha_beta):
    return two_arg_op_primitive(torch.distributions.Beta,alpha_beta)


def exponential(lam):
    return one_arg_op_primitive(torch.distributions.Exponential,lam)


def uniform(low_hi):
    return two_arg_op_primitive(torch.distributions.Uniform,low_hi)


def discrete(prob_vector):
    return one_arg_op_primitive(torch.distributions.Categorical,prob_vector)


distributions_d = {
    'normal': normal,
    'beta': beta,
    'exponential': exponential,
    'uniform': uniform,
    'discrete': discrete,
}

class Function:

    def __init__(self, name, variables, proc):
        self.name = name
        self.variables = variables
        self.proc = proc