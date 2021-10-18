import torch


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


def return_idx_primitive(vector,idx):
    return vector[0][idx]


def first_primitive(vector):
    return return_idx_primitive(vector,idx=0)


def last_primitive(vector):
    return return_idx_primitive(vector,idx=-1)


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

# NB: these functions take a list [c0] or [c0, c1, ..., cn]
# rely on user to not write something non-sensitcal that will fail (e.g. ["+",1,2,3])
primitives_d = {
    '+': add_primitive,
    '-': subtract_primitive,
    '/': divide_primitive,
    '*': multiply_primitive,
    'sqrt': sqrt_primitive,
    'vector': torch.tensor,
    'get' : get_primitive,
    'put' : put_primitive,
    'first' : first_primitive,
    'last' : last_primitive,
    'append' : torch.cat,
    'hash-map' : hash_map_primitive,
    '>':gt_primitive,
    '<':lt_primitive,
    '>=':ge_primitive,
    '<=':le_primitive,
    '==':eq_primitive,
}


def normal(mean_std):
    return two_arg_op_primitive(torch.distributions.Normal,mean_std)


def beta(alpha_beta):
    return two_arg_op_primitive(torch.distributions.Beta,alpha_beta)


def exponential(lam):
    return one_arg_op_primitive(torch.distributions.Exponential,lam)


def uniform(low_hi):
    return two_arg_op_primitive(torch.distributions.Uniform,low_hi)


distributions_d = {
    'normal': normal,
    'beta': beta,
    'exponential': exponential,
    'uniform': uniform,
}

class Function:

    def __init__(self, name, variables, proc):
        self.name = name
        self.variables = variables
        self.proc = proc