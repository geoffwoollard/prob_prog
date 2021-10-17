import torch

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
    
primitives_d = {
    '+': add_primitive,
    '-': subtract_primitive,
    '/': divide_primitive,
    '*': multiply_primitive,
    'sqrt': lambda x: torch.sqrt(x[0]),
    'vector': torch.tensor,
    'get' : get_primitive,
    'put' : put_primitive,
    'first' : first_primitive,
    'last' : last_primitive,
    'append' : torch.cat,
    'hash-map' : hash_map_primitive,
}


def normal(mean_std):
    mean, std = mean_std
    return torch.distributions.Normal(mean,std)

distributions_d = {
    'normal': normal,
}