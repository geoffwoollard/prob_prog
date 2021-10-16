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
	# TODO make work on hash-map
    vector, index, overwritevalue = vector_index_overwritevalue
    vector[index.long()] = overwritevalue
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

primitives_d = {    
    '+': lambda x: torch.add(x[0], x[1]),
    '-': lambda x: torch.subtract(x[0],x[1]),
    '*': lambda x: torch.multiply(x[0], x[1]),
    '/': lambda x: torch.divide(x[0], x[1]),
    'sqrt': lambda x: torch.sqrt(x[0]),
    'vector': torch.tensor,
    'get' : get_primitive,
    'put' : put_primitive,
    'first' : first_primitive,
    'last' : last_primitive,
    'append' : torch.cat,
    'hash-map' : hash_map_primitive,
}