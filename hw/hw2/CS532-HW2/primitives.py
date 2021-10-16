import torch

def get_primitive(vector_and_index):
    vector, index = vector_and_index
    return vector[index.long()]

primitives_d = {
    '/': lambda x: torch.divide(x[0], x[1]),
    '*': lambda x: torch.multiply(x[0], x[1]),
    '+': lambda x: torch.add(x[0], x[1]),
    '-': lambda x: torch.subtract(x[0],x[1]),
    'sqrt': lambda x: torch.sqrt(x[0]),
    'vector': torch.tensor,
    'get' : get_primitive,
}