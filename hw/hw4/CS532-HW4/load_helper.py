import json
import os
from daphne import daphne

def ast_helper(fname,directory='programs'):
    os.chdir('/Users/gw/repos/prob_prog/hw/hw4/CS532-HW4/{}/'.format(directory))
    sugared_fname = '../prob_prog/hw/hw4/CS532-HW4/{}/{}'.format(directory,fname)
    desugared_ast_json_fname = '/Users/gw/repos/prob_prog/' + sugared_fname.replace('.daphne','_ast.json')
    if os.path.isfile(desugared_ast_json_fname):
        with open(desugared_ast_json_fname) as f:
            ast = json.load(f)
    else:
        #note: the sugared path that goes into daphne desugar should be with respect to the daphne path!
        ast = daphne(['desugar', '-i', sugared_fname]) 
        with open(desugared_ast_json_fname, 'w') as f:
            json.dump(ast, f)
    return ast


def graph_helper(fname,directory='programs'):
    os.chdir('/Users/gw/repos/prob_prog/hw/hw4/CS532-HW4/{}/'.format(directory))
    sugared_fname = '../prob_prog/hw/hw4/CS532-HW4/{}/{}'.format(directory,fname)
    graph_json_fname = '/Users/gw/repos/prob_prog/' + sugared_fname.replace('.daphne','_graph.json')
    if os.path.isfile(graph_json_fname):
        with open(graph_json_fname) as f:
            graph = json.load(f)
    else:
        #note: the sugared path that goes into daphne desugar should be with respect to the daphne path!
        graph = daphne(['graph', '-i', sugared_fname]) 
        with open(graph_json_fname, 'w') as f:
            json.dump(graph, f)
    return graph