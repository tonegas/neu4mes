
def init_constant(indexes, params_size, dict_param = {'value':1}):
    return dict_param['value']

def init_negexp(indexes, params_size, dict_param = {'size_index':0, 'first_value':1, 'lambda':3}):
    import numpy as np
    size_index = dict_param['size_index']
    x = 0 if params_size[size_index]-1 == 0 else indexes[size_index]/(params_size[size_index]-1)
    return dict_param['first_value']*np.exp(-dict_param['lambda']*(1-x))

def init_lin(indexes, params_size, dict_param = {'size_index':0, 'first_value':1, 'last_value':0}):
    size_index = dict_param['size_index']
    x = 0 if params_size[size_index]-1 == 0 else indexes[size_index]/(params_size[size_index]-1)
    return (dict_param['last_value'] - dict_param['first_value']) *(1-x) + dict_param['first_value']