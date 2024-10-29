
def init_constant(indexes, params_size, dict_param = {'value':1}):
    return dict_param['value']

def init_negexp(indexes, params_size, dict_param = {'size_index':0, 'first_value':1, 'lambda':3}):
    import numpy as np
    size_index = dict_param['size_index']
    # check if the size of the list of parameters is 1, to avoid a division by zero
    x = 1 if params_size[size_index]-1 == 0 else indexes[size_index]/(params_size[size_index]-1)
    return dict_param['first_value']*np.exp(-dict_param['lambda']*(1-x))

def init_exp(indexes, params_size, dict_param = {'size_index':0, 'max_value':1, 'lambda':3, 'monotonicity':'decreasing'}):
    import numpy as np
    size_index = dict_param['size_index']
    monotonicity = dict_param['monotonicity']
    if monotonicity == 'increasing':
        # increasing exponential, the 'max_value' is the value at x=1, i.e, at the end of the range
        x = 1 if params_size[size_index]-1 == 0 else indexes[size_index]/(params_size[size_index]-1)
        out = dict_param['max_value']*np.exp(dict_param['lambda']*(x-1))
    elif monotonicity == 'decreasing':
        # decreasing exponential, the 'max_value' is the value at x=0, i.e, at the beginning of the range
        x = 0 if params_size[size_index]-1 == 0 else indexes[size_index]/(params_size[size_index]-1)
        out = dict_param['max_value']*np.exp(-dict_param['lambda']*x)
    else:
        raise ValueError('The parameter monotonicity must be either increasing or decreasing.')
    return out

def init_lin(indexes, params_size, dict_param = {'size_index':0, 'first_value':1, 'last_value':0}):
    size_index = dict_param['size_index']
    x = 0 if params_size[size_index]-1 == 0 else indexes[size_index]/(params_size[size_index]-1)
    return (dict_param['last_value'] - dict_param['first_value']) * x + dict_param['first_value']
