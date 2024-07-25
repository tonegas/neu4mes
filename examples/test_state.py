import sys
import os
# append a new directory to sys.path
sys.path.append(os.getcwd())

from neu4mes import *

example = 4

if example == 1:
    print('#### EXAMPLE 1 - NON Recurrent Training ####')
    x = Input('x') 
    F = Input('F')
    x_state = State('x_state')
    x_out = Fir(x_state.tw(0.5))+F.last()
    x_out.update(x_state)
    out = Output('out',x_out)

    mass_spring_damper = Neu4mes()
    mass_spring_damper.addModel(out)
    mass_spring_damper.minimizeError('error', out, x.next())

    ## FAKE JSON 
    # mass_spring_damper.model_def = {'Functions': {},
    #                                 'Inputs': {'F': {'dim': 1, 'sw': [-1, 0], 'tw': [0, 0]},
    #                                            'x': {'dim': 1, 'sw': [0, 1], 'tw': [0, 0]}},
    #                                 'States': {'x_state': {'dim': 1, 'sw': [0, 0], 'tw': [-0.5, 0], 'update':'Add8'}},
    #                                 'Outputs': {'out': 'Add8'},
    #                                 'Parameters': {'PFir3': {'dim': 1, 'tw': 0.5}},
    #                                 'Relations': {'Add8': ['Add', ['Fir5', 'SamplePart7']],
    #                                             'Fir5': ['Fir', ['TimePart4'], 'PFir3'],
    #                                             'SamplePart11': ['SamplePart', ['x'], [0, 1]],
    #                                             'SamplePart7': ['SamplePart', ['F'], [-1, 0]],
    #                                             'TimePart4': ['TimePart', ['x_state'], [-0.5, 0]]},
    #                                 'SampleTime': 0.1}
    mass_spring_damper.neuralizeModel(0.1)

    data_struct = ['time',('x','x_state'),'x_s','F']
    data_folder = './data/'
    mass_spring_damper.loadData(name='dataset', source=data_folder, skiplines=1, format=data_struct)

    print('F (first):', mass_spring_damper.data['dataset']['F'][0])
    print('F (last):', mass_spring_damper.data['dataset']['F'][-1])
    print('x (first):', mass_spring_damper.data['dataset']['x'][0])
    print('x (last):', mass_spring_damper.data['dataset']['x'][-1])
    print('x_state (first):', mass_spring_damper.data['dataset']['x_state'][0])
    print('x_state (last):', mass_spring_damper.data['dataset']['x_state'][-1])
    
    # Training non ricorrente
    params = {'num_of_epochs': 1, 
          'train_batch_size': 4, 
          'val_batch_size':4, 
          'test_batch_size':1, 
          'learning_rate':0.001}
    mass_spring_damper.trainModel(splits=[70,20,10], shuffle_data=False, training_params=params)

elif example == 2:
    print('#### EXAMPLE 2 - Recurrent Training ####')
    x = Input('x') 
    F = Input('F')
    x_state = State('x_state')
    x_out = Fir(x_state.tw(0.5))+F.last()
    x_out.update(x_state)
    out = Output('out',x_out)

    mass_spring_damper = Neu4mes()
    mass_spring_damper.addModel(out)
    mass_spring_damper.minimizeError('error', out, x.next())

    ## FAKE JSON 
    # mass_spring_damper.model_def = {'Functions': {},
    #                                 'Inputs': {'F': {'dim': 1, 'sw': [-1, 0], 'tw': [0, 0]},
    #                                            'x': {'dim': 1, 'sw': [0, 1], 'tw': [0, 0]}},
    #                                 'States': {'x_state': {'dim': 1, 'sw': [0, 0], 'tw': [-0.5, 0], 'update':'Add8'}},
    #                                 'Outputs': {'out': 'Add8'},
    #                                 'Parameters': {'PFir3': {'dim': 1, 'tw': 0.5}},
    #                                 'Relations': {'Add8': ['Add', ['Fir5', 'SamplePart7']],
    #                                             'Fir5': ['Fir', ['TimePart4'], 'PFir3'],
    #                                             'SamplePart11': ['SamplePart', ['x'], [0, 1]],
    #                                             'SamplePart7': ['SamplePart', ['F'], [-1, 0]],
    #                                             'TimePart4': ['TimePart', ['x_state'], [-0.5, 0]]},
    #                                 'SampleTime': 0.1}
    mass_spring_damper.neuralizeModel(0.1)

    data_struct = ['time',('x','x_state'),'x_s','F']
    data_folder = './data/'
    mass_spring_damper.loadData(name='dataset', source=data_folder, skiplines=1, format=data_struct)

    print('F (first):', mass_spring_damper.data['dataset']['F'][0])
    print('F (last):', mass_spring_damper.data['dataset']['F'][-1])
    print('x (first):', mass_spring_damper.data['dataset']['x'][0])
    print('x (last):', mass_spring_damper.data['dataset']['x'][-1])
    print('x_state (first):', mass_spring_damper.data['dataset']['x_state'][0])
    print('x_state (last):', mass_spring_damper.data['dataset']['x_state'][-1])
    
    # Training ricorrente
    params = {'num_of_epochs': 1, 
          'train_batch_size': 4, 
          'val_batch_size':4, 
          'test_batch_size':1, 
          'learning_rate':0.001}
    mass_spring_damper.trainModel(splits=[70,20,10], prediction_horizon=0.2, shuffle_data=False, training_params=params)

elif example == 3:
    print('#### EXAMPLE 3 - NON Recurrent Training (2 state variables) ####')
    x = Input('x') 
    F = Input('F')
    x_state = State('x_state')
    y_state = State('y_state')
    x_out = Fir(x_state.tw(0.5))
    y_out = Fir(y_state.tw(0.5))
    x_out.update(x_state)
    y_out.update(y_state)
    out = Output('out',x_out+y_out)

    mass_spring_damper = Neu4mes()
    mass_spring_damper.addModel(out)
    mass_spring_damper.minimizeError('error', out, x.next())

    ## FAKE JSON 
    # mass_spring_damper.model_def = {'Functions': {},
    #                                 'Inputs': {'x': {'dim': 1, 'sw': [0, 1], 'tw': [0, 0]}},
    #                                 'States': {'x_state': {'dim': 1, 'sw': [0, 0], 'tw': [-0.5, 0], 'update':'Fir5'},
    #                                            'y_state': {'dim': 1, 'sw': [0, 0], 'tw': [-0.5, 0], 'update':'Fir8'}},
    #                                 'Outputs': {'out': 'Add9'},
    #                                 'Parameters': {'PFir3': {'dim': 1, 'tw': 0.5}, 'PFir4': {'dim': 1, 'tw': 0.5}},
    #                                 'Relations': {'Add9': ['Add', ['Fir5', 'Fir8']],
    #                                             'Fir5': ['Fir', ['TimePart4'], 'PFir3'],
    #                                             'Fir8': ['Fir', ['TimePart7'], 'PFir4'],
    #                                             'SamplePart12': ['SamplePart', ['x'], [0, 1]],
    #                                             'TimePart4': ['TimePart', ['x_state'], [-0.5, 0]],
    #                                             'TimePart7': ['TimePart', ['y_state'], [-0.5, 0]]},
    #                                 'SampleTime': 0.1}
    mass_spring_damper.neuralizeModel(0.1)

    data_struct = ['time',('x','x_state'),'x_s','F']
    data_folder = './data/'
    mass_spring_damper.loadData(name='dataset', source=data_folder, skiplines=1, format=data_struct)

    print('x (first):', mass_spring_damper.data['dataset']['x'][0])
    print('x (last):', mass_spring_damper.data['dataset']['x'][-1])
    print('x_state (first):', mass_spring_damper.data['dataset']['x_state'][0])
    print('x_state (last):', mass_spring_damper.data['dataset']['x_state'][-1])
    
    # Training non ricorrente
    params = {'num_of_epochs': 1, 
          'train_batch_size': 4, 
          'val_batch_size':4, 
          'test_batch_size':1, 
          'learning_rate':0.001}
    mass_spring_damper.trainModel(splits=[70,20,10], shuffle_data=False, training_params=params)

elif example == 4:
    print('#### EXAMPLE 4 - Recurrent Training (2 state variables) ####')
    x = Input('x') 
    F = Input('F')
    x_state = State('x_state')
    y_state = State('y_state')
    x_out = Fir(x_state.tw(0.5))
    y_out = Fir(y_state.tw(0.5))
    #x_out.update(x_state)
    y_out.update(y_state)
    out = Output('out',x_out+y_out)

    mass_spring_damper = Neu4mes()
    mass_spring_damper.addModel(out)
    mass_spring_damper.minimizeError('error', out, x.next())

    ## FAKE JSON 
    # mass_spring_damper.model_def = {'Functions': {},
    #                                 'Inputs': {'x': {'dim': 1, 'sw': [0, 1], 'tw': [0, 0]}},
    #                                 'States': {'x_state': {'dim': 1, 'sw': [0, 0], 'tw': [-0.5, 0], 'update':'Fir5'},
    #                                            'y_state': {'dim': 1, 'sw': [0, 0], 'tw': [-0.5, 0], 'update':'Fir8'}},
    #                                 'Outputs': {'out': 'Add9'},
    #                                 'Parameters': {'PFir3': {'dim': 1, 'tw': 0.5}, 'PFir4': {'dim': 1, 'tw': 0.5}},
    #                                 'Relations': {'Add9': ['Add', ['Fir5', 'Fir8']],
    #                                             'Fir5': ['Fir', ['TimePart4'], 'PFir3'],
    #                                             'Fir8': ['Fir', ['TimePart7'], 'PFir4'],
    #                                             'SamplePart12': ['SamplePart', ['x'], [0, 1]],
    #                                             'TimePart4': ['TimePart', ['x_state'], [-0.5, 0]],
    #                                             'TimePart7': ['TimePart', ['y_state'], [-0.5, 0]]},
    #                                 'SampleTime': 0.1}
    mass_spring_damper.neuralizeModel(0.1)

    data_struct = ['time',('x','x_state'),'x_s','F']
    data_folder = './data/'
    mass_spring_damper.loadData(name='dataset', source=data_folder, skiplines=1, format=data_struct)

    print('x (first):', mass_spring_damper.data['dataset']['x'][0])
    print('x (last):', mass_spring_damper.data['dataset']['x'][-1])
    print('x_state (first):', mass_spring_damper.data['dataset']['x_state'][0])
    print('x_state (last):', mass_spring_damper.data['dataset']['x_state'][-1])
    
    # Training non ricorrente
    params = {'num_of_epochs': 1, 
          'train_batch_size': 4, 
          'val_batch_size':4, 
          'test_batch_size':1, 
          'learning_rate':0.01}
    mass_spring_damper.trainModel(splits=[70,20,10], prediction_horizon=0.3, shuffle_data=False, training_params=params)
