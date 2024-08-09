import sys
import os
# append a new directory to sys.path
sys.path.append(os.getcwd())

from neu4mes import *

example = 5

if example == 1:
    print('#### EXAMPLE 1 - NON Recurrent Training ####')
    x = Input('x') 
    F = Input('F')
    x_state = State('x_state')
    x_out = Fir(x_state.tw(0.5))+F.last()
    x_out.update(x_state)
    out = Output('out',x_out)

    mass_spring_damper = Neu4mes(seed=42)
    mass_spring_damper.addModel(out)
    mass_spring_damper.addMinimize('error', out, x.next())

    mass_spring_damper.neuralizeModel(0.1)

    data_struct = ['time',('x','x_state'),'x_s','F']
    data_folder = './examples/data/'
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
    mass_spring_damper.addMinimize('error', out, x.next())

    mass_spring_damper.neuralizeModel(0.1)

    data_struct = ['time',('x','x_state'),'x_s','F']
    data_folder = './examples/data/'
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

    print('finale state: ', mass_spring_damper.model.states)
    mass_spring_damper.clear_state()
    print('state clear: ', mass_spring_damper.model.states)

elif example == 3:
    print('#### EXAMPLE 3 - NON Recurrent Training (2 state variables) ####')
    x = Input('x') 
    x_state = State('x_state')
    y_state = State('y_state')
    x_out = Fir(x_state.tw(0.5))
    y_out = Fir(y_state.tw(0.5))
    x_out.update(x_state)
    y_out.update(y_state)
    out = Output('out',x_out+y_out)

    mass_spring_damper = Neu4mes(seed=42)
    mass_spring_damper.addModel(out)
    mass_spring_damper.addMinimize('error', out, x.next())

    mass_spring_damper.neuralizeModel(0.1)

    data_struct = ['time',('x','x_state'),'x_s','F']
    data_folder = './examples/data/'
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
    mass_spring_damper.trainModel(splits=[100,0,0], shuffle_data=False, training_params=params)

elif example == 4:
    print('#### EXAMPLE 4 - Recurrent Training (2 state variables) ####')
    x = Input('x') 
    F = Input('F')
    x_state = State('x_state')
    y_state = State('y_state')
    x_out = Fir(x_state.tw(0.5))
    y_out = Fir(y_state.tw(0.5))
    x_out.update(x_state)
    y_out.update(y_state)
    out = Output('out',x_out+y_out)

    mass_spring_damper = Neu4mes(seed=42)
    mass_spring_damper.addModel(out)
    mass_spring_damper.addMinimize('error', out, x.next())

    mass_spring_damper.neuralizeModel(0.1)

    data_struct = ['time',('x','x_state'),'x_s','F']
    data_folder = './examples/data/'
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

elif example == 5:
    print('#### EXAMPLE 5 - Recurrent Training with multi-dimensional output and multi-window ####')
    x = Input('x', dimensions=3) 
    F = Input('F')
    x_state = State('x_state', dimensions=3)
    y_state = State('y_state', dimensions=3)
    x_out = Linear(output_dimension=3)(x_state.tw(0.5))
    y_out = Linear(output_dimension=3)(y_state.tw(0.5))
    x_out.update(x_state)
    y_out.update(y_state)
    out = Output('out',x_out+y_out)

    mass_spring_damper = Neu4mes()
    mass_spring_damper.addModel(out)
    mass_spring_damper.addMinimize('error', out, x.next())

    mass_spring_damper.neuralizeModel(0.1)

    data_struct = ['time',('x','x_state'),'x_s','F']
    data_folder = './examples/data/'
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

elif example == 6:
    print('#### EXAMPLE 6 - Recurrent Training with state variables and close_loop ####')

    x = Input('x') 
    F = Input('F')
    x_state = State('x_state')
    y_state = State('y_state')
    x_out = Fir(x_state.tw(0.3))
    y_out = Fir(y_state.tw(0.3))
    x_out.update(x_state)
    y_out.update(y_state)
    out = Output('out',x_out+y_out+F.last())

    mass_spring_damper = Neu4mes()
    mass_spring_damper.addModel(out)
    mass_spring_damper.addMinimize('error', out, x.next())

    mass_spring_damper.neuralizeModel(0.1)

    data_struct = ['time',('x','x_state'),'x_s','F']
    data_folder = './examples/data/'
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
    mass_spring_damper.trainModel(splits=[70,20,10], close_loop={'F':'out'}, prediction_horizon=0.3, shuffle_data=False, training_params=params)