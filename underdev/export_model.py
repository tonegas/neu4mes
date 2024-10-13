import sys
import os
import torch
# append a new directory to sys.path
sys.path.append(os.getcwd())

from neu4mes import *

# Create neu4mes structure
result_path = os.path.join(os.getcwd(), "results")
test = Neu4mes(seed=0, workspace=result_path)

example = 3

## Test Parameter, FuzzyFy, ParamFun
if example == 1:
    data_x = np.arange(10,20,0.01)
    data_y = np.arange(20,30,0.01)
    a,b = -3.0, 5.0
    dataset = {'x': data_x, 'y': data_y, 'z':a*data_x+b*data_y}

    x = Input('x')
    y = Input('y')
    z = Input('z')

    ## create the relations
    def myFun(K1,p1):
        return K1*p1

    K_x = Parameter('k_x', dimensions=1, tw=1)
    K_y = Parameter('k_y', dimensions=1, tw=1)
    w = Parameter('w', dimensions=1, tw=1)
    t = Parameter('t', dimensions=1, tw=1)
    w_5 = Parameter('w_5', dimensions=1, tw=5)
    t_5 = Parameter('t_5', dimensions=1, tw=5)
    parfun_x = ParamFun(myFun, parameters = [K_x])
    parfun_y = ParamFun(myFun, parameters = [K_y])
    fir_w = Fir(parameter=w_5)(x.tw(5))
    fir_t = Fir(parameter=t_5)(y.tw(5))
    time_part = TimePart(x.tw(5),i=1,j=3)
    sample_select = SampleSelect(x.sw(5),i=1)

    def fuzzyfun(x):
        return torch.tan(x)
    fuzzy = Fuzzify(output_dimension=4, range=[0,4], functions=fuzzyfun)(x.tw(1))

    out = Output('out', parfun_x(x.tw(1))+parfun_y(y.tw(1)))
    out2 = Output('out2', Add(w,x.tw(1))+Add(t,y.tw(1)))
    out3 = Output('out3', Add(fir_w, fir_t))
    out4 = Output('out4', Linear(output_dimension=1)(fuzzy))
    out5 = Output('out5', Fir(time_part)+Fir(sample_select))

    test.addModel('model',[out,out2,out3,out4,out5])
    test.addMinimize('error', z.last(), out, loss_function='rmse')
    test.neuralizeModel()
    test.loadData(name='dataset', source=dataset)

    test.exportJSON()

    params = {'num_of_epochs': 100,
            'train_batch_size': 8, 
            'val_batch_size': 8, 
            'test_batch_size': 1, 
            'learning_rate': 0.01}
    test.trainModel(training_params=params)

    ## Neural network Predict
    sample = test.get_random_samples(dataset='dataset', window=1)
    result = test(sample, sampled=True)
    print('Predicted z: ', result['out'])
    print('True z: ', sample['z'])
    print(f'parameters a = {test.model.all_parameters.k_x} : b = {test.model.all_parameters.k_y}')

    test.exportTracer()

    file_path = os.path.join(test.folder_path, 'tracer_model.py')
    test.importTracer(file_path=file_path)

    result = test(sample, sampled=True)
    print('Predicted z: ', result['out'])
    print('True z: ', sample['z'])
    print(f'parameters a = {test.model.all_parameters.k_x} : b = {test.model.all_parameters.k_y}')

elif example == 2:  ## Test Close Loop
    data_x = np.arange(10,20,0.01)
    data_y = np.arange(20,30,0.01)
    dataset = {'x': data_x, 'y': data_y}

    x = Input('x')
    y = Input('y')

    out = Output('out', Fir(x.last())+Fir(y.last()))

    test.addModel('model',[out])
    test.addMinimize('error', x.next(), out, loss_function='rmse')
    test.neuralizeModel(clear_model=True)
    test.loadData(name='dataset', source=dataset)

    test.exportJSON()

    params = {'num_of_epochs': 20,
            'train_batch_size': 1, 
            'val_batch_size': 1, 
            'test_batch_size': 1, 
            'learning_rate': 0.01}
    test.trainModel(training_params=params, close_loop={'x':'out'})
    test.exportTracer()

elif example == 3: ## Test State Variables
    data_x = np.arange(10,20,0.01)
    data_y = np.arange(20,30,0.01)
    a,b = -3.0, 5.0
    dataset = {'x': data_x, 'y': data_y, 'z':a*data_x+b*data_y}

    x = Input('x') 
    y = Input('y')
    z = Input('z')
    x_state = State('x_state')
    y_state = State('y_state')
    x_out = Fir(x_state.tw(0.5))
    y_out = Fir(y_state.tw(0.5))

    out = Output('out',x_out+y_out)

    result_path = os.path.join(os.getcwd(), "results")
    test = Neu4mes(seed=0, workspace=result_path)
    test.addClosedLoop(x_out, x_state)
    test.addClosedLoop(y_out, y_state)
    test.addModel('model', out)
    test.addMinimize('error', out, z.last())

    test.neuralizeModel(0.1)
    test.exportJSON()
    test.loadData('dataset', source=dataset)
    
    params = {'num_of_epochs':20, 
          'train_batch_size': 1, 
          'val_batch_size':1, 
          'test_batch_size':1, }
    test.trainModel(splits=[70,20,10], lr=0.01, prediction_samples=3, shuffle_data=False, training_params=params)
    test.exportTracer()
