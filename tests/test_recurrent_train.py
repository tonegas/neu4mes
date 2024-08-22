import unittest

import sys
import os
# append a new directory to sys.path
sys.path.append(os.getcwd())
from neu4mes import *
relation.CHECK_NAMES = False

import torch

# Linear function
def linear_fun(x,a,b):
    return x*a+b

data_x = np.random.rand(500)*20-10
data_a = 2
data_b = -3
dataset = {'in1': data_x, 'out': linear_fun(data_x,data_a,data_b)}
data_folder = '/tests/data/'

class Neu4mesTrainingTest(unittest.TestCase):
    def test_build_dataset_batch(self):
        input1 = Input('in1')
        out = Input('out')
        rel1 = Fir(input1.tw(0.05))
        y = Output('y', rel1)

        test = Neu4mes(visualizer=None, seed=42)
        test.addModel('y',y)
        test.addMinimize('pos', out.next(), y)
        test.neuralizeModel(0.01)

        test.loadData(name='dataset',source=dataset)

        training_params = {}
        training_params['train_batch_size'] = 4
        training_params['val_batch_size'] = 4
        training_params['test_batch_size'] = 1
        training_params['learning_rate'] = 0.1
        training_params['num_of_epochs'] = 5
        test.trainModel(splits=[70,20,10], close_loop={'in1':'y'}, prediction_samples=5, step=1, training_params = training_params)

        self.assertEqual(346,test.n_samples_train) ## ((500 - 5) * 0.7)  = 346
        self.assertEqual(99,test.n_samples_val) ## ((500 - 5) * 0.2)  = 99
        self.assertEqual(50,test.n_samples_test) ## ((500 - 5) * 0.1)  = 50
        self.assertEqual(495,test.num_of_samples['dataset'])
        self.assertEqual(4,test.train_batch_size)
        self.assertEqual(4,test.val_batch_size)
        self.assertEqual(1,test.test_batch_size)
        self.assertEqual(5,test.num_of_epochs)
        self.assertEqual(0.1,test.learning_rate)

    def test_recurrent_train_one_variable(self):
        x = Input('in1')
        p = Parameter('p', dimensions=1, sw=1, values=[[1.0]])
        fir = Fir(parameter=p)(x.last())
        out = Output('out', fir)

        test = Neu4mes(visualizer=None, seed=42)
        test.addModel('out',out)
        test.addMinimize('pos', x.next(), out)
        test.neuralizeModel(0.01)

        test.loadData(name='dataset',source=dataset)

        training_params = {}
        training_params['train_batch_size'] = 4
        training_params['val_batch_size'] = 4
        training_params['test_batch_size'] = 1
        training_params['learning_rate'] = 0.1
        training_params['num_of_epochs'] = 50
        test.trainModel(splits=[100,0,0], close_loop={'in1':'out'}, prediction_samples=3, step=1, training_params = training_params)
    
    def test_recurrent_train_single_close_loop(self):
        data_x = np.array(list(range(1,101,1)), dtype=np.float32)
        dataset = {'x': data_x, 'y':2*data_x}
        
        x = Input('x')
        y = Input('y')
        out = Output('out', Fir(x.last()))

        test = Neu4mes(visualizer=None, seed=42)
        test.addModel('out',out)
        test.addMinimize('pos', y.last(), out)
        test.neuralizeModel(0.01)

        test.loadData(name='dataset',source=dataset)

        training_params = {}
        training_params['train_batch_size'] = 4
        training_params['val_batch_size'] = 4
        training_params['test_batch_size'] = 1
        training_params['learning_rate'] = 0.01
        training_params['num_of_epochs'] = 50
        test.trainModel(splits=[80,20,0], close_loop={'x':'out'}, prediction_samples=3, step=3, training_params = training_params)
    
    def test_recurrent_train_multiple_close_loop(self):
        data_x = np.array(list(range(1,101,1)), dtype=np.float32)
        dataset = {'x': data_x, 'y':2*data_x}
        
        x = Input('x')
        y = Input('y')
        out_x = Output('out_x', Fir(x.last()))
        out_y = Output('out_y', Fir(y.last()))

        test = Neu4mes(visualizer=None, seed=42)
        test.addModel('out_x',out_x)
        test.addModel('out_y',out_y)
        test.addMinimize('pos_x', x.next(), out_x)
        test.addMinimize('pos_y', y.next(), out_y)
        test.neuralizeModel(0.01)

        test.loadData(name='dataset',source=dataset)

        training_params = {}
        training_params['train_batch_size'] = 4
        training_params['val_batch_size'] = 4
        training_params['test_batch_size'] = 1
        training_params['learning_rate'] = 0.01
        training_params['num_of_epochs'] = 50

        print('test before train: ', test(inputs={'x':[100,101,102,103,104], 'y':[200,202,204,206,208]}))
        test.trainModel(splits=[80,20,0], close_loop={'x':'out_x', 'y':'out_y'}, prediction_samples=3, step=1, training_params = training_params)
        print('test after train: ', test(inputs={'x':[100,101,102,103,104], 'y':[200,202,204,206,208]}))

    def test_recurrent_train_one_state_variable(self):
        x = Input('x')
        x_state = State('x_state')
        p = Parameter('p', dimensions=1, sw=1, values=[[1.0]])
        rel_x = Fir(parameter=p)(x_state.last())
        rel_x.update(x_state)
        out = Output('out', rel_x)

        test = Neu4mes(visualizer=None, seed=42)
        test.addModel('out',out)
        test.addMinimize('pos_x', x.next(), out)
        test.neuralizeModel(0.01)

        result = test(inputs={'x': [2], 'x_state':[1]})
        self.assertEqual(test.model.states['x_state'], torch.tensor(result['out']))
        result = test(inputs={'x': [2]})
        self.assertEqual(test.model.states['x_state'], torch.tensor(1.0))

    def test_recurrent_train_only_state_variables(self):
        x_state = State('x_state')
        p = Parameter('p', dimensions=1, tw=0.03, values=[[1.0], [1.0], [1.0]])
        rel_x = Fir(parameter=p)(x_state.tw(0.03))
        rel_x.update(x_state)
        out = Output('out', rel_x)

        test = Neu4mes(visualizer=None, seed=42)
        test.addModel('out',out)
        test.neuralizeModel(0.01)

        result = test(inputs={'x_state':[1, 2, 3]})
        self.assertEqual(test.model.states['x_state'].numpy().tolist(), [[[2.],[3.],[6.]]])
        result = test()
        self.assertEqual(test.model.states['x_state'].numpy().tolist(), [[[3.],[6.],[11.]]])

if __name__ == '__main__':
    unittest.main()