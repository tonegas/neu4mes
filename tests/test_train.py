import unittest
import torch

import sys
import os
# append a new directory to sys.path
sys.path.append(os.getcwd())
from neu4mes import *
from neu4mes import relation

relation.CHECK_NAMES = False

data_folder = os.path.join(os.path.dirname(__file__), 'data/')


def funIn(x, w):
    return x * w


def funOut(x, w):
    return x / w

class Neu4mesTrainingTest(unittest.TestCase):
    def test_network_mass_spring_damper(self):
        x = Input('x')  # Position
        F = Input('F')  # Force

        # List the output of the model
        x_z = Output('x_z', Fir(x.tw(0.3)) + Fir(F.last()))

        # Add the neural model to the neu4mes structure and neuralization of the model
        test = Neu4mes(visualizer=None)
        test.addModel('x_z',x_z)
        test.addMinimize('next-pos', x.z(-1), x_z, 'mse')

        # Create the neural network
        test.neuralizeModel(sample_time=0.05)  # The sampling time depends to the dataset

        # Data load
        data_struct = ['x','F','x2','y2','','A1x','A1y','B1x','B1y','','A2x','A2y','B2x','out','','x3','in1','in2','time']
        test.loadData(name='dataset', source=data_folder, format=data_struct, skiplines=4, delimiter='\t', header=None)
        test.trainModel(splits=[80,10,10])

        self.assertEqual((15-6), test.num_of_samples['dataset'])
        self.assertEqual(round((15-6)*80/100),test.run_training_params['n_samples_train'])
        self.assertEqual(round((15-6)*10/100),test.run_training_params['n_samples_val'])
        self.assertEqual(round((15-6)*10/100),test.run_training_params['n_samples_test'])
        self.assertEqual(round((15-6)*80/100),test.run_training_params['train_batch_size'])
        self.assertEqual(1,test.run_training_params['val_batch_size'])
        self.assertEqual(1,test.run_training_params['test_batch_size'])
        self.assertEqual(100,test.run_training_params['num_of_epochs'])
        self.assertEqual(0.001,test.run_training_params['optimizer_defaults']['lr'])
    
    def test_build_dataset_batch(self):
        input1 = Input('in1')
        output = Input('out')
        rel1 = Fir(input1.tw(0.05))

        test = Neu4mes(visualizer=None)
        test.addMinimize('out', output.z(-1), rel1)
        test.neuralizeModel(0.01)

        data_struct = ['x','F','x2','y2','','A1x','A1y','B1x','B1y','','A2x','A2y','B2x','out','','x3','in1','in2','time']
        test.loadData(name='dataset', source=data_folder, format=data_struct, skiplines=4, delimiter='\t', header=None)
        self.assertEqual((10,5,1),test.data['dataset']['in1'].shape)

        training_params = {}
        training_params['train_batch_size'] = 1
        training_params['val_batch_size'] = 1
        training_params['test_batch_size'] = 1
        training_params['lr'] = 0.1
        training_params['num_of_epochs'] = 5
        test.trainModel(splits=[70,20,10],training_params = training_params)

        # 15 lines in the dataset
        # 5 lines for input + 1 for output -> total of sample 10
        # 10 / 1 * 0.7 = 7 for training
        # 10 / 1 * 0.2 = 2 for validation
        # 10 / 1 * 0.1 = 1 for test

        self.assertEqual(7,test.run_training_params['n_samples_train'])
        self.assertEqual(2,test.run_training_params['n_samples_val'])
        self.assertEqual(1,test.run_training_params['n_samples_test'])
        self.assertEqual(10,test.num_of_samples['dataset'])
        self.assertEqual(1,test.run_training_params['train_batch_size'])
        self.assertEqual(1,test.run_training_params['val_batch_size'])
        self.assertEqual(1,test.run_training_params['test_batch_size'])
        self.assertEqual(5,test.run_training_params['num_of_epochs'])
        self.assertEqual(0.1,test.run_training_params['optimizer_defaults']['lr'])
    
    def test_build_dataset_batch2(self):
        input1 = Input('in1')
        output = Input('out')
        rel1 = Fir(input1.tw(0.05))

        test = Neu4mes(visualizer=None)
        test.addMinimize('out', output.z(-1), rel1)
        test.neuralizeModel(0.01)

        data_struct = ['x','F','x2','y2','','A1x','A1y','B1x','B1y','','A2x','A2y','B2x','out','','x3','in1','in2','time']
        test.loadData(name='dataset',source=data_folder, format=data_struct, skiplines=4, delimiter='\t', header=None)
        self.assertEqual((10,5,1),test.data['dataset']['in1'].shape)

        training_params = {}
        training_params['train_batch_size'] = 25
        training_params['val_batch_size'] = 25
        training_params['test_batch_size'] = 25
        training_params['lr'] = 0.1
        training_params['num_of_epochs'] = 5
        test.trainModel(splits=[50,0,50],training_params = training_params)

        # 15 lines in the dataset
        # 5 lines for input + 1 for output -> total of sample 10 
        # batch_size > 5 use batch_size = 1
        # 10 / 1 * 0.5 = 5 for training
        # 10 / 1 * 0.0 = 0 for validation
        # 10 / 1 * 0.5 = 5 for test
        self.assertEqual((15 - 5), test.num_of_samples['dataset'])
        self.assertEqual(round((15 - 5) * 50 / 100), test.run_training_params['n_samples_train'])
        self.assertEqual(round((15 - 5) * 0 / 100), test.run_training_params['n_samples_val'])
        self.assertEqual(round((15 - 5) * 50 / 100), test.run_training_params['n_samples_test'])
        self.assertEqual(round((15 - 5) * 50 / 100), test.run_training_params['train_batch_size'])
        self.assertEqual(0, test.run_training_params['val_batch_size'])
        self.assertEqual(round((15 - 5) * 50 / 100), test.run_training_params['test_batch_size'])
        self.assertEqual(5, test.run_training_params['num_of_epochs'])
        self.assertEqual(0.1, test.run_training_params['optimizer_defaults']['lr'])

    def test_build_dataset_batch3(self):
        input1 = Input('in1')
        output = Input('out')
        rel1 = Fir(input1.tw(0.05))

        test = Neu4mes(visualizer=None)
        test.addMinimize('out', output.next(), rel1)
        test.neuralizeModel(0.01)

        data_struct = ['x','F','x2','y2','','A1x','A1y','B1x','B1y','','A2x','A2y','B2x','out','','x3','in1','in2','time']
        test.loadData(name='dataset', source=data_folder, format=data_struct, skiplines=4, delimiter='\t', header=None)
        self.assertEqual((10,5,1),test.data['dataset']['in1'].shape)

        training_params = {}
        training_params['train_batch_size'] = 2
        training_params['val_batch_size'] = 2
        training_params['test_batch_size'] = 2
        training_params['lr'] = 0.1
        training_params['num_of_epochs'] = 5
        test.trainModel(splits=[40,30,30], training_params = training_params)

        # 15 lines in the dataset
        # 5 lines for input + 1 for output -> total of sample 10 
        # batch_size > 5 -> NO
        # num_of_training_sample must be multiple of batch_size
        # num_of_test_sample must be multiple of batch_size and at least 50%
        # 10 * 0.4 = 2 for training
        # 10 * 0.3 = 1 for validation
        # 10 * 0.3 = 1 for test
        self.assertEqual((15 - 5), test.num_of_samples['dataset'])
        self.assertEqual(round((15 - 5) * 40 / 100), test.run_training_params['n_samples_train'])
        self.assertEqual(round((15 - 5) * 30 / 100), test.run_training_params['n_samples_val'])
        self.assertEqual(round((15 - 5) * 30 / 100), test.run_training_params['n_samples_test'])
        self.assertEqual(2, test.run_training_params['train_batch_size'])
        self.assertEqual(2, test.run_training_params['val_batch_size'])
        self.assertEqual(2, test.run_training_params['test_batch_size'])
        self.assertEqual(5, test.run_training_params['num_of_epochs'])
        self.assertEqual(0.1, test.run_training_params['optimizer_defaults']['lr'])

    def test_build_dataset_batch4(self):
        input1 = Input('in1')
        output = Input('out')
        rel1 = Fir(input1.tw(0.05))

        test = Neu4mes(visualizer=None)
        test.addMinimize('out', output.z(-1), rel1)
        test.neuralizeModel(0.01)

        data_struct = ['x','F','x2','y2','','A1x','A1y','B1x','B1y','','A2x','A2y','B2x','out','','x3','in1','in2','time']
        test.loadData(name='dataset', source=data_folder, format=data_struct, skiplines=4, delimiter='\t', header=None)
        self.assertEqual((10,5,1),test.data['dataset']['in1'].shape)

        training_params = {}
        training_params['train_batch_size'] = 2
        training_params['val_batch_size'] = 2
        training_params['test_batch_size'] = 2
        training_params['lr'] = 0.1
        training_params['num_of_epochs'] = 5
        test.trainModel(splits=[80,10,10], training_params = training_params)

        # 15 lines in the dataset
        # 5 lines for input + 1 for output -> total of sample 10
        # batch_size > 1 -> YES
        # num_of_training_sample must be multiple of batch_size
        # num_of_test_sample must be multiple of batch_size and at least 10%
        # 10 * 0.8 = 8 for training
        # 10 * 0.1 = 1 for validation
        # 10 * 0.1 = 1 for test
        self.assertEqual((15 - 5), test.num_of_samples['dataset'])
        self.assertEqual(round((15 - 5) * 80 / 100), test.run_training_params['n_samples_train'])
        self.assertEqual(round((15 - 5) * 10 / 100), test.run_training_params['n_samples_val'])
        self.assertEqual(round((15 - 5) * 10 / 100), test.run_training_params['n_samples_test'])
        self.assertEqual(2, test.run_training_params['train_batch_size'])
        self.assertEqual(1, test.run_training_params['val_batch_size'])
        self.assertEqual(1, test.run_training_params['test_batch_size'])
        self.assertEqual(5, test.run_training_params['num_of_epochs'])
        self.assertEqual(0.1, test.run_training_params['optimizer_defaults']['lr'])
    
    def test_build_dataset_from_code(self):
        input1 = Input('in1')
        output = Input('out')
        rel1 = Fir(input1.tw(0.05))

        test = Neu4mes(visualizer=None)
        test.addMinimize('out', output.next(), rel1)
        test.neuralizeModel(0.01)

        x_size = 20
        data_x = np.random.rand(x_size)*20-10
        data_a = 2
        data_b = -3
        dataset = {'in1': data_x, 'out': data_x*data_a+data_b}

        test.loadData(name='dataset', source=dataset, skiplines=0)
        self.assertEqual((15,5,1),test.data['dataset']['in1'].shape)  ## 20 data - 5 tw = 15 sample | 0.05/0.01 = 5 in1

        training_params = {}
        training_params['train_batch_size'] = 2
        training_params['val_batch_size'] = 2
        training_params['test_batch_size'] = 2
        training_params['lr'] = 0.1
        training_params['num_of_epochs'] = 5
        test.trainModel(splits=[80,20,0], training_params = training_params)

        # 20 lines in the dataset
        # 5 lines for input + 1 for output -> total of sample (20 - 5 - 1) = 16
        # batch_size > 1 -> YES
        # num_of_training_sample must be multiple of batch_size
        # num_of_test_sample must be multiple of batch_size and at least 10%
        # 15 * 0.8 = 12 for training
        # 15 * 0.2 = 3 for validation
        # 15 * 0.0 = 0 for test
        self.assertEqual((20 - 5), test.num_of_samples['dataset'])
        self.assertEqual(round((20 - 5) * 80 / 100), test.run_training_params['n_samples_train'])
        self.assertEqual(round((20 - 5) * 20 / 100), test.run_training_params['n_samples_val'])
        self.assertEqual(round((20 - 5) * 0 / 100), test.run_training_params['n_samples_test'])
        self.assertEqual(2, test.run_training_params['train_batch_size'])
        self.assertEqual(2, test.run_training_params['val_batch_size'])
        self.assertEqual(0, test.run_training_params['test_batch_size'])
        self.assertEqual(5, test.run_training_params['num_of_epochs'])
        self.assertEqual(0.1, test.run_training_params['optimizer_defaults']['lr'])
    
    def test_network_multi_dataset(self):
        train_folder = os.path.join(os.path.dirname(__file__), 'data/')
        val_folder = os.path.join(os.path.dirname(__file__), 'val_data/')
        test_folder = os.path.join(os.path.dirname(__file__), 'test_data/')

        x = Input('x')  # Position
        F = Input('F')  # Force

        # List the output of the model
        x_z = Output('x_z', Fir(x.tw(0.3)) + Fir(F.last()))

        # Add the neural model to the neu4mes structure and neuralization of the model
        test = Neu4mes(visualizer=None)
        test.addModel('x_z',x_z)
        test.addMinimize('next-pos', x.z(-1), x_z, 'mse')

        # Create the neural network
        test.neuralizeModel(sample_time=0.05)  # The sampling time depends to the dataset

        # Data load
        data_struct = ['x','F','x2','y2','','A1x','A1y','B1x','B1y','','A2x','A2y','B2x','out','','x3','in1','in2','time']
        test.loadData(name='train_dataset', source=train_folder, format=data_struct, skiplines=4, delimiter='\t', header=None)
        test.loadData(name='validation_dataset', source=val_folder, format=data_struct, skiplines=4, delimiter='\t', header=None)
        test.loadData(name='test_dataset', source=test_folder, format=data_struct, skiplines=4, delimiter='\t', header=None)

        training_params = {}
        training_params['train_batch_size'] = 3
        training_params['val_batch_size'] = 2
        training_params['test_batch_size'] = 1
        training_params['lr'] = 0.1
        training_params['num_of_epochs'] = 5
        test.trainModel(train_dataset='train_dataset', validation_dataset='validation_dataset', test_dataset='test_dataset', training_params=training_params)

        self.assertEqual(9,test.num_of_samples['train_dataset'])
        self.assertEqual(5,test.num_of_samples['validation_dataset'])
        self.assertEqual(7,test.num_of_samples['test_dataset'])
        self.assertEqual( 9, test.run_training_params['n_samples_train'])
        self.assertEqual(5, test.run_training_params['n_samples_val'])
        self.assertEqual(7, test.run_training_params['n_samples_test'])
        self.assertEqual(3, test.run_training_params['train_batch_size'])
        self.assertEqual(2, test.run_training_params['val_batch_size'])
        self.assertEqual(1, test.run_training_params['test_batch_size'])
        self.assertEqual(5, test.run_training_params['num_of_epochs'])
        self.assertEqual(0.1, test.run_training_params['optimizer_defaults']['lr'])
    
    def test_train_vector_input(self):
        x = Input('x', dimensions=4)
        y = Input('y', dimensions=3)
        k = Input('k', dimensions=2)
        w = Input('w')

        out = Output('out', Fir(Linear(Linear(3)(x.tw(0.02)) + y.tw(0.02))))
        out2 = Output('out2', Fir(Linear(k.last() + Fir(2)(w.tw(0.05,offset=-0.02)))))

        test = Neu4mes(visualizer=None)
        test.addMinimize('out', out, out2)
        test.neuralizeModel(0.01)

        data_folder = os.path.join(os.path.dirname(__file__), 'vector_data/')
        data_struct = ['x', 'y', '','', '', '', 'k', '', '', '', 'w']
        test.loadData(name='dataset', source=data_folder, format=data_struct, skiplines=1, delimiter='\t', header=None)

        training_params = {}
        training_params['train_batch_size'] = 1
        training_params['val_batch_size'] = 1
        training_params['test_batch_size'] = 1
        training_params['lr'] = 0.01
        training_params['num_of_epochs'] = 7
        test.trainModel(splits=[80,10,10],  training_params=training_params)

        self.assertEqual(22,test.num_of_samples['dataset'])
        self.assertEqual( 18, test.run_training_params['n_samples_train'])
        self.assertEqual(2, test.run_training_params['n_samples_val'])
        self.assertEqual(2, test.run_training_params['n_samples_test'])
        self.assertEqual(1, test.run_training_params['train_batch_size'])
        self.assertEqual(1, test.run_training_params['val_batch_size'])
        self.assertEqual(1, test.run_training_params['test_batch_size'])
        self.assertEqual(7, test.run_training_params['num_of_epochs'])
        self.assertEqual(0.01, test.run_training_params['optimizer_defaults']['lr'])

        training_params = {}
        training_params['train_batch_size'] = 6
        training_params['val_batch_size'] = 2
        training_params['test_batch_size'] = 2
        test.trainModel(splits=[80,10,10],  training_params=training_params)

        self.assertEqual(22,test.num_of_samples['dataset'])
        self.assertEqual( 18, test.run_training_params['n_samples_train'])
        self.assertEqual(2, test.run_training_params['n_samples_val'])
        self.assertEqual(2, test.run_training_params['n_samples_test'])
        self.assertEqual(6, test.run_training_params['train_batch_size'])
        self.assertEqual(2, test.run_training_params['val_batch_size'])
        self.assertEqual(2, test.run_training_params['test_batch_size'])
        self.assertEqual(100, test.run_training_params['num_of_epochs'])
        self.assertEqual(0.001, test.run_training_params['optimizer_defaults']['lr'])

    def test_optimizer_configuration(self):
        ## Model1
        input1 = Input('in1')
        a = Parameter('a', dimensions=1, tw=0.05, values=[[1], [1], [1], [1], [1]])
        shared_w = Parameter('w', values=[[5]])
        output1 = Output('out1',
                         Fir(parameter=a)(input1.tw(0.05)) + ParamFun(funIn, n_input=1, parameters={'w': shared_w})(
                             input1.last()))

        test = Neu4mes(visualizer = None, seed = 42)
        test.addModel('model1', output1)
        test.addMinimize('error1', input1.last(), output1)

        ## Model2
        input2 = Input('in2')
        b = Parameter('b', dimensions=1, tw=0.05, values=[[1], [1], [1], [1], [1]])
        output2 = Output('out2',
                         Fir(parameter=b)(input2.tw(0.05)) + ParamFun(funOut, n_input=1, parameters={'w': shared_w})(
                             input2.last()))

        test.addModel('model2', output2)
        test.addMinimize('error2', input2.last(), output2)
        test.neuralizeModel(0.01)

        # Dataset for train
        data_in1 = np.linspace(0, 5, 60)
        data_in2 = np.linspace(10, 15, 60)
        data_out1 = 2
        data_out2 = -3
        dataset = {'in1': data_in1, 'in2': data_in2, 'out1': data_in1 * data_out1, 'out2': data_in2 * data_out2}
        test.loadData(name='dataset1', source=dataset)

        data_in1 = np.linspace(0, 5, 100)
        data_in2 = np.linspace(10, 15, 100)
        data_out1 = 2
        data_out2 = -3
        dataset = {'in1': data_in1, 'in2': data_in2, 'out1': data_in1 * data_out1, 'out2': data_in2 * data_out2}
        test.loadData(name='dataset2', source=dataset)

        # Optimizer
        # Basic usage
        # Standard optimizer with standard configuration
        # We train all the models with split [70,20,10], lr =0.01 and epochs = 100
        # TODO if more than one dataset is loaded I use all the dataset
        test.trainModel()
        self.assertEqual(['model1', 'model2'], test.run_training_params['models'])
        self.assertEqual( 39, test.run_training_params['n_samples_train'])
        self.assertEqual(11, test.run_training_params['n_samples_val'])
        self.assertEqual(6, test.run_training_params['n_samples_test'])
        self.assertEqual(100, test.run_training_params['num_of_epochs'])
        self.assertEqual(0.001, test.run_training_params['optimizer_defaults']['lr'])

        # We train only model1 with split [100,0,0]
        # TODO Learning rate automoatically optimized based on the mean and variance of the output
        # TODO num_of_epochs automatically defined
        # now is 0.001 for learning rate and 100 for the epochs and optimizer Adam
        test.trainModel(models='model1', splits=[100, 0, 0])
        self.assertEqual('model1', test.run_training_params['models'])
        self.assertEqual(100, test.run_training_params['num_of_epochs'])
        self.assertEqual( 56, test.run_training_params['n_samples_train'])
        self.assertEqual(0, test.run_training_params['n_samples_val'])
        self.assertEqual(0, test.run_training_params['n_samples_test'])

        # Set number of epoch and learning rate via parameters it works only for standard parameters
        test.trainModel(models='model1', splits=[100, 0, 0], lr=0.5, num_of_epochs=5)
        self.assertEqual('model1', test.run_training_params['models'])
        self.assertEqual(5, test.run_training_params['num_of_epochs'])
        self.assertEqual( 56, test.run_training_params['n_samples_train'])
        self.assertEqual(0, test.run_training_params['n_samples_val'])
        self.assertEqual(0, test.run_training_params['n_samples_test'])
        self.assertEqual(0.5, test.run_training_params['optimizer_defaults']['lr'])

        # Set number of epoch and learning rate via parameters it works only for standard parameters and use two different dataset one for train and one for validation
        test.trainModel(models='model1', train_dataset='dataset1', validation_dataset='dataset2', lr=0.6, num_of_epochs=10)
        self.assertEqual('model1', test.run_training_params['models'])
        self.assertEqual(10, test.run_training_params['num_of_epochs'])
        self.assertEqual(56, test.run_training_params['n_samples_train'])
        self.assertEqual(96, test.run_training_params['n_samples_val'])
        self.assertEqual(0, test.run_training_params['n_samples_test'])
        self.assertEqual(0.6, test.run_training_params['optimizer_defaults']['lr'])

        # Use dictionary for set number of epoch, learning rate, etc.. This configuration works only standard parameters (all the parameters that are input of the trainModel).
        training_params = {
            'models': ['model2'],
            'splits': [55, 40, 5],
            'num_of_epochs': 20,
            'lr': 0.7
        }
        test.trainModel(training_params=training_params)
        self.assertEqual(['model2'], test.run_training_params['models'])
        self.assertEqual(20, test.run_training_params['num_of_epochs'])
        self.assertEqual(round(56*55/100), test.run_training_params['n_samples_train'])
        self.assertEqual(round(56*40/100), test.run_training_params['n_samples_val'])
        self.assertEqual(round(56*5/100), test.run_training_params['n_samples_test'])
        self.assertEqual(0.7, test.run_training_params['optimizer_defaults']['lr'])

        # If I add a function parameter it has the priority
        # In this case apply train parameter but on a different model
        test.trainModel(models='model1', training_params=training_params)
        self.assertEqual('model1', test.run_training_params['models'])
        self.assertEqual(20, test.run_training_params['num_of_epochs'])
        self.assertEqual(round(56*55/100), test.run_training_params['n_samples_train'])
        self.assertEqual(round(56*40/100), test.run_training_params['n_samples_val'])
        self.assertEqual(round(56*5/100), test.run_training_params['n_samples_test'])
        self.assertEqual(0.7, test.run_training_params['optimizer_defaults']['lr'])

        ##################################
        # Modify additional parameters in the optimizer that are not present in the standard parameter
        # In this case I modify the learning rate and the betas of the Adam optimizer
        # For the optimizer parameter the priority is the following
        # max priority to the function parameter ('lr' : 0.2)
        # then the standard_optimizer_parameters ('lr' : 0.1)
        # finally the standard_train_parameters  ('lr' : 0.5)
        optimizer_defaults = {
            'lr': 0.1,
            'betas': (0.5, 0.99)
        }
        test.trainModel(training_params=training_params, optimizer_defaults=optimizer_defaults, lr=0.2)
        self.assertEqual(['model2'], test.run_training_params['models'])
        self.assertEqual(20, test.run_training_params['num_of_epochs'])
        self.assertEqual(round(56*55/100), test.run_training_params['n_samples_train'])
        self.assertEqual(round(56*40/100), test.run_training_params['n_samples_val'])
        self.assertEqual(round(56*5/100), test.run_training_params['n_samples_test'])
        self.assertEqual(0.2, test.run_training_params['optimizer_defaults']['lr'])
        self.assertEqual((0.5, 0.99), test.run_training_params['optimizer_defaults']['betas'])

        test.trainModel(training_params=training_params, optimizer_defaults=optimizer_defaults)
        self.assertEqual(['model2'], test.run_training_params['models'])
        self.assertEqual(20, test.run_training_params['num_of_epochs'])
        self.assertEqual(round(56*55/100), test.run_training_params['n_samples_train'])
        self.assertEqual(round(56*40/100), test.run_training_params['n_samples_val'])
        self.assertEqual(round(56*5/100), test.run_training_params['n_samples_test'])
        self.assertEqual(0.1, test.run_training_params['optimizer_defaults']['lr'])
        self.assertEqual((0.5, 0.99), test.run_training_params['optimizer_defaults']['betas'])

        test.trainModel(training_params=training_params)
        self.assertEqual(['model2'], test.run_training_params['models'])
        self.assertEqual(20, test.run_training_params['num_of_epochs'])
        self.assertEqual(round(56*55/100), test.run_training_params['n_samples_train'])
        self.assertEqual(round(56*40/100), test.run_training_params['n_samples_val'])
        self.assertEqual(round(56*5/100), test.run_training_params['n_samples_test'])
        self.assertEqual(0.7, test.run_training_params['optimizer_defaults']['lr'])
        ##################################

        # Modify the non standard args of the optimizer using the optimizer_defaults
        # In this case use the SGD with 0.2 of momentum
        optimizer_defaults = {
            'momentum': 0.002
        }
        test.trainModel(optimizer='SGD', training_params=training_params, optimizer_defaults=optimizer_defaults, lr=0.2)
        self.assertEqual(['model2'], test.run_training_params['models'])
        self.assertEqual('SGD', test.run_training_params['optimizer'])
        self.assertEqual(20, test.run_training_params['num_of_epochs'])
        self.assertEqual(round(56*55/100), test.run_training_params['n_samples_train'])
        self.assertEqual(round(56*40/100), test.run_training_params['n_samples_val'])
        self.assertEqual(round(56*5/100), test.run_training_params['n_samples_test'])
        self.assertEqual(0.2, test.run_training_params['optimizer_defaults']['lr'])
        self.assertEqual(0.002, test.run_training_params['optimizer_defaults']['momentum'])


        # Modify standard optimizer parameter for each training parameter
        training_params = {
            'models': ['model1'],
            'splits': [100, 0, 0],
            'num_of_epochs': 30,
            'lr': 0.5,
            'lr_param': {'a': 0.1}
        }
        test.trainModel(training_params=training_params)
        self.assertEqual(['model1'], test.run_training_params['models'])
        self.assertEqual(30, test.run_training_params['num_of_epochs'])
        self.assertEqual(round(56*100/100), test.run_training_params['n_samples_train'])
        self.assertEqual(round(56*0/100), test.run_training_params['n_samples_val'])
        self.assertEqual(round(56*0/100), test.run_training_params['n_samples_test'])
        self.assertEqual(0.5, test.run_training_params['optimizer_defaults']['lr'])
        self.assertEqual([{'lr': 0.1, 'params': 'a'},
                               {'lr': 0.0, 'params': 'b'},
                               {'params': 'w'}], test.run_training_params['optimizer_params'])

        ##################################
        # Modify standard optimizer parameter for each training parameter using optimizer_params
        # The priority is the following
        # max priority to the function parameter ( 'lr_param'={'a': 0.4})
        # then the optimizer_params ( {'params':'a','lr':0.6} )
        # then the optimizer_params inside the train_parameters ( {'params':['a'],'lr':0.7} )
        # finally the train_parameters  ( 'lr_param'={'a': 0.1})
        training_params = {
            'models': ['model1'],
            'splits': [100, 0, 0],
            'num_of_epochs': 40,
            'lr': 0.5,
            'lr_param': {'a': 0.1},
            'optimizer_params': [{'params': ['a'], 'lr': 0.7}],
            'optimizer_defaults': {'lr': 0.12}
        }
        optimizer_params = [
            {'params': ['a'], 'lr': 0.6}
        ]
        optimizer_defaults = {
            'lr': 0.2
        }
        test.trainModel(training_params=training_params, optimizer_params=optimizer_params, optimizer_defaults=optimizer_defaults, lr_param={'a': 0.4})
        self.assertEqual(['model1'], test.run_training_params['models'])
        self.assertEqual(40, test.run_training_params['num_of_epochs'])
        self.assertEqual(round(56*100/100), test.run_training_params['n_samples_train'])
        self.assertEqual(round(56*0/100), test.run_training_params['n_samples_val'])
        self.assertEqual(round(56*0/100), test.run_training_params['n_samples_test'])
        self.assertEqual(0.2, test.run_training_params['optimizer_defaults']['lr'])
        self.assertEqual([{'lr': 0.4, 'params': 'a'}], test.run_training_params['optimizer_params'])

        test.trainModel(training_params=training_params, optimizer_params=optimizer_params, optimizer_defaults=optimizer_defaults)
        self.assertEqual(0.2, test.run_training_params['optimizer_defaults']['lr'])
        self.assertEqual([{'lr': 0.6, 'params': 'a'}], test.run_training_params['optimizer_params'])

        test.trainModel(training_params=training_params, optimizer_params=optimizer_params)
        self.assertEqual(0.12, test.run_training_params['optimizer_defaults']['lr'])
        self.assertEqual([{'lr': 0.6, 'params': 'a'}], test.run_training_params['optimizer_params'])

        test.trainModel(training_params=training_params)
        self.assertEqual(0.12, test.run_training_params['optimizer_defaults']['lr'])
        self.assertEqual([{'params': 'a', 'lr': 0.7}], test.run_training_params['optimizer_params'])

        del training_params['optimizer_defaults']
        test.trainModel(training_params=training_params)
        self.assertEqual(0.5, test.run_training_params['optimizer_defaults']['lr'])
        self.assertEqual([{'params': 'a', 'lr': 0.7}], test.run_training_params['optimizer_params'])

        del training_params['optimizer_params']
        test.trainModel(training_params=training_params)
        self.assertEqual(0.5, test.run_training_params['optimizer_defaults']['lr'])
        self.assertEqual([{'lr': 0.1, 'params': 'a'},
                               {'lr': 0.0, 'params': 'b'},
                               {'params': 'w'}], test.run_training_params['optimizer_params'])

        test.trainModel()
        self.assertEqual(0.001, test.run_training_params['optimizer_defaults']['lr'])
        self.assertEqual([{'params': 'a'},
                               {'params': 'b'},
                               {'params': 'w'}], test.run_training_params['optimizer_params'])
        ##################################

        ##################################
        # Maximum level of configuration I define a custom optimizer with defaults
        # For the optimizer default the priority is the following
        # max priority to the function parameter ('lr'= 0.4)
        # then the optimizer_defaults ('lr':0.1)
        # then the optimizer_defaults inside the train_parameters ('lr'= 0.12)
        # finally the train_parameters  ('lr'= 0.5)
        class RMSprop(Optimizer):
            def __init__(self, optimizer_defaults={}, optimizer_params=[]):
                super(RMSprop, self).__init__('RMSprop', optimizer_defaults, optimizer_params)
            def get_torch_optimizer(self):
                import torch
                return torch.optim.RMSprop(self.replace_key_with_params(), **self.optimizer_defaults)
        training_params = {
            'models': ['model1'],
            'splits': [100, 0, 0],
            'num_of_epochs': 40,
            'lr': 0.5,
            'lr_param': {'a': 0.1},
            'optimizer_params': [{'params': ['a'], 'lr': 0.7}],
            'optimizer_defaults': {'lr': 0.12}
        }
        optimizer_defaults = {
            'alpha': 0.8
        }
        optimizer = RMSprop(optimizer_defaults)
        test.trainModel(optimizer=optimizer, training_params=training_params, optimizer_defaults={'lr': 0.3}, lr=0.4)
        self.assertEqual(['model1'], test.run_training_params['models'])
        self.assertEqual('RMSprop', test.run_training_params['optimizer'])
        self.assertEqual(40, test.run_training_params['num_of_epochs'])
        self.assertEqual(round(56*100/100), test.run_training_params['n_samples_train'])
        self.assertEqual(round(56*0/100), test.run_training_params['n_samples_val'])
        self.assertEqual(round(56*0/100), test.run_training_params['n_samples_test'])
        self.assertEqual({'lr': 0.4}, test.run_training_params['optimizer_defaults'])
        self.assertEqual([{'lr': 0.7, 'params': 'a'}], test.run_training_params['optimizer_params'])

        test.trainModel(optimizer=optimizer, training_params=training_params, optimizer_defaults={'lr': 0.1})
        self.assertEqual({'lr': 0.1}, test.run_training_params['optimizer_defaults'])
        self.assertEqual([{'lr': 0.7, 'params': 'a'}], test.run_training_params['optimizer_params'])

        test.trainModel(optimizer=optimizer, training_params=training_params)
        self.assertEqual({'lr': 0.12}, test.run_training_params['optimizer_defaults'])
        self.assertEqual([{'lr': 0.7, 'params': 'a'}], test.run_training_params['optimizer_params'])

        del training_params['optimizer_defaults']
        test.trainModel(optimizer=optimizer, training_params=training_params)
        self.assertEqual({'alpha': 0.8, 'lr': 0.5}, test.run_training_params['optimizer_defaults'])
        self.assertEqual([{'lr': 0.7, 'params': 'a'}], test.run_training_params['optimizer_params'])

        del training_params['optimizer_params']
        test.trainModel(optimizer=optimizer, training_params=training_params)
        self.assertEqual({'alpha': 0.8, 'lr': 0.5}, test.run_training_params['optimizer_defaults'])
        self.assertEqual([{'lr': 0.1, 'params': 'a'}, {'lr': 0.0, 'params': 'b'}, {'params': 'w'}], test.run_training_params['optimizer_params'])

        test.trainModel(optimizer=optimizer)
        self.assertEqual({'alpha': 0.8, 'lr': 0.001}, test.run_training_params['optimizer_defaults'])
        self.assertEqual([{'params': 'a'}, {'params': 'b'}, {'params': 'w'}], test.run_training_params['optimizer_params'])
        ##################################

        ##################################
        # Maximum level of configuration I define a custom optimizer with custom value for each params
        # The priority is the following
        # max priority to the function parameter ( 'lr_param'={'a': 0.2})
        # then the optimizer_params ( [{'params':['a'],'lr':1.0}] )
        # then the optimizer_params inside the train_parameters (  [{'params':['a'],'lr':0.7}] )
        # then the train_parameters  ( 'lr_param'={'a': 0.1} )
        # finnaly the optimizer_paramsat the time of the optimizer initialization [{'params':['a'],'lr':0.6}]
        training_params = {
            'models': ['model1'],
            'splits': [100, 0, 0],
            'num_of_epochs': 40,
            'lr': 0.5,
            'lr_param': {'a': 0.1},
            'optimizer_params': [{'params': ['a'], 'lr': 0.7}],
            'optimizer_defaults': {'lr': 0.12}
        }
        optimizer_defaults = {
            'alpha': 0.8
        }
        optimizer_params = [
            {'params': ['a'], 'lr': 0.6},{'params': 'w', 'lr': 0.12, 'alpha': 0.02}
        ]
        optimizer = RMSprop(optimizer_defaults, optimizer_params)
        test.trainModel(optimizer=optimizer, training_params=training_params, optimizer_defaults={'lr': 0.3},
                        optimizer_params=[{'params': ['a'], 'lr': 1.0},{'params': ['b'], 'lr': 1.2}], lr_param={'a': 0.2})
        self.assertEqual(['model1'], test.run_training_params['models'])
        self.assertEqual('RMSprop', test.run_training_params['optimizer'])
        self.assertEqual(40, test.run_training_params['num_of_epochs'])
        self.assertEqual(round(56*100/100), test.run_training_params['n_samples_train'])
        self.assertEqual(round(56*0/100), test.run_training_params['n_samples_val'])
        self.assertEqual(round(56*0/100), test.run_training_params['n_samples_test'])
        self.assertEqual({'lr': 0.3}, test.run_training_params['optimizer_defaults'])
        self.assertEqual([{'params': 'a', 'lr': 0.2},{'params': 'b', 'lr': 1.2}], test.run_training_params['optimizer_params'])

        test.trainModel(optimizer=optimizer, training_params=training_params, optimizer_defaults={'lr': 0.3}, optimizer_params=[{'params': ['a'], 'lr': 0.1},{'params': ['b'], 'lr': 0.2}])
        self.assertEqual({'lr': 0.3}, test.run_training_params['optimizer_defaults'])
        self.assertEqual([{'params': 'a', 'lr': 0.1},{'params': 'b', 'lr': 0.2}], test.run_training_params['optimizer_params'])

        test.trainModel(optimizer=optimizer, training_params=training_params, optimizer_defaults={'lr': 0.3})
        self.assertEqual({'lr': 0.3}, test.run_training_params['optimizer_defaults'])
        self.assertEqual([{'params': 'a', 'lr': 0.7}], test.run_training_params['optimizer_params'])

        test.trainModel(optimizer=optimizer, training_params=training_params)
        self.assertEqual({'lr': 0.12}, test.run_training_params['optimizer_defaults'])
        self.assertEqual([{'params': 'a', 'lr': 0.7}], test.run_training_params['optimizer_params'])

        del training_params['optimizer_defaults']
        test.trainModel(optimizer=optimizer, training_params=training_params)
        self.assertEqual({'alpha': 0.8, 'lr': 0.5}, test.run_training_params['optimizer_defaults'])
        self.assertEqual([{'params': 'a', 'lr': 0.7}], test.run_training_params['optimizer_params'])

        del training_params['optimizer_params']
        test.trainModel(optimizer=optimizer, training_params=training_params)
        self.assertEqual({'alpha': 0.8, 'lr': 0.5}, test.run_training_params['optimizer_defaults'])
        self.assertEqual([{'lr': 0.1, 'params': 'a'}, {'alpha': 0.02, 'lr': 0.12, 'params': 'w'}], test.run_training_params['optimizer_params'])

        test.trainModel(optimizer=optimizer)
        self.assertEqual({'alpha': 0.8, 'lr': 0.001}, test.run_training_params['optimizer_defaults'])
        self.assertEqual([{'params': 'a', 'lr': 0.6},{'params': 'w', 'lr': 0.12, 'alpha': 0.02}], test.run_training_params['optimizer_params'])
        ##################################

        ##################################
        # Maximum level of configuration I define a custom optimizer and add some parameter over the defaults
        # The priority is the following
        # max priority to the function parameter ( 'lr_param'={'a': 0.2})
        # then the optimizer_params ( [{'params':['a'],'lr':1.0}] )
        # then the optimizer_params inside the train_parameters (  [{'params':['a'],'lr':0.7}] )
        # then the train_parameters  ( 'lr_param'={'a': 0.1} )
        # The other parameters are the defaults
        training_params = {
            'models': ['model1'],
            'splits': [100, 0, 0],
            'num_of_epochs': 40,
            'lr': 0.5,
            'lr_param': {'a': 0.1},
            'add_optimizer_params': [{'params': ['a'], 'lr': 0.7}],
            'add_optimizer_defaults': {'lr': 0.12}
        }
        optimizer = RMSprop()
        test.trainModel(optimizer=optimizer, training_params=training_params, add_optimizer_defaults={'lr': 0.3},
                        add_optimizer_params=[{'params': ['a'], 'lr': 1.0},{'params': ['b'], 'lr': 1.2}], lr_param={'a': 0.2})
        self.assertEqual(['model1'], test.run_training_params['models'])
        self.assertEqual('RMSprop', test.run_training_params['optimizer'])
        self.assertEqual(40, test.run_training_params['num_of_epochs'])
        self.assertEqual(round(56*100/100), test.run_training_params['n_samples_train'])
        self.assertEqual(round(56*0/100), test.run_training_params['n_samples_val'])
        self.assertEqual(round(56*0/100), test.run_training_params['n_samples_test'])
        self.assertEqual({'lr': 0.3}, test.run_training_params['optimizer_defaults'])
        self.assertEqual([{'params': 'a', 'lr': 0.2},{'params': 'b', 'lr': 1.2}, {'params': 'w'}], test.run_training_params['optimizer_params'])

        test.trainModel(optimizer=optimizer, training_params=training_params, add_optimizer_defaults={'lr': 0.3}, add_optimizer_params=[{'params': ['a'], 'lr': 0.1},{'params': ['b'], 'lr': 0.2}])
        self.assertEqual({'lr': 0.3}, test.run_training_params['optimizer_defaults'])
        self.assertEqual([{'params': 'a', 'lr': 0.1},{'params': 'b', 'lr': 0.2}, {'params': 'w'}], test.run_training_params['optimizer_params'])

        test.trainModel(optimizer=optimizer, training_params=training_params, add_optimizer_defaults={'lr': 0.3})
        self.assertEqual({'lr': 0.3}, test.run_training_params['optimizer_defaults'])
        self.assertEqual([{'params': 'a', 'lr': 0.7}, {'lr': 0.0, 'params': 'b'}, {'params': 'w'}], test.run_training_params['optimizer_params'])

        test.trainModel(optimizer=optimizer, training_params=training_params)
        self.assertEqual({'lr': 0.12}, test.run_training_params['optimizer_defaults'])
        self.assertEqual([{'params': 'a', 'lr': 0.7}, {'lr': 0.0, 'params': 'b'}, {'params': 'w'}], test.run_training_params['optimizer_params'])

        del training_params['add_optimizer_defaults']
        test.trainModel(optimizer=optimizer, training_params=training_params)
        self.assertEqual({'lr': 0.5}, test.run_training_params['optimizer_defaults'])
        self.assertEqual([{'params': 'a', 'lr': 0.7}, {'lr': 0.0, 'params': 'b'}, {'params': 'w'}], test.run_training_params['optimizer_params'])

        del training_params['add_optimizer_params']
        test.trainModel(optimizer=optimizer, training_params=training_params)
        self.assertEqual({ 'lr': 0.5}, test.run_training_params['optimizer_defaults'])
        self.assertEqual([{'lr': 0.1, 'params': 'a'}, {'lr': 0.0, 'params': 'b'}, {'params': 'w'}], test.run_training_params['optimizer_params'])

        test.trainModel(optimizer=optimizer)
        self.assertEqual({'lr': 0.001}, test.run_training_params['optimizer_defaults'])
        self.assertEqual([{'params': 'a'}, {'params': 'b'}, {'params': 'w'}], test.run_training_params['optimizer_params'])

    def test_training_values_fir(self):
        input1 = Input('in1')
        target = Input('out1')
        a = Parameter('a', values=[[1]])
        output1 = Output('out', Fir(parameter=a)(input1.last()))

        test = Neu4mes(visualizer=None,seed=42)
        test.addModel('model', output1)
        test.addMinimize('error', target.last(), output1)
        test.neuralizeModel()

        dataset = {'in1': [1], 'out1': [2]}
        test.loadData(name='dataset', source=dataset)

        self.assertListEqual([[1.0]],test.model.all_parameters['a'].data.numpy().tolist())
        test.trainModel(optimizer='SGD', splits=[100, 0, 0], lr=1, num_of_epochs=1)
        self.assertListEqual([[3.0]],test.model.all_parameters['a'].data.numpy().tolist())
        test.trainModel(optimizer='SGD', splits=[100, 0, 0], lr=1, num_of_epochs=1)
        self.assertListEqual([[1.0]],test.model.all_parameters['a'].data.numpy().tolist())
        test.trainModel(optimizer='SGD', splits=[100, 0, 0], lr=1, num_of_epochs=2)
        self.assertListEqual([[1.0]],test.model.all_parameters['a'].data.numpy().tolist())

    def test_training_values_linear(self):
        input1 = Input('in1')
        target = Input('out1')
        W = Parameter('W', values=[[[1]]])
        b = Parameter('b', values=[[1]])
        output1 = Output('out', Linear(W=W,b=b)(input1.last()))

        test = Neu4mes(visualizer=None,seed=42)
        test.addModel('model', output1)
        test.addMinimize('error', target.last(), output1)
        test.neuralizeModel()

        dataset = {'in1': [1], 'out1': [3]}
        test.loadData(name='dataset', source=dataset)

        self.assertListEqual([[[1.0]]],test.model.all_parameters['W'].data.numpy().tolist())
        self.assertListEqual([[1.0]], test.model.all_parameters['b'].data.numpy().tolist())
        test.trainModel(optimizer='SGD', splits=[100, 0, 0], lr=1, num_of_epochs=1)
        self.assertListEqual([[[3.0]]], test.model.all_parameters['W'].data.numpy().tolist())
        self.assertListEqual([[3.0]], test.model.all_parameters['b'].data.numpy().tolist())
        test.trainModel(optimizer='SGD', splits=[100, 0, 0], lr=1, num_of_epochs=1)
        self.assertListEqual([[[-3.0]]],test.model.all_parameters['W'].data.numpy().tolist())
        self.assertListEqual([[-3.0]], test.model.all_parameters['b'].data.numpy().tolist())

    def test_training_clear_model(self):
        input1 = Input('in1')
        target = Input('out1')
        a = Parameter('a', values=[[1]])
        output1 = Fir(parameter=a)(input1.last())

        W = Parameter('W', values=[[[1]]])
        b = Parameter('b', values=[[1]])
        output2 = Output('out2', Linear(W=W,b=b)(output1))

        test = Neu4mes(seed=42)
        test.addModel('model', output2)
        test.addMinimize('error', target.last(), output2)
        test.neuralizeModel()

        dataset = {'in1': [1], 'out1': [3]}
        test.loadData(name='dataset', source=dataset)

        self.assertListEqual([[[1.0]]],test.model.all_parameters['W'].data.numpy().tolist())
        self.assertListEqual([[1.0]], test.model.all_parameters['b'].data.numpy().tolist())
        self.assertListEqual([[1.0]], test.model.all_parameters['a'].data.numpy().tolist())
        test.trainModel(optimizer='SGD', splits=[100, 0, 0], lr=1, num_of_epochs=1)
        self.assertListEqual([[[3.0]]], test.model.all_parameters['W'].data.numpy().tolist())
        self.assertListEqual([[3.0]], test.model.all_parameters['b'].data.numpy().tolist())
        self.assertListEqual([[3.0]], test.model.all_parameters['a'].data.numpy().tolist())
        test.trainModel(optimizer='SGD', splits=[100, 0, 0], lr=1, num_of_epochs=1)
        self.assertListEqual([[[-51.0]]],test.model.all_parameters['W'].data.numpy().tolist())
        self.assertListEqual([[-15.0]], test.model.all_parameters['b'].data.numpy().tolist())
        self.assertListEqual([[-51.0]], test.model.all_parameters['a'].data.numpy().tolist())

        test.neuralizeModel(clear_model=True)
        self.assertListEqual([[[1.0]]],test.model.all_parameters['W'].data.numpy().tolist())
        self.assertListEqual([[1.0]], test.model.all_parameters['b'].data.numpy().tolist())
        self.assertListEqual([[1.0]], test.model.all_parameters['a'].data.numpy().tolist())
        test.trainModel(optimizer='SGD', splits=[100, 0, 0], lr=1, num_of_epochs=1)
        self.assertListEqual([[[3.0]]], test.model.all_parameters['W'].data.numpy().tolist())
        self.assertListEqual([[3.0]], test.model.all_parameters['b'].data.numpy().tolist())
        self.assertListEqual([[3.0]], test.model.all_parameters['a'].data.numpy().tolist())
        test.trainModel(optimizer='SGD', splits=[100, 0, 0], lr=1, num_of_epochs=1)
        self.assertListEqual([[[-51.0]]],test.model.all_parameters['W'].data.numpy().tolist())
        self.assertListEqual([[-15.0]], test.model.all_parameters['b'].data.numpy().tolist())
        self.assertListEqual([[-51.0]], test.model.all_parameters['a'].data.numpy().tolist())

    def test_training_values_fir_connect_linear(self):
        input1 = Input('in1')
        target = Input('out1')
        a = Parameter('a', values=[[1]])
        output1 = Output('out1',Fir(parameter=a)(input1.last()))

        inout = State('inout')
        W = Parameter('W', values=[[[1]]])
        b = Parameter('b', values=[[1]])
        output2 = Output('out2', Linear(W=W,b=b)(inout.last()))

        test = Neu4mes(seed=42)
        test.addModel('model', [output1,output2])
        test.addMinimize('error', target.last(), output2)
        test.addConnect(output1, inout)
        test.neuralizeModel()

        dataset = {'in1': [1], 'out1': [3]}
        test.loadData(name='dataset', source=dataset)
        print(test({'in1': [1]},connect={'inout':'out1'}))

        self.assertListEqual([[[1.0]]],test.model.all_parameters['W'].data.numpy().tolist())
        self.assertListEqual([[1.0]], test.model.all_parameters['b'].data.numpy().tolist())
        self.assertListEqual([[1.0]], test.model.all_parameters['a'].data.numpy().tolist())
        test.trainModel(optimizer='SGD', splits=[100, 0, 0], lr=1, num_of_epochs=1)
        self.assertListEqual([[[3.0]]], test.model.all_parameters['W'].data.numpy().tolist())
        self.assertListEqual([[3.0]], test.model.all_parameters['b'].data.numpy().tolist())
        self.assertListEqual([[3.0]], test.model.all_parameters['a'].data.numpy().tolist())
        test.trainModel(optimizer='SGD', splits=[100, 0, 0], lr=1, num_of_epochs=1)
        self.assertListEqual([[[-51.0]]],test.model.all_parameters['W'].data.numpy().tolist())
        self.assertListEqual([[-15.0]], test.model.all_parameters['b'].data.numpy().tolist())
        self.assertListEqual([[-51.0]], test.model.all_parameters['a'].data.numpy().tolist())
    def test_training_values_fir_train_connect_linear(self):
        input1 = Input('in1')
        target = Input('out1')
        a = Parameter('a', values=[[1]])
        output1 = Output('out1',Fir(parameter=a)(input1.last()))

        inout = Input('inout')
        W = Parameter('W', values=[[[1]]])
        b = Parameter('b', values=[[1]])
        output2 = Output('out2', Linear(W=W,b=b)(inout.last()))

        test = Neu4mes(seed=42)
        test.addModel('model', [output1,output2])
        test.addMinimize('error', target.last(), output2)
        test.neuralizeModel()

        dataset = {'in1': [1], 'out1': [3]}
        test.loadData(name='dataset', source=dataset)
        print(test({'in1': [1]},connect={'inout':'out1'}))

        self.assertListEqual([[[1.0]]],test.model.all_parameters['W'].data.numpy().tolist())
        self.assertListEqual([[1.0]], test.model.all_parameters['b'].data.numpy().tolist())
        self.assertListEqual([[1.0]], test.model.all_parameters['a'].data.numpy().tolist())
        test.trainModel(optimizer='SGD', splits=[100, 0, 0], lr=1, num_of_epochs=1, connect={'inout':'out1'})
        self.assertListEqual([[[3.0]]], test.model.all_parameters['W'].data.numpy().tolist())
        self.assertListEqual([[3.0]], test.model.all_parameters['b'].data.numpy().tolist())
        self.assertListEqual([[3.0]], test.model.all_parameters['a'].data.numpy().tolist())
        test.trainModel(optimizer='SGD', splits=[100, 0, 0], lr=1, num_of_epochs=1, connect={'inout':'out1'})
        self.assertListEqual([[[-51.0]]],test.model.all_parameters['W'].data.numpy().tolist())
        self.assertListEqual([[-15.0]], test.model.all_parameters['b'].data.numpy().tolist())
        self.assertListEqual([[-51.0]], test.model.all_parameters['a'].data.numpy().tolist())

    def test_multimodel_with_loss_gain_and_lr_gain(self):
        ## Model1
        input1 = Input('in1')
        a = Parameter('a', dimensions=1, tw=0.05, values=[[1],[1],[1],[1],[1]])
        output1 = Output('out1', Fir(parameter=a)(input1.tw(0.05)))

        test = Neu4mes(visualizer=None, seed=42)
        test.addModel('model1', output1)
        test.addMinimize('error1', input1.next(), output1)

        ## Model2
        input2 = Input('in2')
        b = Parameter('b', dimensions=1, tw=0.05, values=[[1],[1],[1],[1],[1]])
        output2 = Output('out2', Fir(parameter=b)(input2.tw(0.05)))

        test.addModel('model2', output2)
        test.addMinimize('error2', input2.next(), output2)
        test.neuralizeModel(0.01)

        data_in1 = np.linspace(0,5,6)
        data_in2 = np.linspace(10,15,6)
        data_out1 = 2
        data_out2 = -3
        dataset = {'in1': data_in1, 'in2': data_in2, 'out1': data_in1*data_out1, 'out2': data_in2*data_out2}

        test.loadData(name='dataset', source=dataset)

        ## Train only model1
        self.assertListEqual(test.model.all_parameters['a'].data.numpy().tolist(), [[1],[1],[1],[1],[1]])
        self.assertListEqual(test.model.all_parameters['b'].data.numpy().tolist(), [[1],[1],[1],[1],[1]])

        # Massimo livello di configurazione il livello è come pytorch
        # oppure posso anche crearmi il mio ottimizzatore
        # Quando optimizer è un optimizer
        opt_params = {
            'lr': 0.1
        }
        opt_params_per_parameter = {
            'a': {'lr':0.1, 'weight_decay':0}
        }
        opt = Optimizer(opt_params, opt_params_per_parameter)
        test.trainModel(models='model1', splits=[100, 0, 0], optimizer=opt, train_parameters=params)

        #livello intermadio vorrei poter aggiungere dei parametriche non sono presenti di default dentro l'ottimizzare

        # Livello di configurazione base
        # Voglio una configurazione abbastanza specifica anche per parametro
        # Quando optimizer è una stringa
        test.trainModel(models='model1', splits=[100, 0, 0], optimizer='SGD', lr = 0.5, lr_param={'a':0.1}, weight_decay=0, weight_decay_gain={'a':.1}, num_of_epochs= 100)

        # Vorrei che fosse facile aggiungere un altro ottimizzatore

        params['lr'] = 0.1
        params['lr_gain'] = {'a':0.1}
        test.trainModel(models='model1', splits=[100, 0, 0], training_params=params)

        # Così passo direttamente i parametri all'ottimizzatore???
        test.trainModel(models='model1', splits=[100, 0, 0], optimizer_params=params)

        training_params = {},

        optimizer = torch.optim.Adam,
        lr_gain = {},
        optimizer_params = {}

        test.trainModel(models='model1', splits=[100,0,0], optimizer=torch.optim.SGD, training_params=params, optimizer_params=optimizer_params)
        self.assertListEqual(test.model.all_parameters['a'].data.numpy().tolist(), [[0.20872743427753448],[0.20891857147216797],[0.20914430916309357],[0.20934967696666718],[0.20958690345287323]])
        self.assertListEqual(test.model.all_parameters['b'].data.numpy().tolist(), [[1],[1],[1],[1],[1]])

        test.neuralizeModel(0.01, clear_model=True)
        ## Train only model2
        self.assertListEqual(test.model.all_parameters['a'].data.numpy().tolist(), [[1],[1],[1],[1],[1]])
        self.assertListEqual(test.model.all_parameters['b'].data.numpy().tolist(), [[1],[1],[1],[1],[1]])
        test.trainModel(models='model2', splits=[100,0,0], training_params=params)
        self.assertListEqual(test.model.all_parameters['a'].data.numpy().tolist(), [[1],[1],[1],[1],[1]])
        self.assertListEqual(test.model.all_parameters['b'].data.numpy().tolist(), [[0.21510866284370422], [0.21509192883968353], [0.21507103741168976], [0.21505486965179443], [0.21503786742687225]])
        
        test.neuralizeModel(0.01, clear_model=True)
        ## Train both models
        self.assertListEqual(test.model.all_parameters['a'].data.numpy().tolist(), [[1],[1],[1],[1],[1]])
        self.assertListEqual(test.model.all_parameters['b'].data.numpy().tolist(), [[1],[1],[1],[1],[1]])
        test.trainModel(models=['model1','model2'], splits=[100,0,0], training_params=params)
        self.assertListEqual(test.model.all_parameters['a'].data.numpy().tolist(), [[0.2097606211900711],[0.20982888340950012],[0.20994682610034943],[0.21001523733139038],[0.21013548970222473]])
        self.assertListEqual(test.model.all_parameters['b'].data.numpy().tolist(), [[0.21503083407878876],[0.2150345891714096],[0.21503330767154694],[0.21502918004989624],[0.21503430604934692]])

        test.neuralizeModel(0.01, clear_model=True)
        ## Train both models but set the gain of a to zero and the gain of b to double
        self.assertListEqual(test.model.all_parameters['a'].data.numpy().tolist(), [[1],[1],[1],[1],[1]])
        self.assertListEqual(test.model.all_parameters['b'].data.numpy().tolist(), [[1],[1],[1],[1],[1]])
        test.trainModel(models=['model1','model2'], splits=[100,0,0], training_params=params, lr_gain={'a':0, 'b':2})
        self.assertListEqual(test.model.all_parameters['a'].data.numpy().tolist(), [[1],[1],[1],[1],[1]])
        self.assertListEqual(test.model.all_parameters['b'].data.numpy().tolist(), [[0.21878866851329803],[0.21873562037944794],[0.2186843752861023],[0.2186216115951538],[0.21856670081615448]])

        test.neuralizeModel(0.01, clear_model=True)
        ## Train both models but set the minimize gain of error1 to zero and the minimize gain of error2 to double
        self.assertListEqual(test.model.all_parameters['a'].data.numpy().tolist(), [[1],[1],[1],[1],[1]])
        self.assertListEqual(test.model.all_parameters['b'].data.numpy().tolist(), [[1],[1],[1],[1],[1]])
        train_loss, _, _ = test.trainModel(models=['model1','model2'], splits=[100,0,0], training_params=params, minimize_gain={'error1':0, 'error2':2})
        self.assertListEqual(train_loss['error1'], [0.0 for i in range(test.num_of_epochs)])
        self.assertListEqual(test.model.all_parameters['a'].data.numpy().tolist(), [[1],[1],[1],[1],[1]])
        self.assertListEqual(train_loss['error2'], [3894.033935546875, 1339.185546875, 182.2778778076172, 68.74202728271484, 359.0768127441406, 499.3464050292969, 368.579345703125, 146.50833129882812, 16.411840438842773, 18.409034729003906,
                                                    70.48310089111328, 85.69669342041016, 52.13644790649414, 12.842440605163574, 1.1463165283203125, 10.94512939453125, 18.03199005126953, 12.468052864074707, 3.2584829330444336, 0.3026407063007355])
        self.assertListEqual(test.model.all_parameters['b'].data.numpy().tolist(), [[0.21512822806835175],[0.21512599289417267],[0.2151111513376236],[0.21510501205921173],[0.21509882807731628]])
    
    def test_multimodel_with_connect(self):
        ## Model1
        input1 = Input('in1')
        a = Parameter('a', dimensions=1, tw=0.05, values=[[1],[1],[1],[1],[1]])
        output1 = Output('out1', Fir(parameter=a)(input1.tw(0.05)))

        test = Neu4mes(visualizer=None, seed=42)
        test.addModel('model1', output1)
        test.addMinimize('error1', input1.next(), output1)
        test.neuralizeModel(0.01)

        ## Model2
        input2 = Input('in2')
        input3 = Input('in3')
        b = Parameter('b', dimensions=1, tw=0.05, values=[[1],[1],[1],[1],[1]])
        c = Parameter('c', dimensions=1, tw=0.03, values=[[1],[1],[1]])
        output2 = Output('out2', Fir(parameter=b)(input2.tw(0.05))+Fir(parameter=c)(input3.tw(0.03)))

        test.addModel('model2', output2)
        test.addMinimize('error2', input2.next(), output2)
        test.neuralizeModel(0.01)

        data_struct = ['x','F','x2','y2','','A1x','A1y','B1x','B1y','','A2x','A2y','B2x','out','','x3','in1','in2','time']
        test.loadData(name='dataset', source=data_folder, format=data_struct, skiplines=4, delimiter='\t', header=None)

        params = {'num_of_epochs': 1, 
          'train_batch_size': 3, 
          'val_batch_size': 1, 
          'test_batch_size':1, 
          'lr':0.1}
        
        test.trainModel(splits=[100,0,0], training_params=params, lr_param={'a':0, 'b':0, 'c':0}, prediction_samples=3, connect={'in3':'out1'}, shuffle_data=False)
    
    
if __name__ == '__main__':
    unittest.main()