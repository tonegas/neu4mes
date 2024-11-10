import unittest, os, sys
sys.path.append(os.getcwd())
from neu4mes import *
from neu4mes import relation
relation.CHECK_NAMES = False

from neu4mes.logger import logging, Neu4MesLogger
log = Neu4MesLogger(__name__, logging.CRITICAL)
log.setAllLevel(logging.CRITICAL)

data_folder = os.path.join(os.path.dirname(__file__), 'data/')

# 13 Tests
# Test the train parameter and the optimizer options

def funIn(x, w):
    return x * w

def funOut(x, w):
    return x / w

def linear_fun(x,a,b):
    return x*a+b

class Neu4mesTrainingTestParameter(unittest.TestCase):
    def TestAlmostEqual(self, data1, data2, precision=4):
        assert np.asarray(data1, dtype=np.float32).ndim == np.asarray(data2, dtype=np.float32).ndim, f'Inputs must have the same dimension! Received {type(data1)} and {type(data2)}'
        if type(data1) == type(data2) == list:
            for pred, label in zip(data1, data2):
                self.TestAlmostEqual(pred, label, precision=precision)
        else:
            self.assertAlmostEqual(data1, data2, places=precision)

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
        self.assertEqual(1, test.run_training_params['update_per_epochs'])
        self.assertEqual(0, test.run_training_params['unused_samples'])
        self.assertEqual(1,test.run_training_params['val_batch_size'])
        self.assertEqual(1,test.run_training_params['test_batch_size'])
        self.assertEqual(100,test.run_training_params['num_of_epochs'])
        self.assertEqual(0.001,test.run_training_params['optimizer_defaults']['lr'])

    def test_build_dataset_batch_connect(self):
        data_x = np.random.rand(500) * 20 - 10
        data_a = 2
        data_b = -3
        dataset = {'in1': data_x, 'out': linear_fun(data_x, data_a, data_b)}

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
        training_params['lr'] = 0.1
        training_params['num_of_epochs'] = 5
        test.trainModel(splits=[70,20,10], closed_loop={'in1':'y'}, prediction_samples=5, step=1, training_params = training_params)

        self.assertEqual(346,test.run_training_params['n_samples_train']) ## ((500 - 5) * 0.7)  = 346
        self.assertEqual(99,test.run_training_params['n_samples_val']) ## ((500 - 5) * 0.2)  = 99
        self.assertEqual(50,test.run_training_params['n_samples_test']) ## ((500 - 5) * 0.1)  = 50
        self.assertEqual(495,test.num_of_samples['dataset'])
        self.assertEqual(4,test.run_training_params['train_batch_size'])
        self.assertEqual(4,test.run_training_params['val_batch_size'])
        self.assertEqual(1,test.run_training_params['test_batch_size'])
        self.assertEqual(5,test.run_training_params['num_of_epochs'])
        self.assertEqual(5, test.run_training_params['prediction_samples'])
        self.assertEqual(1, test.run_training_params['step'])
        self.assertEqual({'in1':'y'}, test.run_training_params['closed_loop'])
        self.assertEqual(0.1,test.run_training_params['optimizer_defaults']['lr'])

        n_samples = test.run_training_params['n_samples_train']
        batch_size = test.run_training_params['train_batch_size']
        prediction_samples = test.run_training_params['prediction_samples']
        step = test.run_training_params['step']
        list_of_batch_indexes = range(0, n_samples - batch_size - prediction_samples + 1, (batch_size + step - 1))
        self.assertEqual(len(list_of_batch_indexes), test.run_training_params['update_per_epochs'])
        #n_samples - list_of_batch_indexes[-1] - batch_size - prediction_samples
        self.assertEqual(1, test.run_training_params['unused_samples'])

    def test_recurrent_train_closed_loop(self):
        data_x = np.random.rand(500) * 20 - 10
        data_a = 2
        data_b = -3
        dataset = {'in1': data_x, 'out': linear_fun(data_x, data_a, data_b)}

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
        training_params['lr'] = 0.1
        training_params['num_of_epochs'] = 50

        test.trainModel(splits=[100,0,0], closed_loop={'in1':'out'}, prediction_samples=3, step=1, training_params = training_params)

        self.assertEqual((len(data_x)-1)*100/100,test.run_training_params['n_samples_train']) ## ((500 - 1) * 1)  = 499
        self.assertEqual(0,test.run_training_params['n_samples_val']) ## ((500 - 5) * 0)  = 0
        self.assertEqual(0,test.run_training_params['n_samples_test']) ## ((500 - 5) * 0)  = 0
        self.assertEqual((len(data_x)-1)*100/100,test.num_of_samples['dataset'])
        self.assertEqual(4,test.run_training_params['train_batch_size'])
        self.assertEqual(0,test.run_training_params['val_batch_size'])
        self.assertEqual(0,test.run_training_params['test_batch_size'])
        self.assertEqual(50,test.run_training_params['num_of_epochs'])
        self.assertEqual(3, test.run_training_params['prediction_samples'])
        self.assertEqual(1, test.run_training_params['step'])
        self.assertEqual({'in1':'out'}, test.run_training_params['closed_loop'])
        self.assertEqual(0.1,test.run_training_params['optimizer_defaults']['lr'])

        n_samples = test.run_training_params['n_samples_train']
        batch_size = test.run_training_params['train_batch_size']
        prediction_samples = test.run_training_params['prediction_samples']
        step = test.run_training_params['step']
        list_of_batch_indexes = range(0, n_samples - batch_size - prediction_samples + 1, (batch_size + step - 1))
        self.assertEqual(len(list_of_batch_indexes), test.run_training_params['update_per_epochs'])
        self.assertEqual(n_samples - list_of_batch_indexes[-1] - batch_size - prediction_samples, test.run_training_params['unused_samples'])

    def test_recurrent_train_single_close_loop(self):
        data_x = np.array(list(range(1, 101, 1)), dtype=np.float32)
        dataset = {'x': data_x, 'y': 2 * data_x}

        x = Input('x')
        y = Input('y')
        out = Output('out', Fir(x.last()))

        test = Neu4mes(visualizer=None, seed=42)
        test.addModel('out', out)
        test.addMinimize('pos', y.last(), out)
        test.neuralizeModel(0.01)

        test.loadData(name='dataset', source=dataset)

        training_params = {}
        training_params['train_batch_size'] = 4
        training_params['val_batch_size'] = 4
        training_params['test_batch_size'] = 1
        training_params['lr'] = 0.01
        training_params['num_of_epochs'] = 50
        test.trainModel(splits=[80, 20, 0], closed_loop={'x': 'out'}, prediction_samples=3, step=3,
                        training_params=training_params)

        self.assertEqual(round((len(data_x) - 0) * 80 / 100), test.run_training_params['n_samples_train'])
        self.assertEqual((len(data_x) - 0) * 20 / 100, test.run_training_params['n_samples_val'])
        self.assertEqual(0, test.run_training_params['n_samples_test'])
        self.assertEqual((len(data_x) - 0) * 100 / 100, test.num_of_samples['dataset'])
        self.assertEqual(4, test.run_training_params['train_batch_size'])
        self.assertEqual(4, test.run_training_params['val_batch_size'])
        self.assertEqual(0, test.run_training_params['test_batch_size'])
        self.assertEqual(50, test.run_training_params['num_of_epochs'])
        self.assertEqual(3, test.run_training_params['prediction_samples'])
        self.assertEqual(3, test.run_training_params['step'])
        self.assertEqual({'x': 'out'}, test.run_training_params['closed_loop'])
        self.assertEqual(0.01, test.run_training_params['optimizer_defaults']['lr'])

        n_samples = test.run_training_params['n_samples_train']
        batch_size = test.run_training_params['train_batch_size']
        prediction_samples = test.run_training_params['prediction_samples']
        step = test.run_training_params['step']
        list_of_batch_indexes = range(0, n_samples - batch_size - prediction_samples + 1, (batch_size + step - 1))
        self.assertEqual(len(list_of_batch_indexes), test.run_training_params['update_per_epochs'])
        self.assertEqual(n_samples - list_of_batch_indexes[-1] - batch_size - prediction_samples, test.run_training_params['unused_samples'])

    def test_recurrent_train_multiple_close_loop(self):
        data_x = np.array(list(range(1, 101, 1)), dtype=np.float32)
        dataset = {'x': data_x, 'y': 2 * data_x}

        x = Input('x')
        y = Input('y')
        out_x = Output('out_x', Fir(x.last()))
        out_y = Output('out_y', Fir(y.last()))

        test = Neu4mes(visualizer=None, seed=42)
        test.addModel('out_x', out_x)
        test.addModel('out_y', out_y)
        test.addMinimize('pos_x', x.next(), out_x)
        test.addMinimize('pos_y', y.next(), out_y)
        test.neuralizeModel(0.01)

        test.loadData(name='dataset', source=dataset)

        training_params = {}
        training_params['train_batch_size'] = 4
        training_params['val_batch_size'] = 4
        training_params['test_batch_size'] = 1
        training_params['lr'] = 0.01
        training_params['num_of_epochs'] = 32

        test.trainModel(splits=[80, 20, 0], closed_loop={'x': 'out_x', 'y': 'out_y'}, prediction_samples=3, step=1,
                        training_params=training_params)

        self.assertEqual(round((len(data_x) - 1) * 80 / 100), test.run_training_params['n_samples_train'])
        self.assertEqual(round((len(data_x) - 1) * 20 / 100), test.run_training_params['n_samples_val'])
        self.assertEqual(0, test.run_training_params['n_samples_test'])
        self.assertEqual((len(data_x) - 1) * 100 / 100, test.num_of_samples['dataset'])
        self.assertEqual(4, test.run_training_params['train_batch_size'])
        self.assertEqual(4, test.run_training_params['val_batch_size'])
        self.assertEqual(0, test.run_training_params['test_batch_size'])
        self.assertEqual(32, test.run_training_params['num_of_epochs'])
        self.assertEqual(3, test.run_training_params['prediction_samples'])
        self.assertEqual(1, test.run_training_params['step'])
        self.assertEqual({'x': 'out_x', 'y': 'out_y'}, test.run_training_params['closed_loop'])
        self.assertEqual(0.01, test.run_training_params['optimizer_defaults']['lr'])

        n_samples = test.run_training_params['n_samples_train']
        batch_size = test.run_training_params['train_batch_size']
        prediction_samples = test.run_training_params['prediction_samples']
        step = test.run_training_params['step']
        list_of_batch_indexes = range(0, n_samples - batch_size - prediction_samples + 1, (batch_size + step - 1))
        self.assertEqual(len(list_of_batch_indexes), test.run_training_params['update_per_epochs'])
        self.assertEqual(n_samples - list_of_batch_indexes[-1] - batch_size - prediction_samples, test.run_training_params['unused_samples'])

        # print('test before train: ', test(inputs={'x': [100, 101, 102, 103, 104], 'y': [200, 202, 204, 206, 208]}))
        # print('test after train: ', test(inputs={'x': [100, 101, 102, 103, 104], 'y': [200, 202, 204, 206, 208]}))

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

        n_samples = test.run_training_params['n_samples_train']
        batch_size = test.run_training_params['train_batch_size']
        list_of_batch_indexes = range(0, n_samples - batch_size + 1, batch_size)
        self.assertEqual(len(list_of_batch_indexes), test.run_training_params['update_per_epochs'])
        self.assertEqual(n_samples - list_of_batch_indexes[-1] - batch_size, test.run_training_params['unused_samples'])

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

        n_samples = test.run_training_params['n_samples_train']
        batch_size = test.run_training_params['train_batch_size']
        list_of_batch_indexes = range(0, n_samples - batch_size + 1, (batch_size - 1))
        self.assertEqual(len(list_of_batch_indexes), test.run_training_params['update_per_epochs'])
        self.assertEqual(n_samples - list_of_batch_indexes[-1] - batch_size, test.run_training_params['unused_samples'])

    def test_build_dataset_batch3(self):
        input1 = Input('in1')
        output = Input('out')
        rel1 = Fir(input1.tw(0.05))

        test = Neu4mes(visualizer=None)
        test.addMinimize('out', output.next(), rel1)
        test.neuralizeModel(0.01)

        data_struct = ['x', 'F', 'x2', 'y2', '', 'A1x', 'A1y', 'B1x', 'B1y', '', 'A2x', 'A2y', 'B2x', 'out', '', 'x3',
                       'in1', 'in2', 'time']
        test.loadData(name='dataset', source=data_folder, format=data_struct, skiplines=4, delimiter='\t', header=None)
        self.assertEqual((10, 5, 1), test.data['dataset']['in1'].shape)

        training_params = {}
        training_params['train_batch_size'] = 2
        training_params['val_batch_size'] = 2
        training_params['test_batch_size'] = 2
        training_params['lr'] = 0.1
        training_params['num_of_epochs'] = 5
        test.trainModel(splits=[40, 30, 30], training_params=training_params)

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

        n_samples = test.run_training_params['n_samples_train']
        batch_size = test.run_training_params['train_batch_size']
        list_of_batch_indexes = range(0, n_samples - batch_size + 1, batch_size)
        self.assertEqual(len(list_of_batch_indexes), test.run_training_params['update_per_epochs'])
        self.assertEqual(n_samples - list_of_batch_indexes[-1] - batch_size, test.run_training_params['unused_samples'])

    def test_build_dataset_batch4(self):
        input1 = Input('in1')
        output = Input('out')
        rel1 = Fir(input1.tw(0.05))

        test = Neu4mes(visualizer=None)
        test.addMinimize('out', output.z(-1), rel1)
        test.neuralizeModel(0.01)

        data_struct = ['x', 'F', 'x2', 'y2', '', 'A1x', 'A1y', 'B1x', 'B1y', '', 'A2x', 'A2y', 'B2x', 'out', '', 'x3',
                       'in1', 'in2', 'time']
        test.loadData(name='dataset', source=data_folder, format=data_struct, skiplines=4, delimiter='\t', header=None)
        self.assertEqual((10, 5, 1), test.data['dataset']['in1'].shape)

        training_params = {}
        training_params['train_batch_size'] = 2
        training_params['val_batch_size'] = 2
        training_params['test_batch_size'] = 2
        training_params['lr'] = 0.1
        training_params['num_of_epochs'] = 5
        test.trainModel(splits=[80, 10, 10], training_params=training_params)

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

        n_samples = test.run_training_params['n_samples_train']
        batch_size = test.run_training_params['train_batch_size']
        list_of_batch_indexes = range(0, n_samples - batch_size + 1, batch_size)
        self.assertEqual(len(list_of_batch_indexes), test.run_training_params['update_per_epochs'])
        self.assertEqual(n_samples - list_of_batch_indexes[-1] - batch_size, test.run_training_params['unused_samples'])

    def test_build_dataset_from_code(self):
        input1 = Input('in1')
        output = Input('out')
        rel1 = Fir(input1.tw(0.05))

        test = Neu4mes(visualizer=None)
        test.addMinimize('out', output.next(), rel1)
        test.neuralizeModel(0.01)

        x_size = 20
        data_x = np.random.rand(x_size) * 20 - 10
        data_a = 2
        data_b = -3
        dataset = {'in1': data_x, 'out': data_x * data_a + data_b}

        test.loadData(name='dataset', source=dataset, skiplines=0)
        self.assertEqual((15, 5, 1),
                         test.data['dataset']['in1'].shape)  ## 20 data - 5 tw = 15 sample | 0.05/0.01 = 5 in1

        training_params = {}
        training_params['train_batch_size'] = 2
        training_params['val_batch_size'] = 2
        training_params['test_batch_size'] = 2
        training_params['lr'] = 0.1
        training_params['num_of_epochs'] = 5
        test.trainModel(splits=[80, 20, 0], training_params=training_params)

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

        n_samples = test.run_training_params['n_samples_train']
        batch_size = test.run_training_params['train_batch_size']
        list_of_batch_indexes = range(0, (n_samples - batch_size + 1), batch_size)
        self.assertEqual(len(list_of_batch_indexes), test.run_training_params['update_per_epochs'])
        self.assertEqual(n_samples - list_of_batch_indexes[-1] - batch_size, test.run_training_params['unused_samples'])

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
        test.addModel('x_z', x_z)
        test.addMinimize('next-pos', x.z(-1), x_z, 'mse')

        # Create the neural network
        test.neuralizeModel(sample_time=0.05)  # The sampling time depends to the dataset

        # Data load
        data_struct = ['x', 'F', 'x2', 'y2', '', 'A1x', 'A1y', 'B1x', 'B1y', '', 'A2x', 'A2y', 'B2x', 'out', '', 'x3',
                       'in1', 'in2', 'time']
        test.loadData(name='train_dataset', source=train_folder, format=data_struct, skiplines=4, delimiter='\t',
                      header=None)
        test.loadData(name='validation_dataset', source=val_folder, format=data_struct, skiplines=4, delimiter='\t',
                      header=None)
        test.loadData(name='test_dataset', source=test_folder, format=data_struct, skiplines=4, delimiter='\t',
                      header=None)

        training_params = {}
        training_params['train_batch_size'] = 3
        training_params['val_batch_size'] = 2
        training_params['test_batch_size'] = 1
        training_params['lr'] = 0.1
        training_params['num_of_epochs'] = 5
        test.trainModel(train_dataset='train_dataset', validation_dataset='validation_dataset',
                        test_dataset='test_dataset', training_params=training_params)

        self.assertEqual(9, test.num_of_samples['train_dataset'])
        self.assertEqual(5, test.num_of_samples['validation_dataset'])
        self.assertEqual(7, test.num_of_samples['test_dataset'])
        self.assertEqual(9, test.run_training_params['n_samples_train'])
        self.assertEqual(5, test.run_training_params['n_samples_val'])
        self.assertEqual(7, test.run_training_params['n_samples_test'])
        self.assertEqual(3, test.run_training_params['train_batch_size'])
        self.assertEqual(2, test.run_training_params['val_batch_size'])
        self.assertEqual(1, test.run_training_params['test_batch_size'])
        self.assertEqual(5, test.run_training_params['num_of_epochs'])
        self.assertEqual(0.1, test.run_training_params['optimizer_defaults']['lr'])

        n_samples = test.run_training_params['n_samples_train']
        batch_size = test.run_training_params['train_batch_size']
        list_of_batch_indexes = range(0, n_samples - batch_size + 1, batch_size)
        self.assertEqual(len(list_of_batch_indexes), test.run_training_params['update_per_epochs'])
        self.assertEqual(n_samples - list_of_batch_indexes[-1] - batch_size, test.run_training_params['unused_samples'])

    def test_train_vector_input(self):
        x = Input('x', dimensions=4)
        y = Input('y', dimensions=3)
        k = Input('k', dimensions=2)
        w = Input('w')

        out = Output('out', Fir(Linear(Linear(3)(x.tw(0.02)) + y.tw(0.02))))
        out2 = Output('out2', Fir(Linear(k.last() + Fir(2)(w.tw(0.05, offset=-0.02)))))

        test = Neu4mes(visualizer=None)
        test.addMinimize('out', out, out2)
        test.neuralizeModel(0.01)

        data_folder = os.path.join(os.path.dirname(__file__), 'vector_data/')
        data_struct = ['x', 'y', '', '', '', '', 'k', '', '', '', 'w']
        test.loadData(name='dataset', source=data_folder, format=data_struct, skiplines=1, delimiter='\t', header=None)

        training_params = {}
        training_params['train_batch_size'] = 1
        training_params['val_batch_size'] = 1
        training_params['test_batch_size'] = 1
        training_params['lr'] = 0.01
        training_params['num_of_epochs'] = 7
        test.trainModel(splits=[80, 10, 10], training_params=training_params)

        self.assertEqual(22, test.num_of_samples['dataset'])
        self.assertEqual(18, test.run_training_params['n_samples_train'])
        self.assertEqual(2, test.run_training_params['n_samples_val'])
        self.assertEqual(2, test.run_training_params['n_samples_test'])
        self.assertEqual(1, test.run_training_params['train_batch_size'])
        self.assertEqual(1, test.run_training_params['val_batch_size'])
        self.assertEqual(1, test.run_training_params['test_batch_size'])
        self.assertEqual(7, test.run_training_params['num_of_epochs'])
        self.assertEqual(0.01, test.run_training_params['optimizer_defaults']['lr'])

        n_samples = test.run_training_params['n_samples_train']
        batch_size = test.run_training_params['train_batch_size']
        list_of_batch_indexes = range(0, n_samples - batch_size + 1, batch_size)
        self.assertEqual(len(list_of_batch_indexes), test.run_training_params['update_per_epochs'])
        self.assertEqual(n_samples - list_of_batch_indexes[-1] - batch_size, test.run_training_params['unused_samples'])

        training_params = {}
        training_params['train_batch_size'] = 6
        training_params['val_batch_size'] = 2
        training_params['test_batch_size'] = 2
        test.trainModel(splits=[80, 10, 10], training_params=training_params)

        self.assertEqual(22, test.num_of_samples['dataset'])
        self.assertEqual(18, test.run_training_params['n_samples_train'])
        self.assertEqual(2, test.run_training_params['n_samples_val'])
        self.assertEqual(2, test.run_training_params['n_samples_test'])
        self.assertEqual(6, test.run_training_params['train_batch_size'])
        self.assertEqual(2, test.run_training_params['val_batch_size'])
        self.assertEqual(2, test.run_training_params['test_batch_size'])
        self.assertEqual(100, test.run_training_params['num_of_epochs'])
        self.assertEqual(0.001, test.run_training_params['optimizer_defaults']['lr'])

        n_samples = test.run_training_params['n_samples_train']
        batch_size = test.run_training_params['train_batch_size']
        list_of_batch_indexes = range(0, n_samples - batch_size + 1, batch_size)
        self.assertEqual(len(list_of_batch_indexes), test.run_training_params['update_per_epochs'])
        self.assertEqual(n_samples - list_of_batch_indexes[-1] - batch_size, test.run_training_params['unused_samples'])

    def test_optimizer_configuration(self):
        ## Model1
        input1 = Input('in1')
        a = Parameter('a', dimensions=1, tw=0.05, values=[[1], [1], [1], [1], [1]])
        shared_w = Parameter('w', values=[[5]])
        output1 = Output('out1',
                         Fir(parameter=a)(input1.tw(0.05)) + ParamFun(funIn, parameters={'w': shared_w})(
                             input1.last()))

        test = Neu4mes(visualizer=None, seed=42)
        test.addModel('model1', output1)
        test.addMinimize('error1', input1.last(), output1)

        ## Model2
        input2 = Input('in2')
        b = Parameter('b', dimensions=1, tw=0.05, values=[[1], [1], [1], [1], [1]])
        output2 = Output('out2',
                         Fir(parameter=b)(input2.tw(0.05)) + ParamFun(funOut, parameters={'w': shared_w})(
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
        self.assertEqual(39, test.run_training_params['n_samples_train'])
        self.assertEqual(11, test.run_training_params['n_samples_val'])
        self.assertEqual(6, test.run_training_params['n_samples_test'])
        self.assertEqual(100, test.run_training_params['num_of_epochs'])
        self.assertEqual(0.001, test.run_training_params['optimizer_defaults']['lr'])

        n_samples = test.run_training_params['n_samples_train']
        batch_size = test.run_training_params['train_batch_size']
        list_of_batch_indexes = range(0, n_samples - batch_size + 1, batch_size)
        self.assertEqual(len(list_of_batch_indexes), test.run_training_params['update_per_epochs'])
        self.assertEqual(n_samples - list_of_batch_indexes[-1] - batch_size, test.run_training_params['unused_samples'])

        # We train only model1 with split [100,0,0]
        # TODO Learning rate automoatically optimized based on the mean and variance of the output
        # TODO num_of_epochs automatically defined
        # now is 0.001 for learning rate and 100 for the epochs and optimizer Adam
        test.trainModel(models='model1', splits=[100, 0, 0])
        self.assertEqual('model1', test.run_training_params['models'])
        self.assertEqual(100, test.run_training_params['num_of_epochs'])
        self.assertEqual(56, test.run_training_params['n_samples_train'])
        self.assertEqual(0, test.run_training_params['n_samples_val'])
        self.assertEqual(0, test.run_training_params['n_samples_test'])

        n_samples = test.run_training_params['n_samples_train']
        batch_size = test.run_training_params['train_batch_size']
        list_of_batch_indexes = range(0, n_samples - batch_size + 1, batch_size)
        self.assertEqual(len(list_of_batch_indexes), test.run_training_params['update_per_epochs'])
        self.assertEqual(n_samples - list_of_batch_indexes[-1] - batch_size, test.run_training_params['unused_samples'])

        # Set number of epoch and learning rate via parameters it works only for standard parameters
        test.trainModel(models='model1', splits=[100, 0, 0], lr=0.5, num_of_epochs=5)
        self.assertEqual('model1', test.run_training_params['models'])
        self.assertEqual(5, test.run_training_params['num_of_epochs'])
        self.assertEqual(56, test.run_training_params['n_samples_train'])
        self.assertEqual(0, test.run_training_params['n_samples_val'])
        self.assertEqual(0, test.run_training_params['n_samples_test'])
        self.assertEqual(0.5, test.run_training_params['optimizer_defaults']['lr'])

        n_samples = test.run_training_params['n_samples_train']
        batch_size = test.run_training_params['train_batch_size']
        list_of_batch_indexes = range(0, n_samples - batch_size + 1, batch_size)
        self.assertEqual(len(list_of_batch_indexes), test.run_training_params['update_per_epochs'])
        self.assertEqual(n_samples - list_of_batch_indexes[-1] - batch_size, test.run_training_params['unused_samples'])

        # Set number of epoch and learning rate via parameters it works only for standard parameters and use two different dataset one for train and one for validation
        test.trainModel(models='model1', train_dataset='dataset1', validation_dataset='dataset2', lr=0.6,
                        num_of_epochs=10)
        self.assertEqual('model1', test.run_training_params['models'])
        self.assertEqual(10, test.run_training_params['num_of_epochs'])
        self.assertEqual(56, test.run_training_params['n_samples_train'])
        self.assertEqual(96, test.run_training_params['n_samples_val'])
        self.assertEqual(0, test.run_training_params['n_samples_test'])
        self.assertEqual(0.6, test.run_training_params['optimizer_defaults']['lr'])

        n_samples = test.run_training_params['n_samples_train']
        batch_size = test.run_training_params['train_batch_size']
        list_of_batch_indexes = range(0, n_samples - batch_size + 1, batch_size)
        self.assertEqual(len(list_of_batch_indexes), test.run_training_params['update_per_epochs'])
        self.assertEqual(n_samples - list_of_batch_indexes[-1] - batch_size, test.run_training_params['unused_samples'])

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
        self.assertEqual(round(56 * 55 / 100), test.run_training_params['n_samples_train'])
        self.assertEqual(round(56 * 40 / 100), test.run_training_params['n_samples_val'])
        self.assertEqual(round(56 * 5 / 100), test.run_training_params['n_samples_test'])
        self.assertEqual(0.7, test.run_training_params['optimizer_defaults']['lr'])

        n_samples = test.run_training_params['n_samples_train']
        batch_size = test.run_training_params['train_batch_size']
        list_of_batch_indexes = range(0, n_samples - batch_size + 1, batch_size)
        self.assertEqual(len(list_of_batch_indexes), test.run_training_params['update_per_epochs'])
        self.assertEqual(n_samples - list_of_batch_indexes[-1] - batch_size, test.run_training_params['unused_samples'])

        # If I add a function parameter it has the priority
        # In this case apply train parameter but on a different model
        test.trainModel(models='model1', training_params=training_params)
        self.assertEqual('model1', test.run_training_params['models'])
        self.assertEqual(20, test.run_training_params['num_of_epochs'])
        self.assertEqual(round(56 * 55 / 100), test.run_training_params['n_samples_train'])
        self.assertEqual(round(56 * 40 / 100), test.run_training_params['n_samples_val'])
        self.assertEqual(round(56 * 5 / 100), test.run_training_params['n_samples_test'])
        self.assertEqual(0.7, test.run_training_params['optimizer_defaults']['lr'])

        n_samples = test.run_training_params['n_samples_train']
        batch_size = test.run_training_params['train_batch_size']
        list_of_batch_indexes = range(0, n_samples - batch_size + 1, batch_size)
        self.assertEqual(len(list_of_batch_indexes), test.run_training_params['update_per_epochs'])
        self.assertEqual(n_samples - list_of_batch_indexes[-1] - batch_size, test.run_training_params['unused_samples'])

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
        self.assertEqual(round(56 * 55 / 100), test.run_training_params['n_samples_train'])
        self.assertEqual(round(56 * 40 / 100), test.run_training_params['n_samples_val'])
        self.assertEqual(round(56 * 5 / 100), test.run_training_params['n_samples_test'])
        self.assertEqual(0.2, test.run_training_params['optimizer_defaults']['lr'])
        self.assertEqual((0.5, 0.99), test.run_training_params['optimizer_defaults']['betas'])

        n_samples = test.run_training_params['n_samples_train']
        batch_size = test.run_training_params['train_batch_size']
        list_of_batch_indexes = range(0, n_samples - batch_size + 1, batch_size)
        self.assertEqual(len(list_of_batch_indexes), test.run_training_params['update_per_epochs'])
        self.assertEqual(n_samples - list_of_batch_indexes[-1] - batch_size, test.run_training_params['unused_samples'])

        test.trainModel(training_params=training_params, optimizer_defaults=optimizer_defaults)
        self.assertEqual(['model2'], test.run_training_params['models'])
        self.assertEqual(20, test.run_training_params['num_of_epochs'])
        self.assertEqual(round(56 * 55 / 100), test.run_training_params['n_samples_train'])
        self.assertEqual(round(56 * 40 / 100), test.run_training_params['n_samples_val'])
        self.assertEqual(round(56 * 5 / 100), test.run_training_params['n_samples_test'])
        self.assertEqual(0.1, test.run_training_params['optimizer_defaults']['lr'])
        self.assertEqual((0.5, 0.99), test.run_training_params['optimizer_defaults']['betas'])

        n_samples = test.run_training_params['n_samples_train']
        batch_size = test.run_training_params['train_batch_size']
        list_of_batch_indexes = range(0, n_samples - batch_size + 1, batch_size)
        self.assertEqual(len(list_of_batch_indexes), test.run_training_params['update_per_epochs'])
        self.assertEqual(n_samples - list_of_batch_indexes[-1] - batch_size, test.run_training_params['unused_samples'])

        test.trainModel(training_params=training_params)
        self.assertEqual(['model2'], test.run_training_params['models'])
        self.assertEqual(20, test.run_training_params['num_of_epochs'])
        self.assertEqual(round(56 * 55 / 100), test.run_training_params['n_samples_train'])
        self.assertEqual(round(56 * 40 / 100), test.run_training_params['n_samples_val'])
        self.assertEqual(round(56 * 5 / 100), test.run_training_params['n_samples_test'])
        self.assertEqual(0.7, test.run_training_params['optimizer_defaults']['lr'])

        n_samples = test.run_training_params['n_samples_train']
        batch_size = test.run_training_params['train_batch_size']
        list_of_batch_indexes = range(0, n_samples - batch_size + 1, batch_size)
        self.assertEqual(len(list_of_batch_indexes), test.run_training_params['update_per_epochs'])
        self.assertEqual(n_samples - list_of_batch_indexes[-1] - batch_size, test.run_training_params['unused_samples'])
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
        self.assertEqual(round(56 * 55 / 100), test.run_training_params['n_samples_train'])
        self.assertEqual(round(56 * 40 / 100), test.run_training_params['n_samples_val'])
        self.assertEqual(round(56 * 5 / 100), test.run_training_params['n_samples_test'])
        self.assertEqual(0.2, test.run_training_params['optimizer_defaults']['lr'])
        self.assertEqual(0.002, test.run_training_params['optimizer_defaults']['momentum'])

        n_samples = test.run_training_params['n_samples_train']
        batch_size = test.run_training_params['train_batch_size']
        list_of_batch_indexes = range(0, n_samples - batch_size + 1, batch_size)
        self.assertEqual(len(list_of_batch_indexes), test.run_training_params['update_per_epochs'])
        self.assertEqual(n_samples - list_of_batch_indexes[-1] - batch_size, test.run_training_params['unused_samples'])

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
        self.assertEqual(round(56 * 100 / 100), test.run_training_params['n_samples_train'])
        self.assertEqual(round(56 * 0 / 100), test.run_training_params['n_samples_val'])
        self.assertEqual(round(56 * 0 / 100), test.run_training_params['n_samples_test'])
        self.assertEqual(0.5, test.run_training_params['optimizer_defaults']['lr'])
        self.assertEqual([{'lr': 0.1, 'params': 'a'},
                          {'lr': 0.0, 'params': 'b'},
                          {'params': 'w'}], test.run_training_params['optimizer_params'])

        n_samples = test.run_training_params['n_samples_train']
        batch_size = test.run_training_params['train_batch_size']
        list_of_batch_indexes = range(0, n_samples - batch_size + 1, batch_size)
        self.assertEqual(len(list_of_batch_indexes), test.run_training_params['update_per_epochs'])
        self.assertEqual(n_samples - list_of_batch_indexes[-1] - batch_size, test.run_training_params['unused_samples'])

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
        test.trainModel(training_params=training_params, optimizer_params=optimizer_params,
                        optimizer_defaults=optimizer_defaults, lr_param={'a': 0.4})
        self.assertEqual(['model1'], test.run_training_params['models'])
        self.assertEqual(40, test.run_training_params['num_of_epochs'])
        self.assertEqual(round(56 * 100 / 100), test.run_training_params['n_samples_train'])
        self.assertEqual(round(56 * 0 / 100), test.run_training_params['n_samples_val'])
        self.assertEqual(round(56 * 0 / 100), test.run_training_params['n_samples_test'])
        self.assertEqual(0.2, test.run_training_params['optimizer_defaults']['lr'])
        self.assertEqual([{'lr': 0.4, 'params': 'a'}], test.run_training_params['optimizer_params'])

        n_samples = test.run_training_params['n_samples_train']
        batch_size = test.run_training_params['train_batch_size']
        list_of_batch_indexes = range(0, n_samples - batch_size + 1, batch_size)
        self.assertEqual(len(list_of_batch_indexes), test.run_training_params['update_per_epochs'])
        self.assertEqual(n_samples - list_of_batch_indexes[-1] - batch_size, test.run_training_params['unused_samples'])

        test.trainModel(training_params=training_params, optimizer_params=optimizer_params,
                        optimizer_defaults=optimizer_defaults)
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

        n_samples = test.run_training_params['n_samples_train']
        batch_size = test.run_training_params['train_batch_size']
        list_of_batch_indexes = range(0, n_samples - batch_size + 1, batch_size)
        self.assertEqual(len(list_of_batch_indexes), test.run_training_params['update_per_epochs'])
        self.assertEqual(n_samples - list_of_batch_indexes[-1] - batch_size, test.run_training_params['unused_samples'])

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
        self.assertEqual(round(56 * 100 / 100), test.run_training_params['n_samples_train'])
        self.assertEqual(round(56 * 0 / 100), test.run_training_params['n_samples_val'])
        self.assertEqual(round(56 * 0 / 100), test.run_training_params['n_samples_test'])
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
        self.assertEqual([{'lr': 0.1, 'params': 'a'}, {'lr': 0.0, 'params': 'b'}, {'params': 'w'}],
                         test.run_training_params['optimizer_params'])

        test.trainModel(optimizer=optimizer)
        self.assertEqual({'alpha': 0.8, 'lr': 0.001}, test.run_training_params['optimizer_defaults'])
        self.assertEqual([{'params': 'a'}, {'params': 'b'}, {'params': 'w'}],
                         test.run_training_params['optimizer_params'])

        n_samples = test.run_training_params['n_samples_train']
        batch_size = test.run_training_params['train_batch_size']
        list_of_batch_indexes = range(0, n_samples - batch_size + 1, batch_size)
        self.assertEqual(len(list_of_batch_indexes), test.run_training_params['update_per_epochs'])
        self.assertEqual(n_samples - list_of_batch_indexes[-1] - batch_size, test.run_training_params['unused_samples'])
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
            {'params': ['a'], 'lr': 0.6}, {'params': 'w', 'lr': 0.12, 'alpha': 0.02}
        ]
        optimizer = RMSprop(optimizer_defaults, optimizer_params)
        test.trainModel(optimizer=optimizer, training_params=training_params, optimizer_defaults={'lr': 0.3},
                        optimizer_params=[{'params': ['a'], 'lr': 1.0}, {'params': ['b'], 'lr': 1.2}],
                        lr_param={'a': 0.2})
        self.assertEqual(['model1'], test.run_training_params['models'])
        self.assertEqual('RMSprop', test.run_training_params['optimizer'])
        self.assertEqual(40, test.run_training_params['num_of_epochs'])
        self.assertEqual(round(56 * 100 / 100), test.run_training_params['n_samples_train'])
        self.assertEqual(round(56 * 0 / 100), test.run_training_params['n_samples_val'])
        self.assertEqual(round(56 * 0 / 100), test.run_training_params['n_samples_test'])
        self.assertEqual({'lr': 0.3}, test.run_training_params['optimizer_defaults'])
        self.assertEqual([{'params': 'a', 'lr': 0.2}, {'params': 'b', 'lr': 1.2}],
                         test.run_training_params['optimizer_params'])

        test.trainModel(optimizer=optimizer, training_params=training_params, optimizer_defaults={'lr': 0.3},
                        optimizer_params=[{'params': ['a'], 'lr': 0.1}, {'params': ['b'], 'lr': 0.2}])
        self.assertEqual({'lr': 0.3}, test.run_training_params['optimizer_defaults'])
        self.assertEqual([{'params': 'a', 'lr': 0.1}, {'params': 'b', 'lr': 0.2}],
                         test.run_training_params['optimizer_params'])

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
        self.assertEqual([{'lr': 0.1, 'params': 'a'}, {'alpha': 0.02, 'lr': 0.12, 'params': 'w'}],
                         test.run_training_params['optimizer_params'])

        test.trainModel(optimizer=optimizer)
        self.assertEqual({'alpha': 0.8, 'lr': 0.001}, test.run_training_params['optimizer_defaults'])
        self.assertEqual([{'params': 'a', 'lr': 0.6}, {'params': 'w', 'lr': 0.12, 'alpha': 0.02}],
                         test.run_training_params['optimizer_params'])

        n_samples = test.run_training_params['n_samples_train']
        batch_size = test.run_training_params['train_batch_size']
        list_of_batch_indexes = range(0, n_samples - batch_size + 1, batch_size)
        self.assertEqual(len(list_of_batch_indexes), test.run_training_params['update_per_epochs'])
        self.assertEqual(n_samples - list_of_batch_indexes[-1] - batch_size, test.run_training_params['unused_samples'])
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
                        add_optimizer_params=[{'params': ['a'], 'lr': 1.0}, {'params': ['b'], 'lr': 1.2}],
                        lr_param={'a': 0.2})
        self.assertEqual(['model1'], test.run_training_params['models'])
        self.assertEqual('RMSprop', test.run_training_params['optimizer'])
        self.assertEqual(40, test.run_training_params['num_of_epochs'])
        self.assertEqual(round(56 * 100 / 100), test.run_training_params['n_samples_train'])
        self.assertEqual(round(56 * 0 / 100), test.run_training_params['n_samples_val'])
        self.assertEqual(round(56 * 0 / 100), test.run_training_params['n_samples_test'])
        self.assertEqual({'lr': 0.3}, test.run_training_params['optimizer_defaults'])
        self.assertEqual([{'params': 'a', 'lr': 0.2}, {'params': 'b', 'lr': 1.2}, {'params': 'w'}],
                         test.run_training_params['optimizer_params'])

        test.trainModel(optimizer=optimizer, training_params=training_params, add_optimizer_defaults={'lr': 0.3},
                        add_optimizer_params=[{'params': ['a'], 'lr': 0.1}, {'params': ['b'], 'lr': 0.2}])
        self.assertEqual({'lr': 0.3}, test.run_training_params['optimizer_defaults'])
        self.assertEqual([{'params': 'a', 'lr': 0.1}, {'params': 'b', 'lr': 0.2}, {'params': 'w'}],
                         test.run_training_params['optimizer_params'])

        test.trainModel(optimizer=optimizer, training_params=training_params, add_optimizer_defaults={'lr': 0.3})
        self.assertEqual({'lr': 0.3}, test.run_training_params['optimizer_defaults'])
        self.assertEqual([{'params': 'a', 'lr': 0.7}, {'lr': 0.0, 'params': 'b'}, {'params': 'w'}],
                         test.run_training_params['optimizer_params'])

        test.trainModel(optimizer=optimizer, training_params=training_params)
        self.assertEqual({'lr': 0.12}, test.run_training_params['optimizer_defaults'])
        self.assertEqual([{'params': 'a', 'lr': 0.7}, {'lr': 0.0, 'params': 'b'}, {'params': 'w'}],
                         test.run_training_params['optimizer_params'])

        del training_params['add_optimizer_defaults']
        test.trainModel(optimizer=optimizer, training_params=training_params)
        self.assertEqual({'lr': 0.5}, test.run_training_params['optimizer_defaults'])
        self.assertEqual([{'params': 'a', 'lr': 0.7}, {'lr': 0.0, 'params': 'b'}, {'params': 'w'}],
                         test.run_training_params['optimizer_params'])

        del training_params['add_optimizer_params']
        test.trainModel(optimizer=optimizer, training_params=training_params)
        self.assertEqual({'lr': 0.5}, test.run_training_params['optimizer_defaults'])
        self.assertEqual([{'lr': 0.1, 'params': 'a'}, {'lr': 0.0, 'params': 'b'}, {'params': 'w'}],
                         test.run_training_params['optimizer_params'])

        test.trainModel(optimizer=optimizer)
        self.assertEqual({'lr': 0.001}, test.run_training_params['optimizer_defaults'])
        self.assertEqual([{'params': 'a'}, {'params': 'b'}, {'params': 'w'}],
                         test.run_training_params['optimizer_params'])

        n_samples = test.run_training_params['n_samples_train']
        batch_size = test.run_training_params['train_batch_size']
        list_of_batch_indexes = range(0, n_samples - batch_size + 1, batch_size)
        self.assertEqual(len(list_of_batch_indexes), test.run_training_params['update_per_epochs'])
        self.assertEqual(n_samples - list_of_batch_indexes[-1] - batch_size, test.run_training_params['unused_samples'])