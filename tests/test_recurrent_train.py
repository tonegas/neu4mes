import unittest, os, sys
sys.path.append(os.getcwd())
from neu4mes import *
from neu4mes import relation
relation.CHECK_NAMES = False

from neu4mes.logger import logging, Neu4MesLogger
log = Neu4MesLogger(__name__, logging.CRITICAL)
log.setAllLevel(logging.CRITICAL)

# 14 Tests
# Test the value of the weight after the recurrent training

# Linear function
def linear_fun(x,a,b):
    return x*a+b

data_x = np.random.rand(500)*20-10
data_a = 2
data_b = -3
dataset = {'in1': data_x, 'out': linear_fun(data_x,data_a,data_b)}
data_folder = '/tests/data/'

class Neu4mesTrainingTest(unittest.TestCase):
    def assertAlmostEqual(self, data1, data2, precision=3):
        if type(data1) == type(data2) == list:
            assert np.asarray(data1, dtype=np.float32).ndim == np.asarray(data2,dtype=np.float32).ndim, f'Inputs must have the same dimension! Received {type(data1)} and {type(data2)}'
            for pred, label in zip(data1, data2):
                self.assertAlmostEqual(pred, label, precision=precision)
        elif type(data1) == type(data2) == dict:
                for (pred_key,pred_value), (label_key,label_value) in zip(data1.items(), data2.items()):
                    self.assertAlmostEqual(pred_value, label_value, precision=precision)
        else:
            super().assertAlmostEqual(data1, data2, places=precision)
    def test_training_values_fir_connect_linear(self):
        NeuObj.reset_count()
        input1 = Input('in1')
        target = Input('out1')
        a = Parameter('a', values=[[1]])
        output1 = Output('out1',Fir(parameter=a)(input1.last()))

        inout = State('inout')
        W = Parameter('W', values=[[[1]]])
        b = Parameter('b', values=[[1]])
        output2 = Output('out2', Linear(W=W,b=b)(inout.last()))

        test = Neu4mes(visualizer=None, seed=42)
        test.addModel('model', [output1,output2])
        test.addMinimize('error', target.last(), output2)
        test.addConnect(output1, inout)
        test.neuralizeModel()
        self.assertEqual({'out1': [1.0], 'out2': [2.0]}, test({'in1': [1]}))

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
        self.assertListEqual([[[1.0]]], test.model.all_parameters['W'].data.numpy().tolist())
        self.assertListEqual([[1.0]], test.model.all_parameters['b'].data.numpy().tolist())
        self.assertListEqual([[1.0]], test.model.all_parameters['a'].data.numpy().tolist())
        test.trainModel(optimizer='SGD', splits=[100, 0, 0], lr=1, num_of_epochs=1)
        self.assertListEqual([[[3.0]]], test.model.all_parameters['W'].data.numpy().tolist())
        self.assertListEqual([[3.0]], test.model.all_parameters['b'].data.numpy().tolist())
        self.assertListEqual([[3.0]], test.model.all_parameters['a'].data.numpy().tolist())
        test.trainModel(optimizer='SGD', splits=[100, 0, 0], lr=1, num_of_epochs=1)
        self.assertListEqual([[[-51.0]]], test.model.all_parameters['W'].data.numpy().tolist())
        self.assertListEqual([[-15.0]], test.model.all_parameters['b'].data.numpy().tolist())
        self.assertListEqual([[-51.0]], test.model.all_parameters['a'].data.numpy().tolist())

        test.neuralizeModel(clear_model=True)
        self.assertListEqual([[[1.0]]], test.model.all_parameters['W'].data.numpy().tolist())
        self.assertListEqual([[1.0]], test.model.all_parameters['b'].data.numpy().tolist())
        self.assertListEqual([[1.0]], test.model.all_parameters['a'].data.numpy().tolist())
        test.trainModel(optimizer='SGD', splits=[100, 0, 0], lr=1, num_of_epochs=2)
        self.assertListEqual([[[-51.0]]], test.model.all_parameters['W'].data.numpy().tolist())
        self.assertListEqual([[-15.0]], test.model.all_parameters['b'].data.numpy().tolist())
        self.assertListEqual([[-51.0]], test.model.all_parameters['a'].data.numpy().tolist())

        dataset = {'in1': [1,1], 'out1': [3,3]}
        test.loadData(name='dataset2', source=dataset)
        test.neuralizeModel(clear_model=True)
        # the out is 3.0 due the mean of the error is not the same of two epochs
        test.trainModel(train_dataset='dataset2', optimizer='SGD', lr=1, num_of_epochs=1)
        self.assertListEqual([[[3.0]]], test.model.all_parameters['W'].data.numpy().tolist())
        self.assertListEqual([[3.0]], test.model.all_parameters['b'].data.numpy().tolist())
        self.assertListEqual([[3.0]], test.model.all_parameters['a'].data.numpy().tolist())
        test.neuralizeModel(clear_model=True)
        test.trainModel(train_dataset='dataset2', optimizer='SGD', lr=1, num_of_epochs=2)
        self.assertListEqual([[[-51.0]]], test.model.all_parameters['W'].data.numpy().tolist())
        self.assertListEqual([[-15.0]], test.model.all_parameters['b'].data.numpy().tolist())
        self.assertListEqual([[-51.0]], test.model.all_parameters['a'].data.numpy().tolist())

    def test_training_values_fir_train_connect_linear(self):
        NeuObj.reset_count()
        input1 = Input('in1')
        target = Input('out1')
        a = Parameter('a', values=[[1]])
        output1 = Output('out1',Fir(parameter=a)(input1.last()))

        inout = Input('inout')
        W = Parameter('W', values=[[[1]]])
        b = Parameter('b', values=[[1]])
        output2 = Output('out2', Linear(W=W,b=b)(inout.last()))

        test = Neu4mes(visualizer=None, seed=42)
        test.addModel('model', [output1,output2])
        test.addMinimize('error', target.last(), output2)
        test.neuralizeModel()
        self.assertEqual({'out1': [1.0], 'out2': [2.0]}, test({'in1': [1]}, connect={'inout': 'out1'}))

        dataset = {'in1': [1], 'out1': [3]}
        test.loadData(name='dataset', source=dataset)

        self.assertListEqual([[[1.0]]],test.model.all_parameters['W'].data.numpy().tolist())
        self.assertListEqual([[1.0]], test.model.all_parameters['b'].data.numpy().tolist())
        self.assertListEqual([[1.0]], test.model.all_parameters['a'].data.numpy().tolist())
        test.trainModel(optimizer='SGD', splits=[100, 0, 0], lr=1, num_of_epochs=1, connect={'inout': 'out1'})
        self.assertListEqual([[[3.0]]], test.model.all_parameters['W'].data.numpy().tolist())
        self.assertListEqual([[3.0]], test.model.all_parameters['b'].data.numpy().tolist())
        self.assertListEqual([[3.0]], test.model.all_parameters['a'].data.numpy().tolist())
        test.trainModel(optimizer='SGD', splits=[100, 0, 0], lr=1, num_of_epochs=1, connect={'inout': 'out1'})
        self.assertListEqual([[[-51.0]]],test.model.all_parameters['W'].data.numpy().tolist())
        self.assertListEqual([[-15.0]], test.model.all_parameters['b'].data.numpy().tolist())
        self.assertListEqual([[-51.0]], test.model.all_parameters['a'].data.numpy().tolist())

        test.neuralizeModel(clear_model=True)
        self.assertListEqual([[[1.0]]], test.model.all_parameters['W'].data.numpy().tolist())
        self.assertListEqual([[1.0]], test.model.all_parameters['b'].data.numpy().tolist())
        self.assertListEqual([[1.0]], test.model.all_parameters['a'].data.numpy().tolist())
        test.trainModel(optimizer='SGD', splits=[100, 0, 0], lr=1, num_of_epochs=1, connect={'inout': 'out1'})
        self.assertListEqual([[[3.0]]], test.model.all_parameters['W'].data.numpy().tolist())
        self.assertListEqual([[3.0]], test.model.all_parameters['b'].data.numpy().tolist())
        self.assertListEqual([[3.0]], test.model.all_parameters['a'].data.numpy().tolist())
        test.trainModel(optimizer='SGD', splits=[100, 0, 0], lr=1, num_of_epochs=1, connect={'inout': 'out1'})
        self.assertListEqual([[[-51.0]]], test.model.all_parameters['W'].data.numpy().tolist())
        self.assertListEqual([[-15.0]], test.model.all_parameters['b'].data.numpy().tolist())
        self.assertListEqual([[-51.0]], test.model.all_parameters['a'].data.numpy().tolist())

        test.neuralizeModel(clear_model=True)
        self.assertListEqual([[[1.0]]], test.model.all_parameters['W'].data.numpy().tolist())
        self.assertListEqual([[1.0]], test.model.all_parameters['b'].data.numpy().tolist())
        self.assertListEqual([[1.0]], test.model.all_parameters['a'].data.numpy().tolist())
        test.trainModel(optimizer='SGD', splits=[100, 0, 0], lr=1, num_of_epochs=2, connect={'inout': 'out1'})
        self.assertListEqual([[[-51.0]]], test.model.all_parameters['W'].data.numpy().tolist())
        self.assertListEqual([[-15.0]], test.model.all_parameters['b'].data.numpy().tolist())
        self.assertListEqual([[-51.0]], test.model.all_parameters['a'].data.numpy().tolist())

        dataset = {'in1': [1,1], 'out1': [3,3]}
        test.loadData(name='dataset2', source=dataset)
        test.neuralizeModel(clear_model=True)
        # the out is 3.0 due the mean of the error is not the same of two epochs
        test.trainModel(train_dataset='dataset2', optimizer='SGD', lr=1, num_of_epochs=1, connect={'inout': 'out1'})
        self.assertListEqual([[[3.0]]], test.model.all_parameters['W'].data.numpy().tolist())
        self.assertListEqual([[3.0]], test.model.all_parameters['b'].data.numpy().tolist())
        self.assertListEqual([[3.0]], test.model.all_parameters['a'].data.numpy().tolist())
        test.neuralizeModel(clear_model=True)
        test.trainModel(train_dataset='dataset2', optimizer='SGD', lr=1, num_of_epochs=2, connect={'inout': 'out1'})
        self.assertListEqual([[[-51.0]]], test.model.all_parameters['W'].data.numpy().tolist())
        self.assertListEqual([[-15.0]], test.model.all_parameters['b'].data.numpy().tolist())
        self.assertListEqual([[-51.0]], test.model.all_parameters['a'].data.numpy().tolist())

    def test_training_values_fir_connect_linear_only_model(self):
        NeuObj.reset_count()
        input1 = Input('in1')
        target = Input('out1')
        a = Parameter('a', values=[[1]])
        output1 = Output('out1',Fir(parameter=a)(input1.last()))

        inout = State('inout')
        W = Parameter('W', values=[[[1]]])
        b = Parameter('b', values=[[1]])
        output2 = Output('out2', Linear(W=W,b=b)(inout.last()))

        test = Neu4mes(visualizer=None, seed=42)
        test.addModel('model1', output1)
        test.addModel('model2', output2)
        test.addMinimize('error', target.last(), output2)
        test.addConnect(output1, inout)
        test.neuralizeModel()
        self.assertEqual({'out1': [1.0], 'out2': [2.0]}, test({'in1': [1]}))

        dataset = {'in1': [1], 'out1': [3]}
        test.loadData(name='dataset', source=dataset)

        test.trainModel(optimizer='SGD', splits=[100, 0, 0], lr=1, num_of_epochs=1)
        self.assertListEqual([[[3.0]]], test.model.all_parameters['W'].data.numpy().tolist())
        self.assertListEqual([[3.0]], test.model.all_parameters['b'].data.numpy().tolist())
        self.assertListEqual([[3.0]], test.model.all_parameters['a'].data.numpy().tolist())
        test.trainModel(optimizer='SGD', splits=[100, 0, 0], lr=1, num_of_epochs=1)
        self.assertListEqual([[[-51.0]]],test.model.all_parameters['W'].data.numpy().tolist())
        self.assertListEqual([[-15.0]], test.model.all_parameters['b'].data.numpy().tolist())
        self.assertListEqual([[-51.0]], test.model.all_parameters['a'].data.numpy().tolist())

        test.neuralizeModel(clear_model=True)
        test.trainModel(models='model1', optimizer='SGD', splits=[100, 0, 0], lr=1, num_of_epochs=1)
        self.assertListEqual([[[1.0]]], test.model.all_parameters['W'].data.numpy().tolist())
        self.assertListEqual([[1.0]], test.model.all_parameters['b'].data.numpy().tolist())
        self.assertListEqual([[3.0]], test.model.all_parameters['a'].data.numpy().tolist())
        test.trainModel(models='model1', optimizer='SGD', splits=[100, 0, 0], lr=1, num_of_epochs=1)
        self.assertListEqual([[[1.0]]], test.model.all_parameters['W'].data.numpy().tolist())
        self.assertListEqual([[1.0]], test.model.all_parameters['b'].data.numpy().tolist())
        self.assertListEqual([[1.0]], test.model.all_parameters['a'].data.numpy().tolist())

        test.neuralizeModel(clear_model=True)
        test.trainModel(models='model2', optimizer='SGD', splits=[100, 0, 0], lr=1, num_of_epochs=1)
        self.assertListEqual([[[3.0]]], test.model.all_parameters['W'].data.numpy().tolist())
        self.assertListEqual([[3.0]], test.model.all_parameters['b'].data.numpy().tolist())
        self.assertListEqual([[1.0]], test.model.all_parameters['a'].data.numpy().tolist())
        test.trainModel(models='model2', optimizer='SGD', splits=[100, 0, 0], lr=1, num_of_epochs=1)
        self.assertListEqual([[[-3.0]]],test.model.all_parameters['W'].data.numpy().tolist())
        self.assertListEqual([[-3.0]], test.model.all_parameters['b'].data.numpy().tolist())
        self.assertListEqual([[1.0]], test.model.all_parameters['a'].data.numpy().tolist())

    def test_training_values_fir_train_connect_linear_only_model(self):
        NeuObj.reset_count()
        input1 = Input('in1')
        target = Input('out1')
        a = Parameter('a', values=[[1]])
        output1 = Output('out1',Fir(parameter=a)(input1.last()))

        inout = Input('inout')
        W = Parameter('W', values=[[[1]]])
        b = Parameter('b', values=[[1]])
        output2 = Output('out2', Linear(W=W,b=b)(inout.last()))

        test = Neu4mes(visualizer=None, seed=42)
        test.addModel('model1', output1)
        test.addModel('model2', output2)
        test.addMinimize('error', target.last(), output2)
        test.neuralizeModel()
        self.assertEqual({'out1': [1.0], 'out2': [1.0]}, test({'in1': [1]}))
        self.assertEqual({'out1': [1.0], 'out2': [2.0]}, test({'in1': [1]}, connect={'inout': 'out1'}))

        dataset = {'in1': [1], 'out1': [3]}
        test.loadData(name='dataset', source=dataset)

        test.trainModel(optimizer='SGD', splits=[100, 0, 0], lr=1, num_of_epochs=1, connect={'inout': 'out1'})
        self.assertListEqual([[[3.0]]], test.model.all_parameters['W'].data.numpy().tolist())
        self.assertListEqual([[3.0]], test.model.all_parameters['b'].data.numpy().tolist())
        self.assertListEqual([[3.0]], test.model.all_parameters['a'].data.numpy().tolist())
        test.trainModel(optimizer='SGD', splits=[100, 0, 0], lr=1, num_of_epochs=1, connect={'inout': 'out1'})
        self.assertListEqual([[[-51.0]]],test.model.all_parameters['W'].data.numpy().tolist())
        self.assertListEqual([[-15.0]], test.model.all_parameters['b'].data.numpy().tolist())
        self.assertListEqual([[-51.0]], test.model.all_parameters['a'].data.numpy().tolist())

        test.neuralizeModel(clear_model=True)
        test.trainModel(models='model1', optimizer='SGD', splits=[100, 0, 0], lr=1, num_of_epochs=1, connect={'inout': 'out1'})
        self.assertListEqual([[[1.0]]], test.model.all_parameters['W'].data.numpy().tolist())
        self.assertListEqual([[1.0]], test.model.all_parameters['b'].data.numpy().tolist())
        self.assertListEqual([[3.0]], test.model.all_parameters['a'].data.numpy().tolist())
        test.trainModel(models='model1', optimizer='SGD', splits=[100, 0, 0], lr=1, num_of_epochs=1, connect={'inout': 'out1'})
        self.assertListEqual([[[1.0]]], test.model.all_parameters['W'].data.numpy().tolist())
        self.assertListEqual([[1.0]], test.model.all_parameters['b'].data.numpy().tolist())
        self.assertListEqual([[1.0]], test.model.all_parameters['a'].data.numpy().tolist())

        test.neuralizeModel(clear_model=True)
        test.trainModel(models='model2', optimizer='SGD', splits=[100, 0, 0], lr=1, num_of_epochs=1, connect={'inout': 'out1'})
        self.assertListEqual([[[3.0]]], test.model.all_parameters['W'].data.numpy().tolist())
        self.assertListEqual([[3.0]], test.model.all_parameters['b'].data.numpy().tolist())
        self.assertListEqual([[1.0]], test.model.all_parameters['a'].data.numpy().tolist())
        test.trainModel(models='model2', optimizer='SGD', splits=[100, 0, 0], lr=1, num_of_epochs=1, connect={'inout': 'out1'})
        self.assertListEqual([[[-3.0]]],test.model.all_parameters['W'].data.numpy().tolist())
        self.assertListEqual([[-3.0]], test.model.all_parameters['b'].data.numpy().tolist())
        self.assertListEqual([[1.0]], test.model.all_parameters['a'].data.numpy().tolist())

    def test_training_values_fir_connect_linear_more_prediction(self):
        NeuObj.reset_count()
        input1 = Input('in1')
        target = Input('out1')
        a = Parameter('a', values=[[1]])
        output1 = Output('out1',Fir(parameter=a)(input1.last()))

        inout = State('inout')
        W = Parameter('W', values=[[[1]]])
        b = Parameter('b', values=[[1]])
        output2 = Output('out2', Linear(W=W,b=b)(inout.last()))

        test = Neu4mes(visualizer=None,seed=42)
        test.addModel('model', [output1,output2])
        test.addMinimize('error', target.last(), output2)
        test.addConnect(output1, inout)
        test.neuralizeModel()
        self.assertEqual({'out1': [1.0], 'out2': [2.0]}, test({'in1': [1]}))

        dataset = {'in1': [0,2,7,1], 'out1': [3,4,5,1], 'inout': [1,1,2,2]}
        test.loadData(name='dataset2', source=dataset)

        self.assertListEqual([[[1.0]]],test.model.all_parameters['W'].data.numpy().tolist())
        self.assertListEqual([[1.0]], test.model.all_parameters['b'].data.numpy().tolist())
        self.assertListEqual([[1.0]], test.model.all_parameters['a'].data.numpy().tolist())
        test.trainModel(train_dataset='dataset2', optimizer='SGD', lr=1, num_of_epochs=1, prediction_samples=0)
        self.assertListEqual([[[-9.0]]], test.model.all_parameters['W'].data.numpy().tolist())
        self.assertListEqual([[0.5]], test.model.all_parameters['b'].data.numpy().tolist())
        self.assertListEqual([[-9.0]], test.model.all_parameters['a'].data.numpy().tolist())
        with self.assertRaises(ValueError):
            test.trainModel(train_dataset='dataset2', optimizer='SGD', lr=1, num_of_epochs=1, prediction_samples=10)
        # test.trainModel(train_dataset='dataset2', optimizer='SGD', lr=1, num_of_epochs=1, prediction_samples=1) # TODO add this test
        test.neuralizeModel(clear_model=True)
        self.assertListEqual([[[1.0]]],test.model.all_parameters['W'].data.numpy().tolist())
        self.assertListEqual([[1.0]], test.model.all_parameters['b'].data.numpy().tolist())
        self.assertListEqual([[1.0]], test.model.all_parameters['a'].data.numpy().tolist())
        test.trainModel(train_dataset='dataset2', optimizer='SGD', lr=1, num_of_epochs=1, train_batch_size=1, prediction_samples=3)
        self.assertListEqual([[[-9.0]]], test.model.all_parameters['W'].data.numpy().tolist())
        self.assertListEqual([[0.5]], test.model.all_parameters['b'].data.numpy().tolist())
        self.assertListEqual([[-9.0]], test.model.all_parameters['a'].data.numpy().tolist())

        dataset = {'in1': [0, 2, 7, 1, 5, 0, 2], 'out1': [1, 4, 8, 2, 6, 1, 1]}
        test.loadData(name='dataset3', source=dataset)

        test.neuralizeModel(clear_model=True)
        self.assertListEqual([[[1.0]]],test.model.all_parameters['W'].data.numpy().tolist())
        self.assertListEqual([[1.0]], test.model.all_parameters['b'].data.numpy().tolist())
        self.assertListEqual([[1.0]], test.model.all_parameters['a'].data.numpy().tolist())
        test.trainModel(train_dataset='dataset3', optimizer='SGD', shuffle_data=False, lr=1, num_of_epochs=1, train_batch_size=2, prediction_samples=3)
        self.assertListEqual([[[-162.75]]], test.model.all_parameters['W'].data.numpy().tolist())
        self.assertListEqual([[-15.75]], test.model.all_parameters['b'].data.numpy().tolist())
        self.assertListEqual([[-162.75]], test.model.all_parameters['a'].data.numpy().tolist())

        # Because is a connect and the window is 1 the initialization of the state is overwritten by the out1
        test.loadData(name='dataset4', source=dataset|{'inout': [0,0,0,0,0,0,0]})
        test.neuralizeModel(clear_model=True)
        test.trainModel(train_dataset='dataset4', optimizer='SGD', shuffle_data=False, lr=1, num_of_epochs=1, train_batch_size=2, prediction_samples=3)
        self.assertListEqual([[[-162.75]]], test.model.all_parameters['W'].data.numpy().tolist())
        self.assertListEqual([[-15.75]], test.model.all_parameters['b'].data.numpy().tolist())
        self.assertListEqual([[-162.75]], test.model.all_parameters['a'].data.numpy().tolist())

    def test_training_values_fir_train_connect_linear_more_prediction(self):
        NeuObj.reset_count()
        input1 = Input('in1')
        target = Input('out1')
        a = Parameter('a', values=[[1]])
        output1 = Output('out1',Fir(parameter=a)(input1.last()))

        inout = Input('inout')
        W = Parameter('W', values=[[[1]]])
        b = Parameter('b', values=[[1]])
        output2 = Output('out2', Linear(W=W,b=b)(inout.last()))

        test = Neu4mes(visualizer=None, seed=42)
        test.addModel('model', [output1,output2])
        test.addMinimize('error', target.last(), output2)
        test.neuralizeModel()
        self.assertEqual({'out1': [1.0], 'out2': [2.0]}, test({'in1': [1]},connect={'inout': 'out1'}))

        dataset = {'in1': [0,2,7,1], 'out1': [3,4,5,1]}
        test.loadData(name='dataset2', source=dataset)

        self.assertListEqual([[[1.0]]],test.model.all_parameters['W'].data.numpy().tolist())
        self.assertListEqual([[1.0]], test.model.all_parameters['b'].data.numpy().tolist())
        self.assertListEqual([[1.0]], test.model.all_parameters['a'].data.numpy().tolist())
        test.trainModel(train_dataset='dataset2', optimizer='SGD', lr=1, num_of_epochs=1, prediction_samples=0, connect={'inout': 'out1'})
        self.assertListEqual([[[-9.0]]], test.model.all_parameters['W'].data.numpy().tolist())
        self.assertListEqual([[0.5]], test.model.all_parameters['b'].data.numpy().tolist())
        self.assertListEqual([[-9.0]], test.model.all_parameters['a'].data.numpy().tolist())
        with self.assertRaises(ValueError):
            test.trainModel(train_dataset='dataset2', optimizer='SGD', lr=1, num_of_epochs=1, prediction_samples=10, connect={'inout': 'out1'})
        # test.trainModel(train_dataset='dataset2', optimizer='SGD', lr=1, num_of_epochs=1, prediction_samples=1, connect={'inout': 'out1'}) # TODO add this test
        test.neuralizeModel(clear_model=True)
        self.assertListEqual([[[1.0]]],test.model.all_parameters['W'].data.numpy().tolist())
        self.assertListEqual([[1.0]], test.model.all_parameters['b'].data.numpy().tolist())
        self.assertListEqual([[1.0]], test.model.all_parameters['a'].data.numpy().tolist())
        test.trainModel(train_dataset='dataset2', optimizer='SGD', lr=1, num_of_epochs=1, train_batch_size=1, prediction_samples=3, connect={'inout': 'out1'})
        self.assertListEqual([[[-9.0]]], test.model.all_parameters['W'].data.numpy().tolist())
        self.assertListEqual([[0.5]], test.model.all_parameters['b'].data.numpy().tolist())
        self.assertListEqual([[-9.0]], test.model.all_parameters['a'].data.numpy().tolist())

        # Because is a connect and the window is 1 the initialization of the state is overwritten by the out1
        dataset = {'in1': [0,2,7,1], 'out1': [3,4,5,1], 'inout':[1,1,2,2]}
        test.loadData(name='dataset3', source=dataset)
        test.neuralizeModel(clear_model=True)
        test.trainModel(train_dataset='dataset3', optimizer='SGD', lr=1, num_of_epochs=1, train_batch_size=1, prediction_samples=3, connect={'inout': 'out1'})
        self.assertListEqual([[[-9.0]]], test.model.all_parameters['W'].data.numpy().tolist())
        self.assertListEqual([[0.5]], test.model.all_parameters['b'].data.numpy().tolist())
        self.assertListEqual([[-9.0]], test.model.all_parameters['a'].data.numpy().tolist())
        test.neuralizeModel(clear_model=True)
        test.trainModel(train_dataset='dataset3', optimizer='SGD', shuffle_data=False, lr=1, num_of_epochs=1, train_batch_size=1, prediction_samples=3)
        self.assertListEqual([[[-273.0]]], test.model.all_parameters['W'].data.numpy().tolist())
        self.assertListEqual([[-137.0]], test.model.all_parameters['b'].data.numpy().tolist())
        self.assertListEqual([[1.0]], test.model.all_parameters['a'].data.numpy().tolist())

        dataset = {'in1': [0, 2, 7, 1, 5, 0, 2], 'out1': [1, 4, 8, 2, 6, 1, 1]}
        test.loadData(name='dataset4', source=dataset)
        test.neuralizeModel(clear_model=True)
        self.assertListEqual([[[1.0]]], test.model.all_parameters['W'].data.numpy().tolist())
        self.assertListEqual([[1.0]], test.model.all_parameters['b'].data.numpy().tolist())
        self.assertListEqual([[1.0]], test.model.all_parameters['a'].data.numpy().tolist())
        test.trainModel(train_dataset='dataset4', optimizer='SGD', shuffle_data=False, lr=1, num_of_epochs=1,
                        train_batch_size=2, prediction_samples=3, connect={'inout': 'out1'})
        self.assertListEqual([[[-162.75]]], test.model.all_parameters['W'].data.numpy().tolist())
        self.assertListEqual([[-15.75]], test.model.all_parameters['b'].data.numpy().tolist())
        self.assertListEqual([[-162.75]], test.model.all_parameters['a'].data.numpy().tolist())

        # Because is a connect and the window is 1 the initialization of the state is overwritten by the out1
        test.loadData(name='dataset5', source=dataset | {'inout': [0, 0, 0, 0, 0, 0, 0]})
        test.neuralizeModel(clear_model=True)
        test.trainModel(train_dataset='dataset5', optimizer='SGD', shuffle_data=False, lr=1, num_of_epochs=1,
                        train_batch_size=2, prediction_samples=3, connect={'inout': 'out1'})
        self.assertListEqual([[[-162.75]]], test.model.all_parameters['W'].data.numpy().tolist())
        self.assertListEqual([[-15.75]], test.model.all_parameters['b'].data.numpy().tolist())
        self.assertListEqual([[-162.75]], test.model.all_parameters['a'].data.numpy().tolist())

    def test_training_values_fir_connect_linear_more_window(self):
        NeuObj.reset_count()
        input1 = Input('in1', dimensions=2)
        W = Parameter('W', values=[[[-1], [-5]]])
        b = Parameter('b', values=[[1]])
        lin_out = Linear(W=W, b=b)(input1.sw(2))
        output1 = Output('out1', lin_out)

        inout = State('inout')
        a = Parameter('a', values=[[4], [5]])
        a_big = Parameter('ab', values=[[1], [2], [3], [4], [5]])
        output2 = Output('out2', Fir(parameter=a)(inout.sw(2)))
        output3 = Output('out3', Fir(parameter=a_big)(inout.sw(5)))
        output4 = Output('out4', Fir(parameter=a)(lin_out))

        target = Input('target')

        test = Neu4mes(visualizer=None, seed=42, log_internal=True)
        test.addModel('model', [output1, output2, output3, output4])
        test.addConnect(output1, inout)
        test.addMinimize('error2', target.last(), output2)
        #test.addMinimize('error3', target.last(), output3)
        #test.addMinimize('error4', target.last(), output4)
        test.neuralizeModel()

        # Dataset with only one sample
        dataset = {'in1': [[0,1],[2,3],[7,4],[1,3],[4,2]], 'target': [3,4,5,1,3]}
        self.assertEqual({'out1': [[-4.0, -16.0], [-16.0, -26.0], [-26.0, -15.0], [-15.0, -13.0]],
                               'out2': [-96.0, -194.0, -179.0, -125.0],
                               'out3': [-96.0, -206.0, -235.0, -239.0],
                               'out4': [-96.0, -194.0, -179.0, -125.0]
                          }, test(dataset))
        test.loadData(name='dataset', source=dataset)
        # TODO add and error
        # dataset = {'in1': [[0,1],[2,3],[7,4],[1,3],[4,2]], 'target': [3,4,5,1]}
        # test.loadData(name='dataset2', source=dataset)

        self.assertListEqual([[[-1], [-5]]],test.model.all_parameters['W'].data.numpy().tolist())
        self.assertListEqual([[1]], test.model.all_parameters['b'].data.numpy().tolist())
        self.assertListEqual([[4], [5]], test.model.all_parameters['a'].data.numpy().tolist())
        self.assertListEqual([[1], [2], [3], [4], [5]], test.model.all_parameters['ab'].data.numpy().tolist())
        test.trainModel(train_dataset='dataset', optimizer='SGD', lr=1, num_of_epochs=1, prediction_samples=0)
        self.assertListEqual([[[6143], [5627]]],test.model.all_parameters['W'].data.numpy().tolist())
        self.assertListEqual([[2305]], test.model.all_parameters['b'].data.numpy().tolist())
        self.assertListEqual([[-3836], [-3323]], test.model.all_parameters['a'].data.numpy().tolist())
        self.assertListEqual([[1], [2], [3], [4], [5]], test.model.all_parameters['ab'].data.numpy().tolist())
        with self.assertRaises(ValueError):
            test.trainModel(train_dataset='dataset', optimizer='SGD', lr=1, num_of_epochs=1, prediction_samples=10)
        with self.assertRaises(ValueError):
            test.trainModel(train_dataset='dataset', optimizer='SGD', lr=1, num_of_epochs=1, prediction_samples=1)

        # Data set with more samples
        test.neuralizeModel(clear_model=True)
        dataset2 = {'in1': [[0,1],[2,3],[7,4],[1,3],[4,2],[6,5],[4,5],[0,0]], 'target': [3,4,5,1,3,0,1,0]}
        test.loadData(name='dataset2', source=dataset2)
        self.maxDiff = None
        self.assertEqual({'out1': [[-4.0, -16.0], [-16.0, -26.0], [-26.0, -15.0], [-15.0, -13.0], [-13.0, -30.0], [-30.0, -28.0], [-28.0, 1.0]],
                          'out2': [-96.0, -194.0, -179.0, -125.0, -202.0, -260.0, -107.0],
                          'out3': [-96.0, -206.0, -235.0, -239.0, -315.0, -355.0, -238.0],
                          'out4': [-96.0, -194.0, -179.0, -125.0, -202.0, -260.0, -107.0]
                          }, test(dataset2))

        # Use a train_batch_size of 4
        # test.trainModel(train_dataset='dataset2', optimizer='SGD', lr=1, num_of_epochs=1, prediction_samples=1) # TODO add this test
        test.trainModel(train_dataset='dataset2', optimizer='SGD', lr=1, num_of_epochs=1, prediction_samples=0)
        self.assertListEqual([[[12779], [11678.5]]],test.model.all_parameters['W'].data.numpy().tolist())
        self.assertListEqual([[3142]], test.model.all_parameters['b'].data.numpy().tolist())
        self.assertListEqual([[-7682], [-7457.5]], test.model.all_parameters['a'].data.numpy().tolist())
        self.assertListEqual([[1], [2], [3], [4], [5]], test.model.all_parameters['ab'].data.numpy().tolist())
        test.neuralizeModel(clear_model=True)
        test.trainModel(train_dataset='dataset2', optimizer='SGD', lr=1, num_of_epochs=1, train_batch_size=4)
        self.assertListEqual([[[12779], [11678.5]]],test.model.all_parameters['W'].data.numpy().tolist())
        self.assertListEqual([[3142]], test.model.all_parameters['b'].data.numpy().tolist())
        self.assertListEqual([[-7682], [-7457.5]], test.model.all_parameters['a'].data.numpy().tolist())
        self.assertListEqual([[1], [2], [3], [4], [5]], test.model.all_parameters['ab'].data.numpy().tolist())

        # Use a small batch but with a prediction sample of 3 = to 4 samples
        test.neuralizeModel(clear_model=True)
        with self.assertRaises(ValueError):
            test.trainModel(train_dataset='dataset2', optimizer='SGD', lr=1, num_of_epochs=1, train_batch_size=1, prediction_samples=4)
        test.trainModel(train_dataset='dataset2', optimizer='SGD', lr=1, num_of_epochs=1, train_batch_size=1, prediction_samples=3)
        self.assertListEqual([[[12779], [11678.5]]],test.model.all_parameters['W'].data.numpy().tolist())
        self.assertListEqual([[3142]], test.model.all_parameters['b'].data.numpy().tolist())
        self.assertListEqual([[-7682], [-7457.5]], test.model.all_parameters['a'].data.numpy().tolist())
        self.assertListEqual([[1], [2], [3], [4], [5]], test.model.all_parameters['ab'].data.numpy().tolist())

        # Different minimize
        test.removeMinimize('error2')
        test.addMinimize('error3', target.last(), output3)
        test.neuralizeModel(clear_model=True)
        self.assertListEqual([[[-1], [-5]]],test.model.all_parameters['W'].data.numpy().tolist())
        self.assertListEqual([[1]], test.model.all_parameters['b'].data.numpy().tolist())
        self.assertListEqual([[4], [5]], test.model.all_parameters['a'].data.numpy().tolist())
        self.assertListEqual([[1], [2], [3], [4], [5]], test.model.all_parameters['ab'].data.numpy().tolist())

        # Use a train_batch_size of 4
        test.trainModel(train_dataset='dataset2', optimizer='SGD', lr=1, num_of_epochs=1, prediction_samples=0)
        self.assertListEqual([[[12779], [11678.5]]],test.model.all_parameters['W'].data.numpy().tolist())
        self.assertListEqual([[3142]], test.model.all_parameters['b'].data.numpy().tolist())
        self.assertListEqual([[4], [5]], test.model.all_parameters['a'].data.numpy().tolist())
        self.assertListEqual([[1], [2], [3], [-7682], [-7457.5]], test.model.all_parameters['ab'].data.numpy().tolist())
        test.neuralizeModel(clear_model=True)
        test.trainModel(train_dataset='dataset2', optimizer='SGD', lr=1, num_of_epochs=1, train_batch_size=4)
        self.assertListEqual([[[12779], [11678.5]]],test.model.all_parameters['W'].data.numpy().tolist())
        self.assertListEqual([[3142]], test.model.all_parameters['b'].data.numpy().tolist())
        self.assertListEqual([[4], [5]], test.model.all_parameters['a'].data.numpy().tolist())
        self.assertListEqual([[1], [2], [3], [-7682], [-7457.5]], test.model.all_parameters['ab'].data.numpy().tolist())

        test.neuralizeModel(clear_model=True)
        test.internals = {}
        test.trainModel(train_dataset='dataset2', optimizer='SGD', lr=1, shuffle_data=False, num_of_epochs=1, train_batch_size=1, prediction_samples=3)
        self.assertDictEqual({'in1': [[[1.0, 3.0], [4.0, 2.0]]], 'target': [[[3.0]]]}, test.internals['inout_0_0']['XY'])
        self.assertDictEqual({'in1': [[[4.0, 2.0], [6.0, 5.0]]], 'target': [[[0.0]]]}, test.internals['inout_0_1']['XY'])
        self.assertDictEqual({'in1': [[[6.0, 5.0], [4.0, 5.0]]], 'target': [[[1.0]]]}, test.internals['inout_0_2']['XY'])
        self.assertDictEqual({'in1': [[[4.0, 5.0], [0.0, 0.0]]], 'target': [[[0.0]]]}, test.internals['inout_0_3']['XY'])
        self.assertDictEqual({'out1': [[[-15.0], [-13.0]]], 'out2': [[[-125.0]]], 'out3': [[[-125.0]]], 'out4': [[[-125.0]]]}, test.internals['inout_0_0']['out'])
        self.assertDictEqual({'out1': [[[-13.0], [-30.0]]], 'out2': [[[-202.0]]], 'out3': [[[-247.0]]], 'out4': [[[-202.0]]]}, test.internals['inout_0_1']['out'])
        self.assertDictEqual({'out1': [[[4*-1+2*-5+1.0], [6*-1+5*-5+1.0]]],
                                  'out2': [[[(4*-1+2*-5+1.0)*4+(6*-1+5*-5+1.0)*5]]],
                                  'out3': [[[(-15)*3.0+(4*-1+2*-5+1.0)*4+(6*-1+5*-5+1.0)*5]]],
                                  'out4': [[[(4*-1+2*-5+1.0)*4+(6*-1+5*-5+1.0)*5]]]}, test.internals['inout_0_1']['out'])
        self.assertDictEqual({'out1': [[[6*-1+5*-5+1.0], [4*-1+5*-5+1.0]]],
                                   'out2': [[[(6*-1+5*-5+1.0)*4.0+(4*-1+5*-5+1.0)*5.0]]],
                                   'out3': [[[(-15)*2.0+(4*-1+2*-5+1.0)*3.0+(6*-1+5*-5+1.0)*4.0+(4*-1+5*-5+1.0)*5.0]]],
                                   'out4': [[[(6*-1+5*-5+1.0)*4.0+(4*-1+5*-5+1.0)*5.0]]]}, test.internals['inout_0_2']['out'])
        self.assertDictEqual({'out1': [[[4*-1+5*-5+1.0], [0*-1+0*-5+1.0]]],
                                  'out2': [[[(4*-1+5*-5+1.0)*4.0+(0*-1+0*-5+1.0)*5.0]]],
                                  'out3': [[[(-15)*1.0+(4*-1+2*-5+1.0)*2.0+(6*-1+5*-5+1.0)*3.0+(4*-1+5*-5+1.0)*4.0+(0*-1+0*-5+1.0)*5.0]]],
                                  'out4': [[[(4*-1+5*-5+1.0)*4.0+(0*-1+0*-5+1.0)*5.0]]]}, test.internals['inout_0_3']['out'])
        self.assertDictEqual({'inout': [[[0.0],[0.0], [0.0], [-15.0], [-13.0]]]}, test.internals['inout_0_0']['state'])
        self.assertDictEqual({'inout': [[[0.0],[0.0], [-15.0], [-13.0], [-30.0]]]}, test.internals['inout_0_1']['state'])
        self.assertDictEqual({'inout': [[[0.0], [-15.0], [-13.0], [-30.0], [-28.0]]]}, test.internals['inout_0_2']['state'])
        self.assertDictEqual({'inout': [[[-15.0], [-13.0], [-30.0], [-28.0], [1.0]]]}, test.internals['inout_0_3']['state'])
        # Replace instead of roll
        # self.assertDictEqual({'inout': [[[0.0], [0.0], [-15.0], [-13.0], [0.0]]]}, test.internals['inout_0_0']['state'])
        # self.assertDictEqual({'inout': [[[0.0], [-15.0], [-13.0], [-30.0], [0.0]]]}, test.internals['inout_0_1']['state'])
        # self.assertDictEqual({'inout': [[[-15.0], [-13.0], [-30.0], [-28.0], [0.0]]]}, test.internals['inout_0_2']['state'])
        # self.assertDictEqual({'inout': [[[-13.0], [-30.0], [-28.0], [1.0], [-15.0]]]}, test.internals['inout_0_3']['state'])
        self.assertListEqual([[[22273.5], [20993.0]]],test.model.all_parameters['W'].data.numpy().tolist())
        self.assertListEqual([[6154.0]], test.model.all_parameters['b'].data.numpy().tolist())
        self.assertListEqual([[4], [5]], test.model.all_parameters['a'].data.numpy().tolist())
        self.assertListEqual([[-1784.0], [-4020.0], [-7564.5], [-10843.5], [-9033.0]], test.model.all_parameters['ab'].data.numpy().tolist())
        with self.assertRaises(KeyError):
            test.internals['inout_1_0']

        test.neuralizeModel(clear_model=True)
        test.internals = {}
        test.trainModel(train_dataset='dataset2', optimizer='SGD', lr=0.01, shuffle_data=False,  num_of_epochs=1, train_batch_size=1,
                         prediction_samples=2)
        self.assertDictEqual({'in1': [[[1.0, 3.0], [4.0, 2.0]]], 'target': [[[3.0]]]}, test.internals['inout_0_0']['XY'])
        self.assertDictEqual({'in1': [[[4.0, 2.0], [6.0, 5.0]]], 'target': [[[0.0]]]}, test.internals['inout_0_1']['XY'])
        self.assertDictEqual({'in1': [[[6.0, 5.0], [4.0, 5.0]]], 'target': [[[1.0]]]}, test.internals['inout_0_2']['XY'])
        self.assertDictEqual({'out1': [[[-15.0], [-13.0]]], 'out2': [[[-125.0]]], 'out3': [[[-125.0]]], 'out4': [[[-125.0]]]}, test.internals['inout_0_0']['out'])
        self.assertDictEqual({'out1': [[[-13.0], [-30.0]]], 'out2': [[[-202.0]]], 'out3': [[[-247.0]]], 'out4': [[[-202.0]]]}, test.internals['inout_0_1']['out'])
        self.assertDictEqual({'out1': [[[4*-1+2*-5+1.0], [6*-1+5*-5+1.0]]],
                              'out2': [[[(4*-1+2*-5+1.0)*4+(6*-1+5*-5+1.0)*5]]],
                              'out3': [[[(-15)*3.0+(4*-1+2*-5+1.0)*4+(6*-1+5*-5+1.0)*5]]],
                              'out4': [[[(4*-1+2*-5+1.0)*4+(6*-1+5*-5+1.0)*5]]]}, test.internals['inout_0_1']['out'])
        self.assertDictEqual({'out1': [[[6*-1+5*-5+1.0], [4*-1+5*-5+1.0]]],
                                   'out2': [[[(6*-1+5*-5+1.0)*4.0+(4*-1+5*-5+1.0)*5.0]]],
                                   'out3': [[[(-15)*2.0+(4*-1+2*-5+1.0)*3.0+(6*-1+5*-5+1.0)*4.0+(4*-1+5*-5+1.0)*5.0]]],
                                   'out4': [[[(6*-1+5*-5+1.0)*4.0+(4*-1+5*-5+1.0)*5.0]]]}, test.internals['inout_0_2']['out'])
        self.assertDictEqual({'inout': [[[0.0], [0.0], [0.0], [-15.0], [-13.0]]]}, test.internals['inout_0_0']['state'])
        self.assertDictEqual({'inout': [[[0.0], [0.0], [-15.0], [-13.0], [-30.0]]]}, test.internals['inout_0_1']['state'])
        self.assertDictEqual({'inout': [[[0.0], [-15.0], [-13.0], [-30.0], [-28.0]]]}, test.internals['inout_0_2']['state'])
        # Replace instead of rolling
        # self.assertDictEqual({'inout': [[[0.0], [0.0], [-15.0], [-13.0], [0.0]]]}, test.internals['inout_0_0']['state'])
        # self.assertDictEqual({'inout': [[[0.0], [-15.0], [-13.0], [-30.0], [0.0]]]}, test.internals['inout_0_1']['state'])
        # self.assertDictEqual({'inout': [[[-15.0], [-13.0], [-30.0], [-28.0], [0.0]]]}, test.internals['inout_0_2']['state'])
        W = test.internals['inout_1_0']['param']['W']
        b = test.internals['inout_1_0']['param']['b']
        self.assertDictEqual({'in1': [[[4.0, 2.0], [6.0, 5.0]]], 'target': [[[0.0]]]}, test.internals['inout_1_0']['XY'])
        self.assertDictEqual({'in1': [[[6.0, 5.0], [4.0, 5.0]]], 'target': [[[1.0]]]}, test.internals['inout_1_1']['XY'])
        self.assertDictEqual({'in1': [[[4.0, 5.0], [0.0, 0.0]]], 'target': [[[0.0]]]},test.internals['inout_1_2']['XY'])
        self.assertAlmostEqual({'inout': [[[0.0], [0.0], [0.0], [W[0][0][0]*4.0+W[0][1][0]*2.0+b[0][0]], [W[0][0][0]*6.0+W[0][1][0]*5.0+b[0][0]]]]}, test.internals['inout_1_0']['state'])
        self.assertAlmostEqual({'inout': [[[0.0], [0.0], [W[0][0][0]*4.0+W[0][1][0]*2.0+b[0][0]], [W[0][0][0]*6.0+W[0][1][0]*5.0+b[0][0]], [W[0][0][0]*4.0+W[0][1][0]*5.0+b[0][0]]]]}, test.internals['inout_1_1']['state'])
        self.assertAlmostEqual({'inout': [[[0.0], [W[0][0][0]*4.0+W[0][1][0]*2.0+b[0][0]], [W[0][0][0]*6.0+W[0][1][0]*5.0+b[0][0]], [W[0][0][0]*4.0+W[0][1][0]*5.0+b[0][0]],  [W[0][0][0]*0.0+W[0][1][0]*0.0+b[0][0]]]]}, test.internals['inout_1_2']['state'])
        # Replace instead of rolling
        # self.assertAlmostEqual({'inout': [[[0.0], [0.0], [W[0][0][0] * 4.0 + W[0][1][0] * 2.0 + b[0][0]],
        #                                    [W[0][0][0] * 6.0 + W[0][1][0] * 5.0 + b[0][0]], [0.0]]]},
        #                        test.internals['inout_1_0']['state'])
        # self.assertAlmostEqual({'inout': [
        #     [[0.0], [W[0][0][0] * 4.0 + W[0][1][0] * 2.0 + b[0][0]], [W[0][0][0] * 6.0 + W[0][1][0] * 5.0 + b[0][0]],
        #      [W[0][0][0] * 4.0 + W[0][1][0] * 5.0 + b[0][0]], [0.0]]]}, test.internals['inout_1_1']['state'])
        # self.assertAlmostEqual({'inout': [
        #     [[W[0][0][0] * 4.0 + W[0][1][0] * 2.0 + b[0][0]], [W[0][0][0] * 6.0 + W[0][1][0] * 5.0 + b[0][0]],
        #      [W[0][0][0] * 4.0 + W[0][1][0] * 5.0 + b[0][0]], [W[0][0][0] * 0.0 + W[0][1][0] * 0.0 + b[0][0]], [0.0]]]},
        #                        test.internals['inout_1_2']['state'])

        with self.assertRaises(KeyError):
            test.internals['inout_2_0']

        test.neuralizeModel(clear_model=True)
        test.internals = {}
        test.trainModel(train_dataset='dataset2', optimizer='SGD', lr=0.01, shuffle_data=False, num_of_epochs=1, train_batch_size=2, prediction_samples=1)
        self.assertDictEqual({'in1': [[[1.0, 3.0], [4.0, 2.0]], [[4.0, 2.0], [6.0, 5.0]]], 'target': [[[3.0]],[[0.0]]]}, test.internals['inout_0_0']['XY'])
        self.assertDictEqual({'in1': [[[4.0, 2.0], [6.0, 5.0]], [[6.0, 5.0], [4.0, 5.0]]], 'target': [[[0.0]], [[1.0]]]}, test.internals['inout_0_1']['XY'])
        with self.assertRaises(KeyError):
            test.internals['inout_1_0']

        test.neuralizeModel(clear_model=True)
        dataset3 = {'in1': [[0,1],[2,3],[7,4],[1,3],[4,2],[6,5],[4,5],[0,0]], 'target': [3,4,5,1,3,0,1,0], 'inout':[9,8,7,6,5,4,3,2]}
        test.loadData(name='dataset3', source=dataset3)
        test.internals = {}
        test.trainModel(train_dataset='dataset3', optimizer='SGD', lr=0.01, shuffle_data=False, num_of_epochs=1,
                        train_batch_size=1,
                        prediction_samples=2)
        self.assertDictEqual({'inout': [[[8.0], [7.0], [6.0], [-15.0], [-13.0]]]}, test.internals['inout_0_0']['state'])
        self.assertDictEqual({'inout': [[[7.0], [6.0], [-15.0], [-13.0], [-30.0]]]}, test.internals['inout_0_1']['state'])
        self.assertDictEqual({'inout': [[[6.0], [-15.0], [-13.0], [-30.0], [-28.0]]]}, test.internals['inout_0_2']['state'])
        # Replace insead of rolling
        # self.assertDictEqual({'inout': [[[8.0], [7.0], [-15.0], [-13.0], [9.0]]]}, test.internals['inout_0_0']['state'])
        # self.assertDictEqual({'inout': [[[7.0], [-15.0], [-13.0], [-30.0], [8.0]]]}, test.internals['inout_0_1']['state'])
        # self.assertDictEqual({'inout': [[[-15.0], [-13.0], [-30.0], [-28.0], [7.0]]]}, test.internals['inout_0_2']['state'])
        W = test.internals['inout_1_0']['param']['W']
        b = test.internals['inout_1_0']['param']['b']
        self.assertAlmostEqual({'inout': [[[7.0], [6.0], [5.0], [W[0][0][0] * 4.0 + W[0][1][0] * 2.0 + b[0][0]],
                                           [W[0][0][0] * 6.0 + W[0][1][0] * 5.0 + b[0][0]]]]},
                               test.internals['inout_1_0']['state'])
        self.assertAlmostEqual({'inout': [[[6.0], [5.0], [W[0][0][0] * 4.0 + W[0][1][0] * 2.0 + b[0][0]],
                                           [W[0][0][0] * 6.0 + W[0][1][0] * 5.0 + b[0][0]],
                                           [W[0][0][0] * 4.0 + W[0][1][0] * 5.0 + b[0][0]]]]},
                               test.internals['inout_1_1']['state'])
        self.assertAlmostEqual({'inout': [
            [[5.0], [W[0][0][0] * 4.0 + W[0][1][0] * 2.0 + b[0][0]], [W[0][0][0] * 6.0 + W[0][1][0] * 5.0 + b[0][0]],
             [W[0][0][0] * 4.0 + W[0][1][0] * 5.0 + b[0][0]], [W[0][0][0] * 0.0 + W[0][1][0] * 0.0 + b[0][0]]]]},
                               test.internals['inout_1_2']['state'])
        # replace insead of rolling
        # self.assertAlmostEqual({'inout': [[[7.0], [6.0], [W[0][0][0] * 4.0 + W[0][1][0] * 2.0 + b[0][0]],
        #                                    [W[0][0][0] * 6.0 + W[0][1][0] * 5.0 + b[0][0]], [8.0]]]},
        #                        test.internals['inout_1_0']['state'])
        # self.assertAlmostEqual({'inout': [[[6.0], [W[0][0][0] * 4.0 + W[0][1][0] * 2.0 + b[0][0]],
        #                                    [W[0][0][0] * 6.0 + W[0][1][0] * 5.0 + b[0][0]],
        #                                    [W[0][0][0] * 4.0 + W[0][1][0] * 5.0 + b[0][0]], [7.0]]]},
        #                        test.internals['inout_1_1']['state'])
        # self.assertAlmostEqual({'inout': [
        #     [[W[0][0][0] * 4.0 + W[0][1][0] * 2.0 + b[0][0]], [W[0][0][0] * 6.0 + W[0][1][0] * 5.0 + b[0][0]],
        #      [W[0][0][0] * 4.0 + W[0][1][0] * 5.0 + b[0][0]], [W[0][0][0] * 0.0 + W[0][1][0] * 0.0 + b[0][0]], [6.0]]]},
        #                        test.internals['inout_1_2']['state'])

    def test_training_values_fir_connect_train_linear_more_window(self):
        NeuObj.reset_count()
        input1 = Input('in1', dimensions=2)
        W = Parameter('W', values=[[[-1], [-5]]])
        b = Parameter('b', values=[[1]])
        lin_out = Linear(W=W, b=b)(input1.sw(2))
        output1 = Output('out1', lin_out)

        inout = Input('inout')
        a = Parameter('a', values=[[4], [5]])
        a_big = Parameter('ab', values=[[1], [2], [3], [4], [5]])
        output2 = Output('out2', Fir(parameter=a)(inout.sw(2)))
        output3 = Output('out3', Fir(parameter=a_big)(inout.sw(5)))
        output4 = Output('out4', Fir(parameter=a)(lin_out))

        target = Input('target')

        test = Neu4mes(visualizer=None, seed=42, log_internal=True)
        test.addModel('model', [output1, output2, output3, output4])
        test.addMinimize('error2', target.last(), output2)
        #test.addMinimize('error3', target.last(), output3)
        #test.addMinimize('error4', target.last(), output4)
        test.neuralizeModel()

        # Dataset with only one sample
        dataset = {'in1': [[0,1],[2,3],[7,4],[1,3],[4,2]], 'target': [3,4,5,1,3]}
        self.assertEqual({'out1': [[-4.0, -16.0], [-16.0, -26.0], [-26.0, -15.0], [-15.0, -13.0]],
                               'out2': [-96.0, -194.0, -179.0, -125.0],
                               'out3': [-96.0, -206.0, -235.0, -239.0],
                               'out4': [-96.0, -194.0, -179.0, -125.0]
                          }, test(dataset, connect={'inout':'out1'}))
        test.loadData(name='dataset', source=dataset)
        # TODO add and error
        # dataset = {'in1': [[0,1],[2,3],[7,4],[1,3],[4,2]], 'target': [3,4,5,1]}
        # test.loadData(name='dataset2', source=dataset)

        self.assertListEqual([[[-1], [-5]]],test.model.all_parameters['W'].data.numpy().tolist())
        self.assertListEqual([[1]], test.model.all_parameters['b'].data.numpy().tolist())
        self.assertListEqual([[4], [5]], test.model.all_parameters['a'].data.numpy().tolist())
        self.assertListEqual([[1], [2], [3], [4], [5]], test.model.all_parameters['ab'].data.numpy().tolist())
        test.trainModel(train_dataset='dataset', optimizer='SGD', lr=1, num_of_epochs=1, prediction_samples=0, connect={'inout':'out1'})
        self.assertListEqual([[[6143], [5627]]],test.model.all_parameters['W'].data.numpy().tolist())
        self.assertListEqual([[2305]], test.model.all_parameters['b'].data.numpy().tolist())
        self.assertListEqual([[-3836], [-3323]], test.model.all_parameters['a'].data.numpy().tolist())
        self.assertListEqual([[1], [2], [3], [4], [5]], test.model.all_parameters['ab'].data.numpy().tolist())
        with self.assertRaises(KeyError):
            test.trainModel(train_dataset='dataset', optimizer='SGD', lr=1, num_of_epochs=1, prediction_samples=10)
        with self.assertRaises(ValueError):
            test.trainModel(train_dataset='dataset', optimizer='SGD', lr=1, num_of_epochs=1, prediction_samples=10, connect={'inout':'out1'})
        with self.assertRaises(ValueError):
            test.trainModel(train_dataset='dataset', optimizer='SGD', lr=1, num_of_epochs=1, prediction_samples=1, connect={'inout':'out1'})

        # Data set with more samples
        test.neuralizeModel(clear_model=True)
        dataset2 = {'in1': [[0,1],[2,3],[7,4],[1,3],[4,2],[6,5],[4,5],[0,0]], 'target': [3,4,5,1,3,0,1,0]}
        test.loadData(name='dataset2', source=dataset2)
        self.maxDiff = None
        self.assertEqual({'out1': [[-4.0, -16.0], [-16.0, -26.0], [-26.0, -15.0], [-15.0, -13.0], [-13.0, -30.0], [-30.0, -28.0], [-28.0, 1.0]],
                          'out2': [-96.0, -194.0, -179.0, -125.0, -202.0, -260.0, -107.0],
                          'out3': [-96.0, -206.0, -235.0, -239.0, -315.0, -355.0, -238.0],
                          'out4': [-96.0, -194.0, -179.0, -125.0, -202.0, -260.0, -107.0]
                          }, test(dataset2, connect={'inout':'out1'}))

        # Use a train_batch_size of 4
        # test.trainModel(train_dataset='dataset2', optimizer='SGD', lr=1, num_of_epochs=1, prediction_samples=1, connect={'inout':'out1'}) TODO add this test
        test.trainModel(train_dataset='dataset2', optimizer='SGD', lr=1, num_of_epochs=1, prediction_samples=0, connect={'inout':'out1'})
        self.assertListEqual([[[12779], [11678.5]]],test.model.all_parameters['W'].data.numpy().tolist())
        self.assertListEqual([[3142]], test.model.all_parameters['b'].data.numpy().tolist())
        self.assertListEqual([[-7682], [-7457.5]], test.model.all_parameters['a'].data.numpy().tolist())
        self.assertListEqual([[1], [2], [3], [4], [5]], test.model.all_parameters['ab'].data.numpy().tolist())
        test.neuralizeModel(clear_model=True)
        test.trainModel(train_dataset='dataset2', optimizer='SGD', lr=1, num_of_epochs=1, train_batch_size=4, connect={'inout':'out1'})
        self.assertListEqual([[[12779], [11678.5]]],test.model.all_parameters['W'].data.numpy().tolist())
        self.assertListEqual([[3142]], test.model.all_parameters['b'].data.numpy().tolist())
        self.assertListEqual([[-7682], [-7457.5]], test.model.all_parameters['a'].data.numpy().tolist())
        self.assertListEqual([[1], [2], [3], [4], [5]], test.model.all_parameters['ab'].data.numpy().tolist())

        # Use a small batch but with a prediction sample of 3 = to 4 samples
        test.neuralizeModel(clear_model=True)
        with self.assertRaises(ValueError):
            test.trainModel(train_dataset='dataset2', optimizer='SGD', lr=1, num_of_epochs=1, train_batch_size=1, prediction_samples=4, connect={'inout':'out1'})
        test.trainModel(train_dataset='dataset2', optimizer='SGD', lr=1, num_of_epochs=1, train_batch_size=1, prediction_samples=3, connect={'inout':'out1'})
        self.assertListEqual([[[12779], [11678.5]]],test.model.all_parameters['W'].data.numpy().tolist())
        self.assertListEqual([[3142]], test.model.all_parameters['b'].data.numpy().tolist())
        self.assertListEqual([[-7682], [-7457.5]], test.model.all_parameters['a'].data.numpy().tolist())
        self.assertListEqual([[1], [2], [3], [4], [5]], test.model.all_parameters['ab'].data.numpy().tolist())

        # Different minimize
        test.removeMinimize('error2')
        test.addMinimize('error3', target.last(), output3)
        test.neuralizeModel(clear_model=True)
        self.assertListEqual([[[-1], [-5]]],test.model.all_parameters['W'].data.numpy().tolist())
        self.assertListEqual([[1]], test.model.all_parameters['b'].data.numpy().tolist())
        self.assertListEqual([[4], [5]], test.model.all_parameters['a'].data.numpy().tolist())
        self.assertListEqual([[1], [2], [3], [4], [5]], test.model.all_parameters['ab'].data.numpy().tolist())

        # Use a train_batch_size of 4
        test.trainModel(train_dataset='dataset2', optimizer='SGD', lr=1, num_of_epochs=1, prediction_samples=0, connect={'inout':'out1'})
        self.assertListEqual([[[12779], [11678.5]]],test.model.all_parameters['W'].data.numpy().tolist())
        self.assertListEqual([[3142]], test.model.all_parameters['b'].data.numpy().tolist())
        self.assertListEqual([[4], [5]], test.model.all_parameters['a'].data.numpy().tolist())
        self.assertListEqual([[1], [2], [3], [-7682], [-7457.5]], test.model.all_parameters['ab'].data.numpy().tolist())
        test.neuralizeModel(clear_model=True)
        test.trainModel(train_dataset='dataset2', optimizer='SGD', lr=1, num_of_epochs=1, train_batch_size=4, connect={'inout':'out1'})
        self.assertListEqual([[[12779], [11678.5]]],test.model.all_parameters['W'].data.numpy().tolist())
        self.assertListEqual([[3142]], test.model.all_parameters['b'].data.numpy().tolist())
        self.assertListEqual([[4], [5]], test.model.all_parameters['a'].data.numpy().tolist())
        self.assertListEqual([[1], [2], [3], [-7682], [-7457.5]], test.model.all_parameters['ab'].data.numpy().tolist())

        test.neuralizeModel(clear_model=True)
        test.trainModel(train_dataset='dataset2', optimizer='SGD', lr=1, num_of_epochs=1, train_batch_size=1, prediction_samples=3, connect={'inout':'out1'})
        self.assertDictEqual({'in1': [[[1.0, 3.0], [4.0, 2.0]]], 'target': [[[3.0]]]},
                             test.internals['inout_0_0']['XY'])
        self.assertDictEqual({'in1': [[[4.0, 2.0], [6.0, 5.0]]], 'target': [[[0.0]]]},
                             test.internals['inout_0_1']['XY'])
        self.assertDictEqual({'in1': [[[6.0, 5.0], [4.0, 5.0]]], 'target': [[[1.0]]]},
                             test.internals['inout_0_2']['XY'])
        self.assertDictEqual({'in1': [[[4.0, 5.0], [0.0, 0.0]]], 'target': [[[0.0]]]},
                             test.internals['inout_0_3']['XY'])
        self.assertDictEqual(
            {'out1': [[[-15.0], [-13.0]]], 'out2': [[[-125.0]]], 'out3': [[[-125.0]]], 'out4': [[[-125.0]]]},
            test.internals['inout_0_0']['out'])
        self.assertDictEqual(
            {'out1': [[[-13.0], [-30.0]]], 'out2': [[[-202.0]]], 'out3': [[[-247.0]]], 'out4': [[[-202.0]]]},
            test.internals['inout_0_1']['out'])
        self.assertDictEqual({'out1': [[[4 * -1 + 2 * -5 + 1.0], [6 * -1 + 5 * -5 + 1.0]]],
                              'out2': [[[(4 * -1 + 2 * -5 + 1.0) * 4 + (6 * -1 + 5 * -5 + 1.0) * 5]]],
                              'out3': [[[(-15) * 3.0 + (4 * -1 + 2 * -5 + 1.0) * 4 + (6 * -1 + 5 * -5 + 1.0) * 5]]],
                              'out4': [[[(4 * -1 + 2 * -5 + 1.0) * 4 + (6 * -1 + 5 * -5 + 1.0) * 5]]]},
                             test.internals['inout_0_1']['out'])
        self.assertDictEqual({'out1': [[[6 * -1 + 5 * -5 + 1.0], [4 * -1 + 5 * -5 + 1.0]]],
                              'out2': [[[(6 * -1 + 5 * -5 + 1.0) * 4.0 + (4 * -1 + 5 * -5 + 1.0) * 5.0]]],
                              'out3': [[[(-15) * 2.0 + (4 * -1 + 2 * -5 + 1.0) * 3.0 + (6 * -1 + 5 * -5 + 1.0) * 4.0 + (
                                          4 * -1 + 5 * -5 + 1.0) * 5.0]]],
                              'out4': [[[(6 * -1 + 5 * -5 + 1.0) * 4.0 + (4 * -1 + 5 * -5 + 1.0) * 5.0]]]},
                             test.internals['inout_0_2']['out'])
        self.assertDictEqual({'out1': [[[4 * -1 + 5 * -5 + 1.0], [0 * -1 + 0 * -5 + 1.0]]],
                              'out2': [[[(4 * -1 + 5 * -5 + 1.0) * 4.0 + (0 * -1 + 0 * -5 + 1.0) * 5.0]]],
                              'out3': [[[(-15) * 1.0 + (4 * -1 + 2 * -5 + 1.0) * 2.0 + (6 * -1 + 5 * -5 + 1.0) * 3.0 + (
                                          4 * -1 + 5 * -5 + 1.0) * 4.0 + (0 * -1 + 0 * -5 + 1.0) * 5.0]]],
                              'out4': [[[(4 * -1 + 5 * -5 + 1.0) * 4.0 + (0 * -1 + 0 * -5 + 1.0) * 5.0]]]},
                             test.internals['inout_0_3']['out'])
        self.assertListEqual([[[22273.5], [20993.0]]], test.model.all_parameters['W'].data.numpy().tolist())
        self.assertListEqual([[6154.0]], test.model.all_parameters['b'].data.numpy().tolist())
        self.assertListEqual([[4], [5]], test.model.all_parameters['a'].data.numpy().tolist())
        self.assertListEqual([[-1784.0], [-4020.0], [-7564.5], [-10843.5], [-9033.0]],
                             test.model.all_parameters['ab'].data.numpy().tolist())
        with self.assertRaises(KeyError):
            test.internals['inout_1_0']

        test.neuralizeModel(clear_model=True)
        test.internals = {}
        test.trainModel(train_dataset='dataset2', optimizer='SGD', lr=0.01, shuffle_data=False, num_of_epochs=1,
                        train_batch_size=1,
                        prediction_samples=2, connect={'inout':'out1'})
        self.assertDictEqual({'in1': [[[1.0, 3.0], [4.0, 2.0]]], 'target': [[[3.0]]]},
                             test.internals['inout_0_0']['XY'])
        self.assertDictEqual({'in1': [[[4.0, 2.0], [6.0, 5.0]]], 'target': [[[0.0]]]},
                             test.internals['inout_0_1']['XY'])
        self.assertDictEqual({'in1': [[[6.0, 5.0], [4.0, 5.0]]], 'target': [[[1.0]]]},
                             test.internals['inout_0_2']['XY'])
        self.assertDictEqual(
            {'out1': [[[-15.0], [-13.0]]], 'out2': [[[-125.0]]], 'out3': [[[-125.0]]], 'out4': [[[-125.0]]]},
            test.internals['inout_0_0']['out'])
        self.assertDictEqual(
            {'out1': [[[-13.0], [-30.0]]], 'out2': [[[-202.0]]], 'out3': [[[-247.0]]], 'out4': [[[-202.0]]]},
            test.internals['inout_0_1']['out'])
        self.assertDictEqual({'out1': [[[4 * -1 + 2 * -5 + 1.0], [6 * -1 + 5 * -5 + 1.0]]],
                              'out2': [[[(4 * -1 + 2 * -5 + 1.0) * 4 + (6 * -1 + 5 * -5 + 1.0) * 5]]],
                              'out3': [[[(-15) * 3.0 + (4 * -1 + 2 * -5 + 1.0) * 4 + (6 * -1 + 5 * -5 + 1.0) * 5]]],
                              'out4': [[[(4 * -1 + 2 * -5 + 1.0) * 4 + (6 * -1 + 5 * -5 + 1.0) * 5]]]},
                             test.internals['inout_0_1']['out'])
        self.assertDictEqual({'out1': [[[6 * -1 + 5 * -5 + 1.0], [4 * -1 + 5 * -5 + 1.0]]],
                              'out2': [[[(6 * -1 + 5 * -5 + 1.0) * 4.0 + (4 * -1 + 5 * -5 + 1.0) * 5.0]]],
                              'out3': [[[(-15) * 2.0 + (4 * -1 + 2 * -5 + 1.0) * 3.0 + (6 * -1 + 5 * -5 + 1.0) * 4.0 + (
                                          4 * -1 + 5 * -5 + 1.0) * 5.0]]],
                              'out4': [[[(6 * -1 + 5 * -5 + 1.0) * 4.0 + (4 * -1 + 5 * -5 + 1.0) * 5.0]]]},
                             test.internals['inout_0_2']['out'])
        W = test.internals['inout_1_0']['param']['W']
        b = test.internals['inout_1_0']['param']['b']
        self.assertDictEqual({'in1': [[[4.0, 2.0], [6.0, 5.0]]], 'target': [[[0.0]]]},
                             test.internals['inout_1_0']['XY'])
        self.assertDictEqual({'in1': [[[6.0, 5.0], [4.0, 5.0]]], 'target': [[[1.0]]]},
                             test.internals['inout_1_1']['XY'])
        self.assertDictEqual({'in1': [[[4.0, 5.0], [0.0, 0.0]]], 'target': [[[0.0]]]},
                             test.internals['inout_1_2']['XY'])
        self.assertAlmostEqual({'inout': [[[0.0], [0.0], [0.0], [W[0][0][0] * 4.0 + W[0][1][0] * 2.0 + b[0][0]],
                                         [W[0][0][0] * 6.0 + W[0][1][0] * 5.0 + b[0][0]]]]},
                             test.internals['inout_1_0']['state'])
        self.assertAlmostEqual({'inout': [[[0.0], [0.0], [W[0][0][0] * 4.0 + W[0][1][0] * 2.0 + b[0][0]],
                                         [W[0][0][0] * 6.0 + W[0][1][0] * 5.0 + b[0][0]],
                                         [W[0][0][0] * 4.0 + W[0][1][0] * 5.0 + b[0][0]]]]},
                             test.internals['inout_1_1']['state'])
        self.assertAlmostEqual({'inout': [
            [[0.0], [W[0][0][0] * 4.0 + W[0][1][0] * 2.0 + b[0][0]], [W[0][0][0] * 6.0 + W[0][1][0] * 5.0 + b[0][0]],
             [W[0][0][0] * 4.0 + W[0][1][0] * 5.0 + b[0][0]], [W[0][0][0] * 0.0 + W[0][1][0] * 0.0 + b[0][0]]]]},
                             test.internals['inout_1_2']['state'])
        with self.assertRaises(KeyError):
            test.internals['inout_2_0']

        test.neuralizeModel(clear_model=True)
        test.internals = {}
        test.trainModel(train_dataset='dataset2', optimizer='SGD', lr=0.01, shuffle_data=False, num_of_epochs=1,
                        train_batch_size=2, prediction_samples=1, connect={'inout':'out1'})
        self.assertDictEqual(
            {'in1': [[[1.0, 3.0], [4.0, 2.0]], [[4.0, 2.0], [6.0, 5.0]]], 'target': [[[3.0]], [[0.0]]]},
            test.internals['inout_0_0']['XY'])
        self.assertDictEqual(
            {'in1': [[[4.0, 2.0], [6.0, 5.0]], [[6.0, 5.0], [4.0, 5.0]]], 'target': [[[0.0]], [[1.0]]]},
            test.internals['inout_0_1']['XY'])
        with self.assertRaises(KeyError):
            test.internals['inout_1_0']

        test.neuralizeModel(clear_model=True)
        dataset3 = {'in1': [[0,1],[2,3],[7,4],[1,3],[4,2],[6,5],[4,5],[0,0]], 'target': [3,4,5,1,3,0,1,0], 'inout':[9,8,7,6,5,4,3,2]}
        test.loadData(name='dataset3', source=dataset3)
        test.internals = {}
        test.trainModel(train_dataset='dataset3', optimizer='SGD', lr=0.01, shuffle_data=False, num_of_epochs=1,
                        train_batch_size=1,
                        prediction_samples=2, connect={'inout':'out1'})
        self.assertDictEqual({'inout': [[[8.0], [7.0], [6.0], [-15.0], [-13.0]]]}, test.internals['inout_0_0']['connect'])
        self.assertDictEqual({'inout': [[[7.0], [6.0],  [-15.0], [-13.0], [-30.0]]]}, test.internals['inout_0_1']['connect'])
        self.assertDictEqual({'inout': [[[6.0], [-15.0], [-13.0], [-30.0], [-28.0]]]}, test.internals['inout_0_2']['connect'])
        # Replace instead of rolling
        # self.assertDictEqual({'inout': [[[8.0], [7.0], [-15.0], [-13.0], [9.0]]]}, test.internals['inout_0_0']['connect'])
        # self.assertDictEqual({'inout': [[[7.0], [-15.0], [-13.0], [-30.0], [8.0]]]}, test.internals['inout_0_1']['connect'])
        # self.assertDictEqual({'inout': [[[-15.0], [-13.0], [-30.0], [-28.0], [7.0]]]}, test.internals['inout_0_2']['connect'])
        W = test.internals['inout_1_0']['param']['W']
        b = test.internals['inout_1_0']['param']['b']
        self.assertAlmostEqual({'inout': [[[7.0], [6.0], [5.0], [W[0][0][0] * 4.0 + W[0][1][0] * 2.0 + b[0][0]],
                                           [W[0][0][0] * 6.0 + W[0][1][0] * 5.0 + b[0][0]]]]},
                               test.internals['inout_1_0']['connect'])
        self.assertAlmostEqual({'inout': [[[6.0], [5.0], [W[0][0][0] * 4.0 + W[0][1][0] * 2.0 + b[0][0]],
                                           [W[0][0][0] * 6.0 + W[0][1][0] * 5.0 + b[0][0]],
                                           [W[0][0][0] * 4.0 + W[0][1][0] * 5.0 + b[0][0]]]]},
                               test.internals['inout_1_1']['connect'])
        self.assertAlmostEqual({'inout': [
            [[5.0], [W[0][0][0] * 4.0 + W[0][1][0] * 2.0 + b[0][0]], [W[0][0][0] * 6.0 + W[0][1][0] * 5.0 + b[0][0]],
             [W[0][0][0] * 4.0 + W[0][1][0] * 5.0 + b[0][0]], [W[0][0][0] * 0.0 + W[0][1][0] * 0.0 + b[0][0]]]]},
                               test.internals['inout_1_2']['connect'])
        # Replace insead of rolling
        # self.assertAlmostEqual({'inout': [[[7.0], [6.0], [W[0][0][0] * 4.0 + W[0][1][0] * 2.0 + b[0][0]],
        #                                    [W[0][0][0] * 6.0 + W[0][1][0] * 5.0 + b[0][0]], [8.0]]]},
        #                        test.internals['inout_1_0']['connect'])
        # self.assertAlmostEqual({'inout': [[[6.0], [W[0][0][0] * 4.0 + W[0][1][0] * 2.0 + b[0][0]],
        #                                    [W[0][0][0] * 6.0 + W[0][1][0] * 5.0 + b[0][0]],
        #                                    [W[0][0][0] * 4.0 + W[0][1][0] * 5.0 + b[0][0]], [7.0]]]},
        #                        test.internals['inout_1_1']['connect'])
        # self.assertAlmostEqual({'inout': [
        #     [[W[0][0][0] * 4.0 + W[0][1][0] * 2.0 + b[0][0]], [W[0][0][0] * 6.0 + W[0][1][0] * 5.0 + b[0][0]],
        #      [W[0][0][0] * 4.0 + W[0][1][0] * 5.0 + b[0][0]], [W[0][0][0] * 0.0 + W[0][1][0] * 0.0 + b[0][0]], [6.0]]]},
        #                        test.internals['inout_1_2']['connect'])

    def test_training_values_fir_and_liner_closed_loop(self):
        NeuObj.reset_count()
        input1 = State('in1')
        target_out1 = Input('out1')
        a = Parameter('a', values=[[1]])
        output1 = Output('out1',Fir(parameter=a)(input1.last()))

        input2 = State('in2')
        target_out2 = Input('out2')
        W = Parameter('W', values=[[[1]]])
        b = Parameter('b', values=[[1]])
        output2 = Output('out2', Linear(W=W,b=b)(input2.last()))

        test = Neu4mes(visualizer=None, seed=42)
        test.addModel('model', [output1,output2])
        test.addMinimize('error1', target_out1.last(), output1)
        test.addMinimize('error2', target_out2.last(), output2)
        test.addClosedLoop(output1, input1)
        test.addClosedLoop(output2, input2)
        test.neuralizeModel()
        self.assertEqual({'out1': [0.0], 'out2': [1.0]}, test())
        self.assertEqual({'out1': [1.0], 'out2': [2.0]}, test({'in1': [1.0],'in2': [1.0]}))
        self.assertEqual({'out1': [1.0], 'out2': [3.0]}, test())
        test.resetStates()
        self.assertEqual({'out1': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 'out2':  [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]}, test(prediction_samples=5, num_of_samples=6))

        dataset = {'in1': [1], 'in2': [1.0], 'out1': [3], 'out2': [3]}
        test.loadData(name='dataset', source=dataset)

        self.assertListEqual([[[1.0]]],test.model.all_parameters['W'].data.numpy().tolist())
        self.assertListEqual([[1.0]], test.model.all_parameters['b'].data.numpy().tolist())
        self.assertListEqual([[1.0]], test.model.all_parameters['a'].data.numpy().tolist())
        test.trainModel(optimizer='SGD', splits=[100, 0, 0], lr=1, num_of_epochs=1)
        self.assertListEqual([[[3.0]]], test.model.all_parameters['W'].data.numpy().tolist())
        self.assertListEqual([[3.0]], test.model.all_parameters['b'].data.numpy().tolist())
        self.assertListEqual([[5.0]], test.model.all_parameters['a'].data.numpy().tolist())
        test.trainModel(optimizer='SGD', splits=[100, 0, 0], lr=1, num_of_epochs=1)
        self.assertListEqual([[[-3.0]]],test.model.all_parameters['W'].data.numpy().tolist())
        self.assertListEqual([[-3.0]], test.model.all_parameters['b'].data.numpy().tolist())
        self.assertListEqual([[1.0]], test.model.all_parameters['a'].data.numpy().tolist())

        test.neuralizeModel(clear_model=True)
        self.assertListEqual([[[1.0]]], test.model.all_parameters['W'].data.numpy().tolist())
        self.assertListEqual([[1.0]], test.model.all_parameters['b'].data.numpy().tolist())
        self.assertListEqual([[1.0]], test.model.all_parameters['a'].data.numpy().tolist())
        test.trainModel(optimizer='SGD', splits=[100, 0, 0], lr=1, num_of_epochs=1)
        self.assertListEqual([[[3.0]]], test.model.all_parameters['W'].data.numpy().tolist())
        self.assertListEqual([[3.0]], test.model.all_parameters['b'].data.numpy().tolist())
        self.assertListEqual([[5.0]], test.model.all_parameters['a'].data.numpy().tolist())
        test.trainModel(optimizer='SGD', splits=[100, 0, 0], lr=1, num_of_epochs=1)
        self.assertListEqual([[[-3.0]]],test.model.all_parameters['W'].data.numpy().tolist())
        self.assertListEqual([[-3.0]], test.model.all_parameters['b'].data.numpy().tolist())
        self.assertListEqual([[1.0]], test.model.all_parameters['a'].data.numpy().tolist())

        test.neuralizeModel(clear_model=True)
        self.assertListEqual([[[1.0]]], test.model.all_parameters['W'].data.numpy().tolist())
        self.assertListEqual([[1.0]], test.model.all_parameters['b'].data.numpy().tolist())
        self.assertListEqual([[1.0]], test.model.all_parameters['a'].data.numpy().tolist())
        test.trainModel(optimizer='SGD', splits=[100, 0, 0], lr=1, num_of_epochs=2)
        self.assertListEqual([[[-3.0]]],test.model.all_parameters['W'].data.numpy().tolist())
        self.assertListEqual([[-3.0]], test.model.all_parameters['b'].data.numpy().tolist())
        self.assertListEqual([[1.0]], test.model.all_parameters['a'].data.numpy().tolist())

        dataset = {'in1': [1.0,1.0], 'in2': [1.0,1.0], 'out1': [3.0,3.0], 'out2': [3.0,3.0]}
        test.loadData(name='dataset2', source=dataset)
        test.neuralizeModel(clear_model=True)
        # the out is 3.0 due the mean of the error is not the same of two epochs
        test.trainModel(train_dataset='dataset2', optimizer='SGD', lr=1, num_of_epochs=1)
        self.assertListEqual([[[3.0]]], test.model.all_parameters['W'].data.numpy().tolist())
        self.assertListEqual([[3.0]], test.model.all_parameters['b'].data.numpy().tolist())
        self.assertListEqual([[5.0]], test.model.all_parameters['a'].data.numpy().tolist())
        test.neuralizeModel(clear_model=True)
        test.trainModel(train_dataset='dataset2', optimizer='SGD', lr=1, num_of_epochs=2)
        self.assertListEqual([[[-3.0]]],test.model.all_parameters['W'].data.numpy().tolist())
        self.assertListEqual([[-3.0]], test.model.all_parameters['b'].data.numpy().tolist())
        self.assertListEqual([[1.0]], test.model.all_parameters['a'].data.numpy().tolist())

    def test_training_values_fir_and_liner_train_closed_loop(self):
        NeuObj.reset_count()
        input1 = Input('in1')
        target_out1 = Input('out1')
        a = Parameter('a', values=[[1]])
        output1 = Output('out1',Fir(parameter=a)(input1.last()))

        input2 = Input('in2')
        target_out2 = Input('out2')
        W = Parameter('W', values=[[[1]]])
        b = Parameter('b', values=[[1]])
        output2 = Output('out2', Linear(W=W,b=b)(input2.last()))

        test = Neu4mes(visualizer=None, seed=42)
        test.addModel('model', [output1,output2])
        test.addMinimize('error1', target_out1.last(), output1)
        test.addMinimize('error2', target_out2.last(), output2)
        test.neuralizeModel()
        self.assertEqual({'out1': [0.0], 'out2': [1.0]}, test(closed_loop={'in1':'out1','in2':'out2'}))
        self.assertEqual({'out1': [1.0], 'out2': [2.0]}, test({'in1': [1.0],'in2': [1.0]},closed_loop={'in1':'out1','in2':'out2'}))
        # # The memory is reset for each call
        self.assertEqual({'out1': [0.0], 'out2': [1.0]}, test(closed_loop={'in1':'out1', 'in2':'out2'}))

        dataset = {'in1': [1], 'in2': [1.0], 'out1': [3], 'out2': [3]}
        test.loadData(name='dataset', source=dataset)

        self.assertListEqual([[[1.0]]],test.model.all_parameters['W'].data.numpy().tolist())
        self.assertListEqual([[1.0]], test.model.all_parameters['b'].data.numpy().tolist())
        self.assertListEqual([[1.0]], test.model.all_parameters['a'].data.numpy().tolist())
        test.trainModel(optimizer='SGD', splits=[100, 0, 0], lr=1, num_of_epochs=1, closed_loop={'in1':'out1','in2':'out2'})
        self.assertListEqual([[[3.0]]], test.model.all_parameters['W'].data.numpy().tolist())
        self.assertListEqual([[3.0]], test.model.all_parameters['b'].data.numpy().tolist())
        self.assertListEqual([[5.0]], test.model.all_parameters['a'].data.numpy().tolist())
        test.trainModel(optimizer='SGD', splits=[100, 0, 0], lr=1, num_of_epochs=1, closed_loop={'in1':'out1','in2':'out2'})
        self.assertListEqual([[[-3.0]]],test.model.all_parameters['W'].data.numpy().tolist())
        self.assertListEqual([[-3.0]], test.model.all_parameters['b'].data.numpy().tolist())
        self.assertListEqual([[1.0]], test.model.all_parameters['a'].data.numpy().tolist())

        test.neuralizeModel(clear_model=True)
        self.assertListEqual([[[1.0]]], test.model.all_parameters['W'].data.numpy().tolist())
        self.assertListEqual([[1.0]], test.model.all_parameters['b'].data.numpy().tolist())
        self.assertListEqual([[1.0]], test.model.all_parameters['a'].data.numpy().tolist())
        test.trainModel(optimizer='SGD', splits=[100, 0, 0], lr=1, num_of_epochs=1, closed_loop={'in1':'out1','in2':'out2'})
        self.assertListEqual([[[3.0]]], test.model.all_parameters['W'].data.numpy().tolist())
        self.assertListEqual([[3.0]], test.model.all_parameters['b'].data.numpy().tolist())
        self.assertListEqual([[5.0]], test.model.all_parameters['a'].data.numpy().tolist())
        test.trainModel(optimizer='SGD', splits=[100, 0, 0], lr=1, num_of_epochs=1, closed_loop={'in1':'out1','in2':'out2'})
        self.assertListEqual([[[-3.0]]],test.model.all_parameters['W'].data.numpy().tolist())
        self.assertListEqual([[-3.0]], test.model.all_parameters['b'].data.numpy().tolist())
        self.assertListEqual([[1.0]], test.model.all_parameters['a'].data.numpy().tolist())

        test.neuralizeModel(clear_model=True)
        self.assertListEqual([[[1.0]]], test.model.all_parameters['W'].data.numpy().tolist())
        self.assertListEqual([[1.0]], test.model.all_parameters['b'].data.numpy().tolist())
        self.assertListEqual([[1.0]], test.model.all_parameters['a'].data.numpy().tolist())
        test.trainModel(optimizer='SGD', splits=[100, 0, 0], lr=1, num_of_epochs=2, closed_loop={'in1':'out1','in2':'out2'})
        self.assertListEqual([[[-3.0]]],test.model.all_parameters['W'].data.numpy().tolist())
        self.assertListEqual([[-3.0]], test.model.all_parameters['b'].data.numpy().tolist())
        self.assertListEqual([[1.0]], test.model.all_parameters['a'].data.numpy().tolist())

        dataset = {'in1': [1.0,1.0], 'in2': [1.0,1.0], 'out1': [3.0,3.0], 'out2': [3.0,3.0]}
        test.loadData(name='dataset2', source=dataset)
        test.neuralizeModel(clear_model=True)
        # the out is 3.0 due the mean of the error is not the same of two epochs
        test.trainModel(train_dataset='dataset2', optimizer='SGD', lr=1, num_of_epochs=1, closed_loop={'in1':'out1','in2':'out2'})
        self.assertListEqual([[[3.0]]], test.model.all_parameters['W'].data.numpy().tolist())
        self.assertListEqual([[3.0]], test.model.all_parameters['b'].data.numpy().tolist())
        self.assertListEqual([[5.0]], test.model.all_parameters['a'].data.numpy().tolist())
        test.neuralizeModel(clear_model=True)
        test.trainModel(train_dataset='dataset2', optimizer='SGD', lr=1, num_of_epochs=2, closed_loop={'in1':'out1','in2':'out2'})
        self.assertListEqual([[[-3.0]]],test.model.all_parameters['W'].data.numpy().tolist())
        self.assertListEqual([[-3.0]], test.model.all_parameters['b'].data.numpy().tolist())
        self.assertListEqual([[1.0]], test.model.all_parameters['a'].data.numpy().tolist())

    def test_training_values_fir_and_linear_closed_loop_more_prediction(self):
        NeuObj.reset_count()
        input1 = State('in1')
        target_out1 = Input('out1')
        a = Parameter('a', values=[[1]])
        output1 = Output('out1',Fir(parameter=a)(input1.last()))

        input2 = State('in2')
        target_out2 = Input('out2')
        W = Parameter('W', values=[[[1]]])
        b = Parameter('b', values=[[1]])
        output2 = Output('out2', Linear(W=W,b=b)(input2.last()))

        test = Neu4mes(visualizer=None, seed=42)
        test.addModel('model', [output1,output2])
        test.addMinimize('error1', target_out1.last(), output1)
        test.addMinimize('error2', target_out2.last(), output2)
        test.addClosedLoop(output1, input1)
        test.addClosedLoop(output2, input2)
        test.neuralizeModel()
        self.assertEqual({'out1': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 'out2': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]},
                         test(prediction_samples=5, num_of_samples=6))
        #self.assertEqual({'out1': [1.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0], 'out2': [7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0]},
        #                 test({'in1':[1.0,2.0]},prediction_samples=5))
        self.assertEqual({'out1': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0], 'out2': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 1.0]},
                          test({'in1': [1.0, 2.0]}, prediction_samples=5, num_of_samples=7))
        #self.assertEqual({'out1': [2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0], 'out2': [0.0, -1.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0]},
        #                 test({'in2':[-1.0,-2.0,-3.0]},prediction_samples=5))
        self.assertEqual({'out1': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 'out2': [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 1.0, 2.0]},
                          test({'in2':[-1.0,-2.0,-3.0]}, prediction_samples=5, num_of_samples=8))

        dataset = {'in1': [0,2,7,1], 'in2': [-1,0,-3,7], 'out1': [3,4,5,1], 'out2': [-3,-4,-5,-1]}
        test.loadData(name='dataset2', source=dataset)

        self.assertListEqual([[[1.0]]],test.model.all_parameters['W'].data.numpy().tolist())
        self.assertListEqual([[1.0]], test.model.all_parameters['b'].data.numpy().tolist())
        self.assertListEqual([[1.0]], test.model.all_parameters['a'].data.numpy().tolist())
        test.trainModel(train_dataset='dataset2', optimizer='SGD', lr=1, num_of_epochs=1, prediction_samples=0)
        self.assertListEqual([[[-24.5]]], test.model.all_parameters['W'].data.numpy().tolist())
        self.assertListEqual([[-9.0]], test.model.all_parameters['b'].data.numpy().tolist())
        self.assertListEqual([[-4.0]], test.model.all_parameters['a'].data.numpy().tolist())
        with self.assertRaises(ValueError):
            test.trainModel(train_dataset='dataset2', optimizer='SGD', lr=1, num_of_epochs=1, prediction_samples=10)
        # test.neuralizeModel(clear_model=True)
        # test.trainModel(train_dataset='dataset2', optimizer='SGD', lr=1, num_of_epochs=1, prediction_samples=1)
        # self.assertListEqual([[[1.0]]], test.model.all_parameters['W'].data.numpy().tolist())
        # self.assertListEqual([[1.0]], test.model.all_parameters['b'].data.numpy().tolist())
        # self.assertListEqual([[1.0]], test.model.all_parameters['a'].data.numpy().tolist())
        test.neuralizeModel(clear_model=True)
        self.assertListEqual([[[1.0]]],test.model.all_parameters['W'].data.numpy().tolist())
        self.assertListEqual([[1.0]], test.model.all_parameters['b'].data.numpy().tolist())
        self.assertListEqual([[1.0]], test.model.all_parameters['a'].data.numpy().tolist())
        test.trainModel(train_dataset='dataset2', optimizer='SGD', lr=1, num_of_epochs=1, train_batch_size=1, prediction_samples=3)
        self.assertListEqual([[[1.0]]], test.model.all_parameters['W'].data.numpy().tolist())
        self.assertListEqual([[-24.0]], test.model.all_parameters['b'].data.numpy().tolist())
        self.assertListEqual([[1.0]], test.model.all_parameters['a'].data.numpy().tolist())

        # test.neuralizeModel(clear_model=True)
        # self.assertListEqual([[[1.0]]],test.model.all_parameters['W'].data.numpy().tolist())
        # self.assertListEqual([[1.0]], test.model.all_parameters['b'].data.numpy().tolist())
        # self.assertListEqual([[1.0]], test.model.all_parameters['a'].data.numpy().tolist())
        # test.trainModel(train_dataset='dataset2', optimizer='SGD', lr=1, num_of_epochs=2, train_batch_size=2, prediction_samples=2)
        # self.assertListEqual([[[1.0]]], test.model.all_parameters['W'].data.numpy().tolist())
        # self.assertListEqual([[-24.0]], test.model.all_parameters['b'].data.numpy().tolist())
        # self.assertListEqual([[1.0]], test.model.all_parameters['a'].data.numpy().tolist())

    def test_training_values_fir_and_linear_train_closed_loop_more_prediction(self):
        NeuObj.reset_count()
        input1 = Input('in1')
        target_out1 = Input('out1')
        a = Parameter('a', values=[[1]])
        output1 = Output('out1',Fir(parameter=a)(input1.last()))

        input2 = Input('in2')
        target_out2 = Input('out2')
        W = Parameter('W', values=[[[1]]])
        b = Parameter('b', values=[[1]])
        output2 = Output('out2', Linear(W=W,b=b)(input2.last()))

        test = Neu4mes(visualizer=None, seed=42)
        test.addModel('model', [output1,output2])
        test.addMinimize('error1', target_out1.last(), output1)
        test.addMinimize('error2', target_out2.last(), output2)
        test.neuralizeModel()
        # self.assertEqual({'out1': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 'out2': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]},
        #                  test(prediction_samples=5, closed_loop={'in2':'out2','in1':'out1'}, num_of_samples=6))
        self.assertEqual({'out1': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0], 'out2': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 1.0]},
                         test({'in1':[1.0, 2.0]}, prediction_samples=5, closed_loop={'in2':'out2','in1':'out1'}, num_of_samples=7))
        self.assertEqual({'out1': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 'out2': [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 1.0, 2.0]},
                         test({'in2':[-1.0,-2.0,-3.0]}, prediction_samples=5, closed_loop={'in2':'out2','in1':'out1'}, num_of_samples=8))

        dataset = {'in1': [0,2,7,1], 'in2': [-1,0,-3,7], 'out1': [3,4,5,1], 'out2': [-3,-4,-5,-1]}
        test.loadData(name='dataset2', source=dataset)

        self.assertListEqual([[[1.0]]],test.model.all_parameters['W'].data.numpy().tolist())
        self.assertListEqual([[1.0]], test.model.all_parameters['b'].data.numpy().tolist())
        self.assertListEqual([[1.0]], test.model.all_parameters['a'].data.numpy().tolist())
        test.trainModel(train_dataset='dataset2', optimizer='SGD', lr=1, num_of_epochs=1, prediction_samples=0, closed_loop={'in2':'out2','in1':'out1'})
        self.assertListEqual([[[-24.5]]], test.model.all_parameters['W'].data.numpy().tolist())
        self.assertListEqual([[-9.0]], test.model.all_parameters['b'].data.numpy().tolist())
        self.assertListEqual([[-4.0]], test.model.all_parameters['a'].data.numpy().tolist())
        with self.assertRaises(ValueError):
            test.trainModel(train_dataset='dataset2', optimizer='SGD', lr=1, num_of_epochs=1, prediction_samples=10, closed_loop={'in2':'out2','in1':'out1'})
        # test.trainModel(train_dataset='dataset2', optimizer='SGD', lr=1, num_of_epochs=1, prediction_samples=1,  closed_loop={'in2':'out2','in1':'out1'}) # TODO add this test
        test.neuralizeModel(clear_model=True)
        self.assertListEqual([[[1.0]]],test.model.all_parameters['W'].data.numpy().tolist())
        self.assertListEqual([[1.0]], test.model.all_parameters['b'].data.numpy().tolist())
        self.assertListEqual([[1.0]], test.model.all_parameters['a'].data.numpy().tolist())
        test.trainModel(train_dataset='dataset2', optimizer='SGD', lr=1, num_of_epochs=1, train_batch_size=1, prediction_samples=3,  closed_loop={'in2':'out2','in1':'out1'})
        self.assertListEqual([[[1.0]]], test.model.all_parameters['W'].data.numpy().tolist())
        self.assertListEqual([[-24.0]], test.model.all_parameters['b'].data.numpy().tolist())
        self.assertListEqual([[1.0]], test.model.all_parameters['a'].data.numpy().tolist())

        # test.neuralizeModel(clear_model=True) # TODO add this test
        # self.assertListEqual([[[1.0]]],test.model.all_parameters['W'].data.numpy().tolist())
        # self.assertListEqual([[1.0]], test.model.all_parameters['b'].data.numpy().tolist())
        # self.assertListEqual([[1.0]], test.model.all_parameters['a'].data.numpy().tolist())
        # test.trainModel(train_dataset='dataset2', optimizer='SGD', lr=1, num_of_epochs=2, train_batch_size=2, prediction_samples=2,  closed_loop={'in2':'out2','in1':'out1'})
        # self.assertListEqual([[[1.0]]], test.model.all_parameters['W'].data.numpy().tolist())
        # self.assertListEqual([[-24.0]], test.model.all_parameters['b'].data.numpy().tolist())
        # self.assertListEqual([[1.0]], test.model.all_parameters['a'].data.numpy().tolist())

    def test_training_values_fir_and_liner_closed_loop_bigger_window(self):
        NeuObj.reset_count()
        input1 = State('in1',dimensions=2)
        W = Parameter('W', values=[[[-1],[-5]]])
        b = Parameter('b', values=[[1]])
        output1 = Output('out1', Linear(W=W, b=b)(input1.sw(2)))

        input2 = State('in2')
        a = Parameter('a', values=[[1,3],[2,4],[3,5],[4,6]])
        output2 = Output('out2', Fir(output_dimension=2,parameter=a)(input2.sw(4)))

        target1 = Input('target1')
        target2 = Input('target2', dimensions=2)

        test = Neu4mes(visualizer=None, seed=42)
        test.addModel('model', [output1,output2])
        test.addClosedLoop(output1, input2)
        test.addClosedLoop(output2, input1)
        test.addMinimize('error1', output1, target1.sw(2))
        test.addMinimize('error2', output2, target2.last())
        test.neuralizeModel()

        self.assertEqual({'out1': [[-10.0, -16.0], [-16.0, -33.0], [-33.0, -4.0]], 'out2': [[[-49.0, -107.0]], [[-12.0, -46.0]], [[13.0, 15.0]]]},
                          test({'in1': [[1.0, 2.0], [2.0, 3.0], [4.0, 6.0], [0.0, 1.0]], 'in2': [-10, -16, -5, 2, 2, 2]}))

        dataset = {'in1': [[1.0, 2.0], [2.0, 3.0], [4.0, 6.0], [0.0, 1.0]], 'in2': [-10, -16, -5, 2],
                   'target1': [-11, -17, -12, -20],
                   'target2': [[-34.0, -86.0], [-31.0, -90.0], [-32.0, -86.0], [-33.0, -84.0]]}
        test.loadData(name='dataset', source=dataset)
        self.assertEqual({'out1': [[-10.0, -16.0], [-16.0, -33.0], [-33.0, -4.0]],
                          'out2': [[[-49.0, -107.0]], [[-120.0, -214.0]], [[-205.0, -333.0]]]},
                         test(dataset))

        self.assertListEqual([[[-1],[-5]]],test.model.all_parameters['W'].data.numpy().tolist())
        self.assertListEqual([[1.0]], test.model.all_parameters['b'].data.numpy().tolist())
        self.assertListEqual([[1,3],[2,4],[3,5],[4,6]], test.model.all_parameters['a'].data.numpy().tolist())
        test.trainModel(train_dataset='dataset', optimizer='SGD', lr=1, num_of_epochs=1, prediction_samples=0)
        self.assertListEqual([[[83.0],[105.0]]],test.model.all_parameters['W'].data.numpy().tolist())
        self.assertListEqual([[6.0]], test.model.all_parameters['b'].data.numpy().tolist())
        self.assertListEqual([[-159.0, -227.0], [-254., -364.], [-77., -110.], [36., 52.]], test.model.all_parameters['a'].data.numpy().tolist())
        with self.assertRaises(ValueError):
            test.trainModel(train_dataset='dataset', optimizer='SGD', lr=1, num_of_epochs=1, prediction_samples=10)
        with self.assertRaises(ValueError):
            test.trainModel(train_dataset='dataset', optimizer='SGD', lr=1, num_of_epochs=1, prediction_samples=1)
        test.neuralizeModel(clear_model=True)
        test.trainModel(train_dataset='dataset', optimizer='SGD', lr=1, num_of_epochs=1, prediction_samples=0, minimize_gain={'error1':0})
        self.assertListEqual([[[-1],[-5]]],test.model.all_parameters['W'].data.numpy().tolist())
        self.assertListEqual([[1.0]], test.model.all_parameters['b'].data.numpy().tolist())
        self.assertListEqual([[-159.0, -227.0], [-254., -364.], [-77., -110.], [36., 52.]], test.model.all_parameters['a'].data.numpy().tolist())
        test.neuralizeModel(clear_model=True)
        test.trainModel(train_dataset='dataset', optimizer='SGD', lr=1, num_of_epochs=1, prediction_samples=0, minimize_gain={'error2':0})
        self.assertListEqual([[[83.0],[105.0]]],test.model.all_parameters['W'].data.numpy().tolist())
        self.assertListEqual([[6.0]], test.model.all_parameters['b'].data.numpy().tolist())
        self.assertListEqual([[1,3],[2,4],[3,5],[4,6]], test.model.all_parameters['a'].data.numpy().tolist())

        test.neuralizeModel(clear_model=True)
        dataset2 = {'in1': [[1.0, 2.0], [2.0, 3.0], [4.0, 6.0], [0.0, 1.0], [0.0, 0.0], [0.0, 1.0], [1.0, 0.0]], 'in2': [-10, -16, -5, 2, 3, -3, 5],
                   'target1':[-11,-17,-12,-20,5,1,0],
                    'target2':[[-34.0, -86.0],[-31.0, -90.0],[-32.0, -86.0],[-33.0, -84.0],[-31.0, -84.0],[0.0, -84.0],[-31.0, 0.0]]}
        test.loadData(name='dataset2', source=dataset2)
        self.assertEqual({'out1': [[-10.0, -16.0], [-16.0, -33.0], [-33.0, -4.0],[-4.0,1.0],[1.0,-4.0],[-4.0,0.0]],
                          'out2': [[[-49, -107]], [[-8, -40]], [[-4, -10]], [[19, 33]], [[-11, -17]], [[-24, -44]]]},
                         test(dataset2))

        # Use a train_batch_size of 4
        # test.trainModel(train_dataset='dataset2', optimizer='SGD', lr=1, num_of_epochs=1, prediction_samples=1) # TODO Add this test
        test.trainModel(train_dataset='dataset2', optimizer='SGD', lr=1, num_of_epochs=1, prediction_samples=0)
        self.assertListEqual([[[20.0], [21.0]]],test.model.all_parameters['W'].data.numpy().tolist())
        self.assertListEqual([[2.75]], test.model.all_parameters['b'].data.numpy().tolist())
        self.assertListEqual([[23., 197.5], [-68.75, -94.75], [12., -76.5], [-70.75, -1.25]], test.model.all_parameters['a'].data.numpy().tolist())
        test.neuralizeModel(clear_model=True)
        test.trainModel(train_dataset='dataset2', optimizer='SGD', lr=1, num_of_epochs=1, train_batch_size=4)
        self.assertListEqual([[[20.0], [21.0]]],test.model.all_parameters['W'].data.numpy().tolist())
        self.assertListEqual([[2.75]], test.model.all_parameters['b'].data.numpy().tolist())
        self.assertListEqual([[23., 197.5], [-68.75, -94.75], [12., -76.5], [-70.75, -1.25]], test.model.all_parameters['a'].data.numpy().tolist())

        # Use a small batch but with a prediction sample of 3 = to 4 samples
        dataset3 = {'in1': [[1.0, 2.0], [2.0, 3.0], [4.0, 6.0], [0.0, 1.0], [0.0, 0.0], [0.0, 1.0], [1.0, 0.0]], 'in2': [-10, -16, -5, 2, 3, -3, 5],
                   'target1':[-11, -17, -30, -2, 582, 1421, -18975],
                    'target2':[[-34.0, -86.0],[-31.0, -90.0],[-32.0, -86.0],[-48, -106], [-140, -256], [2254, 3341], [7420, 11374]]}
        test.loadData(name='dataset3', source=dataset3)
        test.neuralizeModel(clear_model=True)
        with self.assertRaises(ValueError):
            test.trainModel(train_dataset='dataset2', optimizer='SGD', lr=1, num_of_epochs=1, train_batch_size=1, prediction_samples=4)
        test.trainModel(train_dataset='dataset3', optimizer='SGD', lr=1, num_of_epochs=1, train_batch_size=1, prediction_samples=3)
        self.assertListEqual([[[-3010.5],[-5211.25]]],test.model.all_parameters['W'].data.numpy().tolist())
        self.assertListEqual([[199.5]], test.model.all_parameters['b'].data.numpy().tolist())
        self.assertListEqual([[252.75, 1172.5], [420.5, 1998.25], [-228.5, 627.5], [-626.75, 3036.5]], test.model.all_parameters['a'].data.numpy().tolist())

        # test.neuralizeModel(clear_model=True)
        # test.trainModel(train_dataset='dataset3', optimizer='SGD', lr=0.01, num_of_epochs=2, train_batch_size=2, prediction_samples=2)
        # self.assertListEqual([[[-3010.5],[-5211.25]]],test.model.all_parameters['W'].data.numpy().tolist())
        # self.assertListEqual([[199.5]], test.model.all_parameters['b'].data.numpy().tolist())
        # self.assertListEqual([[252.75, 1172.5], [420.5, 1998.25], [-228.5, 627.5], [-626.75, 3036.5]], test.model.all_parameters['a'].data.numpy().tolist())

    def test_training_values_fir_and_liner_train_closed_loop_bigger_window(self):
        NeuObj.reset_count()
        input1 = Input('in1',dimensions=2)
        W = Parameter('W', values=[[[-1],[-5]]])
        b = Parameter('b', values=[[1]])
        output1 = Output('out1', Linear(W=W, b=b)(input1.sw(2)))

        input2 = Input('in2')
        a = Parameter('a', values=[[1,3],[2,4],[3,5],[4,6]])
        output2 = Output('out2', Fir(output_dimension=2,parameter=a)(input2.sw(4)))

        target1 = Input('target1')
        target2 = Input('target2', dimensions=2)

        test = Neu4mes(visualizer=None, seed=42)
        test.addModel('model', [output1,output2])
        test.addMinimize('error1', output1, target1.sw(2))
        test.addMinimize('error2', output2, target2.last())
        test.neuralizeModel()

        self.assertEqual({'out1': [[-10.0, -16.0], [-16.0, -33.0], [-33.0, -4.0]], 'out2': [[[-49.0, -107.0]], [[-12.0, -46.0]], [[13.0, 15.0]]]},
                          test({'in1': [[1.0, 2.0], [2.0, 3.0], [4.0, 6.0], [0.0, 1.0]], 'in2': [-10, -16, -5, 2, 2, 2]},closed_loop={'in1':'out2','in2':'out1'}))

        dataset = {'in1': [[1.0, 2.0], [2.0, 3.0], [4.0, 6.0], [0.0, 1.0]], 'in2': [-10, -16, -5, 2],
                   'target1': [-11, -17, -12, -20],
                   'target2': [[-34.0, -86.0], [-31.0, -90.0], [-32.0, -86.0], [-33.0, -84.0]]}
        test.loadData(name='dataset', source=dataset)
        self.assertEqual({'out1': [[-10.0, -16.0], [-16.0, -33.0], [-33.0, -4.0]],
                          'out2': [[[-49.0, -107.0]], [[-120.0, -214.0]], [[-205.0, -333.0]]]},
                         test(dataset,closed_loop={'in1':'out2','in2':'out1'}))

        self.assertListEqual([[[-1],[-5]]],test.model.all_parameters['W'].data.numpy().tolist())
        self.assertListEqual([[1.0]], test.model.all_parameters['b'].data.numpy().tolist())
        self.assertListEqual([[1,3],[2,4],[3,5],[4,6]], test.model.all_parameters['a'].data.numpy().tolist())
        test.trainModel(train_dataset='dataset', optimizer='SGD', lr=1, num_of_epochs=1, prediction_samples=0,closed_loop={'in1':'out2','in2':'out1'})
        self.assertListEqual([[[83.0],[105.0]]],test.model.all_parameters['W'].data.numpy().tolist())
        self.assertListEqual([[6.0]], test.model.all_parameters['b'].data.numpy().tolist())
        self.assertListEqual([[-159.0, -227.0], [-254., -364.], [-77., -110.], [36., 52.]], test.model.all_parameters['a'].data.numpy().tolist())
        with self.assertRaises(ValueError):
            test.trainModel(train_dataset='dataset', optimizer='SGD', lr=1, num_of_epochs=1, prediction_samples=10,closed_loop={'in1':'out2','in2':'out1'})
        with self.assertRaises(ValueError):
            test.trainModel(train_dataset='dataset', optimizer='SGD', lr=1, num_of_epochs=1, prediction_samples=1,closed_loop={'in1':'out2','in2':'out1'})
        test.neuralizeModel(clear_model=True)
        test.trainModel(train_dataset='dataset', optimizer='SGD', lr=1, num_of_epochs=1, prediction_samples=0, minimize_gain={'error1':0},closed_loop={'in1':'out2','in2':'out1'})
        self.assertListEqual([[[-1],[-5]]],test.model.all_parameters['W'].data.numpy().tolist())
        self.assertListEqual([[1.0]], test.model.all_parameters['b'].data.numpy().tolist())
        self.assertListEqual([[-159.0, -227.0], [-254., -364.], [-77., -110.], [36., 52.]], test.model.all_parameters['a'].data.numpy().tolist())
        test.neuralizeModel(clear_model=True)
        test.trainModel(train_dataset='dataset', optimizer='SGD', lr=1, num_of_epochs=1, prediction_samples=0, minimize_gain={'error2':0},closed_loop={'in1':'out2','in2':'out1'})
        self.assertListEqual([[[83.0],[105.0]]],test.model.all_parameters['W'].data.numpy().tolist())
        self.assertListEqual([[6.0]], test.model.all_parameters['b'].data.numpy().tolist())
        self.assertListEqual([[1,3],[2,4],[3,5],[4,6]], test.model.all_parameters['a'].data.numpy().tolist())

        test.neuralizeModel(clear_model=True)
        dataset2 = {'in1': [[1.0, 2.0], [2.0, 3.0], [4.0, 6.0], [0.0, 1.0], [0.0, 0.0], [0.0, 1.0], [1.0, 0.0]], 'in2': [-10, -16, -5, 2, 3, -3, 5],
                   'target1':[-11,-17,-12,-20,5,1,0],
                    'target2':[[-34.0, -86.0],[-31.0, -90.0],[-32.0, -86.0],[-33.0, -84.0],[-31.0, -84.0],[0.0, -84.0],[-31.0, 0.0]]}
        test.loadData(name='dataset2', source=dataset2)
        self.assertEqual({'out1': [[-10.0, -16.0], [-16.0, -33.0], [-33.0, -4.0],[-4.0,1.0],[1.0,-4.0],[-4.0,0.0]],
                          'out2': [[[-49, -107]], [[-8, -40]], [[-4, -10]], [[19, 33]], [[-11, -17]], [[-24, -44]]]},
                         test(dataset2,closed_loop={'in1':'out2','in2':'out1'}))

        # Use a train_batch_size of 4
        # test.trainModel(train_dataset='dataset2', optimizer='SGD', lr=1, num_of_epochs=1, prediction_samples=1, closed_loop={'in1':'out2','in2':'out1'}) # TODO Add this test
        test.trainModel(train_dataset='dataset2', optimizer='SGD', lr=1, num_of_epochs=1, prediction_samples=0, closed_loop={'in1':'out2','in2':'out1'})
        self.assertListEqual([[[20.0], [21.0]]],test.model.all_parameters['W'].data.numpy().tolist())
        self.assertListEqual([[2.75]], test.model.all_parameters['b'].data.numpy().tolist())
        self.assertListEqual([[23., 197.5], [-68.75, -94.75], [12., -76.5], [-70.75, -1.25]], test.model.all_parameters['a'].data.numpy().tolist())
        test.neuralizeModel(clear_model=True)
        test.trainModel(train_dataset='dataset2', optimizer='SGD', lr=1, num_of_epochs=1, train_batch_size=4,closed_loop={'in1':'out2','in2':'out1'})
        self.assertListEqual([[[20.0], [21.0]]],test.model.all_parameters['W'].data.numpy().tolist())
        self.assertListEqual([[2.75]], test.model.all_parameters['b'].data.numpy().tolist())
        self.assertListEqual([[23., 197.5], [-68.75, -94.75], [12., -76.5], [-70.75, -1.25]], test.model.all_parameters['a'].data.numpy().tolist())

        # Use a small batch but with a prediction sample of 3 = to 4 samples
        dataset3 = {'in1': [[1.0, 2.0], [2.0, 3.0], [4.0, 6.0], [0.0, 1.0], [0.0, 0.0], [0.0, 1.0], [1.0, 0.0]], 'in2': [-10, -16, -5, 2, 3, -3, 5],
                   'target1':[-11, -17, -30, -2, 582, 1421, -18975],
                    'target2':[[-34.0, -86.0],[-31.0, -90.0],[-32.0, -86.0],[-48, -106], [-140, -256], [2254, 3341], [7420, 11374]]}
        test.loadData(name='dataset3', source=dataset3)
        test.neuralizeModel(clear_model=True)
        with self.assertRaises(ValueError):
            test.trainModel(train_dataset='dataset2', optimizer='SGD', lr=1, num_of_epochs=1, train_batch_size=1, prediction_samples=4,closed_loop={'in1':'out2','in2':'out1'})
        test.trainModel(train_dataset='dataset3', optimizer='SGD', lr=1, num_of_epochs=1, train_batch_size=1, prediction_samples=3,closed_loop={'in1':'out2','in2':'out1'})
        self.assertListEqual([[[-3010.5],[-5211.25]]],test.model.all_parameters['W'].data.numpy().tolist())
        self.assertListEqual([[199.5]], test.model.all_parameters['b'].data.numpy().tolist())
        self.assertListEqual([[252.75, 1172.5], [420.5, 1998.25], [-228.5, 627.5], [-626.75, 3036.5]], test.model.all_parameters['a'].data.numpy().tolist())

        # test.neuralizeModel(clear_model=True)
        # test.trainModel(train_dataset='dataset3', optimizer='SGD', lr=1, num_of_epochs=2, train_batch_size=2, prediction_samples=2,closed_loop={'in1':'out2','in2':'out1'})
        # self.assertListEqual([[[-3010.5],[-5211.25]]],test.model.all_parameters['W'].data.numpy().tolist())
        # self.assertListEqual([[199.5]], test.model.all_parameters['b'].data.numpy().tolist())
        # self.assertListEqual([[252.75, 1172.5], [420.5, 1998.25], [-228.5, 627.5], [-626.75, 3036.5]], test.model.all_parameters['a'].data.numpy().tolist())

    def test_train_compare_state_and_closed_loop(self):
        dataset = {'control': [-1, -1, -5, 2,-1, -1, -5, 2,-1, -1, -5, 2,-1, -1, -5, 2,-1, -1, -5, 2,-1, -1, -1, -2,-1, -1, -1, -2,-1, -1, -1, -2],
                   'target1': [-1, -1, -1, -2,-1, -1, -1, -2,-1, -1, -1, -2,-1, -1, -1, -2,-1, -1, -1, -2,-1, -1, -1, -2,-1, -1, -1, -2,-1, -1, -1, -2],
                   'target2': [[-1.0, -1.0], [-1.0, -2.0], [-1.0, -1.0], [-1.0, -2.0],
                               [-1.0, -1.0], [-1.0, -2.0], [-1.0, -1.0], [-1.0, -2.0],
                               [-1.0, -1.0], [-1.0, -2.0], [-1.0, -1.0], [-1.0, -2.0],
                               [-1.0, -1.0], [-1.0, -2.0], [-1.0, -1.0], [-1.0, -2.0],
                               [-1.0, -1.0], [-1.0, -2.0], [-1.0, -1.0], [-1.0, -2.0],
                               [-1.0, -1.0], [-1.0, -2.0], [-1.0, -1.0], [-1.0, -2.0],
                               [-1.0, -1.0], [-1.0, -2.0], [-1.0, -1.0], [-1.0, -2.0],
                               [-1.0, -1.0], [-1.0, -2.0], [-1.0, -1.0], [-1.0, -2.0]
                               ]}

        feed = Input('control')
        input1 = Input('in1', dimensions=2)
        W = Parameter('W', values=[[[0.1],[0.1]]])
        b = Parameter('b', values=[[0.1]])
        output1 = Output('out1', feed.sw(2)+Linear(W=W, b=b)(input1.sw(2)))

        input2 = Input('in2')
        a = Parameter('a', values=[[0.1,0.3],[0.2,0.4],[0.3,0.5],[0.4,0.6]])
        output2 = Output('out2', Fir(output_dimension=2,parameter=a)(input2.sw(4)))

        target1 = Input('target1')
        target2 = Input('target2', dimensions=2)

        test = Neu4mes(visualizer=TextVisualizer(), seed=42)
        test.addModel('model', [output1,output2])
        test.addMinimize('error1', output1, target1.sw(2))
        test.addMinimize('error2', output2, target2.last())
        test.neuralizeModel()
        test.loadData(name='dataset', source=dataset)
        test.trainModel(splits=[60,40,0], optimizer='SGD', lr=0.001, num_of_epochs=1, prediction_samples=10,
                        closed_loop={'in1': 'out2', 'in2': 'out1'})

        feed = Input('control')
        input1 = State('in1',dimensions=2)
        W = Parameter('W', values=[[[0.1],[0.1]]])
        b = Parameter('b', values=[[0.1]])
        output1 = Output('out1', feed.sw(2) + Linear(W=W, b=b)(input1.sw(2)))

        input2 = State('in2')
        a = Parameter('a', values=[[0.1,0.3],[0.2,0.4],[0.3,0.5],[0.4,0.6]])
        output2 = Output('out2', Fir(output_dimension=2, parameter=a)(input2.sw(4)))

        target1 = Input('target1')
        target2 = Input('target2', dimensions=2)

        test2 = Neu4mes(visualizer=TextVisualizer(), seed=42)
        test2.addModel('model', [output1, output2])
        test2.addMinimize('error1', output1, target1.sw(2))
        test2.addMinimize('error2', output2, target2.last())
        test2.addClosedLoop(output1, input2)
        test2.addClosedLoop(output2, input1)
        test2.neuralizeModel()
        test2.loadData(name='dataset', source=dataset)
        test2.trainModel(splits=[60,40,0], optimizer='SGD', lr=0.001, num_of_epochs=1, prediction_samples=10)

        self.assertListEqual(test2.model.all_parameters['W'].data.numpy().tolist(), test.model.all_parameters['W'].data.numpy().tolist())
        self.assertListEqual(test2.model.all_parameters['a'].data.numpy().tolist(), test.model.all_parameters['a'].data.numpy().tolist())
        self.assertListEqual(test2.model.all_parameters['b'].data.numpy().tolist(), test.model.all_parameters['b'].data.numpy().tolist())
        self.assertListEqual(test2.training['error1']['train'] , test.training['error1']['train'])
        self.assertListEqual(test2.training['error1']['val'], test.training['error1']['val'])
        self.assertListEqual(test2.training['error2']['train'] , test.training['error2']['train'])
        self.assertListEqual(test2.training['error2']['val'], test.training['error2']['val'])

if __name__ == '__main__':
    unittest.main()