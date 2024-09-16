import unittest

import sys
import os
# append a new directory to sys.path
sys.path.append(os.getcwd())
from neu4mes import *
from neu4mes import relation
relation.CHECK_NAMES = False

# 6 Tests
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

        test = Neu4mes(visualizer=None, seed=42)
        test.addModel('model', [output1,output2])
        test.addMinimize('error', target.last(), output2)
        test.addConnect(output1, inout)
        test.neuralizeModel()
        self.assertEqual({'out1': [1.0], 'out2': [2.0]}, test({'in1': [1]}))

        dataset = {'in1': [0,2,7,1], 'out1': [3,4,5,1]}
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
        with self.assertRaises(ValueError):
            test.trainModel(train_dataset='dataset2', optimizer='SGD', lr=1, num_of_epochs=1, prediction_samples=1)
        test.neuralizeModel(clear_model=True)
        self.assertListEqual([[[1.0]]],test.model.all_parameters['W'].data.numpy().tolist())
        self.assertListEqual([[1.0]], test.model.all_parameters['b'].data.numpy().tolist())
        self.assertListEqual([[1.0]], test.model.all_parameters['a'].data.numpy().tolist())
        test.trainModel(train_dataset='dataset2', optimizer='SGD', lr=1, num_of_epochs=1, train_batch_size=1, prediction_samples=3)
        self.assertListEqual([[[-9.0]]], test.model.all_parameters['W'].data.numpy().tolist())
        self.assertListEqual([[0.5]], test.model.all_parameters['b'].data.numpy().tolist())
        self.assertListEqual([[-9.0]], test.model.all_parameters['a'].data.numpy().tolist())

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
        with self.assertRaises(ValueError):
            test.trainModel(train_dataset='dataset2', optimizer='SGD', lr=1, num_of_epochs=1, prediction_samples=1, connect={'inout': 'out1'})
        test.neuralizeModel(clear_model=True)
        self.assertListEqual([[[1.0]]],test.model.all_parameters['W'].data.numpy().tolist())
        self.assertListEqual([[1.0]], test.model.all_parameters['b'].data.numpy().tolist())
        self.assertListEqual([[1.0]], test.model.all_parameters['a'].data.numpy().tolist())
        test.trainModel(train_dataset='dataset2', optimizer='SGD', lr=1, num_of_epochs=1, train_batch_size=1, prediction_samples=3, connect={'inout': 'out1'})
        self.assertListEqual([[[-9.0]]], test.model.all_parameters['W'].data.numpy().tolist())
        self.assertListEqual([[0.5]], test.model.all_parameters['b'].data.numpy().tolist())
        self.assertListEqual([[-9.0]], test.model.all_parameters['a'].data.numpy().tolist())

        dataset = {'in1': [0,2,7,1], 'out1': [3,4,5,1], 'inout':[1,1,2,2]}
        test.loadData(name='dataset3', source=dataset)
        test.neuralizeModel(clear_model=True)
        test.trainModel(train_dataset='dataset2', optimizer='SGD', lr=1, num_of_epochs=1, train_batch_size=1, prediction_samples=3, connect={'inout': 'out1'})
        self.assertListEqual([[[-9.0]]], test.model.all_parameters['W'].data.numpy().tolist())
        self.assertListEqual([[0.5]], test.model.all_parameters['b'].data.numpy().tolist())
        self.assertListEqual([[-9.0]], test.model.all_parameters['a'].data.numpy().tolist())

        test.neuralizeModel(clear_model=True)
        test.trainModel(train_dataset='dataset3', optimizer='SGD', shuffle_data=False, lr=1, num_of_epochs=1, train_batch_size=1, prediction_samples=3)
        self.assertListEqual([[[-273.0]]], test.model.all_parameters['W'].data.numpy().tolist())
        self.assertListEqual([[-137.0]], test.model.all_parameters['b'].data.numpy().tolist())
        self.assertListEqual([[1.0]], test.model.all_parameters['a'].data.numpy().tolist())

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

        test = Neu4mes(seed=42)
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
        with self.assertRaises(ValueError):
            test.trainModel(train_dataset='dataset2', optimizer='SGD', lr=1, num_of_epochs=1, prediction_samples=1)
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
        test.trainModel(train_dataset='dataset2', optimizer='SGD', lr=1, num_of_epochs=1, train_batch_size=1, prediction_samples=3)
        self.assertListEqual([[[12779], [11678.5]]],test.model.all_parameters['W'].data.numpy().tolist())
        self.assertListEqual([[3142]], test.model.all_parameters['b'].data.numpy().tolist())
        self.assertListEqual([[4], [5]], test.model.all_parameters['a'].data.numpy().tolist())
        self.assertListEqual([[1], [2], [3], [-7682], [-7457.5]], test.model.all_parameters['ab'].data.numpy().tolist())

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
        test.clear_state()
        self.assertEqual({'out1': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 'out2':  [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]}, test(prediction_samples=5))

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
                         test(prediction_samples=5))
        self.assertEqual({'out1': [1.0, 2.0, 2.0, 2.0, 2.0, 2.0,2.0], 'out2': [7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0]},
                         test({'in1':[1.0,2.0]},prediction_samples=5))
        self.assertEqual({'out1': [2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0], 'out2': [0.0, -1.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0]},
                         test({'in2':[-1.0,-2.0,-3.0]},prediction_samples=5))

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
        with self.assertRaises(ValueError):
            test.trainModel(train_dataset='dataset2', optimizer='SGD', lr=1, num_of_epochs=1, prediction_samples=1)
        test.neuralizeModel(clear_model=True)
        self.assertListEqual([[[1.0]]],test.model.all_parameters['W'].data.numpy().tolist())
        self.assertListEqual([[1.0]], test.model.all_parameters['b'].data.numpy().tolist())
        self.assertListEqual([[1.0]], test.model.all_parameters['a'].data.numpy().tolist())
        test.trainModel(train_dataset='dataset2', optimizer='SGD', lr=1, num_of_epochs=1, train_batch_size=1, prediction_samples=3)
        self.assertListEqual([[[1.0]]], test.model.all_parameters['W'].data.numpy().tolist())
        self.assertListEqual([[-24.0]], test.model.all_parameters['b'].data.numpy().tolist())
        self.assertListEqual([[1.0]], test.model.all_parameters['a'].data.numpy().tolist())

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
        self.assertEqual({'out1': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 'out2': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]},
                         test(prediction_samples=5, closed_loop={'in2':'out2','in1':'out1'}))
        self.assertEqual({'out1': [1.0, 2.0, 2.0, 2.0, 2.0, 2.0,2.0], 'out2': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]},
                         test({'in1':[1.0,2.0]},prediction_samples=5, closed_loop={'in2':'out2','in1':'out1'}))
        self.assertEqual({'out1': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 'out2': [0.0, -1.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0]},
                         test({'in2':[-1.0,-2.0,-3.0]},prediction_samples=5, closed_loop={'in2':'out2','in1':'out1'}))

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
        with self.assertRaises(ValueError):
            test.trainModel(train_dataset='dataset2', optimizer='SGD', lr=1, num_of_epochs=1, prediction_samples=1,  closed_loop={'in2':'out2','in1':'out1'})
        test.neuralizeModel(clear_model=True)
        self.assertListEqual([[[1.0]]],test.model.all_parameters['W'].data.numpy().tolist())
        self.assertListEqual([[1.0]], test.model.all_parameters['b'].data.numpy().tolist())
        self.assertListEqual([[1.0]], test.model.all_parameters['a'].data.numpy().tolist())
        test.trainModel(train_dataset='dataset2', optimizer='SGD', lr=1, num_of_epochs=1, train_batch_size=1, prediction_samples=3,  closed_loop={'in2':'out2','in1':'out1'})
        self.assertListEqual([[[1.0]]], test.model.all_parameters['W'].data.numpy().tolist())
        self.assertListEqual([[-24.0]], test.model.all_parameters['b'].data.numpy().tolist())
        self.assertListEqual([[1.0]], test.model.all_parameters['a'].data.numpy().tolist())

    # def test_training_other(self):
    #     NeuObj.reset_count()
    #     input1 = Input('in1')
    #     target_out1 = Input('out1')
    #     a = Parameter('a', values=[[1]])
    #     output1 = Output('out1',Fir(parameter=a)(input1.last()))
    #
    #     input2 = Input('in2')
    #     target_out2 = Input('out2')
    #     W = Parameter('W', values=[[[1]]])
    #     b = Parameter('b', values=[[1]])
    #     output2 = Output('out2', Linear(W=W,b=b)(input2.sw(3)))
    #
    #     test = Neu4mes(visualizer=None, seed=42)
    #     test.addModel('model', [output1,output2])
    #     test.addMinimize('error1', target_out1.last(), output1)
    #     test.addMinimize('error2', target_out2.last(), output2)
    #     test.neuralizeModel()
    #     #self.assertEqual({'out1': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 'out2': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]},
    #                      #test(prediction_samples=5, closed_loop={'in2':'out2','in1':'out1'}))
    #
    #     dataset = {'in1': [0,2,7,1], 'in2': [-1,0,-3,7], 'out1': [3,4,5,1], 'out2': [-3,-4,-5,-1]}
    #     test.loadData(name='dataset2', source=dataset)
    #     test.trainModel(train_dataset='dataset2', optimizer='SGD', lr=1, num_of_epochs=1, prediction_samples=0, closed_loop={'in2':'out2','in1':'out1'})


if __name__ == '__main__':
    unittest.main()