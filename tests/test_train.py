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

class Neu4mesTrainingTest(unittest.TestCase):
    def TestAlmostEqual(self, data1, data2, precision=4):
        assert np.asarray(data1, dtype=np.float32).ndim == np.asarray(data2, dtype=np.float32).ndim, f'Inputs must have the same dimension! Received {type(data1)} and {type(data2)}'
        if type(data1) == type(data2) == list:
            for pred, label in zip(data1, data2):
                self.TestAlmostEqual(pred, label, precision=precision)
        else:
            self.assertAlmostEqual(data1, data2, places=precision)

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
        NeuObj.reset_count()
        input1 = Input('in1')
        target = Input('out1')
        W = Parameter('W', values=[[[1]]])
        b = Parameter('b', values=[[1]])
        output1 = Output('out', Linear(W=W,b=b)(input1.last()))

        test = Neu4mes(visualizer=None, seed=42)
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
        NeuObj.reset_count()
        input1 = Input('in1')
        target = Input('out1')
        a = Parameter('a', values=[[1]])
        fir_out = Fir(parameter=a)(input1.last())
        output1 = Output('out1', fir_out)

        W = Parameter('W', values=[[[1]]])
        b = Parameter('b', values=[[1]])
        output2 = Output('out2', Linear(W=W,b=b)(fir_out))

        test = Neu4mes(visualizer=None, seed=42)
        test.addModel('model', [output1,output2])
        test.addMinimize('error', target.last(), output2)
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

    def test_multimodel_with_loss_gain_and_lr_gain(self):
        ## Model1
        input1 = Input('in1')
        a1 = Parameter('a1', dimensions=1, tw=0.05, values=[[1],[1],[1],[1],[1]])
        output11 = Output('out11', Fir(parameter=a1)(input1.tw(0.05)))
        a2 = Parameter('a2', dimensions=1, tw=0.05, values=[[1], [1], [1], [1], [1]])
        output12 = Output('out12', Fir(parameter=a2)(input1.tw(0.05)))
        a3 = Parameter('a3', dimensions=1, tw=0.05, values=[[1], [1], [1], [1], [1]])
        output13 = Output('out13', Fir(parameter=a3)(input1.tw(0.05)))

        test = Neu4mes(visualizer=None, seed=42)
        test.addModel('model1', [output11, output12, output13])
        test.addMinimize('error11', input1.next(), output11)
        test.addMinimize('error12', input1.next(), output12)
        test.addMinimize('error13', input1.next(), output13)

        ## Model2
        input2 = Input('in2')
        b1 = Parameter('b1', dimensions=1, tw=0.05, values=[[1],[1],[1],[1],[1]])
        output21 = Output('out21', Fir(parameter=b1)(input2.tw(0.05)))
        b2 = Parameter('b2', dimensions=1, tw=0.05, values=[[1],[1],[1],[1],[1]])
        output22 = Output('out22', Fir(parameter=b2)(input2.tw(0.05)))
        b3 = Parameter('b3', dimensions=1, tw=0.05, values=[[1],[1],[1],[1],[1]])
        output23 = Output('out23', Fir(parameter=b3)(input2.tw(0.05)))

        test.addModel('model2', [output21, output22, output23])
        test.addMinimize('error21', input2.next(), output21)
        test.addMinimize('error22', input2.next(), output22)
        test.addMinimize('error23', input2.next(), output23)
        test.neuralizeModel(0.01)

        data_in1 = [1, 1, 1, 1, 1, 2]
        data_in2 = [1, 1, 1, 1, 1, 2]
        dataset = {'in1': data_in1, 'in2': data_in2}

        test.loadData(name='dataset', source=dataset)

        ## Train only model1
        self.assertListEqual(test.model.all_parameters['a1'].data.numpy().tolist(), [[1], [1], [1], [1], [1]])
        self.assertListEqual(test.model.all_parameters['b1'].data.numpy().tolist(), [[1], [1], [1], [1], [1]])
        self.assertListEqual(test.model.all_parameters['a2'].data.numpy().tolist(), [[1], [1], [1], [1], [1]])
        self.assertListEqual(test.model.all_parameters['b2'].data.numpy().tolist(), [[1], [1], [1], [1], [1]])
        self.assertListEqual(test.model.all_parameters['a3'].data.numpy().tolist(), [[1], [1], [1], [1], [1]])
        self.assertListEqual(test.model.all_parameters['b3'].data.numpy().tolist(), [[1], [1], [1], [1], [1]])
        test.trainModel(optimizer='SGD', models='model1', splits=[100,0,0], lr=1, num_of_epochs=1)
        self.assertListEqual(test.model.all_parameters['a1'].data.numpy().tolist(), [[-5],[-5],[-5],[-5],[-5]])
        self.assertListEqual(test.model.all_parameters['b1'].data.numpy().tolist(), [[1],[1],[1],[1],[1]])
        self.assertListEqual(test.model.all_parameters['a2'].data.numpy().tolist(), [[-5],[-5],[-5],[-5],[-5]])
        self.assertListEqual(test.model.all_parameters['b2'].data.numpy().tolist(), [[1],[1],[1],[1],[1]])
        self.assertListEqual(test.model.all_parameters['a3'].data.numpy().tolist(), [[-5],[-5],[-5],[-5],[-5]])
        self.assertListEqual(test.model.all_parameters['b3'].data.numpy().tolist(), [[1],[1],[1],[1],[1]])

        ## Train only model2
        test.neuralizeModel(0.01, clear_model=True)
        test.trainModel(optimizer='SGD', models='model2', splits=[100,0,0], lr=1, num_of_epochs=1)
        self.assertListEqual(test.model.all_parameters['a1'].data.numpy().tolist(), [[1],[1],[1],[1],[1]])
        self.assertListEqual(test.model.all_parameters['b1'].data.numpy().tolist(), [[-5],[-5],[-5],[-5],[-5]])
        self.assertListEqual(test.model.all_parameters['a2'].data.numpy().tolist(), [[1],[1],[1],[1],[1]])
        self.assertListEqual(test.model.all_parameters['b2'].data.numpy().tolist(), [[-5],[-5],[-5],[-5],[-5]])
        self.assertListEqual(test.model.all_parameters['a3'].data.numpy().tolist(), [[1],[1],[1],[1],[1]])
        self.assertListEqual(test.model.all_parameters['b3'].data.numpy().tolist(), [[-5],[-5],[-5],[-5],[-5]])

        ## Train both models
        test.neuralizeModel(0.01, clear_model=True)
        test.trainModel(optimizer='SGD', models=['model1','model2'], splits=[100,0,0], lr=1, num_of_epochs=1)
        self.assertListEqual(test.model.all_parameters['a1'].data.numpy().tolist(), [[-5],[-5],[-5],[-5],[-5]])
        self.assertListEqual(test.model.all_parameters['b1'].data.numpy().tolist(), [[-5],[-5],[-5],[-5],[-5]])
        self.assertListEqual(test.model.all_parameters['a2'].data.numpy().tolist(), [[-5],[-5],[-5],[-5],[-5]])
        self.assertListEqual(test.model.all_parameters['b2'].data.numpy().tolist(), [[-5],[-5],[-5],[-5],[-5]])
        self.assertListEqual(test.model.all_parameters['a3'].data.numpy().tolist(), [[-5],[-5],[-5],[-5],[-5]])
        self.assertListEqual(test.model.all_parameters['b3'].data.numpy().tolist(), [[-5],[-5],[-5],[-5],[-5]])

        ## Train both models but set the gain of a to zero and the gain of b to double
        test.neuralizeModel(0.01, clear_model=True)
        test.trainModel(optimizer='SGD', models=['model1','model2'], splits=[100,0,0], lr=1, num_of_epochs=1, lr_param={'a1':0, 'b1':2})
        self.assertListEqual(test.model.all_parameters['a1'].data.numpy().tolist(), [[1],[1],[1],[1],[1]])
        self.assertListEqual(test.model.all_parameters['b1'].data.numpy().tolist(), [[-11],[-11],[-11],[-11],[-11]])
        self.assertListEqual(test.model.all_parameters['a2'].data.numpy().tolist(), [[-5],[-5],[-5],[-5],[-5]])
        self.assertListEqual(test.model.all_parameters['b2'].data.numpy().tolist(), [[-5],[-5],[-5],[-5],[-5]])
        self.assertListEqual(test.model.all_parameters['a3'].data.numpy().tolist(), [[-5],[-5],[-5],[-5],[-5]])
        self.assertListEqual(test.model.all_parameters['b3'].data.numpy().tolist(), [[-5],[-5],[-5],[-5],[-5]])

        ## Train both models but set the minimize gain of error1 to zero and the minimize gain of error2 to double
        test.neuralizeModel(0.01, clear_model=True)
        test.trainModel(optimizer='SGD', models=['model1','model2'], splits=[100,0,0], lr=1, num_of_epochs=1, minimize_gain={'error11':0})
        self.assertListEqual(test.model.all_parameters['a1'].data.numpy().tolist(), [[1],[1],[1],[1],[1]])
        self.assertListEqual(test.model.all_parameters['b1'].data.numpy().tolist(), [[-5],[-5],[-5],[-5],[-5]])
        self.assertListEqual(test.model.all_parameters['a2'].data.numpy().tolist(), [[-5],[-5],[-5],[-5],[-5]])
        self.assertListEqual(test.model.all_parameters['b2'].data.numpy().tolist(), [[-5],[-5],[-5],[-5],[-5]])
        self.assertListEqual(test.model.all_parameters['a3'].data.numpy().tolist(), [[-5],[-5],[-5],[-5],[-5]])
        self.assertListEqual(test.model.all_parameters['b3'].data.numpy().tolist(), [[-5],[-5],[-5],[-5],[-5]])

        ## Train both models but set the minimize gain of error1 to zero and the minimize gain of error2 to double
        test.neuralizeModel(0.01, clear_model=True)
        test.trainModel(optimizer='SGD', models=['model1','model2'], splits=[100,0,0], lr=1, num_of_epochs=1, minimize_gain={'error11':-1,'error22':2})
        self.assertListEqual(test.model.all_parameters['a1'].data.numpy().tolist(), [[7],[7],[7],[7],[7]])
        self.assertListEqual(test.model.all_parameters['b1'].data.numpy().tolist(), [[-5],[-5],[-5],[-5],[-5]])
        self.assertListEqual(test.model.all_parameters['a2'].data.numpy().tolist(), [[-5],[-5],[-5],[-5],[-5]])
        self.assertListEqual(test.model.all_parameters['b2'].data.numpy().tolist(), [[-11],[-11],[-11],[-11],[-11]])
        self.assertListEqual(test.model.all_parameters['a3'].data.numpy().tolist(), [[-5],[-5],[-5],[-5],[-5]])
        self.assertListEqual(test.model.all_parameters['b3'].data.numpy().tolist(), [[-5],[-5],[-5],[-5],[-5]])

if __name__ == '__main__':
    unittest.main()