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

# 3 Tests
# Test the value of the weight after the training

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

    # def test_multimodel_with_loss_gain_and_lr_gain(self):
    #     ## Model1
    #     input1 = Input('in1')
    #     a = Parameter('a', dimensions=1, tw=0.05, values=[[1],[1],[1],[1],[1]])
    #     output1 = Output('out1', Fir(parameter=a)(input1.tw(0.05)))
    #
    #     test = Neu4mes(visualizer=None, seed=42)
    #     test.addModel('model1', output1)
    #     test.addMinimize('error1', input1.next(), output1)
    #
    #     ## Model2
    #     input2 = Input('in2')
    #     b = Parameter('b', dimensions=1, tw=0.05, values=[[1],[1],[1],[1],[1]])
    #     output2 = Output('out2', Fir(parameter=b)(input2.tw(0.05)))
    #
    #     test.addModel('model2', output2)
    #     test.addMinimize('error2', input2.next(), output2)
    #     test.neuralizeModel(0.01)
    #
    #     data_in1 = np.linspace(0,5,6)
    #     data_in2 = np.linspace(10,15,6)
    #     data_out1 = 2
    #     data_out2 = -3
    #     dataset = {'in1': data_in1, 'in2': data_in2, 'out1': data_in1*data_out1, 'out2': data_in2*data_out2}
    #
    #     test.loadData(name='dataset', source=dataset)
    #
    #     ## Train only model1
    #     self.assertListEqual(test.model.all_parameters['a'].data.numpy().tolist(), [[1],[1],[1],[1],[1]])
    #     self.assertListEqual(test.model.all_parameters['b'].data.numpy().tolist(), [[1],[1],[1],[1],[1]])
    #     test.trainModel(models='model1', splits=[100,0,0])
    #     self.assertListEqual(test.model.all_parameters['a'].data.numpy().tolist(), [[0.20872743427753448],[0.20891857147216797],[0.20914430916309357],[0.20934967696666718],[0.20958690345287323]])
    #     self.assertListEqual(test.model.all_parameters['b'].data.numpy().tolist(), [[1],[1],[1],[1],[1]])
    #
    #     test.neuralizeModel(0.01, clear_model=True)
    #     ## Train only model2
    #     self.assertListEqual(test.model.all_parameters['a'].data.numpy().tolist(), [[1],[1],[1],[1],[1]])
    #     self.assertListEqual(test.model.all_parameters['b'].data.numpy().tolist(), [[1],[1],[1],[1],[1]])
    #     test.trainModel(models='model2', splits=[100,0,0])
    #     self.assertListEqual(test.model.all_parameters['a'].data.numpy().tolist(), [[1],[1],[1],[1],[1]])
    #     self.assertListEqual(test.model.all_parameters['b'].data.numpy().tolist(), [[0.21510866284370422], [0.21509192883968353], [0.21507103741168976], [0.21505486965179443], [0.21503786742687225]])
    #
    #     test.neuralizeModel(0.01, clear_model=True)
    #     ## Train both models
    #     self.assertListEqual(test.model.all_parameters['a'].data.numpy().tolist(), [[1],[1],[1],[1],[1]])
    #     self.assertListEqual(test.model.all_parameters['b'].data.numpy().tolist(), [[1],[1],[1],[1],[1]])
    #     test.trainModel(models=['model1','model2'], splits=[100,0,0])
    #     self.assertListEqual(test.model.all_parameters['a'].data.numpy().tolist(), [[0.2097606211900711],[0.20982888340950012],[0.20994682610034943],[0.21001523733139038],[0.21013548970222473]])
    #     self.assertListEqual(test.model.all_parameters['b'].data.numpy().tolist(), [[0.21503083407878876],[0.2150345891714096],[0.21503330767154694],[0.21502918004989624],[0.21503430604934692]])
    #
    #     test.neuralizeModel(0.01, clear_model=True)
    #     ## Train both models but set the gain of a to zero and the gain of b to double
    #     self.assertListEqual(test.model.all_parameters['a'].data.numpy().tolist(), [[1],[1],[1],[1],[1]])
    #     self.assertListEqual(test.model.all_parameters['b'].data.numpy().tolist(), [[1],[1],[1],[1],[1]])
    #     test.trainModel(models=['model1','model2'], splits=[100,0,0], lr_param={'a':0, 'b':2})
    #     self.assertListEqual(test.model.all_parameters['a'].data.numpy().tolist(), [[1],[1],[1],[1],[1]])
    #     self.assertListEqual(test.model.all_parameters['b'].data.numpy().tolist(), [[0.21878866851329803],[0.21873562037944794],[0.2186843752861023],[0.2186216115951538],[0.21856670081615448]])
    #
    #     test.neuralizeModel(0.01, clear_model=True)
    #     ## Train both models but set the minimize gain of error1 to zero and the minimize gain of error2 to double
    #     self.assertListEqual(test.model.all_parameters['a'].data.numpy().tolist(), [[1],[1],[1],[1],[1]])
    #     self.assertListEqual(test.model.all_parameters['b'].data.numpy().tolist(), [[1],[1],[1],[1],[1]])
    #     train_loss, _, _ = test.trainModel(models=['model1','model2'], splits=[100,0,0], minimize_gain={'error1':0, 'error2':2})
    #     self.assertListEqual(train_loss['error1'], [0.0 for i in range(test.num_of_epochs)])
    #     self.assertListEqual(test.model.all_parameters['a'].data.numpy().tolist(), [[1],[1],[1],[1],[1]])
    #     self.assertListEqual(train_loss['error2'], [3894.033935546875, 1339.185546875, 182.2778778076172, 68.74202728271484, 359.0768127441406, 499.3464050292969, 368.579345703125, 146.50833129882812, 16.411840438842773, 18.409034729003906,
    #                                                 70.48310089111328, 85.69669342041016, 52.13644790649414, 12.842440605163574, 1.1463165283203125, 10.94512939453125, 18.03199005126953, 12.468052864074707, 3.2584829330444336, 0.3026407063007355])
    #     self.assertListEqual(test.model.all_parameters['b'].data.numpy().tolist(), [[0.21512822806835175],[0.21512599289417267],[0.2151111513376236],[0.21510501205921173],[0.21509882807731628]])
    #
    # def test_multimodel_with_connect(self):
    #     ## Model1
    #     input1 = Input('in1')
    #     a = Parameter('a', dimensions=1, tw=0.05, values=[[1],[1],[1],[1],[1]])
    #     output1 = Output('out1', Fir(parameter=a)(input1.tw(0.05)))
    #
    #     test = Neu4mes(visualizer=None, seed=42)
    #     test.addModel('model1', output1)
    #     test.addMinimize('error1', input1.next(), output1)
    #     test.neuralizeModel(0.01)
    #
    #     ## Model2
    #     input2 = Input('in2')
    #     input3 = Input('in3')
    #     b = Parameter('b', dimensions=1, tw=0.05, values=[[1],[1],[1],[1],[1]])
    #     c = Parameter('c', dimensions=1, tw=0.03, values=[[1],[1],[1]])
    #     output2 = Output('out2', Fir(parameter=b)(input2.tw(0.05))+Fir(parameter=c)(input3.tw(0.03)))
    #
    #     test.addModel('model2', output2)
    #     test.addMinimize('error2', input2.next(), output2)
    #     test.neuralizeModel(0.01)
    #
    #     data_struct = ['x','F','x2','y2','','A1x','A1y','B1x','B1y','','A2x','A2y','B2x','out','','x3','in1','in2','time']
    #     test.loadData(name='dataset', source=data_folder, format=data_struct, skiplines=4, delimiter='\t', header=None)
    #
    #     params = {'num_of_epochs': 1,
    #       'train_batch_size': 3,
    #       'val_batch_size': 1,
    #       'test_batch_size':1,
    #       'lr':0.1}
    #
    #     test.trainModel(splits=[100,0,0], training_params=params, lr_param={'a':0, 'b':0, 'c':0}, prediction_samples=3, connect={'in3':'out1'}, shuffle_data=False)
    
    
if __name__ == '__main__':
    unittest.main()