import unittest
import torch

import sys
import os
# append a new directory to sys.path
sys.path.append(os.getcwd())
from neu4mes import *
data_folder = os.path.join(os.path.dirname(__file__), 'data/')

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
        training_params['learning_rate'] = 0.1
        training_params['num_of_epochs'] = 5
        test.trainModel(splits=[70,20,10],training_params = training_params)

        # 15 lines in the dataset
        # 5 lines for input + 1 for output -> total of sample 10
        # 10 / 1 * 0.7 = 7 for training
        # 10 / 1 * 0.2 = 2 for validation
        # 10 / 1 * 0.1 = 1 for test
        self.assertEqual(7,test.n_samples_train)
        self.assertEqual(2,test.n_samples_val)
        self.assertEqual(1,test.n_samples_test)
        self.assertEqual(10,test.num_of_samples['dataset'])
        self.assertEqual(1,test.train_batch_size)
        self.assertEqual(1,test.val_batch_size)
        self.assertEqual(1,test.test_batch_size)
        self.assertEqual(5,test.num_of_epochs)
        self.assertEqual(0.1,test.learning_rate)
    
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
        training_params['learning_rate'] = 0.1
        training_params['num_of_epochs'] = 5
        test.trainModel(splits=[50,0,50],training_params = training_params)

        # 15 lines in the dataset
        # 5 lines for input + 1 for output -> total of sample 10 
        # batch_size > 5 use batch_size = 1
        # 10 / 1 * 0.5 = 5 for training
        # 10 / 1 * 0.0 = 0 for validation
        # 10 / 1 * 0.5 = 5 for test
        self.assertEqual(5,test.n_samples_train)
        self.assertEqual(0,test.n_samples_val)
        self.assertEqual(5,test.n_samples_test)
        self.assertEqual(1,test.train_batch_size)
        self.assertEqual(1,test.val_batch_size)
        self.assertEqual(1,test.test_batch_size)
        self.assertEqual(5,test.num_of_epochs)
        self.assertEqual(0.1,test.learning_rate)
    
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
        training_params['learning_rate'] = 0.1
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
        self.assertEqual(4,test.n_samples_train)
        self.assertEqual(3,test.n_samples_val)
        self.assertEqual(3,test.n_samples_test)
        self.assertEqual(10,test.num_of_samples['dataset'])
        self.assertEqual(2,test.train_batch_size)
        self.assertEqual(2,test.val_batch_size)
        self.assertEqual(2,test.test_batch_size)
        self.assertEqual(5,test.num_of_epochs)
        self.assertEqual(0.1,test.learning_rate)
    
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
        training_params['learning_rate'] = 0.1
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
        self.assertEqual(8,test.n_samples_train)
        self.assertEqual(1,test.n_samples_val)
        self.assertEqual(1,test.n_samples_test)
        self.assertEqual(10,test.num_of_samples['dataset'])
        self.assertEqual(2,test.train_batch_size)
        self.assertEqual(1,test.val_batch_size)
        self.assertEqual(1,test.test_batch_size)
        self.assertEqual(5,test.num_of_epochs)
        self.assertEqual(0.1,test.learning_rate)
    
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
        training_params['learning_rate'] = 0.1
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
        self.assertEqual(12,test.n_samples_train)
        self.assertEqual(3,test.n_samples_val)
        self.assertEqual(0,test.n_samples_test)
        self.assertEqual(15,test.num_of_samples['dataset'])
        self.assertEqual(2,test.train_batch_size)
        self.assertEqual(2,test.val_batch_size)
        self.assertEqual(1,test.test_batch_size)
        self.assertEqual(5,test.num_of_epochs)
        self.assertEqual(0.1,test.learning_rate)
    
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
        training_params['learning_rate'] = 0.1
        training_params['num_of_epochs'] = 5
        test.trainModel(train_dataset='train_dataset', validation_dataset='validation_dataset', test_dataset='test_dataset', training_params=training_params)

        self.assertEqual(9,test.num_of_samples['train_dataset'])
        self.assertEqual(9,test.num_of_samples['validation_dataset'])
        self.assertEqual(9,test.num_of_samples['test_dataset'])
        self.assertEqual(3,test.train_batch_size)
        self.assertEqual(2,test.val_batch_size)
        self.assertEqual(1,test.test_batch_size)
        self.assertEqual(9,test.n_samples_train)
        self.assertEqual(9,test.n_samples_val)
        self.assertEqual(9,test.n_samples_test)
        self.assertEqual(5,test.num_of_epochs)
        self.assertEqual(0.1,test.learning_rate)
    
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
        training_params['learning_rate'] = 0.01
        training_params['num_of_epochs'] = 7
        test.trainModel(train_dataset='dataset', splits=[80,10,10],  training_params=training_params)

        self.assertEqual(22, test.num_of_samples['dataset'])
        self.assertEqual(1, test.train_batch_size)
        self.assertEqual(1, test.val_batch_size)
        self.assertEqual(1, test.test_batch_size)
        self.assertEqual(18, test.n_samples_train)
        self.assertEqual(2, test.n_samples_val)
        self.assertEqual(2, test.n_samples_test)
        self.assertEqual(7, test.num_of_epochs)
        self.assertEqual(0.01, test.learning_rate)

        training_params = {}
        training_params['train_batch_size'] = 6
        training_params['val_batch_size'] = 2
        training_params['test_batch_size'] = 2
        test.trainModel(train_dataset='dataset', splits=[80,10,10],  training_params=training_params)

        self.assertEqual(22, test.num_of_samples['dataset'])
        self.assertEqual(6, test.train_batch_size)
        self.assertEqual(2, test.val_batch_size)
        self.assertEqual(2, test.test_batch_size)
        self.assertEqual(18, test.n_samples_train)
        self.assertEqual(2, test.n_samples_val)
        self.assertEqual(2, test.n_samples_test)
    
    def test_multimodel_with_loss_gain_and_lr_gain(self):
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
        b = Parameter('b', dimensions=1, tw=0.05, values=[[1],[1],[1],[1],[1]])
        output2 = Output('out2', Fir(parameter=b)(input2.tw(0.05)))

        test.addModel('model2', output2)
        test.addMinimize('error2', input2.next(), output2)
        test.neuralizeModel(0.01)

        data_struct = ['x','F','x2','y2','','A1x','A1y','B1x','B1y','','A2x','A2y','B2x','out','','x3','in1','in2','time']
        test.loadData(name='dataset', source=data_folder, format=data_struct, skiplines=4, delimiter='\t', header=None)

        params = {'num_of_epochs': 20, 
          'train_batch_size': 3, 
          'val_batch_size': 1, 
          'test_batch_size':1, 
          'learning_rate':0.1}
        
        ## Train only model1
        self.assertListEqual(test.model.all_parameters['a'].data.numpy().tolist(), [[1],[1],[1],[1],[1]])
        self.assertListEqual(test.model.all_parameters['b'].data.numpy().tolist(), [[1],[1],[1],[1],[1]])
        test.trainModel(models='model1', splits=[100,0,0], training_params=params)
        self.assertListEqual(test.model.all_parameters['a'].data.numpy().tolist(), [[0.20872743427753448],[0.20891857147216797],[0.20914430916309357],[0.20934967696666718],[0.20958690345287323]])
        self.assertListEqual(test.model.all_parameters['b'].data.numpy().tolist(), [[1],[1],[1],[1],[1]])

        test.neuralizeModel(0.01)
        ## Train only model2
        self.assertListEqual(test.model.all_parameters['a'].data.numpy().tolist(), [[1],[1],[1],[1],[1]])
        self.assertListEqual(test.model.all_parameters['b'].data.numpy().tolist(), [[1],[1],[1],[1],[1]])
        test.trainModel(models='model2', splits=[100,0,0], training_params=params)
        self.assertListEqual(test.model.all_parameters['a'].data.numpy().tolist(), [[1],[1],[1],[1],[1]])
        self.assertListEqual(test.model.all_parameters['b'].data.numpy().tolist(), [[0.21510866284370422], [0.21509192883968353], [0.21507103741168976], [0.21505486965179443], [0.21503786742687225]])
        
        test.neuralizeModel(0.01)
        ## Train both models
        self.assertListEqual(test.model.all_parameters['a'].data.numpy().tolist(), [[1],[1],[1],[1],[1]])
        self.assertListEqual(test.model.all_parameters['b'].data.numpy().tolist(), [[1],[1],[1],[1],[1]])
        test.trainModel(models=['model1','model2'], splits=[100,0,0], training_params=params)
        self.assertListEqual(test.model.all_parameters['a'].data.numpy().tolist(), [[0.2097606211900711],[0.20982888340950012],[0.20994682610034943],[0.21001523733139038],[0.21013548970222473]])
        self.assertListEqual(test.model.all_parameters['b'].data.numpy().tolist(), [[0.21503083407878876],[0.2150345891714096],[0.21503330767154694],[0.21502918004989624],[0.21503430604934692]])

        test.neuralizeModel(0.01)
        ## Train both models but set the gain of a to zero and the gain of b to double
        self.assertListEqual(test.model.all_parameters['a'].data.numpy().tolist(), [[1],[1],[1],[1],[1]])
        self.assertListEqual(test.model.all_parameters['b'].data.numpy().tolist(), [[1],[1],[1],[1],[1]])
        test.trainModel(models=['model1','model2'], splits=[100,0,0], training_params=params, lr_gain={'a':0, 'b':2})
        self.assertListEqual(test.model.all_parameters['a'].data.numpy().tolist(), [[1],[1],[1],[1],[1]])
        self.assertListEqual(test.model.all_parameters['b'].data.numpy().tolist(), [[0.21878866851329803],[0.21873562037944794],[0.2186843752861023],[0.2186216115951538],[0.21856670081615448]])

        test.neuralizeModel(0.01)
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
          'learning_rate':0.1}
        
        test.trainModel(splits=[100,0,0], training_params=params, lr_gain={'a':0, 'b':0, 'c':0}, prediction_samples=3, connect={'in3':'out1'}, shuffle_data=False)
    
    
if __name__ == '__main__':
    unittest.main()