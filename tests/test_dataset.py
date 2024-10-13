import logging
import sys
import os
# append a new directory to sys.path
sys.path.append(os.getcwd())

import unittest
import numpy as np
from neu4mes import *
from neu4mes import relation
relation.CHECK_NAMES = False


log = logging.getLogger(__name__)
log.setLevel(logging.CRITICAL)

# 16 Tests
# This file test the data loading in particular:
# The shape and the value of the inputs

import os
train_folder = os.path.join(os.path.dirname(__file__), 'data/')
val_folder = os.path.join(os.path.dirname(__file__), 'val_data/')
test_folder = os.path.join(os.path.dirname(__file__), 'test_data/')


class Neu4mesCreateDatasetTest(unittest.TestCase):
    
    def test_build_dataset_simple(self):
        input = Input('in')
        output = Input('out')
        relation = Fir(input.tw(0.05))

        test = Neu4mes(visualizer=None)
        test.addMinimize('out', output.z(-1), relation)
        test.neuralizeModel(0.01)

        data_struct = ['x1','y1','x2','y2','','A1x','A1y','B1x','B1y','','A2x','A2y','B2x','out','','x3','in','theta','time']
        test.loadData(name='dataset_1', source=train_folder, format=data_struct, skiplines=4, delimiter='\t', header=None)
        self.assertEqual((10,5,1),test.data['dataset_1']['in'].shape)
        self.assertEqual([[0.984],[0.983],[0.982],[0.98],[0.977]] , test.data['dataset_1']['in'][0].tolist())

        self.assertEqual((10,1,1),test.data['dataset_1']['out'].shape)
        self.assertEqual([[1.225]], test.data['dataset_1']['out'][0].tolist())
        self.assertEqual([[[1.225]], [[1.224]], [[1.222]], [[1.22]], [[1.217]], [[1.214]], [[1.211]], [[1.207]], [[1.204]], [[1.2]]], test.data['dataset_1']['out'].tolist())

    def test_build_multi_dataset_simple(self):
        input = Input('in')
        output = Input('out')
        relation = Fir(input.tw(0.05))

        test = Neu4mes(visualizer=None)
        test.addMinimize('out', output.z(-1), relation)
        test.neuralizeModel(0.01)

        data_struct = ['x1','y1','x2','y2','','A1x','A1y','B1x','B1y','','A2x','A2y','B2x','out','','x3','in','theta','time']

        test.loadData(name='train_dataset', source=train_folder, format=data_struct, skiplines=4, delimiter='\t', header=None)
        test.loadData(name='validation_dataset', source=val_folder, format=data_struct, skiplines=4, delimiter='\t', header=None)
        test.loadData(name='test_dataset', source=test_folder, format=data_struct, skiplines=4, delimiter='\t', header=None)

        self.assertEqual(3, test.n_datasets)

        self.assertEqual((10,5,1),test.data['train_dataset']['in'].shape)
        self.assertEqual([[0.984],[0.983],[0.982],[0.98],[0.977]] , test.data['train_dataset']['in'][0].tolist())
        self.assertEqual((6,5,1),test.data['validation_dataset']['in'].shape)
        self.assertEqual([[0.884],[0.883],[0.882],[0.88],[0.877]] , test.data['validation_dataset']['in'][0].tolist())
        self.assertEqual((8,5,1),test.data['test_dataset']['in'].shape)
        self.assertEqual([[0.784],[0.783],[0.782],[0.78],[0.777]] , test.data['test_dataset']['in'][0].tolist())

        self.assertEqual((10,1,1),test.data['train_dataset']['out'].shape)
        self.assertEqual([[1.225]], test.data['train_dataset']['out'][0].tolist())
        self.assertEqual([[[1.225]], [[1.224]], [[1.222]], [[1.22]], [[1.217]], [[1.214]], [[1.211]], [[1.207]], [[1.204]], [[1.2]]], test.data['train_dataset']['out'].tolist())
        self.assertEqual((6,1,1),test.data['validation_dataset']['out'].shape)
        self.assertEqual([[2.225]], test.data['validation_dataset']['out'][0].tolist())
        self.assertEqual([[[2.225]], [[2.224]], [[2.222]], [[2.22]], [[2.217]], [[2.214]]], test.data['validation_dataset']['out'].tolist())
        self.assertEqual((8,1,1),test.data['test_dataset']['out'].shape)
        self.assertEqual([[3.225]], test.data['test_dataset']['out'][0].tolist())
        self.assertEqual([[[3.225]], [[3.224]], [[3.222]], [[3.22]], [[3.217]], [[3.214]], [[3.211]], [[3.207]]], test.data['test_dataset']['out'].tolist())
    
    def test_build_dataset_medium1(self):
        input = Input('in')
        output = Input('out')
        rel1 = Fir(input.tw(0.05))
        rel2 = Fir(input.tw(0.01))

        test = Neu4mes(visualizer=None)
        test.addMinimize('out', output.z(-1), rel1 + rel2)
        test.neuralizeModel(0.01)

        data_struct = ['x1','y1','x2','y2','','A1x','A1y','B1x','B1y','','A2x','A2y','B2x','out','','x3','in','theta','time']
        test.loadData(name='dataset', source=train_folder, format=data_struct, skiplines=4, delimiter='\t', header=None)
        self.assertEqual((10,5,1),test.data['dataset']['in'].shape)
        self.assertEqual([[0.984],[0.983],[0.982],[0.98],[0.977]],test.data['dataset']['in'][0].tolist())

        self.assertEqual((10,1,1),test.data['dataset']['out'].shape)
        self.assertEqual([[[1.225]], [[1.224]], [[1.222]], [[1.22]], [[1.217]], [[1.214]], [[1.211]], [[1.207]], [[1.204]], [[1.2]]],test.data['dataset']['out'].tolist())
    
    def test_build_multi_dataset_medium1(self):
        input = Input('in')
        output = Input('out')
        rel1 = Fir(input.tw(0.05))
        rel2 = Fir(input.tw(0.01))

        test = Neu4mes(visualizer=None)
        test.addMinimize('out', output.z(-1), rel1 + rel2)
        test.neuralizeModel(0.01)

        data_struct = ['x1','y1','x2','y2','','A1x','A1y','B1x','B1y','','A2x','A2y','B2x','out','','x3','in','theta','time']

        test.loadData(name='train_dataset', source=train_folder, format=data_struct, skiplines=4, delimiter='\t', header=None)
        test.loadData(name='validation_dataset', source=val_folder, format=data_struct, skiplines=4, delimiter='\t', header=None)
        test.loadData(name='test_dataset', source=test_folder, format=data_struct, skiplines=4, delimiter='\t', header=None)

        self.assertEqual(3, test.n_datasets)

        self.assertEqual((10,5,1),test.data['train_dataset']['in'].shape)
        self.assertEqual([[0.984],[0.983],[0.982],[0.98],[0.977]],test.data['train_dataset']['in'][0].tolist())
        self.assertEqual((6,5,1),test.data['validation_dataset']['in'].shape)
        self.assertEqual([[0.884],[0.883],[0.882],[0.88],[0.877]],test.data['validation_dataset']['in'][0].tolist())
        self.assertEqual((8,5,1),test.data['test_dataset']['in'].shape)
        self.assertEqual([[0.784],[0.783],[0.782],[0.78],[0.777]],test.data['test_dataset']['in'][0].tolist())

        self.assertEqual((10,1,1),test.data['train_dataset']['out'].shape)
        self.assertEqual([[[1.225]], [[1.224]], [[1.222]], [[1.22]], [[1.217]], [[1.214]], [[1.211]], [[1.207]], [[1.204]], [[1.2]]],test.data['train_dataset']['out'].tolist())
        self.assertEqual((6,1,1),test.data['validation_dataset']['out'].shape)
        self.assertEqual([[[2.225]], [[2.224]], [[2.222]], [[2.22]], [[2.217]], [[2.214]]],test.data['validation_dataset']['out'].tolist())
        self.assertEqual((8,1,1),test.data['test_dataset']['out'].shape)
        self.assertEqual([[[3.225]], [[3.224]], [[3.222]], [[3.22]], [[3.217]], [[3.214]], [[3.211]], [[3.207]]],test.data['test_dataset']['out'].tolist())
    
    def test_build_dataset_medium2(self):
        input1 = Input('in1')
        input2 = Input('in2')
        output = Input('out')
        rel1 = Fir(input1.tw(0.05))
        rel2 = Fir(input1.tw(0.01))
        rel3 = Fir(input2.tw(0.02))

        test = Neu4mes(visualizer=None)
        test.addMinimize('out', output.z(-1), rel1 + rel2 + rel3)
        test.neuralizeModel(0.01)

        data_struct = ['x1','y1','x2','y2','','A1x','A1y','B1x','B1y','','A2x','A2y','B2x','out','','x3','in1','in2','time']
        test.loadData(name='dataset', source=train_folder, format=data_struct, skiplines=4, delimiter='\t', header=None)
        self.assertEqual((10,5,1),test.data['dataset']['in1'].shape)
        self.assertEqual([[0.984],[0.983],[0.982],[0.98],[0.977]],test.data['dataset']['in1'][0].tolist())

        self.assertEqual((10,2,1),test.data['dataset']['in2'].shape)
        self.assertEqual([[12.498], [12.502]],test.data['dataset']['in2'][0].tolist())

        self.assertEqual((10,1,1),test.data['dataset']['out'].shape)
        self.assertEqual([[[1.225]], [[1.224]], [[1.222]], [[1.22]], [[1.217]], [[1.214]], [[1.211]], [[1.207]], [[1.204]], [[1.2]]],test.data['dataset']['out'].tolist())
    
    def test_build_dataset_complex1(self):
        input1 = Input('in1')
        output = Input('out')
        rel1 = Fir(input1.tw(0.05))
        rel2 = Fir(input1.tw([-0.01,0.02]))

        test = Neu4mes(visualizer=None)
        test.addMinimize('out', output.z(-1), rel1 + rel2)
        test.neuralizeModel(0.01)

        data_struct = ['x1','y1','x2','y2','','A1x','A1y','B1x','B1y','','A2x','A2y','B2x','out','','x3','in1','in2','time']
        test.loadData(name='dataset', source=train_folder, format=data_struct, skiplines=4, delimiter='\t', header=None)
        self.assertEqual((9,7,1),test.data['dataset']['in1'].shape)
        self.assertEqual([[0.984],[0.983],[0.982],[0.98],[0.977],[0.973],[0.969]],test.data['dataset']['in1'][0].tolist())

        self.assertEqual((9,1,1),test.data['dataset']['out'].shape)
        self.assertEqual([[[1.225]], [[1.224]], [[1.222]], [[1.22]], [[1.217]], [[1.214]], [[1.211]], [[1.207]], [[1.204]]],test.data['dataset']['out'].tolist())

    def test_build_multi_dataset_complex1(self):
        input1 = Input('in1')
        output = Input('out')
        rel1 = Fir(input1.tw(0.05))
        rel2 = Fir(input1.tw([-0.01,0.02]))

        test = Neu4mes(visualizer=None)
        test.addMinimize('out', output.z(-1), rel1 + rel2)
        test.neuralizeModel(0.01)

        data_struct = ['x1','y1','x2','y2','','A1x','A1y','B1x','B1y','','A2x','A2y','B2x','out','','x3','in1','in2','time']

        test.loadData(name='train_dataset', source=train_folder, format=data_struct, skiplines=4, delimiter='\t', header=None)
        test.loadData(name='validation_dataset', source=val_folder, format=data_struct, skiplines=4, delimiter='\t', header=None)
        test.loadData(name='test_dataset', source=test_folder, format=data_struct, skiplines=4, delimiter='\t', header=None)

        self.assertEqual(3, test.n_datasets)

        self.assertEqual((9,7,1),test.data['train_dataset']['in1'].shape)
        self.assertEqual([[0.984],[0.983],[0.982],[0.98],[0.977],[0.973],[0.969]],test.data['train_dataset']['in1'][0].tolist())
        self.assertEqual((5,7,1),test.data['validation_dataset']['in1'].shape)
        self.assertEqual([[0.884],[0.883],[0.882],[0.88],[0.877],[0.873],[0.869]],test.data['validation_dataset']['in1'][0].tolist())
        self.assertEqual((7,7,1),test.data['test_dataset']['in1'].shape)
        self.assertEqual([[0.784],[0.783],[0.782],[0.78],[0.777],[0.773],[0.769]],test.data['test_dataset']['in1'][0].tolist())

        self.assertEqual((9,1,1),test.data['train_dataset']['out'].shape)
        self.assertEqual([[[1.225]], [[1.224]], [[1.222]], [[1.22]], [[1.217]], [[1.214]], [[1.211]], [[1.207]], [[1.204]]],test.data['train_dataset']['out'].tolist())
        self.assertEqual((5,1,1),test.data['validation_dataset']['out'].shape)
        self.assertEqual([[[2.225]], [[2.224]], [[2.222]], [[2.22]], [[2.217]]],test.data['validation_dataset']['out'].tolist())
        self.assertEqual((7,1,1),test.data['test_dataset']['out'].shape)
        self.assertEqual([[[3.225]], [[3.224]], [[3.222]], [[3.22]], [[3.217]], [[3.214]], [[3.211]]],test.data['test_dataset']['out'].tolist())
    
    def test_build_dataset_complex2(self):
        input1 = Input('in1')
        output = Input('out')
        rel1 = Fir(input1.tw(0.05))
        rel2 = Fir(input1.tw([-0.01,0.01]))

        test = Neu4mes(visualizer=None)
        test.addMinimize('out', output.z(-1), rel1 + rel2)
        test.neuralizeModel(0.01)

        data_struct = ['x1','y1','x2','y2','','A1x','A1y','B1x','B1y','','A2x','A2y','B2x','out','','x3','in1','in2','time']
        test.loadData(name='dataset', source=train_folder, format=data_struct, skiplines=4, delimiter='\t', header=None)
        self.assertEqual((10,6,1),test.data['dataset']['in1'].shape)
        self.assertEqual([[0.984],[0.983],[0.982],[0.98],[0.977],[0.973]],test.data['dataset']['in1'][0].tolist())

        self.assertEqual((10,1,1),test.data['dataset']['out'].shape)
        self.assertEqual([[[1.225]], [[1.224]], [[1.222]], [[1.22]], [[1.217]], [[1.214]], [[1.211]], [[1.207]], [[1.204]], [[1.2]]],test.data['dataset']['out'].tolist())
    
    def test_build_dataset_complex3(self):
        input1 = Input('in1')
        input2 = Input('in2')
        output = Input('out')
        rel1 = Fir(input1.tw(0.05))
        rel2 = Fir(input2.tw(0.02))
        rel3 = Fir(input1.tw([-0.01,0.01]))
        rel4 = Fir(input2.last())
        fun = Output('out',rel1+rel2+rel3+rel4)

        test = Neu4mes(visualizer=None)
        test.addModel('fun', fun)
        test.addMinimize('out', output.z(-1), rel1 + rel2 + rel3 + rel4)
        test.neuralizeModel(0.01)

        data_struct = ['x1','y1','x2','y2','','A1x','A1y','B1x','B1y','','A2x','A2y','B2x','out','','x3','in1','in2','time']
        test.loadData(name='dataset', source=train_folder, format=data_struct, skiplines=4, delimiter='\t', header=None)
        self.assertEqual((10,6,1),test.data['dataset']['in1'].shape)
        self.assertEqual([[0.984],[0.983],[0.982],[0.98],[0.977],[0.973]],test.data['dataset']['in1'][0].tolist())
        
        self.assertEqual((10,2,1),test.data['dataset']['in2'].shape)
        self.assertEqual([[12.498], [12.502]],test.data['dataset']['in2'][0].tolist())

        self.assertEqual((10,1,1),test.data['dataset']['out'].shape)
        self.assertEqual([[[1.225]], [[1.224]], [[1.222]], [[1.22]], [[1.217]], [[1.214]], [[1.211]], [[1.207]], [[1.204]], [[1.2]]],test.data['dataset']['out'].tolist())

    def test_build_multi_dataset_complex3(self):
        input1 = Input('in1')
        input2 = Input('in2')
        output = Input('out')
        rel1 = Fir(input1.tw(0.05))
        rel2 = Fir(input2.tw(0.02))
        rel3 = Fir(input1.tw([-0.01,0.01]))
        rel4 = Fir(input2.last())
        fun = Output('out',rel1+rel2+rel3+rel4)

        test = Neu4mes(visualizer=None)
        test.addModel('fun',fun)
        test.addMinimize('out', output.z(-1), rel1 + rel2 + rel3 + rel4)
        test.neuralizeModel(0.01)

        data_struct = ['x1','y1','x2','y2','','A1x','A1y','B1x','B1y','','A2x','A2y','B2x','out','','x3','in1','in2','time']
        test.loadData(name='train_dataset', source=train_folder, format=data_struct, skiplines=4, delimiter='\t', header=None)
        test.loadData(name='validation_dataset', source=val_folder, format=data_struct, skiplines=4, delimiter='\t', header=None)
        test.loadData(name='test_dataset', source=test_folder, format=data_struct, skiplines=4, delimiter='\t', header=None)

        self.assertEqual(3, test.n_datasets)

        self.assertEqual((10,6,1),test.data['train_dataset']['in1'].shape)
        self.assertEqual([[0.984],[0.983],[0.982],[0.98],[0.977],[0.973]],test.data['train_dataset']['in1'][0].tolist())
        self.assertEqual((6,6,1),test.data['validation_dataset']['in1'].shape)
        self.assertEqual([[0.884],[0.883],[0.882],[0.88],[0.877],[0.873]],test.data['validation_dataset']['in1'][0].tolist())
        self.assertEqual((8,6,1),test.data['test_dataset']['in1'].shape)
        self.assertEqual([[0.784],[0.783],[0.782],[0.78],[0.777],[0.773]],test.data['test_dataset']['in1'][0].tolist())
        
        self.assertEqual((10,2,1),test.data['train_dataset']['in2'].shape)
        self.assertEqual([[12.498], [12.502]],test.data['train_dataset']['in2'][0].tolist())
        self.assertEqual((6,2,1),test.data['validation_dataset']['in2'].shape)
        self.assertEqual([[12.498], [12.502]],test.data['validation_dataset']['in2'][0].tolist())
        self.assertEqual((8,2,1),test.data['test_dataset']['in2'].shape)
        self.assertEqual([[12.498], [12.502]],test.data['test_dataset']['in2'][0].tolist())

        self.assertEqual((10,1,1),test.data['train_dataset']['out'].shape)
        self.assertEqual([[[1.225]], [[1.224]], [[1.222]], [[1.22]], [[1.217]], [[1.214]], [[1.211]], [[1.207]], [[1.204]], [[1.2]]],test.data['train_dataset']['out'].tolist())
        self.assertEqual((6,1,1),test.data['validation_dataset']['out'].shape)
        self.assertEqual([[[2.225]], [[2.224]], [[2.222]], [[2.22]], [[2.217]], [[2.214]]],test.data['validation_dataset']['out'].tolist())
        self.assertEqual((8,1,1),test.data['test_dataset']['out'].shape)
        self.assertEqual([[[3.225]], [[3.224]], [[3.222]], [[3.22]], [[3.217]], [[3.214]], [[3.211]], [[3.207]]],test.data['test_dataset']['out'].tolist())
    
    def test_build_dataset_complex5(self):
        input1 = Input('in1')
        output = Input('out')
        rel1 = Fir(input1.tw(0.05))
        rel2 = Fir(input1.tw([-0.01,0.01]))
        rel3 = Fir(input1.tw([-0.02,0.02]))
        fun = Output('out',rel1+rel2+rel3)

        test = Neu4mes(visualizer=None)
        test.addModel('fun', fun)
        test.addMinimize('out', output.z(-1), fun)
        test.neuralizeModel(0.01)

        data_struct = ['x1','y1','x2','y2','','A1x','A1y','B1x','B1y','','A2x','A2y','B2x','out','','x3','in1','in2','time']
        test.loadData(name='dataset', source=train_folder, format=data_struct, skiplines=4, delimiter='\t', header=None)
        self.assertEqual((9,7,1),test.data['dataset']['in1'].shape)
        self.assertEqual([[0.984],[0.983],[0.982],[0.98],[0.977],[0.973],[0.969]],test.data['dataset']['in1'][0].tolist())

        self.assertEqual((9,1,1),test.data['dataset']['out'].shape)
        self.assertEqual([[[1.225]], [[1.224]], [[1.222]], [[1.22]], [[1.217]], [[1.214]], [[1.211]], [[1.207]], [[1.204]]],test.data['dataset']['out'].tolist())
    
    def test_build_dataset_complex6(self):
        input1 = Input('in1')
        output = Input('out')
        rel1 = Fir(input1.tw(0.05))
        rel2 = Fir(input1.tw([-0.01,0.02]))
        rel3 = Fir(input1.tw([-0.05,0.01]))
        fun = Output('out',rel1+rel2+rel3)

        test = Neu4mes(visualizer=None)
        test.addModel('fun',fun)
        test.addMinimize('out', output.z(-1), fun)
        test.neuralizeModel(0.01)

        data_struct = ['x1','y1','x2','y2','','A1x','A1y','B1x','B1y','','A2x','A2y','B2x','out','','x3','in1','in2','time']
        test.loadData(name='dataset', source=train_folder, format=data_struct, skiplines=4, delimiter='\t', header=None)
        self.assertEqual((9,7,1),test.data['dataset']['in1'].shape)
        self.assertEqual([[0.984],[0.983],[0.982],[0.98],[0.977],[0.973],[0.969]],test.data['dataset']['in1'][0].tolist())

        self.assertEqual((9,1,1),test.data['dataset']['out'].shape)
        self.assertEqual([[[1.225]], [[1.224]], [[1.222]], [[1.22]], [[1.217]], [[1.214]], [[1.211]], [[1.207]], [[1.204]]],test.data['dataset']['out'].tolist())

    def test_build_dataset_custom(self):
        input1 = Input('in1')
        output = Input('out')
        rel1 = Fir(input1.tw(0.05))
        rel2 = Fir(input1.tw([-0.01,0.02]))
        rel3 = Fir(input1.tw([-0.05,0.01]))
        fun = Output('out',rel1+rel2+rel3)

        test = Neu4mes(visualizer=None)
        test.addModel('fun',fun)
        test.addMinimize('out', output.z(-1), fun)
        test.neuralizeModel(0.01)

        data_x = np.array(range(10))
        data_a = 2
        data_b = -3
        dataset = {'in1': data_x, 'out': (data_a*data_x) + data_b}
        
        test.loadData(name='dataset',source=dataset)
        self.assertEqual((4,7,1),test.data['dataset']['in1'].shape)
        self.assertEqual([[[0],[1],[2],[3],[4],[5],[6]],
                        [[1],[2],[3],[4],[5],[6],[7]],
                        [[2],[3],[4],[5],[6],[7],[8]],
                        [[3],[4],[5],[6],[7],[8],[9]]],
                        test.data['dataset']['in1'].tolist())

        self.assertEqual((4,1,1),test.data['dataset']['out'].shape)
        self.assertEqual([[[7]],[[9]],[[11]],[[13]]],test.data['dataset']['out'].tolist())

    def test_build_multi_dataset_custom(self):
        input1 = Input('in1')
        output = Input('out')
        rel1 = Fir(input1.tw(0.05))
        rel2 = Fir(input1.tw([-0.01, 0.02]))
        rel3 = Fir(input1.tw([-0.05, 0.01]))
        fun = Output('out', rel1 + rel2 + rel3)

        test = Neu4mes(visualizer=None)
        test.addModel('fun',fun)
        test.addMinimize('out', output.z(-1), fun)
        test.neuralizeModel(0.01)

        train_data_x = np.array(range(10))
        val_data_x = np.array(range(10, 20))
        test_data_x = np.array(range(20, 30))
        data_a = 2
        data_b = -3
        train_dataset = {'in1': train_data_x, 'out': (data_a * train_data_x) + data_b}
        val_dataset = {'in1': val_data_x, 'out': (data_a * val_data_x) + data_b}
        test_dataset = {'in1': test_data_x, 'out': (data_a * test_data_x) + data_b}

        test.loadData(name='train_dataset', source=train_dataset)
        test.loadData(name='val_dataset', source=val_dataset)
        test.loadData(name='test_dataset', source=test_dataset)

        self.assertEqual(3, test.n_datasets)

        self.assertEqual((4, 7, 1), test.data['train_dataset']['in1'].shape)
        self.assertEqual([[[0], [1], [2], [3], [4], [5], [6]],
                          [[1], [2], [3], [4], [5], [6], [7]],
                          [[2], [3], [4], [5], [6], [7], [8]],
                          [[3], [4], [5], [6], [7], [8], [9]]],
                         test.data['train_dataset']['in1'].tolist())
        self.assertEqual((4, 7, 1), test.data['val_dataset']['in1'].shape)
        self.assertEqual([[[10], [11], [12], [13], [14], [15], [16]],
                          [[11], [12], [13], [14], [15], [16], [17]],
                          [[12], [13], [14], [15], [16], [17], [18]],
                          [[13], [14], [15], [16], [17], [18], [19]]],
                         test.data['val_dataset']['in1'].tolist())
        self.assertEqual((4, 7, 1), test.data['test_dataset']['in1'].shape)
        self.assertEqual([[[20], [21], [22], [23], [24], [25], [26]],
                          [[21], [22], [23], [24], [25], [26], [27]],
                          [[22], [23], [24], [25], [26], [27], [28]],
                          [[23], [24], [25], [26], [27], [28], [29]]],
                         test.data['test_dataset']['in1'].tolist())

        self.assertEqual((4, 1, 1), test.data['train_dataset']['out'].shape)
        self.assertEqual([[[7]], [[9]], [[11]], [[13]]], test.data['train_dataset']['out'].tolist())
        self.assertEqual((4, 1, 1), test.data['val_dataset']['out'].shape)
        self.assertEqual([[[27]], [[29]], [[31]], [[33]]], test.data['val_dataset']['out'].tolist())
        self.assertEqual((4, 1, 1), test.data['test_dataset']['out'].shape)
        self.assertEqual([[[47]], [[49]], [[51]], [[53]]], test.data['test_dataset']['out'].tolist())

    def test_vector_input_dataset(self):
        x = Input('x', dimensions=4)
        y = Input('y', dimensions=3)
        k = Input('k', dimensions=2)
        w = Input('w')


        out = Output('out', Fir(Linear(Linear(3)(x.tw(0.02)) + y.tw(0.02))))
        out2 = Output('out2', Fir(Linear(k.last() + Fir(2)(w.tw(0.05,offset=-0.02)))))

        test = Neu4mes(visualizer=None)
        test.addMinimize('out', out, out2)
        test.neuralizeModel(0.01)

        ## Custom dataset
        data_x = np.transpose(np.array(
                 [np.linspace(1,100,100, dtype=np.float32),
                  np.linspace(2, 101, 100, dtype=np.float32),
                  np.linspace(3, 102, 100, dtype=np.float32),
                  np.linspace(4, 103, 100, dtype=np.float32)]))
        data_y = np.transpose(np.array(
                 [np.linspace(1,100,100, dtype=np.float32) + 10,
                  np.linspace(2, 101, 100, dtype=np.float32) + 10,
                  np.linspace(3, 102, 100, dtype=np.float32) + 10]))
        data_k = np.transpose(np.array(
                 [np.linspace(1,100,100, dtype=np.float32) + 20,
                  np.linspace(2, 101, 100, dtype=np.float32) + 20]))
        data_w = np.linspace(1,100,100, dtype=np.float32) + 30
        dataset = {'x': data_x, 'y': data_y, 'w': data_w, 'k': data_k}

        test.loadData(name='dataset', source=dataset)

        self.assertEqual((96, 2, 4),test.data['dataset']['x'].shape)
        self.assertEqual((96, 2, 3),test.data['dataset']['y'].shape)
        self.assertEqual((96, 1, 2),test.data['dataset']['k'].shape)
        self.assertEqual((96, 5, 1),test.data['dataset']['w'].shape)

        self.assertEqual([[4.0, 5.0, 6.0, 7.0], [5.0, 6.0, 7.0, 8.0]],
                        test.data['dataset']['x'][0].tolist())
        self.assertEqual([[5.0, 6.0, 7.0, 8.0],[6.0, 7.0, 8.0, 9.0]],
                        test.data['dataset']['x'][1].tolist())
        self.assertEqual([[99, 100, 101, 102], [100, 101, 102, 103]],
                        test.data['dataset']['x'][-1].tolist())
        self.assertEqual([[98, 99, 100, 101],[99, 100, 101, 102]],
                        test.data['dataset']['x'][-2].tolist())

        self.assertEqual([[14.0, 15.0, 16.0], [15.0, 16.0, 17.0]],
                         test.data['dataset']['y'][0].tolist())
        self.assertEqual([[15.0, 16.0, 17.0], [16.0, 17.0, 18.0]],
                         test.data['dataset']['y'][1].tolist())
        self.assertEqual([[109, 110, 111], [110, 111, 112]],
                         test.data['dataset']['y'][-1].tolist())
        self.assertEqual([[108, 109, 110], [109, 110, 111]],
                         test.data['dataset']['y'][-2].tolist())

        self.assertEqual([[25.0, 26.0]],
                         test.data['dataset']['k'][0].tolist())
        self.assertEqual([[26.0, 27.0]],
                         test.data['dataset']['k'][1].tolist())
        self.assertEqual([[120, 121]],
                         test.data['dataset']['k'][-1].tolist())
        self.assertEqual([[119, 120]],
                         test.data['dataset']['k'][-2].tolist())

        self.assertEqual([[31], [32], [33], [34], [35]],
                         test.data['dataset']['w'][0].tolist())
        self.assertEqual([[32], [33], [34], [35], [36]],
                         test.data['dataset']['w'][1].tolist())
        self.assertEqual([[126], [127], [128], [129], [130]],
                         test.data['dataset']['w'][-1].tolist())
        self.assertEqual([[125], [126], [127], [128], [129]],
                         test.data['dataset']['w'][-2].tolist())

    def test_vector_input_dataset_files(self):
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

        self.assertEqual((22, 2, 4),test.data['dataset']['x'].shape)
        self.assertEqual((22, 2, 3),test.data['dataset']['y'].shape)
        self.assertEqual((22, 1, 2),test.data['dataset']['k'].shape)
        self.assertEqual((22, 5, 1),test.data['dataset']['w'].shape)

        self.assertEqual([[0.804,	0.825,	0.320,	0.488], [0.805,	0.825,	0.322,	0.485]],
                        test.data['dataset']['x'][0].tolist())
        self.assertEqual([[0.805,	0.825,	0.322,	0.485],[0.806,	0.824,	0.325,	0.481]],
                        test.data['dataset']['x'][1].tolist())
        self.assertEqual([[0.806,	0.824,	0.325,	0.481], [0.807,	0.823,	0.329,	0.477]],
                        test.data['dataset']['x'][-1].tolist())
        self.assertEqual([[0.805,	0.825,	0.322,	0.485],[0.806,	0.824,	0.325,	0.481]],
                        test.data['dataset']['x'][-2].tolist())

        self.assertEqual([[0.350,	1.375,	0.586], [0.350,	1.375,	0.585]],
                         test.data['dataset']['y'][0].tolist())
        self.assertEqual([[0.350,	1.375,	0.585], [0.350,	1.375,	0.584]],
                         test.data['dataset']['y'][1].tolist())
        self.assertEqual([[0.350,	1.375,	0.584], [0.350,	1.375,	0.582]],
                         test.data['dataset']['y'][-1].tolist())
        self.assertEqual([[0.350,	1.375,	0.585], [0.350,	1.375,	0.584]],
                         test.data['dataset']['y'][-2].tolist())

        self.assertEqual([[0.714,	1.227]],
                         test.data['dataset']['k'][0].tolist())
        self.assertEqual([[0.712,	1.225]],
                         test.data['dataset']['k'][1].tolist())
        self.assertEqual([[0.710,	1.224]],
                         test.data['dataset']['k'][-1].tolist())
        self.assertEqual([[0.712,	1.225]],
                         test.data['dataset']['k'][-2].tolist())

        self.assertEqual([[12.493], [12.493], [12.495], [12.498], [12.502]],
                         test.data['dataset']['w'][0].tolist())
        self.assertEqual([[12.493], [12.495], [12.498], [12.502], [12.508]],
                         test.data['dataset']['w'][1].tolist())
        self.assertEqual([[12.495], [12.498], [12.502], [12.508], [12.515]],
                         test.data['dataset']['w'][-1].tolist())
        self.assertEqual([[12.493], [12.495], [12.498], [12.502], [12.508]],
                         test.data['dataset']['w'][-2].tolist())

        ## Load from file
        ## Try to train the model
        # test.trainModel(splits=[80, 10, 10],
        #                 training_params={'num_of_epochs': 100, 'train_batch_size': 4, 'test_batch_size': 4})


if __name__ == '__main__':
    unittest.main()