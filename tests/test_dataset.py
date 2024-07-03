import logging
import sys
import os
# append a new directory to sys.path
sys.path.append(os.getcwd())

import unittest
import numpy as np
from neu4mes import *

log = logging.getLogger(__name__)
log.setLevel(logging.CRITICAL)

# This file test the data loading in particular:
# The shape and the value of the inputs

import os
data_folder = os.path.join(os.path.dirname(__file__), 'data/')

class Neu4mesCreateDatasetTest(unittest.TestCase):
    
    def test_build_dataset_simple(self):
        input = Input('in')
        output = Input('out')
        relation = Fir(input.tw(0.05))

        test = Neu4mes(visualizer=None)
        test.minimizeError('out', output.z(-1), relation)
        test.neuralizeModel(0.01)

        data_struct = ['x1','y1','x2','y2','','A1x','A1y','B1x','B1y','','A2x','A2y','B2x','out','','x3','in','theta','time']
        test.loadData(source=data_folder, format=data_struct, skiplines=4, delimiter='\t', header=None)
        self.assertEqual((10,5,1),test.data['in'].shape)
        self.assertEqual([[0.984],[0.983],[0.982],[0.98],[0.977]] , test.data['in'][0].tolist())

        self.assertEqual((10,1,1),test.data['out'].shape)
        self.assertEqual([[1.225]], test.data['out'][0].tolist())
        self.assertEqual([[[1.225]], [[1.224]], [[1.222]], [[1.22]], [[1.217]], [[1.214]], [[1.211]], [[1.207]], [[1.204]], [[1.2]]], test.data['out'].tolist())
    
    def test_build_dataset_medium1(self):
        input = Input('in')
        output = Input('out')
        rel1 = Fir(input.tw(0.05))
        rel2 = Fir(input.tw(0.01))

        test = Neu4mes(visualizer=None)
        test.minimizeError('out', output.z(-1), rel1 + rel2)
        test.neuralizeModel(0.01)

        data_struct = ['x1','y1','x2','y2','','A1x','A1y','B1x','B1y','','A2x','A2y','B2x','out','','x3','in','theta','time']
        test.loadData(source=data_folder, format=data_struct, skiplines=4, delimiter='\t', header=None)
        self.assertEqual((10,5,1),test.data['in'].shape)
        self.assertEqual([[0.984],[0.983],[0.982],[0.98],[0.977]],test.data['in'][0].tolist())

        self.assertEqual((10,1,1),test.data['out'].shape)
        self.assertEqual([[[1.225]], [[1.224]], [[1.222]], [[1.22]], [[1.217]], [[1.214]], [[1.211]], [[1.207]], [[1.204]], [[1.2]]],test.data['out'].tolist())
    
    
    def test_build_dataset_medium2(self):
        input1 = Input('in1')
        input2 = Input('in2')
        output = Input('out')
        rel1 = Fir(input1.tw(0.05))
        rel2 = Fir(input1.tw(0.01))
        rel3 = Fir(input2.tw(0.02))

        test = Neu4mes(visualizer=None)
        test.minimizeError('out', output.z(-1), rel1 + rel2 + rel3)
        test.neuralizeModel(0.01)

        data_struct = ['x1','y1','x2','y2','','A1x','A1y','B1x','B1y','','A2x','A2y','B2x','out','','x3','in1','in2','time']
        test.loadData(source=data_folder, format=data_struct, skiplines=4, delimiter='\t', header=None)
        self.assertEqual((10,5,1),test.data['in1'].shape)
        self.assertEqual([[0.984],[0.983],[0.982],[0.98],[0.977]],test.data['in1'][0].tolist())

        self.assertEqual((10,2,1),test.data['in2'].shape)
        self.assertEqual([[12.498], [12.502]],test.data['in2'][0].tolist())

        self.assertEqual((10,1,1),test.data['out'].shape)
        self.assertEqual([[[1.225]], [[1.224]], [[1.222]], [[1.22]], [[1.217]], [[1.214]], [[1.211]], [[1.207]], [[1.204]], [[1.2]]],test.data['out'].tolist())
    
    def test_build_dataset_complex1(self):
        input1 = Input('in1')
        output = Input('out')
        rel1 = Fir(input1.tw(0.05))
        rel2 = Fir(input1.tw([-0.01,0.02]))

        test = Neu4mes(visualizer=None)
        test.minimizeError('out', output.z(-1), rel1 + rel2)
        test.neuralizeModel(0.01)

        data_struct = ['x1','y1','x2','y2','','A1x','A1y','B1x','B1y','','A2x','A2y','B2x','out','','x3','in1','in2','time']
        test.loadData(source=data_folder, format=data_struct, skiplines=4, delimiter='\t', header=None)
        self.assertEqual((9,7,1),test.data['in1'].shape)
        self.assertEqual([[0.984],[0.983],[0.982],[0.98],[0.977],[0.973],[0.969]],test.data['in1'][0].tolist())

        self.assertEqual((9,1,1),test.data['out'].shape)
        self.assertEqual([[[1.225]], [[1.224]], [[1.222]], [[1.22]], [[1.217]], [[1.214]], [[1.211]], [[1.207]], [[1.204]]],test.data['out'].tolist())
    
    def test_build_dataset_complex2(self):
        input1 = Input('in1')
        output = Input('out')
        rel1 = Fir(input1.tw(0.05))
        rel2 = Fir(input1.tw([-0.01,0.01]))

        test = Neu4mes(visualizer=None)
        test.minimizeError('out', output.z(-1), rel1 + rel2)
        test.neuralizeModel(0.01)

        data_struct = ['x1','y1','x2','y2','','A1x','A1y','B1x','B1y','','A2x','A2y','B2x','out','','x3','in1','in2','time']
        test.loadData(source=data_folder, format=data_struct, skiplines=4, delimiter='\t', header=None)
        self.assertEqual((10,6,1),test.data['in1'].shape)
        self.assertEqual([[0.984],[0.983],[0.982],[0.98],[0.977],[0.973]],test.data['in1'][0].tolist())

        self.assertEqual((10,1,1),test.data['out'].shape)
        self.assertEqual([[[1.225]], [[1.224]], [[1.222]], [[1.22]], [[1.217]], [[1.214]], [[1.211]], [[1.207]], [[1.204]], [[1.2]]],test.data['out'].tolist())
    
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
        test.addModel(fun)
        test.minimizeError('out', output.z(-1), rel1 + rel2 + rel3 + rel4)
        test.neuralizeModel(0.01)

        data_struct = ['x1','y1','x2','y2','','A1x','A1y','B1x','B1y','','A2x','A2y','B2x','out','','x3','in1','in2','time']
        test.loadData(source=data_folder, format=data_struct, skiplines=4, delimiter='\t', header=None)
        self.assertEqual((10,6,1),test.data['in1'].shape)
        self.assertEqual([[0.984],[0.983],[0.982],[0.98],[0.977],[0.973]],test.data['in1'][0].tolist())
        
        self.assertEqual((10,2,1),test.data['in2'].shape)
        self.assertEqual([[12.498], [12.502]],test.data['in2'][0].tolist())

        self.assertEqual((10,1,1),test.data['out'].shape)
        self.assertEqual([[[1.225]], [[1.224]], [[1.222]], [[1.22]], [[1.217]], [[1.214]], [[1.211]], [[1.207]], [[1.204]], [[1.2]]],test.data['out'].tolist())
    
    def test_build_dataset_complex5(self):
        input1 = Input('in1')
        output = Input('out')
        rel1 = Fir(input1.tw(0.05))
        rel2 = Fir(input1.tw([-0.01,0.01]))
        rel3 = Fir(input1.tw([-0.02,0.02]))
        fun = Output('out',rel1+rel2+rel3)

        test = Neu4mes(visualizer=None)
        test.addModel(fun)
        test.minimizeError('out', output.z(-1), fun)
        test.neuralizeModel(0.01)

        data_struct = ['x1','y1','x2','y2','','A1x','A1y','B1x','B1y','','A2x','A2y','B2x','out','','x3','in1','in2','time']
        test.loadData(source=data_folder, format=data_struct, skiplines=4, delimiter='\t', header=None)
        self.assertEqual((9,7,1),test.data['in1'].shape)
        self.assertEqual([[0.984],[0.983],[0.982],[0.98],[0.977],[0.973],[0.969]],test.data['in1'][0].tolist())

        self.assertEqual((9,1,1),test.data['out'].shape)
        self.assertEqual([[[1.225]], [[1.224]], [[1.222]], [[1.22]], [[1.217]], [[1.214]], [[1.211]], [[1.207]], [[1.204]]],test.data['out'].tolist())
    
    def test_build_dataset_complex6(self):
        input1 = Input('in1')
        output = Input('out')
        rel1 = Fir(input1.tw(0.05))
        rel2 = Fir(input1.tw([-0.01,0.02]))
        rel3 = Fir(input1.tw([-0.05,0.01]))
        fun = Output('out',rel1+rel2+rel3)

        test = Neu4mes(visualizer=None)
        test.addModel(fun)
        test.minimizeError('out', output.z(-1), fun)
        test.neuralizeModel(0.01)

        data_struct = ['x1','y1','x2','y2','','A1x','A1y','B1x','B1y','','A2x','A2y','B2x','out','','x3','in1','in2','time']
        test.loadData(source=data_folder, format=data_struct, skiplines=4, delimiter='\t', header=None)
        self.assertEqual((9,7,1),test.data['in1'].shape)
        self.assertEqual([[0.984],[0.983],[0.982],[0.98],[0.977],[0.973],[0.969]],test.data['in1'][0].tolist())

        self.assertEqual((9,1,1),test.data['out'].shape)
        self.assertEqual([[[1.225]], [[1.224]], [[1.222]], [[1.22]], [[1.217]], [[1.214]], [[1.211]], [[1.207]], [[1.204]]],test.data['out'].tolist())

    def test_build_dataset_custom(self):
        input1 = Input('in1')
        output = Input('out')
        rel1 = Fir(input1.tw(0.05))
        rel2 = Fir(input1.tw([-0.01,0.02]))
        rel3 = Fir(input1.tw([-0.05,0.01]))
        fun = Output('out',rel1+rel2+rel3)

        test = Neu4mes(visualizer=None)
        test.addModel(fun)
        test.minimizeError('out', output.z(-1), fun)
        test.neuralizeModel(0.01)

        data_x = np.array(range(10))
        data_a = 2
        data_b = -3
        dataset = {'in1': data_x, 'out': (data_a*data_x) + data_b}
        
        test.loadData(source=dataset)
        self.assertEqual((4,7,1),test.data['in1'].shape)
        self.assertEqual([[[0],[1],[2],[3],[4],[5],[6]],
                        [[1],[2],[3],[4],[5],[6],[7]],
                        [[2],[3],[4],[5],[6],[7],[8]],
                        [[3],[4],[5],[6],[7],[8],[9]]],
                        test.data['in1'].tolist())

        self.assertEqual((4,1,1),test.data['out'].shape)
        self.assertEqual([[[7]],[[9]],[[11]],[[13]]],test.data['out'].tolist())

if __name__ == '__main__':
    unittest.main()