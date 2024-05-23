import sys
import os
# append a new directory to sys.path
sys.path.append(os.getcwd())

import unittest, logging
import numpy as np
from neu4mes import *

import os
data_folder = os.path.join(os.path.dirname(__file__), 'data/')

class Neu4mesCreateDatasetTest(unittest.TestCase):
    def test_build_dataset_simple(self):
        input = Input('in')
        output = Input('out')
        relation = Fir(input.tw(0.05))

        test = Neu4mes()
        test.minimizeError('out',output.z(-1),relation)
        test.neuralizeModel(0.01)

        data_struct = ['x1','y1','x2','y2','','A1x','A1y','B1x','B1y','','A2x','A2y','B2x','out','','x3','in','theta','time']
        test.loadData(folder=data_folder, format=data_struct, skiplines = 4)
        self.assertEqual((10,5),test.inout_asarray['in'].shape)
        self.assertEqual([[0.984,0.983,0.982,0.98,0.977],
                    [0.983,0.982,0.98,0.977,0.973],
                    [0.982,0.98,0.977,0.973,0.969],
                    [0.98,0.977,0.973,0.969,0.963],
                    [0.977,0.973,0.969,0.963,0.957],
                    [0.973,0.969,0.963,0.957,0.95],
                    [0.969,0.963,0.957,0.95,0.942],
                    [0.963,0.957,0.95,0.942,0.933],
                    [0.957,0.95,0.942,0.933,0.923],
                    [0.95,0.942,0.933,0.923,0.912]] , test.inout_asarray['in'].tolist())

        self.assertEqual((10,),test.inout_asarray['out'].shape)
        self.assertEqual([1.225, 1.224, 1.222, 1.22, 1.217, 1.214, 1.211, 1.207, 1.204, 1.200],test.inout_asarray['out'].tolist())

    def test_build_dataset_medium1(self):
        input = Input('in')
        output = Input('out')
        rel1 = Fir(input.tw(0.05))
        rel2 = Fir(input.tw(0.01))

        test = Neu4mes()
        test.minimizeError('out',output.z(-1), rel1+rel2)
        test.neuralizeModel(0.01)

        data_struct = ['x1','y1','x2','y2','','A1x','A1y','B1x','B1y','','A2x','A2y','B2x','out','','x3','in','theta','time']
        test.loadData(folder=data_folder, format=data_struct, skiplines = 4)
        self.assertEqual((10,5),test.inout_asarray['in'].shape)
        self.assertEqual([[0.984,0.983,0.982,0.98,0.977],
                    [0.983,0.982,0.98,0.977,0.973],
                    [0.982,0.98,0.977,0.973,0.969],
                    [0.98,0.977,0.973,0.969,0.963],
                    [0.977,0.973,0.969,0.963,0.957],
                    [0.973,0.969,0.963,0.957,0.95],
                    [0.969,0.963,0.957,0.95,0.942],
                    [0.963,0.957,0.95,0.942,0.933],
                    [0.957,0.95,0.942,0.933,0.923],
                    [0.95,0.942,0.933,0.923,0.912]],test.inout_asarray['in'].tolist())

        self.assertEqual((10,),test.inout_asarray['out'].shape)
        self.assertEqual([1.225, 1.224, 1.222, 1.22, 1.217, 1.214, 1.211, 1.207, 1.204, 1.200],test.inout_asarray['out'].tolist())

    def test_build_dataset_medium2(self):
        input1 = Input('in1')
        input2 = Input('in2')
        output = Input('out')
        rel1 = Fir(input1.tw(0.05))
        rel2 = Fir(input1.tw(0.01))
        rel3 = Fir(input2.tw(0.02))

        test = Neu4mes()
        test.minimizeError('out',output.z(-1), rel1+rel2+rel3)
        test.neuralizeModel(0.01)

        data_struct = ['x1','y1','x2','y2','','A1x','A1y','B1x','B1y','','A2x','A2y','B2x','out','','x3','in1','in2','time']
        test.loadData(folder=data_folder, format=data_struct, skiplines = 4)
        self.assertEqual((10,5),test.inout_asarray['in1'].shape)
        self.assertEqual([[0.984,0.983,0.982,0.98,0.977],
                    [0.983,0.982,0.98,0.977,0.973],
                    [0.982,0.98,0.977,0.973,0.969],
                    [0.98,0.977,0.973,0.969,0.963],
                    [0.977,0.973,0.969,0.963,0.957],
                    [0.973,0.969,0.963,0.957,0.95],
                    [0.969,0.963,0.957,0.95,0.942],
                    [0.963,0.957,0.95,0.942,0.933],
                    [0.957,0.95,0.942,0.933,0.923],
                    [0.95,0.942,0.933,0.923,0.912]],test.inout_asarray['in1'].tolist())

        self.assertEqual((10,2),test.inout_asarray['in2'].shape)
        self.assertEqual([[12.498, 12.502],
                    [12.502, 12.508],
                    [12.508, 12.515],
                    [12.515, 12.523],
                    [12.523, 12.533],
                    [12.533, 12.543],
                    [12.543, 12.556],
                    [12.556, 12.57 ],
                    [12.57 , 12.585],
                    [12.585, 12.602]],test.inout_asarray['in2'].tolist())

        self.assertEqual((10,),test.inout_asarray['out'].shape)
        self.assertEqual([1.225, 1.224, 1.222, 1.22, 1.217, 1.214, 1.211, 1.207, 1.204, 1.200],test.inout_asarray['out'].tolist())

    def test_build_dataset_complex1(self):
        input1 = Input('in1')
        output = Input('out')
        rel1 = Fir(input1.tw(0.05))
        rel2 = Fir(input1.tw([-0.01,0.02]))

        test = Neu4mes()
        test.minimizeError('out', output.z(-1), rel1+rel2)
        test.neuralizeModel(0.01)

        data_struct = ['x1','y1','x2','y2','','A1x','A1y','B1x','B1y','','A2x','A2y','B2x','out','','x3','in1','in2','time']
        test.loadData(folder=data_folder, format=data_struct, skiplines = 4)
        self.assertEqual((9,7),test.inout_asarray['in1'].shape)
        self.assertEqual([[0.984,0.983,0.982,0.98,0.977,0.973,0.969],
                    [0.983,0.982,0.98,0.977,0.973,0.969,0.963],
                    [0.982,0.98,0.977,0.973,0.969,0.963,0.957],
                    [0.98,0.977,0.973,0.969,0.963,0.957,0.95],
                    [0.977,0.973,0.969,0.963,0.957,0.95,0.942],
                    [0.973,0.969,0.963,0.957,0.95,0.942,0.933],
                    [0.969,0.963,0.957,0.95,0.942,0.933,0.923],
                    [0.963,0.957,0.95,0.942,0.933,0.923,0.912],
                    [0.957,0.95,0.942,0.933,0.923,0.912,0.900]],test.inout_asarray['in1'].tolist())

        self.assertEqual((9,),test.inout_asarray['out'].shape)
        self.assertEqual([1.225, 1.224, 1.222, 1.22, 1.217, 1.214, 1.211, 1.207, 1.204],test.inout_asarray['out'].tolist())

    def test_build_dataset_complex2(self):
        input1 = Input('in1')
        output = Input('out')
        rel1 = Fir(input1.tw(0.05))
        rel2 = Fir(input1.tw([-0.01,0.01]))

        test = Neu4mes()
        test.minimizeError('out',output.z(-1), rel1+rel2)
        test.neuralizeModel(0.01)

        data_struct = ['x1','y1','x2','y2','','A1x','A1y','B1x','B1y','','A2x','A2y','B2x','out','','x3','in1','in2','time']
        test.loadData(folder=data_folder, format=data_struct, skiplines = 4)
        self.assertEqual((10,6),test.inout_asarray['in1'].shape)
        self.assertEqual([[0.984,0.983,0.982,0.98,0.977,0.973],
                        [0.983,0.982,0.98,0.977,0.973,0.969],
                        [0.982,0.98,0.977,0.973,0.969,0.963],
                        [0.98,0.977,0.973,0.969,0.963,0.957],
                        [0.977,0.973,0.969,0.963,0.957,0.95],
                        [0.973,0.969,0.963,0.957,0.95,0.942],
                        [0.969,0.963,0.957,0.95,0.942,0.933],
                        [0.963,0.957,0.95,0.942,0.933,0.923],
                        [0.957,0.95,0.942,0.933,0.923,0.912],
                        [0.95,0.942,0.933,0.923,0.912,0.900]],test.inout_asarray['in1'].tolist())

        self.assertEqual((10,),test.inout_asarray['out'].shape)
        self.assertEqual([1.225, 1.224, 1.222, 1.22, 1.217, 1.214, 1.211, 1.207, 1.204, 1.200],test.inout_asarray['out'].tolist())

    def test_build_dataset_complex3(self):
        input1 = Input('in1')
        input2 = Input('in2')
        output = Input('out')
        rel1 = Fir(input1.tw(0.05))
        rel2 = Fir(input2.tw(0.02))
        rel3 = Fir(input1.tw([-0.01,0.01]))
        rel4 = Fir(input2)
        fun = Output('out',rel1+rel2+rel3+rel4)

        test = Neu4mes()
        test.addModel(fun)
        test.minimizeError('out',output.z(-1), rel1+rel2+rel3+rel4)
        test.neuralizeModel(0.01)

        data_struct = ['x1','y1','x2','y2','','A1x','A1y','B1x','B1y','','A2x','A2y','B2x','out','','x3','in1','in2','time']
        test.loadData(folder=data_folder, format=data_struct, skiplines = 4)
        self.assertEqual((10,6),test.inout_asarray['in1'].shape)
        self.assertEqual([[0.984,0.983,0.982,0.98,0.977,0.973],
                        [0.983,0.982,0.98,0.977,0.973,0.969],
                        [0.982,0.98,0.977,0.973,0.969,0.963],
                        [0.98,0.977,0.973,0.969,0.963,0.957],
                        [0.977,0.973,0.969,0.963,0.957,0.95],
                        [0.973,0.969,0.963,0.957,0.95,0.942],
                        [0.969,0.963,0.957,0.95,0.942,0.933],
                        [0.963,0.957,0.95,0.942,0.933,0.923],
                        [0.957,0.95,0.942,0.933,0.923,0.912],
                        [0.95,0.942,0.933,0.923,0.912,0.900]],test.inout_asarray['in1'].tolist())
        
        self.assertEqual((10,2),test.inout_asarray['in2'].shape)
        self.assertEqual([[12.498, 12.502],
                    [12.502, 12.508],
                    [12.508, 12.515],
                    [12.515, 12.523],
                    [12.523, 12.533],
                    [12.533, 12.543],
                    [12.543, 12.556],
                    [12.556, 12.57 ],
                    [12.57 , 12.585],
                    [12.585, 12.602]],test.inout_asarray['in2'].tolist())

        self.assertEqual((10,),test.inout_asarray['out'].shape)
        self.assertEqual([1.225, 1.224, 1.222, 1.22, 1.217, 1.214, 1.211, 1.207, 1.204, 1.200],test.inout_asarray['out'].tolist())

    def test_build_dataset_complex5(self):
        input1 = Input('in1')
        output = Input('out')
        rel1 = Fir(input1.tw(0.05))
        rel2 = Fir(input1.tw([-0.01,0.01]))
        rel3 = Fir(input1.tw([-0.02,0.02]))
        fun = Output('out',rel1+rel2+rel3)

        test = Neu4mes()
        test.addModel(fun)
        test.minimizeError('out',output.z(-1), fun)
        test.neuralizeModel(0.01)

        data_struct = ['x1','y1','x2','y2','','A1x','A1y','B1x','B1y','','A2x','A2y','B2x','out','','x3','in1','in2','time']
        test.loadData(folder=data_folder, format=data_struct, skiplines = 4)
        self.assertEqual((9,7),test.inout_asarray['in1'].shape)
        self.assertEqual([[0.984,0.983,0.982,0.98,0.977,0.973,0.969],
                        [0.983,0.982,0.98,0.977,0.973,0.969,0.963],
                        [0.982,0.98,0.977,0.973,0.969,0.963,0.957],
                        [0.98,0.977,0.973,0.969,0.963,0.957,0.95],
                        [0.977,0.973,0.969,0.963,0.957,0.95,0.942],
                        [0.973,0.969,0.963,0.957,0.95,0.942,0.933],
                        [0.969,0.963,0.957,0.95,0.942,0.933,0.923],
                        [0.963,0.957,0.95,0.942,0.933,0.923,0.912],
                        [0.957,0.95,0.942,0.933,0.923,0.912,0.900]],test.inout_asarray['in1'].tolist())

        self.assertEqual((9,),test.inout_asarray['out'].shape)
        self.assertEqual([1.225, 1.224, 1.222, 1.22, 1.217, 1.214, 1.211, 1.207, 1.204],test.inout_asarray['out'].tolist())

    def test_build_dataset_complex6(self):
        input1 = Input('in1')
        output = Input('out')
        rel1 = Fir(input1.tw(0.05))
        rel2 = Fir(input1.tw([-0.01,0.02]))
        rel3 = Fir(input1.tw([-0.05,0.01]))
        fun = Output('out',rel1+rel2+rel3)

        test = Neu4mes()
        test.addModel(fun)
        test.minimizeError('out',output.z(-1), fun)
        test.neuralizeModel(0.01)

        data_struct = ['x1','y1','x2','y2','','A1x','A1y','B1x','B1y','','A2x','A2y','B2x','out','','x3','in1','in2','time']
        test.loadData(folder=data_folder, format=data_struct, skiplines = 4)
        self.assertEqual((9,7),test.inout_asarray['in1'].shape)
        self.assertEqual([[0.984,0.983,0.982,0.98,0.977,0.973,0.969],
                        [0.983,0.982,0.98,0.977,0.973,0.969,0.963],
                        [0.982,0.98,0.977,0.973,0.969,0.963,0.957],
                        [0.98,0.977,0.973,0.969,0.963,0.957,0.95],
                        [0.977,0.973,0.969,0.963,0.957,0.95,0.942],
                        [0.973,0.969,0.963,0.957,0.95,0.942,0.933],
                        [0.969,0.963,0.957,0.95,0.942,0.933,0.923],
                        [0.963,0.957,0.95,0.942,0.933,0.923,0.912],
                        [0.957,0.95,0.942,0.933,0.923,0.912,0.900]],test.inout_asarray['in1'].tolist())

        self.assertEqual((9,),test.inout_asarray['out'].shape)
        self.assertEqual([1.225, 1.224, 1.222, 1.22, 1.217, 1.214, 1.211, 1.207, 1.204],test.inout_asarray['out'].tolist())

if __name__ == '__main__':
    unittest.main()