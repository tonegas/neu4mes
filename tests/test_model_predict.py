import sys
import os
# append a new directory to sys.path
sys.path.append(os.getcwd())

import unittest
from neu4mes import *
relation.CHECK_NAMES = False

import torch

# This file test the model prediction in particular the output value

def myfun(x, P):
    return x*P

def myfun2(a, b ,c):
    import torch
    return torch.sin(a + b) * c

def myfun3(a, b, p1, p2):
    import torch
    at = torch.transpose(a[:, :, 0:2],1,2)
    bt = torch.transpose(b, 1, 2)
    return torch.matmul(p1,at+bt)+p2.t()

# Dimensions
# The first dimension must indicate the time dimension i.e. how many time samples I asked for
# The second dimension indicates the output time dimension for each sample.
# The third is the size of the signal

class MyTestCase(unittest.TestCase):
    
    def TestAlmostEqual(self, data1, data2, precision=4):
        assert np.asarray(data1, dtype=np.float32).ndim == np.asarray(data2, dtype=np.float32).ndim, f'Inputs must have the same dimension! Received {type(data1)} and {type(data2)}'
        if type(data1) == type(data2) == list:  
            for pred, label in zip(data1, data2):
                self.TestAlmostEqual(pred, label, precision=precision)
        else:
            self.assertAlmostEqual(data1, data2, places=precision)

    def test_single_in(self):
        in1 = Input('in1')
        in2 = Input('in2')
        out_fun = Fir(in1.tw(0.1)) + Fir(in2.last())
        out = Output('out', out_fun)
        test = Neu4mes(visualizer=None, seed=1)
        test.addModel('out',out)
        test.neuralizeModel(0.01)
        results = test({'in1': [[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]],'in2': [[5]]})
        self.assertEqual(1, len(results['out']))
        self.TestAlmostEqual([33.74938201904297], results['out'])
        results = test({'in1': [[1], [2], [3], [4], [5], [6], [7], [8], [9], [10], [11]], 'in2': [[5], [7]]})
        self.assertEqual(2, len(results['out']))
        self.TestAlmostEqual([33.74938201904297, 40.309326171875], results['out'])
        results = test({'in1': [[1], [2], [3], [4], [5], [6], [7], [8], [9], [10], [11], [12]], 'in2': [[5], [7], [9]]})
        self.assertEqual(3, len(results['out']))
        self.TestAlmostEqual([33.74938201904297, 40.309326171875, 46.86927032470703], results['out'])
    
    def test_single_in_window(self):
        # Here there is more sample for each time step but the dimensions of the input is 1
        in1 = Input('in1')

        # Finestre nel tempo
        out1 = Output('x.tw(1)', in1.tw(1))
        out2 = Output('x.tw([-1,0])', in1.tw([-1, 0]))
        out3 = Output('x.tw([-3,0])', in1.tw([-3, 0]))
        out4 = Output('x.tw([1,3])', in1.tw([1, 3]))
        out5 = Output('x.tw([-1,3])', in1.tw([-1, 3]))
        out6 = Output('x.tw([0,1])', in1.tw([0, 1]))
        out7 = Output('x.tw([-3,-2])', in1.tw([-3, -2]))

        ## TODO: adjust the z function
        # Finesatre nei samples 
        out8 = Output('x.z(-1)',  in1.z(-1))
        out9 = Output('x.z(0)',  in1.z(0))
        out10 = Output('x.z(2)',  in1.z(2))
        out11 = Output('x.sw([-1,0])',  in1.sw([-1, 0]))
        out12 = Output('x.sw([1,2])',  in1.sw([1, 2]))
        out13 = Output('x.sw([-3,1])',  in1.sw([-3, 1]))
        out14 = Output('x.sw([-3,-2])',  in1.sw([-3, -2]))
        out15 = Output('x.sw([0,1])',  in1.sw([0, 1]))

        test = Neu4mes(visualizer=None, seed=1)
        #test.addModel('out',[out0,out1,out2,out3,out4,out5,out6,out7,out11,out12,out13,out14,out15])
        test.addModel('out',[out1, out2, out3, out4, out5, out6, out7, out8, out9, out10, out11, out12, out13, out14, out15])

        test.neuralizeModel(1)
        # Time                  -2,-1,0,1,2,3 # zero represent the last passed instant
        results = test({'in1': [[-2],[-1],[0],[1],[7],[3]]})
        # Time window
        self.assertEqual((1,), np.array(results['x.tw(1)']).shape)
        self.TestAlmostEqual([0], results['x.tw(1)'])
        self.assertEqual((1,), np.array(results['x.tw([-1,0])']).shape)
        self.TestAlmostEqual([0], results['x.tw([-1,0])'])
        self.assertEqual((1, 3), np.array(results['x.tw([-3,0])']).shape)
        self.TestAlmostEqual([[-2, -1, 0]], results['x.tw([-3,0])'])
        self.assertEqual((1, 2), np.array(results['x.tw([1,3])']).shape)
        self.TestAlmostEqual([[7, 3]], results['x.tw([1,3])'])
        self.assertEqual((1, 4), np.array(results['x.tw([-1,3])']).shape)
        self.TestAlmostEqual([[0, 1, 7, 3]], results['x.tw([-1,3])'])
        self.assertEqual((1,), np.array(results['x.tw([0,1])']).shape)
        self.TestAlmostEqual([1], results['x.tw([0,1])'])
        self.assertEqual((1,), np.array(results['x.tw([-3,-2])']).shape)
        self.TestAlmostEqual([-2],results['x.tw([-3,-2])'])
        # Sample window
        self.assertEqual((1,), np.array(results['x.z(-1)']).shape)
        self.TestAlmostEqual([1], results['x.z(-1)'])
        self.assertEqual((1,), np.array(results['x.z(0)']).shape)
        self.TestAlmostEqual([0], results['x.z(0)'])
        self.assertEqual((1,), np.array(results['x.z(2)']).shape)
        self.TestAlmostEqual([-2], results['x.z(2)'])
        self.assertEqual((1,), np.array(results['x.sw([-1,0])']).shape)
        self.TestAlmostEqual([0], results['x.sw([-1,0])'])
        self.assertEqual((1,), np.array(results['x.sw([1,2])']).shape)
        self.TestAlmostEqual([7], results['x.sw([1,2])'])
        self.assertEqual((1,4), np.array(results['x.sw([-3,1])']).shape)
        self.TestAlmostEqual([[-2,-1,0,1]], results['x.sw([-3,1])'])
        self.assertEqual((1,), np.array(results['x.sw([-3,-2])']).shape)
        self.TestAlmostEqual([-2], results['x.sw([-3,-2])'])
        self.assertEqual((1,), np.array(results['x.sw([0,1])']).shape)
        self.TestAlmostEqual([1],results['x.sw([0,1])'])
    
    def test_single_in_window_offset(self):
        # Here there is more sample for each time step but the dimensions of the input is 1
        in1 = Input('in1')

        # Finestre nel tempo
        out1 = Output('x.tw(1)', in1.tw(1,offset=-1))
        out2 = Output('x.tw([-1,0])', in1.tw([-1, 0],offset=-1))
        out3 = Output('x.tw([1,3])', in1.tw([1, 3],offset=1))
        out4 = Output('x.tw([-3,-2])', in1.tw([-3, -2],offset=-3))

        # Finesatre nei samples
        out5 = Output('x.sw([-1,0])',  in1.sw([-1, 0],offset=-1))
        out6 = Output('x.sw([-3,1])',  in1.sw([-3, 1],offset=-3))
        out7 = Output('x.sw([0,1])',  in1.sw([0, 1],offset=0))
        out8 = Output('x.sw([-3, 3])', in1.sw([-3, 3], offset=2))
        out9 = Output('x.sw([-3, 3])-2', in1.sw([-3, 3], offset=-1))

        test = Neu4mes(visualizer = None, seed = 1)
        test.addModel('out',[out1,out2,out3,out4,out5,out6,out7,out8,out9])

        test.neuralizeModel(1)
        # Time                  -2,-1,0,1,2,3 # zero represent the last passed instant
        results = test({'in1': [[-2],[-1],[0],[1],[7],[3]]})
        # Time window
        self.assertEqual((1,), np.array(results['x.tw(1)']).shape)
        self.TestAlmostEqual([0], results['x.tw(1)'])
        self.assertEqual((1,), np.array(results['x.tw([-1,0])']).shape)
        self.TestAlmostEqual([0], results['x.tw([-1,0])'])
        self.assertEqual((1,2), np.array(results['x.tw([1,3])']).shape)
        self.TestAlmostEqual([[0, -4]], results['x.tw([1,3])'])
        self.assertEqual((1,), np.array(results['x.tw([-3,-2])']).shape)
        self.TestAlmostEqual([0],results['x.tw([-3,-2])'])
        # # Sample window
        self.assertEqual((1,), np.array(results['x.sw([-1,0])']).shape)
        self.TestAlmostEqual([0], results['x.sw([-1,0])'])
        self.assertEqual((1,4), np.array(results['x.sw([-3,1])']).shape)
        self.TestAlmostEqual([[0,1,2,3]], results['x.sw([-3,1])'])
        self.assertEqual((1,), np.array(results['x.sw([0,1])']).shape)
        self.TestAlmostEqual([0],results['x.sw([0,1])'])
        self.assertEqual((1,6), np.array(results['x.sw([-3, 3])']).shape)
        self.TestAlmostEqual([[-5,-4,-3,-2,4,0]],results['x.sw([-3, 3])'])
        self.assertEqual((1,6), np.array(results['x.sw([-3, 3])-2']).shape)
        self.TestAlmostEqual([[-2,-1,0,1,7,3]],results['x.sw([-3, 3])-2'])
    
    def test_multi_in_window_offset(self):
        # Here there is more sample for each time step but the dimensions of the input is 1
        in1 = Input('in1',dimensions=3)

        # Finestre nel tempo
        out1 = Output('x.tw(1)', in1.tw(1, offset=-1))
        out2 = Output('x.tw([-1,0])', in1.tw([-1, 0], offset=-1))
        out3 = Output('x.tw([1,3])', in1.tw([1, 3], offset=1))
        out4 = Output('x.tw([-3,-2])', in1.tw([-3, -2], offset=-3))

        # Finesatre nei samples
        out5 = Output('x.sw([-1,0])', in1.sw([-1, 0], offset=-1))
        out6 = Output('x.sw([-3,1])', in1.sw([-3, 1], offset=-3))
        out7 = Output('x.sw([0,1])', in1.sw([0, 1], offset=0))
        out8 = Output('x.sw([-3, 3])', in1.sw([-3, 3], offset=2))
        out9 = Output('x.sw([-3, 3])-2', in1.sw([-3, 3], offset=-1))

        test = Neu4mes(visualizer = None, seed = 1)
        test.addModel('out',[out1, out2, out3, out4, out5, out6, out7, out8, out9])

        test.neuralizeModel(1)

        # Single input
        # Time                  -2,         -1,      0,      1,      2,       3 # zero represent the last passed instant
        results = test({'in1': [[-2,3,4],[-1,2,2],[0,0,0],[1,2,3],[2,7,3],[3,3,3]]})
        # Time window
        self.assertEqual((1,1,3), np.array(results['x.tw(1)']).shape)
        self.TestAlmostEqual([[[0,0,0]]], results['x.tw(1)'])
        self.assertEqual((1,1,3), np.array(results['x.tw([-1,0])']).shape)
        self.TestAlmostEqual([[[0,0,0]]], results['x.tw([-1,0])'])
        self.assertEqual((1,2,3), np.array(results['x.tw([1,3])']).shape)
        self.TestAlmostEqual([[[0,0,0], [1,-4,0]]], results['x.tw([1,3])'])
        self.assertEqual((1,1,3), np.array(results['x.tw([-3,-2])']).shape)
        self.TestAlmostEqual([[[0,0,0]]], results['x.tw([-3,-2])'])
        # # Sample window
        self.assertEqual((1,1,3), np.array(results['x.sw([-1,0])']).shape)
        self.TestAlmostEqual([[[0,0,0]]], results['x.sw([-1,0])'])
        self.assertEqual((1,4,3), np.array(results['x.sw([-3,1])']).shape)
        self.TestAlmostEqual([[[0,0,0], [1,-1,-2], [2,-3,-4], [3,-1,-1]]], results['x.sw([-3,1])'])
        self.assertEqual((1,1,3), np.array(results['x.sw([0,1])']).shape)
        self.TestAlmostEqual([[[0,0,0]]], results['x.sw([0,1])'])
        self.assertEqual((1,6,3), np.array(results['x.sw([-3, 3])']).shape)
        self.TestAlmostEqual([[[-5,0,1],[-4,-1,-1], [-3,-3,-3], [-2,-1,0], [-1,4,0], [0,0,0]]], results['x.sw([-3, 3])'])
        self.assertEqual((1,6,3), np.array(results['x.sw([-3, 3])-2']).shape)
        self.TestAlmostEqual([[[-2,3,4],[-1,2,2],[0,0,0],[1,2,3],[2,7,3],[3,3,3]]], results['x.sw([-3, 3])-2'])

        # Multi input
        results = test({'in1': [[-2, 3, 4], [-1, 2, 2], [0, 0, 0], [1, 2, 3], [2, 7, 3], [3, 3, 3], [2, 2, 2]]})
        self.assertEqual((2,6,3), np.array(results['x.sw([-3, 3])']).shape)
        self.TestAlmostEqual([[[-5,0,1], [-4,-1,-1], [-3,-3,-3], [-2,-1,0], [-1,4,0], [0,0,0]],
                                    [[-3,0,0], [-2,-2,-2], [-1,0,1],   [0,5,1],   [1,1,1],  [0,0,0]]], results['x.sw([-3, 3])'])
        self.assertEqual((2,6,3), np.array(results['x.sw([-3, 3])-2']).shape)
        self.TestAlmostEqual([[[-2,3,4],[-1,2,2],[0,0,0],[1,2,3],[2,7,3],[3,3,3]],
                                    [[-2,0,-1],[-1,-2,-3],[0,0,0],[1,5,0],[2,1,0],[1,0,-1]]], results['x.sw([-3, 3])-2'])
    
    def test_single_in_window_offset_aritmetic(self):
        # Elementwise arithmetic, Activation, Trigonometric
        # the dimensions and time window remain unchanged, for the
        # binary operators must be equal

        in1 = Input('in1')
        in2 = Input('in2', dimensions=2)
        out1 = Output('sum', in1.tw(1, offset=-1) + in1.tw([-1,0]))
        out2 = Output('sub', in1.tw([1, 3], offset=1) - in1.tw([-3, -1], offset=-2))
        out3 = Output('mul', in1.tw([-2, 2]) * in1.tw([-3, 1], offset=-2))

        out4 = Output('sum2', in2.tw(1, offset=-1) + in2.tw([-1,0]))
        out5 = Output('sub2', in2.tw([1, 3], offset=1) - in2.tw([-3, -1], offset=-2))
        out6 = Output('mul2', in2.tw([-2, 2]) * in2.tw([-3, 1], offset=-2))

        test = Neu4mes(visualizer=None)
        test.addModel('out',[out1, out2, out3, out4, out5, out6])

        test.neuralizeModel(1)
        # Single input
        #Time                  -2    -1    0    1     2    3              -2       -1      0        1      2        3
        results = test({'in1': [[1], [2], [8], [4], [-1], [6]], 'in2': [[-2, 3], [-1, 2], [0, 5], [1, 2], [2, 7], [3, 3]]})

        self.assertEqual((1,), np.array(results['sum']).shape)
        self.TestAlmostEqual([8], results['sum'])
        self.assertEqual((1,2), np.array(results['sub']).shape) # [-1,6]+1 - [1,2]-2
        self.TestAlmostEqual([[1,7]], results['sub'])
        self.assertEqual((1,4), np.array(results['mul']).shape) #[2, 8, 4, -1]*[1, 2, 8, 4]-2
        self.TestAlmostEqual([[-2,0,24,-2]], results['mul'])#[2, 8, 4, -1]*[-1, 0, 6, 2]

        self.assertEqual((1,1,2), np.array(results['sum2']).shape)
        self.TestAlmostEqual([[[0,5]]], results['sum2'])
        self.assertEqual((1,2,2), np.array(results['sub2']).shape) #[[2,7],[3,3]]-[2,7] - [[-2,3],[-1,2]]-[-1,2]
        self.TestAlmostEqual([[[1,-1],[1,-4]]], results['sub2']) # [[0,0],[1,-4]] - [[-1,1],[0,0]]
        #[[-1, 2], [0, 5], [1, 2], [2, 7]] * [[-2, 3], [-1, 2], [0, 5], [1, 2]]-[-1, 2]
        # [[-1, 2], [0, 5], [1, 2], [2, 7]] * [[-1, 1], [0, 0], [1, 3], [2, 0]]
        self.assertEqual((1,4,2), np.array(results['mul2']).shape)
        self.TestAlmostEqual([[[1, 2], [0, 0], [1, 6], [4, 0]]], results['mul2'])

        # Multi input
        # Time                  -2 -1  0  1  2  3  4             -2       -1      0        1      2        3         4
        results = test({'in1': [1, 2, 8, 4, -1, 6, 9], 'in2': [[-2, 3], [-1, 2], [0, 5], [1, 2], [2, 7], [3, 3], [0, 0]]})
        self.assertEqual((2,), np.array(results['sum']).shape)
        self.TestAlmostEqual([8,4], results['sum'])
        # [6,9]-6 - [2,8]-8 = [0,3] - [-6,0]
        self.assertEqual((2,2), np.array(results['sub']).shape) # [-1,6]+1 - [1,2]-2
        self.TestAlmostEqual([[1,7],[6,3]], results['sub'])
        #[8, 4, -1, 6] * [2, 8, 4, -1]-8 = [8, 4, -1, 6] * [-6, 0, -4, -9]
        self.assertEqual((2,4), np.array(results['mul']).shape)
        self.TestAlmostEqual([[-2,0,24,-2],[-48,0,4,-54]], results['mul'])

        self.assertEqual((2,1,2), np.array(results['sum2']).shape)
        self.TestAlmostEqual([[[0,5]],[[1, 2]]], results['sum2'])
        self.assertEqual((2,2,2), np.array(results['sub2']).shape)
        #[[3, 3], [0, 0]]-[3,3] - [[-1, 2], [0, 5]]-[0, 5] = [[0, 0], [-3, -3]] - [[-1, -3], [0, 0]]
        self.TestAlmostEqual([[[1,-1],[1,-4]],[[1,3],[-3,-3]]], results['sub2'])
        #[[0, 5], [1, 2], [2, 7], [3, 3]] * [[-1, 2], [0, 5], [1, 2], [2, 7]]-[0, 5]
        #[[0, 5], [1, 2], [2, 7], [3, 3]] * [[-1, -3], [0, 0], [1, -3], [2, 2]]
        self.assertEqual((2,4,2), np.array(results['mul2']).shape)
        self.TestAlmostEqual([[[1, 2], [0, 0], [1, 6], [4, 0]],[[0, -15], [0, 0], [2, -21], [6, 6]]], results['mul2'])
    
    def test_single_in_window_offset_fir(self):
        # The input must be scalar and the time dimension is compress to 1,
        # Vector input not allowed, it could be done that a number of fir filters equal to the size of the vector are constructed
        # Should weights be shared or not?
        in1 = Input('in1')
        out1 = Output('Fir3', Fir(3)(in1.last()))
        out2 = Output('Fir5', Fir(5)(in1.tw(1)))#
        out3 = Output('Fir2', Fir(2)(in1.tw([-1,0])))#
        out4 = Output('Fir1', Fir(1)(in1.tw([-3,3])))#
        out5 = Output('Fir7', Fir(7)(in1.tw(3,offset=-1)))#
        out6 = Output('Fir4', Fir(4)(in1.tw([2,3],offset=2)))#
        out7 = Output('Fir6', Fir(6)(in1.sw([-2,-1], offset=-2)))#

        test = Neu4mes(visualizer = None, seed = 1)
        test.addModel('out',[out1,out2,out3,out4,out5,out6,out7])
        test.neuralizeModel(1)
        # Single input
        # Time                 -2    -1    0    1    2    3
        results = test({'in1': [[1], [2], [7], [4], [5], [6]]})
        self.assertEqual((1,1,3), np.array(results['Fir3']).shape)
        self.assertEqual((1,1,5), np.array(results['Fir5']).shape)
        self.assertEqual((1,1,2), np.array(results['Fir2']).shape)
        self.assertEqual((1,), np.array(results['Fir1']).shape)
        self.assertEqual((1,1,7), np.array(results['Fir7']).shape)
        self.assertEqual((1,1,4), np.array(results['Fir4']).shape)
        self.assertEqual((1,1,6), np.array(results['Fir6']).shape)

        # Multi input there are 3 temporal instant
        # Time                 -2 -1  0  1  2  3  4  5
        results = test({'in1': [[1], [2], [7], [4], [5], [6], [7], [8]]})
        self.assertEqual((3,1,3), np.array(results['Fir3']).shape)
        self.assertEqual((3,1,5), np.array(results['Fir5']).shape)
        self.assertEqual((3,1,2), np.array(results['Fir2']).shape)
        self.assertEqual((3,), np.array(results['Fir1']).shape)
        self.assertEqual((3,1,7), np.array(results['Fir7']).shape)
        self.assertEqual((3,1,4), np.array(results['Fir4']).shape)
        self.assertEqual((3,1,6), np.array(results['Fir6']).shape)
    
    def test_fir_and_parameter(self):
        x = Input('x')
        p1 = Parameter('p1', tw=3, values=[[1],[2],[3],[6],[2],[3]])
        with self.assertRaises(TypeError):
            Fir(parameter=p1)(x)
        with self.assertRaises(ValueError):
            Fir(parameter=p1)(x.tw([-3, 1]))
        out1 = Output('out1', Fir(parameter=p1)(x.tw([-2, 1])))

        p2 = Parameter('p2', sw=1, values=[[-2]])
        with self.assertRaises(KeyError):
            Fir(parameter=p2)(x.tw([-2, 1]))
        out2 = Output('out2', Fir(parameter=p2)(x.last()))

        p3 = Parameter('p3', dimensions=2, sw=1, values=[[-2,1]])
        with self.assertRaises(KeyError):
            Fir(parameter=p3)(x.tw([-2, 1]))
        out3 = Output('out3', Fir(parameter=p3)(x.last()))

        p4 = Parameter('p4', dimensions=2, tw=2, values=[[-2,1],[2,0],[0,1],[4,0]])
        with self.assertRaises(KeyError):
            Fir(parameter=p4)(x.sw([-2, 0]))
        out4 = Output('out4', Fir(parameter=p4)(x.tw([-2, 0])))

        p5 = Parameter('p6', sw=2, dimensions=2, values=[[-2,1],[2,0]])
        with self.assertRaises(TypeError):
            Fir(parameter = p5)(x)
        with self.assertRaises(KeyError):
            Fir(parameter = p5)(x.tw([-2,1]))
        with self.assertRaises(ValueError):
            Fir(parameter = p5)(x.sw([-2,1]))
        out5 = Output('out5', Fir(parameter=p5)(x.sw([-2, 0])))

        test = Neu4mes(visualizer=None)
        test.addModel('out',[out1, out2, out3, out4, out5])
        test.neuralizeModel(0.5)
        # Time   -2, -1, 0, 1, 2, 3, 4
        input = [-2, -1, 0, 1, 2, 3, 12]
        results = test({'x': input})
        pprint(results)
        self.assertEqual((2,), np.array(results['out1']).shape)
        self.TestAlmostEqual([15,56], results['out1'])
        self.assertEqual((2,), np.array(results['out2']).shape)
        self.TestAlmostEqual([-2, -4], results['out2'])
        self.assertEqual((2, 1, 2), np.array(results['out3']).shape)
        self.TestAlmostEqual([[[-2,1], [-4,2]]], results['out3'])
        self.assertEqual((2, 1, 2), np.array(results['out4']).shape)
        self.TestAlmostEqual([[[6.0, -2.0]], [[10.0, 0.0]]], results['out4'])
        self.assertEqual((2, 1, 2), np.array(results['out5']).shape)
        self.TestAlmostEqual([[[2.0, 0.0]], [[2.0, 1.0]]], results['out5'])
    
    def test_single_in_window_offset_parametric_function(self):
        # An input dimension is temporal and does not remain unchanged unless redefined on output
        # If there are multiple inputs the function returns an error if the dimensions are not defined
        in1 = Input('in1')
        parfun = ParamFun(myfun)
        out = Output('out', parfun(in1.last()))
        test = Neu4mes(visualizer = None, seed = 1)
        test.addModel('out',out)
        test.neuralizeModel(0.1)

        #results = test({'in1': 1})   
        #self.TestAlmostEqual(results['out'], [0.7576315999031067])
        results = test({'in1': [[1]]})
        self.assertEqual((1,), np.array(results['out']).shape)
        self.TestAlmostEqual(results['out'],[0.7576315999031067])
        results = test({'in1': [2]})
        self.TestAlmostEqual(results['out'],[1.5152631998062134])
        results = test({'in1': [1,2]})
        self.assertEqual((2,), np.array(results['out']).shape)
        self.TestAlmostEqual(results['out'],[0.7576315999031067,1.5152631998062134])

        out = Output('out', ParamFun(myfun)(in1.tw(0.2)))
        test = Neu4mes(visualizer=None, seed = 1)
        test.addModel('out',out)
        test.neuralizeModel(0.1)

        with self.assertRaises(StopIteration):
            results = test({'in1': [1]})
        with self.assertRaises(StopIteration):
            results = test({'in1': [2]})
        results = test({'in1': [1,2]})
        self.assertEqual((1,2), np.array(results['out']).shape)
        self.TestAlmostEqual(results['out'], [[0.7576315999031067, 1.5152631998062134]])
        results = test({'in1': [[1,2]]}, sampled=True)
        self.assertEqual((1,2), np.array(results['out']).shape)
        self.TestAlmostEqual(results['out'], [[0.7576315999031067, 1.5152631998062134]])
        results = test({'in1': [1, 2, 3, 4, 5]})# Qui vengono costruite gli input a due a due con shift di 1
        self.assertEqual((4,2), np.array(results['out']).shape)
        self.TestAlmostEqual(results['out'], [[0.7576315999031067, 1.5152631998062134], [1.5152631998062134, 2.272894859313965], [2.272894859313965, 3.0305263996124268], [3.0305263996124268, 3.7881579399108887]])
        results = test({'in1': [[1, 2], [2, 3], [3, 4], [4, 5]]}, sampled=True)
        self.assertEqual((4,2), np.array(results['out']).shape)
        self.TestAlmostEqual(results['out'], [[0.7576315999031067, 1.5152631998062134], [1.5152631998062134, 2.272894859313965], [2.272894859313965, 3.0305263996124268], [3.0305263996124268, 3.7881579399108887]])

        out = Output('out', ParamFun(myfun)(in1.last(),in1.last()))
        test = Neu4mes(visualizer=None, seed = 1)
        test.addModel('out',out)
        test.neuralizeModel(0.1)

        #results = test({'in1': 1})
        #self.TestAlmostEqual(results['out'],[1])
        results = test({'in1': [1]})
        self.TestAlmostEqual(results['out'],[1])
        results = test({'in1': [2]})
        self.TestAlmostEqual(results['out'],[4])
        results = test({'in1': [1,2]})
        self.TestAlmostEqual(results['out'],[1,4])

        out = Output('out', ParamFun(myfun)(in1.tw(0.1), in1.tw(0.1)))
        test = Neu4mes(visualizer=None)
        test.addModel('out',out)
        test.neuralizeModel(0.1)

        #results = test({'in1': 2})
        #self.TestAlmostEqual(results['out'], [4])
        results = test({'in1': [2]})
        self.TestAlmostEqual(results['out'], [4])
        results = test({'in1': [2,1]})
        self.TestAlmostEqual(results['out'], [4,1])

        out = Output('out', ParamFun(myfun)(in1.tw(0.2), in1.tw(0.2)))
        test = Neu4mes(visualizer=None)
        test.addModel('out',out)
        test.neuralizeModel(0.1)

        results = test({'in1': [2,4]})
        self.assertEqual((1,2), np.array(results['out']).shape)
        self.TestAlmostEqual(results['out'], [[4,16]])
        results = test({'in1': [[1, 2], [3, 2]]}, sampled=True)
        self.assertEqual((2,2), np.array(results['out']).shape)
        self.TestAlmostEqual(results['out'], [[1.0, 4.0], [9.0, 4.0]])

        out = Output('out', ParamFun(myfun)(in1.tw(0.3), in1.tw(0.3)))
        test = Neu4mes(visualizer=None)
        test.addModel('out',out)
        test.neuralizeModel(0.1)

        with self.assertRaises(StopIteration):
            test({'in1': [2]})

        with self.assertRaises(StopIteration):
            test({'in1': [2, 4]})

        results = test({'in1': [3,2,1]})
        self.assertEqual((1,3), np.array(results['out']).shape)
        self.TestAlmostEqual(results['out'], [[9,4,1]])
        results = test({'in1': [[1,2,2],[3,4,5]]}, sampled=True)
        self.assertEqual((2,3), np.array(results['out']).shape)
        self.TestAlmostEqual(results['out'], [[1, 4, 4], [9, 16, 25]])
        results = test({'in1': [[3, 2, 1], [2, 1, 0]]}, sampled=True)
        self.assertEqual((2,3), np.array(results['out']).shape)
        self.TestAlmostEqual(results['out'], [[9, 4, 1],[4, 1, 0]])
        results = test({'in1': [3,2,1,0]})
        self.assertEqual((2,3), np.array(results['out']).shape)
        self.TestAlmostEqual(results['out'], [[9, 4, 1],[4, 1, 0]])
        
        out = Output('out', ParamFun(myfun)(in1.tw(0.4), in1.tw(0.4)))
        test = Neu4mes(visualizer=None)
        test.addModel('out',out)
        test.neuralizeModel(0.1)
        with self.assertRaises(StopIteration):
            test({'in1': [[1, 2, 2], [3, 4, 5]]})
    
    def test_vectorial_input_parametric_function(self):
        # Vector input for parametric function
        in1 = Input('in1', dimensions=3)
        in2 = Input('in2', dimensions=2)
        p1 = Parameter('p1', dimensions=(3,2),values=[[[1,2],[3,4],[5,6]]])
        p2 = Parameter('p2', dimensions=3,values=[[1,2,3]])
        parfun = ParamFun(myfun3, parameters=[p1,p2])
        out = Output('out', parfun(in1.last(),in2.last()))
        test = Neu4mes(visualizer = None, seed = 1)
        test.addModel('out',out)
        test.neuralizeModel(0.1)

        results = test({'in1': [[1,2,3]],'in2':[[5,6]]})
        self.assertEqual((1,3), np.array(results['out']).shape)
        self.TestAlmostEqual(results['out'], [[23,52,81]])

        results = test({'in1': [[1,2,3],[5,6,7]],'in2':[[5,6],[7,8]]})
        self.assertEqual((2,3), np.array(results['out']).shape)
        self.TestAlmostEqual(results['out'], [[23,52,81],[41,94,147]])
    
    def test_parametric_function_and_fir(self):
        torch.manual_seed(1)
        in1 = Input('in1')
        in2 = Input('in2')
        out = Output('out', Fir(ParamFun(myfun2)(in1.tw(0.4))))
        test = Neu4mes(visualizer = None)
        test.addModel('out',out)
        test.neuralizeModel(0.1)

        with self.assertRaises(StopIteration):
            test({'in1': [1, 2, 2]})
        results = test({'in1': [[1], [2], [2], [4]]})
        self.TestAlmostEqual(results['out'][0], 0.47691094875335693)
        results = test({'in1': [[[1], [2], [2], [4]],[[2], [2], [4], [5]]]}, sampled=True)
        self.TestAlmostEqual(results['out'], [0.47691094875335693, 0.12529167532920837])
        results = test({'in1': [[1], [2], [2], [4], [5]]})
        self.TestAlmostEqual(results['out'], [0.47691094875335693, 0.12529167532920837])

        with self.assertRaises(RuntimeError):
            Output('out', Fir(ParamFun(myfun2)(in1.tw(0.4),in2.tw(0.2))))

        in1 = Input('in1')
        in2 = Input('in2')
        out = Output('out', Fir(ParamFun(myfun2)(in1.tw(0.4),in2.tw(0.4))))
        test = Neu4mes(visualizer=None)
        test.addModel('out',out)
        test.neuralizeModel(0.1)
        with self.assertRaises(StopIteration): ## TODO: change to KeyError when checking the inputs
            test({'in1': [[1, 2, 2, 4]]})
        results = test({'in1': [1, 2, 2, 4], 'in2': [1, 2, 2, 4]})

        self.TestAlmostEqual(results['out'], [0.5426889061927795])
        results = test({'in1': [[1, 2, 2, 4]], 'in2': [[1, 2, 2, 4]]}, sampled=True)
        self.TestAlmostEqual(results['out'], [0.5426889061927795])
        results = test({'in1': [1, 2, 2, 4, 5], 'in2': [1, 2, 2, 4]})
        self.TestAlmostEqual(results['out'], [0.5426889061927795])
        results = test({'in1': [[1, 2, 2, 4], [1, 2, 2, 4]], 'in2': [[1, 2, 2, 4]]}, sampled=True)
        self.TestAlmostEqual(results['out'], [0.5426889061927795])
        results = test({'in1': [[1, 2, 2, 4], [1, 2, 2, 4]], 'in2': [[1, 2, 2, 4], [5, 5, 5, 5]]}, sampled=True)
        self.TestAlmostEqual(results['out'], [0.5426889061927795,  0.6200160384178162])

        self.TestAlmostEqual(results['out'], [0.5426889061927795])
        results = test({'in1': [[1, 2, 2, 4]], 'in2': [[1, 2, 2, 4]]}, sampled=True)
        self.TestAlmostEqual(results['out'], [0.5426889061927795])
        results = test({'in1': [1, 2, 2, 4, 5], 'in2': [1, 2, 2, 4]})
        self.TestAlmostEqual(results['out'], [0.5426889061927795])
        results = test({'in1': [[1, 2, 2, 4], [1, 2, 2, 4]], 'in2': [[1, 2, 2, 4]]}, sampled=True)
        self.TestAlmostEqual(results['out'], [0.5426889061927795])
        results = test({'in1': [[1, 2, 2, 4], [1, 2, 2, 4]], 'in2': [[1, 2, 2, 4], [5, 5, 5, 5]]}, sampled=True)
        self.TestAlmostEqual(results['out'], [0.5426889061927795,  0.6200160384178162])

        in1 = Input('in1')
        in2 = Input('in2')
        out = Output('out', Fir(3)(ParamFun(myfun2)(in1.tw(0.4), in2.tw(0.4))))
        test = Neu4mes(visualizer=None)
        test.addModel('out',out)
        test.neuralizeModel(0.1)

        results = test({'in1': [[1, 2, 2, 4], [1, 2, 2, 4]], 'in2': [[1, 2, 2, 4], [1, 2, 2, 4]]}, sampled=True)
        self.assertEqual((2,1,3), np.array(results['out']).shape)
        self.TestAlmostEqual(results['out'], [[[-0.03303150087594986, 0.023659050464630127, 0.0185492392629385]], [[-0.03303150087594986, 0.023659050464630127, 0.0185492392629385]]])

        parfun = ParamFun(myfun2)
        with self.assertRaises(ValueError):
            Output('out', parfun(Fir(3)(parfun(in1.tw(0.4), in2.tw(0.4)))))

        parfun = ParamFun(myfun2)
        out = Output('out', parfun(Fir(3)(parfun(in1.tw(0.4)))))
        test = Neu4mes(visualizer=None)
        test.addModel('out',out)
        test.neuralizeModel(0.1)

        results = test({'in1': [[1, 2, 2, 4], [2, 1, 1, 3]]}, sampled=True)
        self.assertEqual((2,1,3), np.array(results['out']).shape)
        self.TestAlmostEqual(results['out'], [[[0.548146665096283, 0.6267049908638, 0.5457861423492432]], [[0.6165860891342163, 0.6518347859382629, 0.6502876281738281]]])

        results = test({'in1': [1, 2, 2, 4, 3],'in2': [6, 2, 2, 4, 4]})
        self.assertEqual((2,1,3), np.array(results['out']).shape)
        self.TestAlmostEqual(results['out'], [[[0.548146665096283, 0.6267049908638, 0.5457861423492432]], [[0.2763473391532898, 0.45648545026779175, 0.4165944457054138]]])
        results = test({'in1': [[1, 2, 2, 4],[2, 2, 4, 3]],'in2': [[6, 2, 2, 4],[2, 2, 4, 4]]}, sampled=True)
        self.assertEqual((2,1,3), np.array(results['out']).shape)
        self.TestAlmostEqual(results['out'], [[[0.548146665096283, 0.6267049908638, 0.5457861423492432]], [[0.2763473391532898, 0.45648545026779175, 0.4165944457054138]]])
    
    def test_trigonometri_parameter_and_numeric_constant(self):
        torch.manual_seed(1)
        in1 = Input('in1').last()
        par = Parameter('par', values=[[5]])
        in4 = Input('in4', dimensions=4).last()
        par4 = Parameter('par4', values=[[1,2,3,4]])
        add = in1 + par + 5.2
        sub = in1 - par - 5.2
        mul = in1 * par * 5.2
        div = in1 / par / 5.2
        pow = in1 ** par ** 2
        sin1 = Sin(par) + Sin(5.2)
        cos1 = Cos(par) + Cos(5.2)
        tan1 = Tan(par) + Tan(5.2)
        relu1 = Relu(par) + Relu(5.2)
        tanh1 = Tanh(par) + Tanh(5.2)
        tot1 = add + sub + mul + div + pow + sin1 + cos1 + tan1 + relu1 + tanh1

        add4 = in4 + par4 + 5.2
        sub4 = in4 - par4 - 5.2
        mul4 = in4 * par4 * 5.2
        div4 = in4 / par4 / 5.2
        pow4 = in4 ** par4 ** 2
        sin4 = Sin(par4) + Sin(5.2)
        cos4 = Cos(par4) + Cos(5.2)
        tan4 = Tan(par4) + Tan(5.2)
        relu4 = Relu(par4) + Relu(5.2)
        tanh4 = Tanh(par4) + Tanh(5.2)
        tot4 = add4 + sub4 + mul4 + div4 + pow4 + sin4 + cos4 + tan4 + relu4 + tanh4
        out1 = Output('out1', tot1)
        out4 = Output('out4', tot4)
        linW = Parameter('linW',dimensions=(4,1),values=[[[1],[1],[1],[1]]])
        outtot = Output('outtot', tot1 + Linear(W=linW)(tot4))
        test = Neu4mes(visualizer=None)
        test.addModel('out',[out1,out4,outtot])
        test.neuralizeModel()

        results = test({'in1': [1, 2, -2],'in4': [[6, 2, 2, 4], [7, 2, 2, 4], [-6, -5, 5, 4]]})
        self.assertEqual((3,), np.array(results['out1']).shape)
        self.assertEqual((3,1,4), np.array(results['out4']).shape)
        self.assertEqual((3,), np.array(results['outtot']).shape)

        self.TestAlmostEqual([34.8819529, 33554496.0,  -33554480.0], results['out1'] )
        self.TestAlmostEqual([[[58.9539756, 46.1638031, 554.231201171875, 4294967296.0]], [[67.3462829589843, 46.16380310058594, 554.231201171875, 4294967296.0]], [[ -41.75371170043945, 567.6907348632812, 1953220.375, 4294967296.0]]], results['out4'])
        self.TestAlmostEqual([4294967808.0, 4328522240.0,  4263366656.0], results['outtot'])

    def test_parameter_and_linear(self):
        torch.manual_seed(1)
        input = Input('in').last()
        W15 = Parameter('W15', dimensions=(1, 5), values=[[[1,2,3,4,5]]])
        b15 = Parameter('b15', dimensions=5, values=[[1,2,3,4,5]])
        input4 = Input('in4',dimensions=4).last()
        W45 = Parameter('W45', dimensions=(4, 5), values=[[[1,2,3,4,5],[5,3,3,4,5],[1,2,3,4,7],[-8,2,3,4,5]]])
        b45 = Parameter('b45', dimensions=5, values=[[5,2,3,4,5]])

        o = Output('out' , Linear(input) + Linear(input4))
        o3 = Output('out3' , Linear(3)(input) + Linear(3)(input4))
        oW = Output('outW' , Linear(W = W15)(input) + Linear(W = W45)(input4))
        oWb = Output('outWb' , Linear(W = W15,b = b15)(input) + Linear(W = W45, b = b45)(input4))

        n = Neu4mes(visualizer=None)
        n.addModel('out',[o,o3,oW,oWb])
        n.neuralizeModel()
        results = n({'in': [1, 2], 'in4': [[6, 2, 2, 4], [7, 2, 2, 4]]})
        self.assertEqual((2,), np.array(results['out']).shape)
        self.TestAlmostEqual([9.274794578552246,10.3853759765625], results['out'])
        self.assertEqual((2,1,3), np.array(results['out3']).shape)
        self.TestAlmostEqual([[[9.247159004211426, 6.103044033050537,7.719359397888184]],[[10.68740463256836, 6.687504291534424,8.585973739624023]]], results['out3'])
        #W15 = torch.tensor([[1,2,3,4,5]])
        #in1 = torch.tensor([[1, 2]])
        #W45 = torch.tensor([[1, 2, 3, 4, 5], [5, 3, 3, 4, 5], [1, 2, 3, 4, 7], [-8, 2, 3, 4, 5]])
        #in4 = torch.tensor([[6, 2, 2, 4], [7, 2, 2, 4]])
        #torch.matmul(W45.t(),in4.t())+torch.matmul(W15.t(),in1)
        self.assertEqual((2, 1, 5), np.array(results['outW']).shape)
        self.TestAlmostEqual([[[-13.0,32.0,45.0,60.0,79.0]],[[-11.0,36.0,51.,68.,89.]]], results['outW'])
        # W15 = torch.tensor([[1,2,3,4,5]])
        # b15 = torch.tensor([[5, 2, 3, 4, 5]])
        # in1 = torch.tensor([[1, 2]])
        # W45 = torch.tensor([[1, 2, 3, 4, 5], [5, 3, 3, 4, 5], [1, 2, 3, 4, 7], [-8, 2, 3, 4, 5]])
        # b45 = torch.tensor([[1, 2, 3, 4, 5]])
        # in4 = torch.tensor([[6, 2, 2, 4], [7, 2, 2, 4]])
        # oo = torch.matmul(W45.t(),in4.t())+b45.t()+torch.matmul(W15.t(),in1)+b15.t()
        self.assertEqual((2, 1, 5), np.array(results['outWb']).shape)
        self.TestAlmostEqual([[[-7, 36, 51, 68, 89]],[[-5, 40, 57, 76, 99]]], results['outWb'])

        input2 = Input('in').sw([-1,1])
        input42 = Input('in4', dimensions=4).sw([-1,1])

        o = Output('out' , Linear(input2) + Linear(input42))
        o3 = Output('out3' , Linear(3)(input2) + Linear(3)(input42))
        oW = Output('outW' , Linear(W = W15)(input2) + Linear(W = W45)(input42))
        oWb = Output('outWb' , Linear(W = W15,b = b15)(input2) + Linear(W = W45, b = b45)(input42))
        n = Neu4mes(visualizer=None)
        n.addModel('out',[o,o3,oW,oWb])
        n.neuralizeModel()
        results = n({'in': [1, 2], 'in4': [[6, 2, 2, 4], [7, 2, 2, 4]]})
        self.assertEqual((1, 2), np.array(results['out']).shape)
        self.TestAlmostEqual([[9.276955604553223, 10.290032386779785]], results['out'])
        self.assertEqual((1, 2, 3), np.array(results['out3']).shape)
        self.TestAlmostEqual([[[12.529109954833984, 6.662200927734375, 10.58283519744873],
                             [14.46115779876709, 7.2378997802734375, 11.927245140075684]]], results['out3'])
        self.assertEqual((1, 2, 5), np.array(results['outW']).shape)
        self.TestAlmostEqual([[[-13.0, 32.0, 45.0, 60.0, 79.0], [-11.0, 36.0, 51., 68., 89.]]], results['outW'])
        self.assertEqual((1, 2, 5), np.array(results['outWb']).shape)
        self.TestAlmostEqual([[[-7, 36, 51, 68, 89], [-5, 40, 57, 76, 99]]], results['outWb'])

    def test_initialization(self):
        torch.manual_seed(1)
        input = Input('in')
        W = Parameter('W', dimensions=(1,1), init=init_constant)
        b = Parameter('b', dimensions=1, init=init_constant)
        o = Output('out', Linear(W=W,b=b)(input.last()))

        W5 = Parameter('W5', dimensions=(1,1), init=init_constant, init_params={'value':5})
        b2 = Parameter('b2', dimensions=1, init=init_constant, init_params={'value':2})
        o52 = Output('out52', Linear(W=W5,b=b2)(input.last()))

        par = Parameter('par', dimensions=3, sw=2, init=init_constant)
        opar = Output('outpar', Fir(parameter=par)(input.sw(2)))

        par2 = Parameter('par2', dimensions=3, sw=2, init=init_constant, init_params={'value':2})
        opar2 = Output('outpar2', Fir(parameter=par2)(input.sw(2)))

        ol = Output('outl', Linear(output_dimension=1,b=True,W_init=init_constant,b_init=init_constant)(input.last()))
        ol52 = Output('outl52', Linear(output_dimension=1,b=True,W_init=init_constant,b_init=init_constant,W_init_params={'value':5},b_init_params={'value':2})(input.last()))
        ofpar = Output('outfpar', Fir(output_dimension=3,parameter_init=init_constant)(input.sw(2)))
        ofpar2 = Output('outfpar2', Fir(output_dimension=3,parameter_init=init_constant,parameter_init_params={'value':2})(input.sw(2)))

        n = Neu4mes(visualizer=None)
        n.addModel('model',[o,o52,opar,opar2,ol,ol52,ofpar,ofpar2])
        n.neuralizeModel()
        results = n({'in': [1, 1, 2]})
        self.assertEqual((2,), np.array(results['out']).shape)
        self.TestAlmostEqual([2,3], results['out'])
        self.assertEqual((2,), np.array(results['out52']).shape)
        self.TestAlmostEqual([7,12], results['out52'])

        self.assertEqual((2,1,3), np.array(results['outpar']).shape)
        self.TestAlmostEqual([[[2,2,2]],[[3,3,3]]], results['outpar'])
        self.assertEqual((2,1,3), np.array(results['outpar2']).shape)
        self.TestAlmostEqual([[[4,4,4]],[[6,6,6]]], results['outpar2'])

        self.assertEqual((2,), np.array(results['outl']).shape)
        self.TestAlmostEqual([2,3], results['outl'])
        self.assertEqual((2,), np.array(results['outl52']).shape)
        self.TestAlmostEqual([7.0,12.0], results['outl52'])

        self.assertEqual((2,1,3), np.array(results['outfpar']).shape)
        self.TestAlmostEqual([[[2,2,2]],[[3,3,3]]], results['outfpar'])
        self.assertEqual((2,1,3), np.array(results['outfpar2']).shape)
        self.TestAlmostEqual([[[4,4,4]],[[6,6,6]]], results['outfpar2'])

    def test_sample_part_and_select(self):
        in1 = Input('in1')
        # Offset before the sample window
        with self.assertRaises(IndexError):
            in1.sw([-5, -2], offset=-6)
        # Offset after the sample window
        with self.assertRaises(IndexError):
            in1.sw([-5, -2], offset=-2)
        # Offset before the sample window
        with self.assertRaises(IndexError):
            in1.sw([0, 3], offset=-1)
        # Offset after the sample window
        with self.assertRaises(IndexError):
            in1.sw([0, 3], offset=3)

        sw3,sw32 = in1.sw([-5, -2], offset=-4), in1.sw([0, 3], offset=0)
        out_sw3 = Output('in_sw3', sw3)
        out_sw32 = Output('in_sw32', sw32)
        #Get after the window
        with self.assertRaises(ValueError):
            SamplePart(sw3, 0, 4)
        #Empty sample window
        with self.assertRaises(ValueError):
            SamplePart(sw3, 0, 0)
        #Get before the sample window
        with self.assertRaises(ValueError):
            SamplePart(sw3, -1, 0)
        # Get after the window
        with self.assertRaises(ValueError):
            SamplePart(sw32, 0, 4)
        #Empty sample window
        with self.assertRaises(ValueError):
            SamplePart(sw32, 0, 0)
        #Get before the sample window
        with self.assertRaises(ValueError):
            SamplePart(sw32, -1, 0)

        # Offset before the sample window
        with self.assertRaises(IndexError):
            SamplePart(sw3, 0, 3, offset=-1)
        # Offset after the sample window
        with self.assertRaises(IndexError):
            SamplePart(sw3, 0, 3, offset=3)
        # Offset before the sample window
        with self.assertRaises(IndexError):
            SamplePart(sw32, 0, 3, offset=-1)
        # Offset after the sample window
        with self.assertRaises(IndexError):
            SamplePart(sw32, 0, 3, offset=3)
        in_SP1first = Output('in_SP1first', SamplePart(sw3, 0, 1))
        in_SP1mid = Output('in_SP1mid', SamplePart(sw3, 1, 2))
        in_SP1last = Output('in_SP1last', SamplePart(sw3, 2, 3))
        in_SP1all = Output('in_SP1all', SamplePart(sw3, 0, 3))
        in_SP1off1 = Output('in_SP1off1', SamplePart(sw3, 0, 3, offset=0))
        in_SP1off2 = Output('in_SP1off2', SamplePart(sw3, 0, 3, offset=1))
        in_SP1off3 = Output('in_SP1off3', SamplePart(sw3, 0, 3, offset=2))
        with self.assertRaises(ValueError):
            SampleSelect(sw3, -1)
        with self.assertRaises(ValueError):
            SampleSelect(sw3, 3)
        with self.assertRaises(ValueError):
            SampleSelect(sw32, -1)
        with self.assertRaises(ValueError):
            SampleSelect(sw32, 3)
        in_SS1 = Output('in_SS1', SampleSelect(sw3, 0))
        in_SS2 = Output('in_SS2', SampleSelect(sw3, 1))
        in_SS3 = Output('in_SS3', SampleSelect(sw3, 2))
        test = Neu4mes(visualizer=None)
        test.addModel('out',[out_sw3, out_sw32,
                       in_SP1first, in_SP1mid, in_SP1last, in_SP1all, in_SP1off1, in_SP1off2, in_SP1off3,
                       in_SS1, in_SS2, in_SS3])
        test.neuralizeModel()
        results = test({'in1': [0, 1, 2, 3, 4, 5, 6, 7]})

        self.assertEqual((1, 3), np.array(results['in_sw3']).shape)
        self.TestAlmostEqual([[-1,0,1]], results['in_sw3'])
        self.assertEqual((1, 3), np.array(results['in_sw32']).shape)
        self.TestAlmostEqual([[0,1,2]], results['in_sw32'])

        self.assertEqual((1,), np.array(results['in_SP1first']).shape)
        self.TestAlmostEqual([-1], results['in_SP1first'])
        self.assertEqual((1,), np.array(results['in_SP1mid']).shape)
        self.TestAlmostEqual([0], results['in_SP1mid'])
        self.assertEqual((1,), np.array(results['in_SP1last']).shape)
        self.TestAlmostEqual([1], results['in_SP1last'])
        self.assertEqual((1,3), np.array(results['in_SP1all']).shape)
        self.TestAlmostEqual([[-1,0,1]], results['in_SP1all'])
        self.assertEqual((1,3), np.array(results['in_SP1off1']).shape)
        self.TestAlmostEqual([[0,1,2]], results['in_SP1off1'])
        self.assertEqual((1,3), np.array(results['in_SP1off2']).shape)
        self.TestAlmostEqual([[-1,0,1]], results['in_SP1off2'])
        self.assertEqual((1,3), np.array(results['in_SP1off3']).shape)
        self.TestAlmostEqual([[-2,-1,0]], results['in_SP1off3'])

        self.assertEqual((1,), np.array(results['in_SS1']).shape)
        self.TestAlmostEqual([-1], results['in_SS1'])
        self.assertEqual((1,), np.array(results['in_SS2']).shape)
        self.TestAlmostEqual([0], results['in_SS2'])
        self.assertEqual((1,), np.array(results['in_SS3']).shape)
        self.TestAlmostEqual([1], results['in_SS3'])

        results = test({'in1': [0, 1, 2, 3, 4, 5, 6, 7, 10]})

        self.assertEqual((2, 3), np.array(results['in_sw3']).shape)
        self.TestAlmostEqual([[-1, 0, 1],[-1, 0, 1]], results['in_sw3'])
        self.assertEqual((2, 3), np.array(results['in_sw32']).shape)
        self.TestAlmostEqual([[0, 1, 2],[0, 1, 4]], results['in_sw32'])

        self.assertEqual((2,), np.array(results['in_SP1first']).shape)
        self.TestAlmostEqual([-1,-1], results['in_SP1first'])
        self.assertEqual((2,), np.array(results['in_SP1mid']).shape)
        self.TestAlmostEqual([0,0], results['in_SP1mid'])
        self.assertEqual((2,), np.array(results['in_SP1last']).shape)
        self.TestAlmostEqual([1,1], results['in_SP1last'])
        self.assertEqual((2, 3), np.array(results['in_SP1all']).shape)
        self.TestAlmostEqual([[-1, 0, 1],[-1, 0, 1]], results['in_SP1all'])
        self.assertEqual((2, 3), np.array(results['in_SP1off1']).shape)
        self.TestAlmostEqual([[0, 1, 2],[0, 1, 2]], results['in_SP1off1'])
        self.assertEqual((2, 3), np.array(results['in_SP1off2']).shape)
        self.TestAlmostEqual([[-1, 0, 1],[-1, 0, 1]], results['in_SP1off2'])
        self.assertEqual((2, 3), np.array(results['in_SP1off3']).shape)
        self.TestAlmostEqual([[-2, -1, 0],[-2, -1, 0]], results['in_SP1off3'])

        self.assertEqual((2,), np.array(results['in_SS1']).shape)
        self.TestAlmostEqual([-1,-1], results['in_SS1'])
        self.assertEqual((2,), np.array(results['in_SS2']).shape)
        self.TestAlmostEqual([0,0], results['in_SS2'])
        self.assertEqual((2,), np.array(results['in_SS3']).shape)
        self.TestAlmostEqual([1,1], results['in_SS3'])

    def test_time_part(self):
        in1 = Input('in1')
        # Offset before the time window
        with self.assertRaises(IndexError):
            in1.tw([-5, -2], offset=-6)
        # Offset after the time window
        with self.assertRaises(IndexError):
            in1.tw([-5, -2], offset=-2)
        # Offset before the time window
        with self.assertRaises(IndexError):
            in1.tw([0, 3], offset=-1)
        # Offset after the time window
        with self.assertRaises(IndexError):
            in1.tw([0, 3], offset=3)

        tw3, tw32 = in1.tw([-5, -2], offset=-4), in1.tw([0, 3], offset=0)
        out_tw3 = Output('in_tw3', tw3)
        out_tw32 = Output('in_tw32', tw32)
        # Get after the window
        with self.assertRaises(ValueError):
            TimePart(tw3, 0, 4)
        # Empty time window
        with self.assertRaises(ValueError):
            TimePart(tw3, 0, 0)
        # Get before the time window
        with self.assertRaises(ValueError):
            TimePart(tw3, -1, 0)
        # Get after the window
        with self.assertRaises(ValueError):
            TimePart(tw32, 0, 4)
        # Empty sample window
        with self.assertRaises(ValueError):
            TimePart(tw32, 0, 0)
        # Get before the time window
        with self.assertRaises(ValueError):
            TimePart(tw32, -1, 0)

        # Offset before the time window
        with self.assertRaises(IndexError):
            TimePart(tw3, 0, 3, offset=-1)
        # Offset after the time window
        with self.assertRaises(IndexError):
            TimePart(tw3, 0, 3, offset=3)
        # Offset before the time window
        with self.assertRaises(IndexError):
            TimePart(tw32, 0, 3, offset=-1)
        # Offset after the time window
        with self.assertRaises(IndexError):
            TimePart(tw32, 0, 3, offset=3)

        in_TP1first = Output('in_TP1first', TimePart(tw32, 0, 1))
        in_TP1mid = Output('in_TP1mid', TimePart(tw32, 1, 2))
        in_TP1last = Output('in_TP1last', TimePart(tw32, 2, 3))
        in_TP1all = Output('in_TP1all', TimePart(tw32, 0, 3))
        in_TP1off1 = Output('in_TP1off1', TimePart(tw32, 0, 3, offset=0))
        in_TP1off2 = Output('in_TP1off2', TimePart(tw32, 0, 3, offset=1))
        in_TP1off3 = Output('in_TP1off3', TimePart(tw32, 0, 3, offset=2))

        test = Neu4mes(visualizer=None)
        test.addModel('out',[out_tw3, out_tw32,
                       in_TP1first, in_TP1mid, in_TP1last, in_TP1all, in_TP1off1, in_TP1off2, in_TP1off3])
        test.neuralizeModel()
        results = test({'in1': [0, 1, 2, 3, 4, 5, 6, 7]})

        self.assertEqual((1, 3), np.array(results['in_tw3']).shape)
        self.TestAlmostEqual([[-1, 0, 1]], results['in_tw3'])
        self.assertEqual((1, 3), np.array(results['in_tw32']).shape)
        self.TestAlmostEqual([[0, 1, 2]], results['in_tw32'])

        self.assertEqual((1,), np.array(results['in_TP1first']).shape)
        self.TestAlmostEqual([0], results['in_TP1first'])
        self.assertEqual((1,), np.array(results['in_TP1mid']).shape)
        self.TestAlmostEqual([1], results['in_TP1mid'])
        self.assertEqual((1,), np.array(results['in_TP1last']).shape)
        self.TestAlmostEqual([2], results['in_TP1last'])
        self.assertEqual((1, 3), np.array(results['in_TP1all']).shape)
        self.TestAlmostEqual([[0, 1, 2]], results['in_TP1all'])
        self.assertEqual((1, 3), np.array(results['in_TP1off1']).shape)
        self.TestAlmostEqual([[0, 1, 2]], results['in_TP1off1'])
        self.assertEqual((1, 3), np.array(results['in_TP1off2']).shape)
        self.TestAlmostEqual([[-1, 0, 1]], results['in_TP1off2'])
        self.assertEqual((1, 3), np.array(results['in_TP1off3']).shape)
        self.TestAlmostEqual([[-2, -1, 0]], results['in_TP1off3'])

        results = test({'in1': [0, 1, 2, 3, 4, 5, 6, 7, 10]})

        self.assertEqual((2, 3), np.array(results['in_tw3']).shape)
        self.TestAlmostEqual([[-1, 0, 1], [-1, 0, 1]], results['in_tw3'])
        self.assertEqual((2, 3), np.array(results['in_tw32']).shape)
        self.TestAlmostEqual([[0, 1, 2], [0, 1, 4]], results['in_tw32'])

        self.assertEqual((2,), np.array(results['in_TP1first']).shape)
        self.TestAlmostEqual([0, 0], results['in_TP1first'])
        self.assertEqual((2,), np.array(results['in_TP1mid']).shape)
        self.TestAlmostEqual([1, 1], results['in_TP1mid'])
        self.assertEqual((2,), np.array(results['in_TP1last']).shape)
        self.TestAlmostEqual([2, 4], results['in_TP1last'])
        self.assertEqual((2, 3), np.array(results['in_TP1all']).shape)
        self.TestAlmostEqual([[0, 1, 2], [0, 1, 4]], results['in_TP1all'])
        self.assertEqual((2, 3), np.array(results['in_TP1off1']).shape)
        self.TestAlmostEqual([[0, 1, 2], [0, 1, 4]], results['in_TP1off1'])
        self.assertEqual((2, 3), np.array(results['in_TP1off2']).shape)
        self.TestAlmostEqual([[-1, 0, 1], [-1, 0, 3]], results['in_TP1off2'])
        self.assertEqual((2, 3), np.array(results['in_TP1off3']).shape)
        self.TestAlmostEqual([[-2, -1, 0], [-4, -3, 0]], results['in_TP1off3'])
    
    def test_part_and_select(self):
        in1 = Input('in1',dimensions=4)

        tw3, tw32 = in1.tw([-5, -2], offset=-4), in1.tw([0, 3], offset=0)
        out_tw3 = Output('in_tw3', tw3)
        out_tw32 = Output('in_tw32', tw32)
        # Get after the window
        with self.assertRaises(IndexError):
            Part(tw3, 0, 5)
        # Empty time window
        with self.assertRaises(IndexError):
            Part(tw3, 0, 0)
        # Get before the time window
        with self.assertRaises(IndexError):
            Part(tw3, -1, 0)
        # Get after the window
        with self.assertRaises(IndexError):
            Part(tw32, 0, 5)
        # Empty sample window
        with self.assertRaises(IndexError):
            Part(tw32, 0, 0)
        # Get before the time window
        with self.assertRaises(IndexError):
            Part(tw32, -1, 0)

        in_P1first = Output('in_P1first', Part(tw32, 0, 1))
        in_P1mid = Output('in_P1mid', Part(tw32, 1, 2))
        in_P1last = Output('in_P1last', Part(tw32, 2, 4))
        in_P1all = Output('in_P1all', Part(tw32, 0, 4))

        with self.assertRaises(IndexError):
            Select(tw3, -1)
        with self.assertRaises(IndexError):
            Select(tw3, 4)
        with self.assertRaises(IndexError):
            Select(tw32, -1)
        with self.assertRaises(IndexError):
            Select(tw32, 4)
        in_S1 = Output('in_S1', Select(tw3, 0))
        in_S2 = Output('in_S2', Select(tw3, 1))
        in_S3 = Output('in_S3', Select(tw3, 2))
        in_S4 = Output('in_S4', Select(tw3, 3))

        test = Neu4mes(visualizer=None)
        test.addModel('out',[out_tw3, out_tw32,
                       in_P1first, in_P1mid, in_P1last, in_P1all,
                       in_S1, in_S2, in_S3, in_S4])
        test.neuralizeModel()
        results = test({'in1': [[0,1,2,4], [1,3,4,5], [2,5,6,7], [3,3,4,1], [4,4,6,7], [5,6,7,8], [6,7,5,4],[7,2,3,1]]})

        self.assertEqual((1, 3, 4), np.array(results['in_tw3']).shape)
        self.TestAlmostEqual([[[-1,-2,-2,-1], [0,0,0,0], [1,2,2,2]]], results['in_tw3'])
        self.assertEqual((1, 3, 4), np.array(results['in_tw32']).shape)
        self.TestAlmostEqual([[[0,0,0,0], [1,1,-2,-4],[2,-4,-4,-7]]], results['in_tw32'])

        self.assertEqual((1,3), np.array(results['in_P1first']).shape)
        self.TestAlmostEqual([[0,1,2]], results['in_P1first'])
        self.assertEqual((1,3), np.array(results['in_P1mid']).shape)
        self.TestAlmostEqual([[0,1,-4]], results['in_P1mid'])
        self.assertEqual((1,3,2), np.array(results['in_P1last']).shape)
        self.TestAlmostEqual([[[0,0],[-2,-4],[-4,-7]]], results['in_P1last'])
        self.assertEqual((1,3,4), np.array(results['in_P1all']).shape)
        self.TestAlmostEqual([[[0,0,0,0], [1,1,-2,-4],[2,-4,-4,-7]]], results['in_P1all'])

        self.assertEqual((1,3), np.array(results['in_S1']).shape)
        self.TestAlmostEqual([[-1,0,1]], results['in_S1'])
        self.assertEqual((1,3), np.array(results['in_S2']).shape)
        self.TestAlmostEqual([[-2,0,2]], results['in_S2'])
        self.assertEqual((1,3), np.array(results['in_S3']).shape)
        self.TestAlmostEqual([[-2,0,2]], results['in_S3'])
        self.assertEqual((1,3), np.array(results['in_S4']).shape)
        self.TestAlmostEqual([[-1,0,2]], results['in_S4'])

        results = test({'in1': [[0,1,2,4], [1,3,4,5], [2,5,6,7], [3,3,4,1], [4,4,6,7], [5,6,7,8], [6,7,5,4],[7,2,3,1],[0,7,0,0]]})

        self.assertEqual((2, 3, 4), np.array(results['in_tw3']).shape)
        self.TestAlmostEqual([[[-1, -2, -2, -1], [0, 0, 0, 0], [1, 2, 2, 2]],
                                    [[-1, -2, -2, -2], [0, 0, 0, 0], [1, -2, -2, -6]]], results['in_tw3'])
        self.assertEqual((2, 3, 4), np.array(results['in_tw32']).shape)
        self.TestAlmostEqual([[[0, 0, 0, 0], [1, 1, -2, -4], [2, -4, -4, -7]],
                                    [[0, 0, 0, 0], [1, -5, -2, -3], [-6, 0, -5, -4]]], results['in_tw32'])

        self.assertEqual((2, 3), np.array(results['in_P1first']).shape)
        self.TestAlmostEqual([[0, 1, 2],[0, 1, -6]], results['in_P1first'])
        self.assertEqual((2, 3), np.array(results['in_P1mid']).shape)
        self.TestAlmostEqual([[0, 1, -4],[0, -5, 0]], results['in_P1mid'])
        self.assertEqual((2, 3, 2), np.array(results['in_P1last']).shape)
        self.TestAlmostEqual([[[0, 0], [-2, -4], [-4, -7]],
                                    [[0, 0], [-2, -3], [-5, -4]]], results['in_P1last'])
        self.assertEqual((2, 3, 4), np.array(results['in_P1all']).shape)
        self.TestAlmostEqual([[[0, 0, 0, 0], [1, 1, -2, -4], [2, -4, -4, -7]],
                                    [[0, 0, 0, 0], [1, -5, -2, -3], [-6, 0, -5, -4]]], results['in_P1all'])

        self.assertEqual((2, 3), np.array(results['in_S1']).shape)
        self.TestAlmostEqual([[-1, 0, 1],[-1, 0, 1]], results['in_S1'])
        self.assertEqual((2, 3), np.array(results['in_S2']).shape)
        self.TestAlmostEqual([[-2, 0, 2],[-2, 0, -2]], results['in_S2'])
        self.assertEqual((2, 3), np.array(results['in_S3']).shape)
        self.TestAlmostEqual([[-2, 0, 2],[-2, 0, -2]], results['in_S3'])
        self.assertEqual((2, 3), np.array(results['in_S4']).shape)
        self.TestAlmostEqual([[-1, 0, 2],[-2, 0, -6]], results['in_S4'])
    
    def test_recurrent_training(self):
        ## the memory is not shared between different calls
        x = Input('x') 
        F = Input('F')
        p = Parameter('p', tw=0.5, dimensions=1, values=[[1.0],[1.0],[1.0],[1.0],[1.0]])
        x_out = Fir(parameter=p)(x.tw(0.5))+F.last()
        out = Output('out',x_out)
        test = Neu4mes(visualizer=None, seed=42)
        test.addModel('out',out)
        test.neuralizeModel(0.1)

        ## one sample prediction with F initialized with zeros
        result = test(inputs={'x':[1,2,3,4,5]}, close_loop={'F':'out'})
        self.assertEqual(result['out'], [15.0])
        ## 5 samples prediction with F initialized with zero only the first time
        result = test(inputs={'x':[1,2,3,4,5,6,7,8,9]}, close_loop={'F':'out'})
        self.assertEqual(result['out'], [15.0, 35.0, 60.0, 90.0, 125.0])
        ## one sample prediction with F initialized with [1]
        result = test(inputs={'x':[1,2,3,4,5], 'F':[1]}, close_loop={'F':'out'})
        self.assertEqual(result['out'], [16.0])
        ## 5 samples prediction with F initialized with [1] only the first time
        result = test(inputs={'x':[1,2,3,4,5,6,7,8,9], 'F':[1]}, close_loop={'F':'out'})
        self.assertEqual(result['out'], [16.0, 36.0, 61.0, 91.0, 126.0])
        ## 5 samples prediction with F initialized with [1] the first time, [2] the second time and [3] the third time
        result = test(inputs={'x':[1,2,3,4,5,6,7,8,9], 'F':[1,2,3]}, close_loop={'F':'out'})
        self.assertEqual(result['out'], [16.0, 22.0, 28.0, 58.0, 93.0])
        ## one sample prediction with F initialized with [1] (the other values are ignored)
        result = test(inputs={'x':[1,2,3,4,5], 'F':[1,2,3]}, close_loop={'F':'out'})
        self.assertEqual(result['out'], [16.0])

    def test_recurrent_training_complex(self):
        ## the memory is not shared between different calls
        x = Input('x') 
        y = Input('y')
        p = Parameter('p', tw=0.5, dimensions=1, values=[[1.0],[1.0],[1.0],[1.0],[1.0]])
        n = Parameter('n', tw=0.5, dimensions=1, values=[[-1.0],[-1.0],[-1.0],[-1.0],[-1.0]])
        fir_pos = Fir(parameter=p)(x.tw(0.5))
        fir_neg = Fir(parameter=n)(y.tw(0.5))
        out_pos = Output('out_pos', fir_pos)
        out_neg = Output('out_neg', fir_neg)
        out = Output('out',fir_neg+fir_pos)
        test = Neu4mes(visualizer=None, seed=42)
        test.addModel('out', out)
        test.addModel('out_pos',out_pos)
        test.addModel('out_neg',out_neg)
        test.neuralizeModel(0.1)

        ## two sample prediction with x in close loop
        result = test(inputs={'x':[1,2,3,4,5], 'y':[1,2,3,4,5,6]}, close_loop={'x':'out_pos'})
        self.assertEqual(result['out'], [0.0, 9.0])
        self.assertEqual(result['out_pos'], [15.0, 29.0])
        self.assertEqual(result['out_neg'], [-15.0, -20.0])
        ## two sample prediction with y in close loop
        result = test(inputs={'x':[1,2,3,4,5,6], 'y':[1,2,3,4,5]}, close_loop={'y':'out_pos'})
        self.assertEqual(result['out'], [0.0, -9.0])
        self.assertEqual(result['out_pos'], [15.0, 20.0])
        self.assertEqual(result['out_neg'], [-15.0, -29.0])
        ## one sample prediction with both close loops
        ## (!! since all the inputs are recurrent we must specify the prediction horizon (defualt=1))
        result = test(inputs={'x':[1,2,3,4,5], 'y':[1,2,3,4,5]}, close_loop={'x':'out_pos', 'y':'out_neg'})
        self.assertEqual(result['out'], [0.0])
        self.assertEqual(result['out_pos'], [15.0])
        self.assertEqual(result['out_neg'], [-15.0])
        ## three sample prediction with both close loops
        ## (!! since all the inputs are recurrent we must specify the prediction horizon (defualt=1))
        result = test(inputs={'x':[1,2,3,4,5], 'y':[1,2,3,4,5]}, close_loop={'x':'out_pos', 'y':'out_neg'}, prediction_samples=3)
        self.assertEqual(result['out'], [0.0, 30.0, 58.0])
        self.assertEqual(result['out_pos'], [15.0, 29.0, 56.0])
        self.assertEqual(result['out_neg'], [-15.0, 1.0, 2.0])
        ## three sample prediction with both close loops but y gets initialized for 3 steps
        ## (!! since all the inputs are recurrent we must specify the prediction horizon (defualt=1))
        result = test(inputs={'x':[1,2,3,4,5], 'y':[1,2,3,4,5,6,7]}, close_loop={'x':'out_pos', 'y':'out_neg'}, prediction_samples=3)
        self.assertEqual(result['out'], [0.0, 9.0, 31.0])
        self.assertEqual(result['out_pos'], [15.0, 29.0, 56.0])
        self.assertEqual(result['out_neg'], [-15.0, -20.0, -25.0])
    
    def test_state_variables(self):
        ## the state is saved inside the model so the memory is shared between different calls
        x = Input('x') 
        F_state = State('F')
        y_state = State('y')
        z_state = State('z')
        p = Parameter('p', tw=0.5, dimensions=1, values=[[1.0],[1.0],[1.0],[1.0],[1.0]])
        x_out = Fir(parameter=p)(x.tw(0.5))+F_state.last()+y_state.last()+z_state.last()
        x_out.update(F_state)
        x_out.update(y_state)
        x_out.update(z_state)
        out = Output('out',x_out)

        test = Neu4mes(visualizer=None, seed=42)
        test.addModel('out',out)
        test.neuralizeModel(0.1)

        ## one sample prediction with state variables not initialized
        ## (they will have the last valid state)
        result = test(inputs={'x':[1,2,3,4,5]})
        self.assertEqual(result['out'], [15.0])
        ## 5 sample prediction with state variables not initialized
        ## (the first prediction will preserve the state of the previous test [15.0])
        result = test(inputs={'x':[1,2,3,4,5,6,7,8,9]})
        self.assertEqual(result['out'], [60.0, 200.0, 625.0, 1905.0, 5750.0])
        ## one sample prediction with state variables initialized with zero
        test.clear_state()
        result = test(inputs={'x':[1,2,3,4,5]})
        self.assertEqual(result['out'], [15.0])
        ## one sample prediction with F initialized with [1] and the others not initialized (so they will have 15.0 in the memory)
        result = test(inputs={'x':[1,2,3,4,5], 'F':[1]})
        self.assertEqual(result['out'], [46.0])
        ## one sample prediction with all the state variables initialized
        result = test(inputs={'x':[1,2,3,4,5], 'F':[1], 'y':[2], 'z':[3]})
        self.assertEqual(result['out'], [21.0])
        ## 5 samples prediction with state variables initialized as many times as they have values to take
        result = test(inputs={'x':[1,2,3,4,5,6,7,8,9], 'F':[1,2,3], 'y':[2,3], 'z':[3]})
        self.assertEqual(result['out'], [21.0, 46.0, 120.0, 390.0, 1205.0])
    
    def test_state_variables_complex(self):
        ## the state is saved inside the model so the memory is shared between different calls
        x = Input('x') 
        y_state = State('y')
        z_state = State('z')
        x_p = Parameter('x_p', tw=0.5, dimensions=1, values=[[1.0],[1.0],[1.0],[1.0],[1.0]])
        y_p = Parameter('y_p', tw=0.5, dimensions=1, values=[[2.0],[2.0],[2.0],[2.0],[2.0]])
        z_p = Parameter('z_p', tw=0.5, dimensions=1, values=[[3.0],[3.0],[3.0],[3.0],[3.0]])
        x_fir = Fir(parameter=x_p)(x.tw(0.5))
        y_fir = Fir(parameter=y_p)(y_state.tw(0.5))
        z_fir = Fir(parameter=z_p)(z_state.tw(0.5))
        y_fir.update(y_state)
        z_fir.update(z_state)
        out_x = Output('out_x', x_fir)
        out_y = Output('out_y', y_fir)
        out_z = Output('out_z', z_fir)
        out = Output('out',x_fir+y_fir+z_fir)

        test = Neu4mes(visualizer=None, seed=42)
        test.addModel('out_all',[out, out_x, out_y, out_z])
        test.neuralizeModel(0.1)

        ## one sample prediction with state variables not initialized
        ## (they will have the last valid state)
        result = test(inputs={'x':[1,2,3,4,5]})
        self.assertEqual(result['out'], [15.0])
        self.assertEqual(result['out_x'], [15.0])
        self.assertEqual(result['out_y'], [0.0])
        self.assertEqual(result['out_z'], [0.0])
        self.assertEqual(test.model.states['y'].numpy().tolist(), [[[0.0], [0.0], [0.0], [0.0], [0.0]]])
        self.assertEqual(test.model.states['z'].numpy().tolist(), [[[0.0], [0.0], [0.0], [0.0], [0.0]]])
        ## 1 sample prediction with state variables all initialized
        result = test(inputs={'x':[1,2,3,4,5], 'y':[1,2,3,4,5], 'z':[1,2,3,4,5]})
        self.assertEqual(result['out'], [90.0])
        self.assertEqual(result['out_x'], [15.0])
        self.assertEqual(result['out_y'], [30.0])
        self.assertEqual(result['out_z'], [45.0])
        self.assertEqual(test.model.states['y'].numpy().tolist(), [[[2.0], [3.0], [4.0], [5.0], [30.0]]])
        self.assertEqual(test.model.states['z'].numpy().tolist(), [[[2.0], [3.0], [4.0], [5.0], [45.0]]])
        ## clear state of y
        test.clear_state(state='y')
        self.assertEqual(test.model.states['y'].numpy().tolist(), [[[0.0], [0.0], [0.0], [0.0], [0.0]]])
        self.assertEqual(test.model.states['z'].numpy().tolist(), [[[2.0], [3.0], [4.0], [5.0], [45.0]]])
        ## multi-sample prediction with states initialized as many times as they have values
        result = test(inputs={'x':[1,2,3,4,5,6,7,8,9], 'y':[1,2,3,4,5,6,7], 'z':[1,2,3,4,5,6]})
        self.assertEqual(result['out'], [90.0, 120.0, 309.0, 1101.0, 4155.0])
        self.assertEqual(result['out_x'], [15.0, 20.0, 25.0, 30.0, 35.0])
        self.assertEqual(result['out_y'], [30.0, 40.0, 50.0, 144.0, 424.0])
        self.assertEqual(result['out_z'], [45.0, 60.0, 234.0, 927.0, 3696.0])
        self.assertEqual(test.model.states['y'].numpy().tolist(), [[[6.0], [7.0], [50.0], [144.0], [424.0]]])
        self.assertEqual(test.model.states['z'].numpy().tolist(), [[[6.0], [60.0], [234.0], [927.0], [3696.0]]])
        ## Clear all states
        test.clear_state()
        self.assertEqual(test.model.states['y'].numpy().tolist(), [[[0.0], [0.0], [0.0], [0.0], [0.0]]])
        self.assertEqual(test.model.states['z'].numpy().tolist(), [[[0.0], [0.0], [0.0], [0.0], [0.0]]])
    
if __name__ == '__main__':
    unittest.main()

