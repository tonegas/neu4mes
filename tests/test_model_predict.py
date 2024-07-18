import sys
import os
# append a new directory to sys.path
sys.path.append(os.getcwd())

import unittest
from neu4mes import *
import torch

# This file test the model prediction in particular the output value

def myfun(x, P):
    return x*P

def myfun2(a, b ,c):
    import torch
    return torch.sin(a + b) * c

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
        torch.manual_seed(1)
        in1 = Input('in1')
        in2 = Input('in2')
        out_fun = Fir(in1.tw(0.1)) + Fir(in2.last())
        out = Output('out', out_fun)
        test = Neu4mes(visualizer=None)
        test.addModel(out)
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
        torch.manual_seed(1)
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

        test = Neu4mes(visualizer=None)
        #test.addModel([out0,out1,out2,out3,out4,out5,out6,out7,out11,out12,out13,out14,out15])
        test.addModel([out1, out2, out3, out4, out5, out6, out7, out8, out9, out10, out11, out12, out13, out14, out15])

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
        torch.manual_seed(1)
        in1 = Input('in1')

        # Finestre nel tempo
        out1 = Output('x.tw(1)', in1.tw(1,offset=0))
        out2 = Output('x.tw([-1,0])', in1.tw([-1, 0],offset=0))
        out3 = Output('x.tw([1,3])', in1.tw([1, 3],offset=2))
        out4 = Output('x.tw([-3,-2])', in1.tw([-3, -2],offset=-2))

        # Finesatre nei samples
        out5 = Output('x.sw([-1,0])',  in1.sw([-1, 0],offset=0))
        out6 = Output('x.sw([-3,1])',  in1.sw([-3, 1],offset=-2))
        out7 = Output('x.sw([0,1])',  in1.sw([0, 1],offset=1))
        out8 = Output('x.sw([-3, 3])', in1.sw([-3, 3], offset=3))
        out9 = Output('x.sw([-3, 3])-2', in1.sw([-3, 3], offset=0))

        test = Neu4mes(visualizer=None)
        test.addModel([out1,out2,out3,out4,out5,out6,out7,out8,out9])

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
        torch.manual_seed(1)
        in1 = Input('in1',dimensions=3)

        # Finestre nel tempo
        out1 = Output('x.tw(1)', in1.tw(1, offset=0))
        out2 = Output('x.tw([-1,0])', in1.tw([-1, 0], offset=0))
        out3 = Output('x.tw([1,3])', in1.tw([1, 3], offset=2))
        out4 = Output('x.tw([-3,-2])', in1.tw([-3, -2], offset=-2))

        # Finesatre nei samples
        out5 = Output('x.sw([-1,0])', in1.sw([-1, 0], offset=0))
        out6 = Output('x.sw([-3,1])', in1.sw([-3, 1], offset=-2))
        out7 = Output('x.sw([0,1])', in1.sw([0, 1], offset=1))
        out8 = Output('x.sw([-3, 3])', in1.sw([-3, 3], offset=3))
        out9 = Output('x.sw([-3, 3])-2', in1.sw([-3, 3], offset=0))

        test = Neu4mes(visualizer=None)
        test.addModel([out1, out2, out3, out4, out5, out6, out7, out8, out9])

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
        out1 = Output('sum', in1.tw(1, offset=0) + in1.tw([-1,0]))
        out2 = Output('sub', in1.tw([1, 3], offset=2) - in1.tw([-3, -1], offset=-1))
        out3 = Output('mul', in1.tw([-2, 2]) * in1.tw([-3, 1], offset=-1))

        out4 = Output('sum2', in2.tw(1, offset=0) + in2.tw([-1,0]))
        out5 = Output('sub2', in2.tw([1, 3], offset=2) - in2.tw([-3, -1], offset=-1))
        out6 = Output('mul2', in2.tw([-2, 2]) * in2.tw([-3, 1], offset=-1))

        test = Neu4mes(visualizer=None)
        test.addModel([out1, out2, out3, out4, out5, out6])

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
        torch.manual_seed(1)
        in1 = Input('in1')
        out1 = Output('Fir3', Fir(3)(in1.last()))
        out2 = Output('Fir5', Fir(5)(in1.tw(1)))#
        out3 = Output('Fir2', Fir(2)(in1.tw([-1,0])))#
        out4 = Output('Fir1', Fir(1)(in1.tw([-3,3])))#
        out5 = Output('Fir7', Fir(7)(in1.tw(3,offset=0)))#
        out6 = Output('Fir4', Fir(4)(in1.tw([2,3],offset=3)))#
        out7 = Output('Fir6', Fir(6)(in1.sw([-2,-1], offset=-1)))#

        test = Neu4mes(visualizer=None)
        test.addModel([out1,out2,out3,out4,out5,out6,out7])
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
    
    def test_single_in_window_offset_parametric_function(self):
        # An input dimension is temporal and does not remain unchanged unless redefined on output
        # If there are multiple inputs the function returns an error if the dimensions are not defined
        torch.manual_seed(1)
        in1 = Input('in1')
        parfun = ParamFun(myfun)
        out = Output('out', parfun(in1.last()))
        test = Neu4mes(visualizer=None)
        test.addModel(out)
        test.neuralizeModel(0.1)

        #results = test({'in1': 1})   
        #self.TestAlmostEqual(results['out'], [0.7576315999031067])
        results = test({'in1': [[1]]})
        self.assertEqual((1,), np.array(results['out']).shape)
        self.TestAlmostEqual(results['out'],[0.524665892124176])
        results = test({'in1': [2]})
        self.TestAlmostEqual(results['out'],[1.049331784248352])
        results = test({'in1': [1,2]})
        self.assertEqual((2,), np.array(results['out']).shape)
        self.TestAlmostEqual(results['out'],[0.524665892124176,1.049331784248352])

        out = Output('out', ParamFun(myfun)(in1.tw(0.2)))
        test = Neu4mes(visualizer=None)
        test.addModel(out)
        test.neuralizeModel(0.1)

        with self.assertRaises(StopIteration):
            results = test({'in1': [1]})
        with self.assertRaises(StopIteration):
            results = test({'in1': [2]})
        results = test({'in1': [1,2]})
        self.assertEqual((1,2), np.array(results['out']).shape)
        self.TestAlmostEqual(results['out'], [[0.04048508405685425, 0.0809701681137085]])
        results = test({'in1': [[1,2]]}, sampled=True)
        self.assertEqual((1,2), np.array(results['out']).shape)
        self.TestAlmostEqual(results['out'], [[0.04048508405685425, 0.0809701681137085]])
        results = test({'in1': [1, 2, 3, 4, 5]})# Qui vengono costruite gli input a due a due con shift di 1
        self.assertEqual((4,2), np.array(results['out']).shape)
        self.TestAlmostEqual(results['out'], [[0.04048508405685425, 0.0809701681137085], [0.0809701681137085, 0.12145525217056274], [0.12145525217056274, 0.161940336227417], [0.161940336227417, 0.20242542028427124]])
        results = test({'in1': [[1, 2], [2, 3], [3, 4], [4, 5]]}, sampled=True)
        self.assertEqual((4,2), np.array(results['out']).shape)
        self.TestAlmostEqual(results['out'], [[0.04048508405685425, 0.0809701681137085], [0.0809701681137085, 0.12145525217056274], [0.12145525217056274, 0.161940336227417], [0.161940336227417, 0.20242542028427124]])

        out = Output('out', ParamFun(myfun)(in1.last(),in1.last()))
        test = Neu4mes(visualizer=None)
        test.addModel(out)
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
        test.addModel(out)
        test.neuralizeModel(0.1)

        #results = test({'in1': 2})
        #self.TestAlmostEqual(results['out'], [4])
        results = test({'in1': [2]})
        self.TestAlmostEqual(results['out'], [4])
        results = test({'in1': [2,1]})
        self.TestAlmostEqual(results['out'], [4,1])

        out = Output('out', ParamFun(myfun)(in1.tw(0.2), in1.tw(0.2)))
        test = Neu4mes(visualizer=None)
        test.addModel(out)
        test.neuralizeModel(0.1)

        results = test({'in1': [2,4]})
        self.assertEqual((1,2), np.array(results['out']).shape)
        self.TestAlmostEqual(results['out'], [[4,16]])
        results = test({'in1': [[1, 2], [3, 2]]}, sampled=True)
        self.assertEqual((2,2), np.array(results['out']).shape)
        self.TestAlmostEqual(results['out'], [[1.0, 4.0], [9.0, 4.0]])

        out = Output('out', ParamFun(myfun)(in1.tw(0.3), in1.tw(0.3)))
        test = Neu4mes(visualizer=None)
        test.addModel(out)
        test.neuralizeModel(0.1)

        with self.assertRaises(StopIteration):
            results = test({'in1': [2]})

        with self.assertRaises(StopIteration):
            results = test({'in1': [2, 4]})

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
        test.addModel(out)
        test.neuralizeModel(0.1)
        with self.assertRaises(StopIteration):
            test({'in1': [[1, 2, 2], [3, 4, 5]]})
    
    def test_parametric_function_and_fir(self):
        torch.manual_seed(1)
        in1 = Input('in1')
        in2 = Input('in2')
        out = Output('out', Fir(ParamFun(myfun2)(in1.tw(0.4))))
        test = Neu4mes(visualizer=None)
        test.addModel(out)
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
        test.addModel(out)
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
        test.addModel(out)
        test.neuralizeModel(0.1)

        results = test({'in1': [[1, 2, 2, 4], [1, 2, 2, 4]], 'in2': [[1, 2, 2, 4], [1, 2, 2, 4]]}, sampled=True)
        self.assertEqual((2,1,3), np.array(results['out']).shape)
        print(results['out'])
        self.TestAlmostEqual(results['out'], [[[-0.03303150087594986, 0.023659050464630127, 0.0185492392629385]], [[-0.03303150087594986, 0.023659050464630127, 0.0185492392629385]]])

        parfun = ParamFun(myfun2)
        with self.assertRaises(AssertionError):
            Output('out', parfun(Fir(3)(parfun(in1.tw(0.4), in2.tw(0.4)))))

        parfun = ParamFun(myfun2)
        out = Output('out', parfun(Fir(3)(parfun(in1.tw(0.4)))))
        test = Neu4mes(visualizer=None)
        test.addModel(out)
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
        par = Parameter('par', sw=1, values=5)
        in4 = Input('in4', dimensions=4).last()
        par4 = Parameter('par4', dimensions=4, sw=1, values=[1,2,3,4])
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
        test.addModel([out1,out4,outtot])
        test.neuralizeModel()

        results = test({'in1': [1, 2, -2],'in4': [[6, 2, 2, 4], [7, 2, 2, 4], [-6, -5, 5, 4]]})
        self.assertEqual((3,), np.array(results['out1']).shape)
        self.assertEqual((3,1,4), np.array(results['out4']).shape)
        self.assertEqual((3,), np.array(results['outtot']).shape)

        self.TestAlmostEqual([34.8819529, 33554496.0,  -33554480.0], results['out1'] )
        self.TestAlmostEqual([[[58.9539756, 46.1638031, 554.231201171875, 4294967296.0]], [[67.3462829589843, 46.16380310058594, 554.231201171875, 4294967296.0]], [[ -41.75371170043945, 567.6907348632812, 1953220.375, 4294967296.0]]], results['out4'])
        self.TestAlmostEqual([4294967808.0, 4328522240.0,  4263366656.0], results['outtot'])


if __name__ == '__main__':
    unittest.main()

