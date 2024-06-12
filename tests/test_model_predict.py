import sys
import os
# append a new directory to sys.path
sys.path.append(os.getcwd())

import unittest
from neu4mes import *
import torch

# This file test the model prediction in particular the output value

def myfun(x,P):
    return x*P

def myfun2(a, b ,c):
    import torch
    return torch.sin(a + b) * c

class MyTestCase(unittest.TestCase):
    def TestAlmostEqual(self, data1, data2, precision=4):
        assert np.asarray(data1, dtype=np.float32).ndim == np.asarray(data2, dtype=np.float32).ndim, f'Inputs must have the same dimension! Received {type(data1)} and {type(data2)}'
        if type(data1) == type(data2) == list:  
            for pred, label in zip(data1, data2):
                self.TestAlmostEqual(pred, label, precision=precision)
        else:
            self.assertAlmostEqual(data1, data2, places=precision)


    def test_single_in_single_out(self):
        torch.manual_seed(1)
        in1 = Input('in1')
        in2 = Input('in2')
        out_fun = Fir(in1.tw(0.1)) + Fir(in2)
        out = Output('out', out_fun)
        test = Neu4mes(visualizer=None)
        test.addModel(out)
        test.neuralizeModel(0.01)
        results = test({'in1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],'in2': [5]})
        self.assertEqual(1, len(results['out']))
        self.TestAlmostEqual([33.74938201904297], results['out'])
        results = test({'in1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], 'in2': [5, 7]})
        self.assertEqual(2, len(results['out']))
        self.TestAlmostEqual([33.74938201904297, 40.309326171875], results['out'])
        results = test({'in1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], 'in2': [5, 7, 9]})
        self.assertEqual(3, len(results['out']))
        self.TestAlmostEqual([33.74938201904297, 40.309326171875, 46.86927032470703], results['out'])

    def test_single_in_multi_out_window(self):
        # Here there is more sample for each time step but the dimensions of the input is 1
        torch.manual_seed(1)
        in1 = Input('in1')
        #TODO se rimuovo questo togliendo questo modello dalla lista non funziona più
        # -----
        f = Fir(1)
        out_fun = f(in1.tw(1))
        out0 = Output('out', out_fun)
        # -----

        # Finestre nel tempo
        out1 = Output('x.tw(1)', in1.tw(1))
        out2 = Output('x.tw([-1,0])', in1.tw([-1, 0]))
        out3 = Output('x.tw([-3,0])', in1.tw([-3, 0]))
        out4 = Output('x.tw([1,3])', in1.tw([1, 3]))
        out5 = Output('x.tw([-1,3])', in1.tw([-1, 3]))
        out6 = Output('x.tw([0,1])', in1.tw([0, 1]))
        out7 = Output('x.tw([-3,-2])', in1.tw([-3, -2]))

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
        test.addModel([out0,out1,out2,out3,out4,out5,out6,out7,out8,out9,out10,out11,out12,out13,out14,out15])
        # TODO test.addModel([out1, out2, out3, out4, out5, out6, out7, out8, out9, out10, out11, out12, out13, out14, out15])

        test.neuralizeModel(1)
        # Time                  -2,-1,0,1,2,3 # zero represent the last passed instant
        results = test({'in1': [-2,-1,0,1,7,3]})
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

    def test_single_in_multi_out_window_offset(self):
        # Here there is more sample for each time step but the dimensions of the input is 1
        torch.manual_seed(1)
        in1 = Input('in1')
        #TODO se rimuovo questo togliendo questo modello dalla lista non funziona più
        # -----
        f = Fir(1)
        out_fun = f(in1.tw(1))
        out0 = Output('out', out_fun)
        # -----

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
        test.addModel([out0,out1,out2,out3,out4,out5,out6,out7,out8,out9])

        test.neuralizeModel(1)
        # Time                  -2,-1,0,1,2,3 # zero represent the last passed instant
        results = test({'in1': [-2,-1,0,1,7,3]})
        # Time window
        self.assertEqual((1,), np.array(results['x.tw(1)']).shape)
        self.TestAlmostEqual([0], results['x.tw(1)'])
        self.assertEqual((1,), np.array(results['x.tw([-1,0])']).shape)
        self.TestAlmostEqual([0], results['x.tw([-1,0])'])
        self.assertEqual((1, 2), np.array(results['x.tw([1,3])']).shape)
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


    def test_single_in_multi_out(self):
        torch.manual_seed(1)
        in1 = Input('in1')
        f = Fir(3)
        out_fun = f(in1.tw(0.1))
        out1 = Output('out', out_fun)
        out2 = Output('intw2', in1.tw(1))
        out3 = Output('intw3', in1.tw(0.2))
        out4 = Output('intw4', in1.tw([-0.2,0.2]))
        out5 = Output('intw5', in1.tw(0.1))
        test = Neu4mes(visualizer=None)
        test.addModel(out1)
        test.addModel(out2)
        test.addModel(out3)
        test.addModel(out4)
        test.addModel(out5)
        test.neuralizeModel(0.1)
        results = test({'in1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]})
        #TODO
        # La prima dimensione è quella temporale
        # la seconda dimensione è il numero di elementi temporali per ogni istante
        # la terza è la dimensione del segnale
        #self.assertEqual((1,1,3), np.array(results['out']).shape)
        #self.TestAlmostEqual(results['out'],[[[7.576315879821777, 2.7931089401245117, 4.0306925773620605]]])
        self.assertEqual((1, 10), np.array(results['intw2']).shape)
        self.TestAlmostEqual(results['intw2'], [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
        self.assertEqual((1, 2), np.array(results['intw3']).shape)
        self.TestAlmostEqual(results['intw3'], [[9, 10]])
        self.assertEqual((1, 4), np.array(results['intw4']).shape)
        self.TestAlmostEqual(results['intw4'], [[9, 10, 11, 12]])
        self.assertEqual((1,), np.array(results['intw5']).shape)
        self.TestAlmostEqual(results['intw5'], [10])
    
    ## don't know why but the indentation fails when trying to build the function
    def test_parametric_function(self): 
        torch.manual_seed(1)
        in1 = Input('in1')
        parfun = ParamFun(myfun, 1)
        out = Output('out', parfun(in1))
        test = Neu4mes(visualizer=None)
        test.addModel(out)
        test.neuralizeModel(0.1)

        #results = test({'in1': 1})   
        #self.TestAlmostEqual(results['out'], [0.7576315999031067])
        results = test({'in1': [1]})
        self.TestAlmostEqual(results['out'],[0.7576315999031067])
        results = test({'in1': [2]})
        self.TestAlmostEqual(results['out'],[1.5152631998062134])
        results = test({'in1': [1,2]})
        self.TestAlmostEqual(results['out'],[0.7576315999031067,1.5152631998062134])

        out = Output('out', ParamFun(myfun)(in1.tw(0.2)))
        test = Neu4mes(visualizer=None)
        test.addModel(out)
        test.neuralizeModel(0.1)

        with self.assertRaises(StopIteration):#TODO voglio una finestra di almeno 2 sample altrimenti errore
            results = test({'in1': [1]})
        with self.assertRaises(StopIteration):
            results = test({'in1': [2]})
        results = test({'in1': [1,2]})
        self.TestAlmostEqual(results['out'], [[0.2793108820915222, 0.5586217641830444]])
        results = test({'in1': [[1,2]]}, sampled=True)
        self.TestAlmostEqual(results['out'], [[0.2793108820915222, 0.5586217641830444]])
        results = test({'in1': [1, 2, 3, 4, 5]})# Qui vengono costruite gli input a due a due con shift di 1
        self.TestAlmostEqual(results['out'], [[0.2793108820915222, 0.5586217641830444], [0.5586217641830444, 0.8379326462745667], [0.8379326462745667, 1.1172435283660889], [1.1172435283660889, 1.3965544700622559]])
        results = test({'in1': [[1, 2], [2, 3], [3, 4], [4, 5]]}, sampled=True)
        self.TestAlmostEqual(results['out'], [[0.2793108820915222, 0.5586217641830444], [0.5586217641830444, 0.8379326462745667], [0.8379326462745667, 1.1172435283660889], [1.1172435283660889, 1.3965544700622559]])

        out = Output('out', ParamFun(myfun)(in1,in1))
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

        with self.assertRaises(AssertionError):
            Output('out', ParamFun(myfun)(in1.tw(1), in1))

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
        self.TestAlmostEqual(results['out'], [[4,16]])
        results = test({'in1': [[1, 2], [3, 2]]}, sampled=True)
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
        self.TestAlmostEqual(results['out'], [[9,4,1]])
        results = test({'in1': [[1,2,2],[3,4,5]]}, sampled=True)
        self.TestAlmostEqual(results['out'], [[1, 4, 4], [9, 16, 25]])
        results = test({'in1': [[3, 2, 1], [2, 1, 0]]}, sampled=True)
        self.TestAlmostEqual(results['out'], [[9, 4, 1],[4, 1, 0]])
        results = test({'in1': [3,2,1,0]})
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
        results = test({'in1': [1, 2, 2, 4]})
        self.TestAlmostEqual(results['out'][0], -0.03262542188167572)
        results = test({'in1': [[1, 2, 2, 4],[2, 2, 4, 5]]}, sampled=True)
        self.TestAlmostEqual(results['out'], [-0.03262542188167572, -0.001211121678352356])
        results = test({'in1': [1, 2, 2, 4, 5]})
        self.TestAlmostEqual(results['out'], [-0.03262542188167572, -0.001211121678352356])

        with self.assertRaises(AssertionError):
            Output('out', Fir(ParamFun(myfun2)(in1.tw(0.4),in2.tw(0.2))))

        out = Output('out', Fir(ParamFun(myfun2)(in1.tw(0.4),in2.tw(0.4))))
        test = Neu4mes(visualizer=None)
        test.addModel(out)
        test.neuralizeModel(0.1)
        with self.assertRaises(KeyError):
            test({'in1': [[1, 2, 2, 4]]})
        results = test({'in1': [1, 2, 2, 4], 'in2': [1, 2, 2, 4]})

        self.TestAlmostEqual(results['out'], [0.12016649544239044])
        results = test({'in1': [[1, 2, 2, 4]], 'in2': [[1, 2, 2, 4]]}, sampled=True)
        self.TestAlmostEqual(results['out'], [0.12016649544239044])
        results = test({'in1': [1, 2, 2, 4, 5], 'in2': [1, 2, 2, 4]})
        self.TestAlmostEqual(results['out'], [0.12016649544239044])
        results = test({'in1': [[1, 2, 2, 4], [1, 2, 2, 4]], 'in2': [[1, 2, 2, 4]]}, sampled=True)
        self.TestAlmostEqual(results['out'], [0.12016649544239044])
        results = test({'in1': [[1, 2, 2, 4], [1, 2, 2, 4]], 'in2': [[1, 2, 2, 4], [5, 5, 5, 5]]}, sampled=True)
        self.TestAlmostEqual(results['out'], [0.12016649544239044,  0.44285041093826294])

        self.TestAlmostEqual(results['out'], [0.12016649544239044])
        results = test({'in1': [[1, 2, 2, 4]], 'in2': [[1, 2, 2, 4]]}, sampled=True)
        self.TestAlmostEqual(results['out'], [0.12016649544239044])
        results = test({'in1': [1, 2, 2, 4, 5], 'in2': [1, 2, 2, 4]})
        self.TestAlmostEqual(results['out'], [0.12016649544239044])
        results = test({'in1': [[1, 2, 2, 4], [1, 2, 2, 4]], 'in2': [[1, 2, 2, 4]]}, sampled=True)
        self.TestAlmostEqual(results['out'], [0.12016649544239044])
        results = test({'in1': [[1, 2, 2, 4], [1, 2, 2, 4]], 'in2': [[1, 2, 2, 4], [5, 5, 5, 5]]}, sampled=True)
        self.TestAlmostEqual(results['out'], [0.12016649544239044,  0.44285041093826294])


        out = Output('out', Fir(3)(ParamFun(myfun2)(in1.tw(0.4), in2.tw(0.4))))
        test = Neu4mes(visualizer=None)
        test.addModel(out)
        test.neuralizeModel(0.1)

        results = test({'in1': [[1, 2, 2, 4], [1, 2, 2, 4]], 'in2': [[1, 2, 2, 4], [1, 2, 2, 4]]}, sampled=True)
        self.TestAlmostEqual(results['out'], [[0.06674075871706009, 0.04721337556838989, -0.0806228518486023], [0.06674075871706009, 0.04721337556838989, -0.0806228518486023]])

        parfun = ParamFun(myfun2)
        with self.assertRaises(AssertionError):
            Output('out', parfun(Fir(3)(parfun(in1.tw(0.4), in2.tw(0.4)))))

        parfun = ParamFun(myfun2)
        out = Output('out', parfun(Fir(3)(parfun(in1.tw(0.4)))))
        test = Neu4mes(visualizer=None)
        test.addModel(out)
        test.neuralizeModel(0.1)

        results = test({'in1': [[1, 2, 2, 4], [2, 1, 1, 3]]}, sampled=True)
        self.TestAlmostEqual(results['out'], [[0.4886608421802521, 0.4340703785419464, 0.7541031241416931], [0.6608667969703674, 0.6199508905410767, 0.7365500330924988]])

        results = test({'in1': [1, 2, 2, 4, 3],'in2': [6, 2, 2, 4, 4]})
        self.TestAlmostEqual(results['out'], [[0.4886608421802521, 0.4340703785419464, 0.7541031241416931], [0.3563916087150574, -0.10742330551147461, 0.650678277015686]])
        results = test({'in1': [[1, 2, 2, 4],[2, 2, 4, 3]],'in2': [[6, 2, 2, 4],[2, 2, 4, 4]]}, sampled=True)
        self.TestAlmostEqual(results['out'], [[0.4886608421802521, 0.4340703785419464, 0.7541031241416931], [0.3563916087150574, -0.10742330551147461, 0.650678277015686]])


if __name__ == '__main__':
    unittest.main()

