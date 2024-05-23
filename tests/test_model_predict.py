import sys
import os
# append a new directory to sys.path
sys.path.append(os.getcwd())

import unittest
from neu4mes import *
import torch

def myfun(x,P):
    return x*P

def myfun2(a, b ,c):
    import torch
    return torch.sin(a + b) * c

class MyTestCase(unittest.TestCase):
    def test_single_in_single_out(self):
        torch.manual_seed(1)
        in1 = Input('in1')
        in2 = Input('in2')
        out_fun = Fir(in1.tw(0.1)) + Fir(in2)
        out = Output('out', out_fun)
        test = Neu4mes(verbose=False)
        test.addModel(out)
        test.neuralizeModel(0.01)
        results = test({'in1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],'in2': [5]})
        self.assertAlmostEqual(results['out'], [-0.09323513507843018])
        results = test({'in1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], 'in2': [5, 7]})
        for pred, label in zip(results['out'], [-0.09323513507843018, 0.05682480335235596]):
            self.assertAlmostEqual(pred, label)
        results = test({'in1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], 'in2': [5, 7, 9]})
        for pred, label in zip(results['out'], [-0.09323513507843018, 0.05682480335235596, 0.2068847417831421]):
            self.assertAlmostEqual(pred, label)
    
    def test_single_in_multi_out(self):
        torch.manual_seed(1)
        in1 = Input('in1')
        f = Fir(3)
        out_fun = f(in1.tw(0.1))
        out1 = Output('out', out_fun)
        out2 = Output('intw2', in1.tw(1))
        out3 = Output('intw3', in1.tw(0.2))
        out4 = Output('intw4', in1.tw([-0.2,0.2]))
        test = Neu4mes(verbose=False)
        test.addModel(out1)
        test.addModel(out2)
        test.addModel(out3)
        test.addModel(out4)
        test.neuralizeModel(0.1)
        results = test({'in1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]})
        self.assertAlmostEqual(results['out'],[[4.693689346313477, -9.414368629455566, 5.9971723556518555]])
        self.assertAlmostEqual(results['intw2'], [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
        self.assertAlmostEqual(results['intw3'], [[9, 10]])
        self.assertAlmostEqual(results['intw4'], [[9, 10, 11, 12]])
    
    ## don't know why but the indentation fails when trying to build the function
    def test_parametric_function(self): 
        torch.manual_seed(1)
        in1 = Input('in1')
        parfun = ParamFun(myfun, 1)
        out = Output('out', parfun(in1))
        test = Neu4mes(verbose=False)
        test.addModel(out)
        test.neuralizeModel(0.1)

        results = test({'in1': 1})   
        self.assertAlmostEqual(results['out'], [0.7576315999031067])
        results = test({'in1': [1]})
        self.assertAlmostEqual(results['out'],[0.7576315999031067])
        results = test({'in1': [2]})
        self.assertAlmostEqual(results['out'],[1.5152631998062134])
        results = test({'in1': [1,2]})
        self.assertAlmostEqual(results['out'],[0.7576315999031067,1.5152631998062134])

        out = Output('out', ParamFun(myfun)(in1.tw(0.2)))
        test = Neu4mes(verbose=False)
        test.addModel(out)
        test.neuralizeModel(0.1)

        with self.assertRaises(AssertionError):#TODO voglio una finestra di almeno 2 sample altrimenti errore
            results = test({'in1': [1]})
        with self.assertRaises(AssertionError):
            results = test({'in1': [2]})
        results = test({'in1': [1,2]})
        for pred, label in zip(results['out'], [[0.2793108820915222, 0.5586217641830444]]):
            self.assertAlmostEqual(pred, label)
        results = test({'in1': [[1,2]]}, sampled=True)
        for pred, label in zip(results['out'], [[0.2793108820915222, 0.5586217641830444]]):
            self.assertAlmostEqual(pred, label)
        results = test({'in1': [1, 2, 3, 4, 5]})# Qui vengono costruite gli input a due a due con shift di 1
        for pred, label in zip(results['out'], [[0.2793108820915222, 0.5586217641830444], [0.5586217641830444, 0.8379326462745667], [0.8379326462745667, 1.1172435283660889], [1.1172435283660889, 1.3965544700622559]]):
            self.assertAlmostEqual(pred, label)
        results = test({'in1': [[1, 2], [2, 3], [3, 4], [4, 5]]}, sampled=True)
        for pred, label in zip(results['out'], [[0.2793108820915222, 0.5586217641830444], [0.5586217641830444, 0.8379326462745667], [0.8379326462745667, 1.1172435283660889], [1.1172435283660889, 1.3965544700622559]]):
            self.assertAlmostEqual(pred, label)

        out = Output('out', ParamFun(myfun)(in1,in1))
        test = Neu4mes(verbose=False)
        test.addModel(out)
        test.neuralizeModel(0.1)

        results = test({'in1': 1})
        self.assertAlmostEqual(results['out'],[1])
        results = test({'in1': [1]})
        self.assertAlmostEqual(results['out'],[1])
        results = test({'in1': [2]})
        self.assertAlmostEqual(results['out'],[4])
        results = test({'in1': [1,2]})
        self.assertAlmostEqual(results['out'],[1,4])

        with self.assertRaises(AssertionError):
            Output('out', ParamFun(myfun)(in1.tw(1), in1))

        out = Output('out', ParamFun(myfun)(in1.tw(0.1), in1.tw(0.1)))
        test = Neu4mes(verbose=False)
        test.addModel(out)
        test.neuralizeModel(0.1)

        results = test({'in1': 2})
        self.assertAlmostEqual(results['out'], [4])
        results = test({'in1': [2]})
        self.assertAlmostEqual(results['out'], [4])
        results = test({'in1': [2,1]})
        self.assertAlmostEqual(results['out'], [4,1])

        out = Output('out', ParamFun(myfun)(in1.tw(0.2), in1.tw(0.2)))
        test = Neu4mes(verbose=False)
        test.addModel(out)
        test.neuralizeModel(0.1)

        results = test({'in1': [2,4]})
        for pred, label in zip(results['out'], [[4,16]]):
            self.assertAlmostEqual(pred, label)
        results = test({'in1': [[1, 2], [3, 2]]}, sampled=True)
        for pred, label in zip(results['out'], [[1.0, 4.0], [9.0, 4.0]]):
            self.assertAlmostEqual(pred, label)

        out = Output('out', ParamFun(myfun)(in1.tw(0.3), in1.tw(0.3)))
        test = Neu4mes(verbose=False)
        test.addModel(out)
        test.neuralizeModel(0.1)

        with self.assertRaises(AssertionError):
            results = test({'in1': [2]})

        with self.assertRaises(AssertionError):
            results = test({'in1': [2, 4]})

        results = test({'in1': [3,2,1]})
        for pred, label in zip(results['out'], [[9,4,1]]):
            self.assertAlmostEqual(pred, label)
        results = test({'in1': [[1,2,2],[3,4,5]]}, sampled=True)
        for pred, label in zip(results['out'], [[1, 4, 4], [9, 16, 25]]):
            self.assertAlmostEqual(pred, label)
        results = test({'in1': [[3, 2, 1], [2, 1, 0]]}, sampled=True)
        for pred, label in zip(results['out'], [[9, 4, 1],[4, 1, 0]]):
            self.assertAlmostEqual(pred, label)
        results = test({'in1': [3,2,1,0]})
        for pred, label in zip(results['out'], [[9, 4, 1],[4, 1, 0]]):
            self.assertAlmostEqual(pred, label)
        
        out = Output('out', ParamFun(myfun)(in1.tw(0.4), in1.tw(0.4)))
        test = Neu4mes(verbose=False)
        test.addModel(out)
        test.neuralizeModel(0.1)
        with self.assertRaises(AssertionError):
            test({'in1': [[1, 2, 2], [3, 4, 5]]})
    
    def test_parametric_function_and_fir(self):
        torch.manual_seed(1)
        in1 = Input('in1')
        in2 = Input('in2')
        out = Output('out', Fir(ParamFun(myfun2)(in1.tw(0.4))))
        test = Neu4mes(verbose=False)
        test.addModel(out)
        test.neuralizeModel(0.1)

        with self.assertRaises(AssertionError):
            test({'in1': [1, 2, 2]})
        results = test({'in1': [1, 2, 2, 4]})
        self.assertAlmostEqual(results['out'], [0.022739043459296227])
        results = test({'in1': [[1, 2, 2, 4],[2, 2, 4, 5]]}, sampled=True)
        self.assertAlmostEqual(results['out'], [0.022739043459296227, 0.005036544054746628])
        results = test({'in1': [1, 2, 2, 4, 5]})
        self.assertAlmostEqual(results['out'], [0.022739043459296227, 0.005036544054746628])

        with self.assertRaises(AssertionError):
            Output('out', Fir(ParamFun(myfun2)(in1.tw(0.4),in2.tw(0.2))))

        out = Output('out', Fir(ParamFun(myfun2)(in1.tw(0.4),in2.tw(0.4))))
        test = Neu4mes(verbose=False)
        test.addModel(out)
        test.neuralizeModel(0.1)
        with self.assertRaises(AssertionError):
            test({'in1': [[1, 2, 2, 4]]})
        results = test({'in1': [1, 2, 2, 4], 'in2': [1, 2, 2, 4]})
        self.assertAlmostEqual(results['out'], [0.2159799486398697])
        results = test({'in1': [[1, 2, 2, 4]], 'in2': [[1, 2, 2, 4]]}, sampled=True)
        self.assertAlmostEqual(results['out'], [0.2159799486398697])
        results = test({'in1': [1, 2, 2, 4, 5], 'in2': [1, 2, 2, 4]})
        self.assertAlmostEqual(results['out'], [0.2159799486398697])
        results = test({'in1': [[1, 2, 2, 4], [1, 2, 2, 4]], 'in2': [[1, 2, 2, 4]]}, sampled=True)
        self.assertAlmostEqual(results['out'], [0.2159799486398697])
        results = test({'in1': [[1, 2, 2, 4], [1, 2, 2, 4]], 'in2': [[1, 2, 2, 4], [5, 5, 5, 5]]}, sampled=True)
        self.assertAlmostEqual(results['out'], [0.2159799486398697, 0.15265005826950073])

        out = Output('out', Fir(3)(ParamFun(myfun2)(in1.tw(0.4), in2.tw(0.4))))
        test = Neu4mes(verbose=False)
        test.addModel(out)
        test.neuralizeModel(0.1)

        results = test({'in1': [[1, 2, 2, 4], [1, 2, 2, 4]], 'in2': [[1, 2, 2, 4], [1, 2, 2, 4]]}, sampled=True)
        for pred, label in zip(results['out'], [[0.5022241473197937, 0.046179354190826416, 0.2152290940284729], [0.5022241473197937, 0.046179354190826416, 0.2152290940284729]]):
            self.assertAlmostEqual(pred, label)

        parfun = ParamFun(myfun2)
        with self.assertRaises(AssertionError):
            Output('out', parfun(Fir(3)(parfun(in1.tw(0.4), in2.tw(0.4)))))

        parfun = ParamFun(myfun2)
        out = Output('out', parfun(Fir(3)(parfun(in1.tw(0.4)))))
        test = Neu4mes(verbose=False)
        test.addModel(out)
        test.neuralizeModel(0.1)

        results = test({'in1': [[1, 2, 2, 4], [2, 1, 1, 3]]}, sampled=True)
        for pred, label in zip(results['out'], [[0.10244938731193542, 0.4312977194786072, 0.21356363594532013], [0.20510706305503845, 0.3891393542289734, 0.02812940627336502]]):
            self.assertAlmostEqual(pred, label)

        results = test({'in1': [1, 2, 2, 4, 3],'in2': [6, 2, 2, 4, 4]})
        for pred, label in zip(results['out'], [[0.10244938731193542, 0.4312977194786072, 0.21356363594532013], [0.4902294874191284, 0.5814334154129028, 0.3989813029766083]]):
            self.assertAlmostEqual(pred, label)
        results = test({'in1': [[1, 2, 2, 4],[2, 1, 1, 3]],'in2': [[6, 2, 2, 4],[1, 2, 2, 4]]}, sampled=True)
        for pred, label in zip(results['out'], [[0.10244938731193542, 0.4312977194786072, 0.21356363594532013], [0.20510706305503845, 0.3891393542289734, 0.02812940627336502]]):
            self.assertAlmostEqual(pred, label)

if __name__ == '__main__':
    unittest.main()

