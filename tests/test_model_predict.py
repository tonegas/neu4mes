import unittest
from neu4mes import *
import torch

class MyTestCase(unittest.TestCase):
    def test_single_in_single_out(self):
        torch.manual_seed(1)
        in1 = Input('in1')
        in2 = Input('in2')
        out_fun = Fir(in1.tw(0.1)) + Fir(in2)
        out = Output('out', out_fun)
        test = Neu4mes(verbose=True)
        test.addModel(out)
        test.neuralizeModel(0.01)
        results = test({'in1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],'in2': [5]})
        self.assertAlmostEqual(results['out'],-0.09323513507843018)
        results = test({'in1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],'in2': 5})
        self.assertAlmostEqual(results['out'],4.1329298)
        # TODO deve essere gestito come il dataset cioè come funa finestra temporale che scorre
        results = test({'in1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], 'in2': [5, 7]})
        self.assertAlmostEqual(results['out'],[4.1329298, 5.1917152])
        # TODO deve essere gestito come il dataset cioè come funa finestra temporale che scorre
        results = test({'in1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], 'in2': [5, 7, 9]})
        self.assertAlmostEqual(results['out'],[4.1329298, 5.1917152, 6.2505002])

    def test_single_in_multi_out(self):
        torch.manual_seed(1)
        in1 = Input('in1')
        f = Fir(3)
        out_fun = f(in1.tw(0.1))
        out1 = Output('out', out_fun)
        out2 = Output('intw2', in1.tw(1))
        out3 = Output('intw3', in1.tw(0.2))
        out4 = Output('intw4', in1.tw([-0.2,0.2]))
        test = Neu4mes(verbose=True)
        test.addModel(out1)
        test.addModel(out2)
        test.addModel(out3)
        test.addModel(out4)
        test.neuralizeModel(0.1)
        results = test({'in1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]})
        #TODO l'uscita deve avere questa dimensione perché queste uscite corrispondo ad una solo auscita temporale
        self.assertAlmostEqual(results['out'],[[4.693689346313477, -9.414368629455566, 5.9971723556518555]])
        self.assertAlmostEqual(results['intw2'], [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
        self.assertAlmostEqual(results['intw3'], [[9, 10]])
        self.assertAlmostEqual(results['intw4'], [[9, 10, 11, 12]])

    def test_parametric_function(self):
        torch.manual_seed(1)
        in1 = Input('in1')
        def myfun(x,P):
            return x*P

        out = Output('out', ParamFun(myfun)(in1))
        test = Neu4mes(verbose=True)
        test.addModel(out)
        test.neuralizeModel(0.1)

        results = test({'in1': 1})
        self.assertAlmostEqual(results['out'], 0.7576315999031067)
        results = test({'in1': [1]})
        self.assertAlmostEqual(results['out'],0.7576315999031067)
        results = test({'in1': [2]})
        self.assertAlmostEqual(results['out'],1.5152631998062134)
        results = test({'in1': [1,2]})
        self.assertAlmostEqual(results['out'],[0.7576315999031067,1.5152631998062134])

        out = Output('out', ParamFun(myfun)(in1.tw(0.2)))
        test = Neu4mes(verbose=True)
        test.addModel(out)
        test.neuralizeModel(0.1)

        with self.assertRaises(AssertionError):#TODO voglio una finestra di almeno 2 sample altrimenti errore
            results = test({'in1': [1]})
        with self.assertRaises(AssertionError):
            results = test({'in1': [2]})
        results = test({'in1': [1,2]})
        self.assertAlmostEqual(results['out'],[[0.7576315999031067,1.5152631998062134]])
        results = test({'in1': [[1,2]]})
        self.assertAlmostEqual(results['out'],[[0.7576315999031067,1.5152631998062134]])
        results = test({'in1': [1, 2, 2, 3, 4, 5]})# Qui vengono costruite gli input a due a due con shift di 1
        self.assertAlmostEqual(results['out'], [[0.7576315999031067, 1.5152631998062134], [1.5152631998062134, 2.272894859313965], [2.272894859313965, 3.0305263996124268], [3.0305263996124268, 3.7881579399108887]])
        results = test({'in1': [[1, 2], [2, 3], [3, 4], [4, 5]]})
        self.assertAlmostEqual(results['out'], [[0.7576315999031067, 1.5152631998062134], [1.5152631998062134, 2.272894859313965], [2.272894859313965, 3.0305263996124268], [3.0305263996124268, 3.7881579399108887]])


        out = Output('out', ParamFun(myfun)(in1,in1))
        test = Neu4mes(verbose=True)
        test.addModel(out)
        test.neuralizeModel(0.1)

        results = test({'in1': 1})
        self.assertAlmostEqual(results['out'],1)
        results = test({'in1': [1]})
        self.assertAlmostEqual(results['out'],1)
        results = test({'in1': [2]})
        self.assertAlmostEqual(results['out'],4)
        results = test({'in1': [1,2]})
        self.assertAlmostEqual(results['out'],[1,4])

        with self.assertRaises(AssertionError):
            Output('out', ParamFun(myfun)(in1.tw(1), in1))

        out = Output('out', ParamFun(myfun)(in1.tw(0.1), in1.tw(0.1)))
        test = Neu4mes(verbose=True)
        test.addModel(out)
        test.neuralizeModel(0.1)

        results = test({'in1': 2})
        self.assertAlmostEqual(results['out'], 4)
        results = test({'in1': [2]})
        self.assertAlmostEqual(results['out'], 4)
        results = test({'in1': [2,1]})
        self.assertAlmostEqual(results['out'], [4,1])

        out = Output('out', ParamFun(myfun)(in1.tw(0.2), in1.tw(0.2)))
        test = Neu4mes(verbose=True)
        test.addModel(out)
        test.neuralizeModel(0.1)

        results = test({'in1': [2,4]})
        self.assertAlmostEqual(results['out'], [4,16])
        results = test({'in1': [[1, 2], [3, 2]]})
        self.assertAlmostEqual(results['out'], [[1.0, 4.0], [9.0, 4.0]])

        out = Output('out', ParamFun(myfun)(in1.tw(0.3), in1.tw(0.3)))
        test = Neu4mes(verbose=True)
        test.addModel(out)
        test.neuralizeModel(0.1)

        with self.assertRaises(AssertionError):
            results = test({'in1': [2]})

        with self.assertRaises(AssertionError):
            results = test({'in1': [2, 4]})

        results = test({'in1': [3,2,1]})
        self.assertAlmostEqual(results['out'], [9,4,1])
        results = test({'in1': [[1,2,2],[3,4,5]]})
        self.assertAlmostEqual(results['out'], [[1, 4, 4], [9, 16, 25]])
        results = test({'in1': [[3, 2, 1], [2, 1, 0]]})
        self.assertAlmostEqual(results['out'], [[9, 4, 1],[4, 1, 0]])
        results = test({'in1': [3,2,1,0]})
        self.assertAlmostEqual(results['out'], [[9, 4, 1],[4, 1, 0]])

        out = Output('out', ParamFun(myfun)(in1.tw(0.4), in1.tw(0.4)))
        test = Neu4mes(verbose=True)
        test.addModel(out)
        test.neuralizeModel(0.1)
        with self.assertRaises(AssertionError):
            test({'in1': [[1, 2, 2], [3, 4, 5]]})

    def test_parametric_function_and_fir(self):
        torch.manual_seed(1)
        in1 = Input('in1')
        in2 = Input('in2')
        def myfun(a, b ,c):
            return np.sin(a + b) * c

        out = Output('out', Fir(ParamFun(myfun)(in1.tw(0.4))))
        test = Neu4mes(verbose=True)
        test.addModel(out)
        test.neuralizeModel(0.1)

        with self.assertRaises(AssertionError):
            test({'in1': [1, 2, 2]})
        results = test({'in1': [1, 2, 2, 4]})
        self.assertAlmostEqual(results['out'], 0.022739043459296227)
        results = test({'in1': [[1, 2, 2, 4],[2, 2, 4, 5]]})
        self.assertAlmostEqual(results['out'], [0.022739043459296227, 0.005036544054746628])
        results = test({'in1': [[1, 2, 2, 4, 5]]})
        self.assertAlmostEqual(results['out'], [0.022739043459296227, 0.005036544054746628])

        with self.assertRaises(AssertionError):
            Output('out', Fir(ParamFun(myfun)(in1.tw(0.4),in2.tw(0.2))))

        out = Output('out', Fir(ParamFun(myfun)(in1.tw(0.4),in2.tw(0.4))))
        test = Neu4mes(verbose=True)
        test.addModel(out)
        test.neuralizeModel(0.1)
        with self.assertRaises(AssertionError):
            test({'in1': [[1, 2, 2, 4]]})
        results = test({'in1': [[1, 2, 2, 4]], 'in2': [[1, 2, 2, 4]]})
        self.assertAlmostEqual(results['out'], [0.171805739402771])
        with self.assertRaises(AssertionError):
            test({'in1': [[1, 2, 2, 4], [1, 2, 2, 4]], 'in2': [[1, 2, 2, 4]]})

        results = test({'in1': [[1, 2, 2, 4], [1, 2, 2, 4]], 'in2': [[1, 2, 2, 4], [5, 5, 5, 5]]})
        self.assertAlmostEqual(results['out'], [0.171805739402771, 0.03363896161317825])

        out = Output('out', Fir(3)(ParamFun(myfun)(in1.tw(0.4), in2.tw(0.4))))
        test = Neu4mes(verbose=True)
        test.addModel(out)
        test.neuralizeModel(0.1)

        results = test({'in1': [[1, 2, 2, 4], [1, 2, 2, 4]], 'in2': [[1, 2, 2, 4], [1, 2, 2, 4]]})
        self.assertAlmostEqual(results['out'], [[-0.03320024162530899, -0.4807766079902649, -0.15642336010932922], [-0.03320024162530899, -0.4807766079902649, -0.15642336010932922]])

        parfun = ParamFun(myfun)
        with self.assertRaises(AssertionError):
            Output('out', parfun(Fir(3)(parfun(in1.tw(0.4), in2.tw(0.4)))))

        out = Output('out', parfun(Fir(3)(parfun(in1.tw(0.4)))))
        test = Neu4mes(verbose=True)
        test.addModel(out)
        test.neuralizeModel(0.1)

        results = test({'in1': [[1, 2, 2, 4], [2, 1, 1, 3]]})
        self.assertAlmostEqual(results['out'], [[0.190605029463768, 0.2285921424627304, 0.17749454081058502], [0.19276225566864014, 0.20867504179477692, 0.16721397638320923]])

        results = test({'in1': [[1, 2, 2, 4],[2, 1, 1, 3]],'in2': [[6, 2, 2, 4],[1, 2, 2, 4]]})
        self.assertAlmostEqual(results['out'], [[0.190605029463768, 0.2285921424627304, 0.17749454081058502], [0.19276225566864014, 0.20867504179477692, 0.16721397638320923]])



if __name__ == '__main__':
    unittest.main()
