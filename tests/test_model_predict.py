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
        self.assertAlmostEqual(results['out'],4.1329298)
        #results = test({'in1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],'in2': 5})
        #self.assertAlmostEqual(results['out'],4.1329298)
        #with self.assertRaises(AssertionError):
        #    test({'in1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],'in3': 5})
        # TODO deve essere gestito come il dataset cioè come funa finestra temporale che scorre
        #results = test({'in1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], 'in2': [5, 7]})
        #self.assertAlmostEqual(results['out'],[4.1329298, 5.1917152])
        # TODO deve essere gestito come il dataset cioè come funa finestra temporale che scorre
        #results = test({'in1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], 'in2': [5, 7, 9]})
        #self.assertAlmostEqual(results['out'],[4.1329298, 5.1917152, 6.2505002])

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
        self.assertAlmostEqual(results['out'],[5.152631759643555, -4.413782119750977, -1.938614845275879])
        self.assertAlmostEqual(results['intw2'], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        self.assertAlmostEqual(results['intw3'], [9, 10])
        self.assertAlmostEqual(results['intw4'], [9, 10, 11, 12])



if __name__ == '__main__':
    unittest.main()
