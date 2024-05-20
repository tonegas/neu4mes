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
        results = test({'in1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],'in2': 5})
        self.assertAlmostEqual(results['out'],4.1329298)
        with self.assertRaises(AssertionError):
            test({'in1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],'in3': 5})
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
        out = Output('out', out_fun)
        test = Neu4mes(verbose=True)
        test.addModel(out)
        test.neuralizeModel(0.01)
        results = test({'in1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]})
        self.assertAlmostEqual(results['out'],4.1329298)
        results = test({'in1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],'in2': 5})
        self.assertAlmostEqual(results['out'],4.1329298)
        # TODO deve essere gestito come il dataset cioè come funa finestra temporale che scorre
        results = test({'in1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],'in2': [5, 7]})
        self.assertAlmostEqual(results['out'],[4.1329298, 5.1917152])
        # TODO deve essere gestito come il dataset cioè come funa finestra temporale che scorre
        results = test({'in1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],'in2': [5, 7, 9]})
        self.assertAlmostEqual(results['out'],[4.1329298, 5.1917152, 6.2505002])


if __name__ == '__main__':
    unittest.main()
