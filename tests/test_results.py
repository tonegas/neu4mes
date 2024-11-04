import unittest, os, sys

from neu4mes import *
from neu4mes import relation
relation.CHECK_NAMES = False

from neu4mes.logger import logging, Neu4MesLogger
log = Neu4MesLogger(__name__, logging.CRITICAL)
log.setAllLevel(logging.CRITICAL)

# append a new directory to sys.path
sys.path.append(os.getcwd())

data_folder = os.path.join(os.path.dirname(__file__), 'data/')

class Neu4mesTrainingTest(unittest.TestCase):
    def TestAlmostEqual(self, data1, data2, precision=4):
        assert np.asarray(data1, dtype=np.float32).ndim == np.asarray(data2, dtype=np.float32).ndim, f'Inputs must have the same dimension! Received {type(data1)} and {type(data2)}'
        if type(data1) == type(data2) == list:
            for pred, label in zip(data1, data2):
                self.TestAlmostEqual(pred, label, precision=precision)
        else:
            self.assertAlmostEqual(data1, data2, places=precision)

    def test_analysis_results(self):
        input1 = Input('in1')
        target1 = Input('out1')
        target2 = Input('out2')
        a = Parameter('a', values=[[1]])
        output1 = Output('out', Fir(parameter=a)(input1.last()))

        test = Neu4mes(visualizer=None,seed=42)
        test.addModel('model', output1)
        test.addMinimize('error1', target1.last(), output1)
        test.addMinimize('error2', target2.last(), output1)
        test.neuralizeModel()

        dataset = {'in1': [1,1,1,1,1,1,1,1,1,1], 'out1': [2,2,2,2,2,2,2,2,2,2], 'out2': [3,3,3,3,3,3,3,3,3,3]}
        test.loadData(name='dataset', source=dataset)

        # Test prediction
        test.resultAnalysis('dataset')
        self.assertEqual({'A': [[[2.0]],[[2.0]],[[2.0]],[[2.0]],[[2.0]],[[2.0]],[[2.0]],[[2.0]],[[2.0]],[[2.0]]],
                               'B': [[[1.0]],[[1.0]],[[1.0]],[[1.0]],[[1.0]],[[1.0]],[[1.0]],[[1.0]],[[1.0]],[[1.0]]]},
                         test.prediction['dataset']['error1'])
        self.assertEqual((1.0 ** 2) * 10.0 / 10.0, test.performance['dataset']['error1']['mse'])
        self.assertEqual((2.0 ** 2) * 10.0 / 10.0, test.performance['dataset']['error2']['mse'])
        self.assertEqual((1+4)/2.0, test.performance['dataset']['total']['mean_error'])

        test.resultAnalysis('dataset', batch_size = 5)
        self.assertEqual({'A': [[[2.0]],[[2.0]],[[2.0]],[[2.0]],[[2.0]],[[2.0]],[[2.0]],[[2.0]],[[2.0]],[[2.0]]],
                               'B': [[[1.0]],[[1.0]],[[1.0]],[[1.0]],[[1.0]],[[1.0]],[[1.0]],[[1.0]],[[1.0]],[[1.0]]]},
                         test.prediction['dataset']['error1'])
        self.assertEqual((1.0 ** 2) * 10.0 / 10.0, test.performance['dataset']['error1']['mse'])
        self.assertEqual((2.0 ** 2) * 10.0 / 10.0, test.performance['dataset']['error2']['mse'])
        self.assertEqual((1+4)/2.0, test.performance['dataset']['total']['mean_error'])

        test.resultAnalysis('dataset', batch_size = 6)
        self.assertEqual({'A': [[[2.0]],[[2.0]],[[2.0]],[[2.0]],[[2.0]],[[2.0]]],
                               'B': [[[1.0]],[[1.0]],[[1.0]],[[1.0]],[[1.0]],[[1.0]]]},
                         test.prediction['dataset']['error1'])
        self.assertEqual((1.0 ** 2) * 10.0 / 10.0, test.performance['dataset']['error1']['mse'])
        self.assertEqual((2.0 ** 2) * 10.0 / 10.0, test.performance['dataset']['error2']['mse'])
        self.assertEqual((1+4)/2.0, test.performance['dataset']['total']['mean_error'])

        dataset = {'in1': [1,1,1,1,1,1,2,2,3,3], 'out1': [2,2,2,2,2,2,2,2,2,2], 'out2': [3,3,3,3,3,3,3,3,3,3]}
        test.loadData(name='dataset2', source=dataset)

        test.resultAnalysis('dataset2')
        self.assertAlmostEqual((1.0 ** 2.0) * 8.0 / 10.0, test.performance['dataset2']['error1']['mse'], places=6)
        self.assertAlmostEqual(((2.0 ** 2) * 6.0 + (1.0 ** 2) * 2.0) / 10.0, test.performance['dataset2']['error2']['mse'], places=6)
        self.assertAlmostEqual(((1.0 ** 2) * 8.0 / 10.0 + ((2.0 ** 2) * 6.0 + (1.0 ** 2) * 2.0) / 10.0 )/2.0, test.performance['dataset2']['total']['mean_error'], places=6)

        test.resultAnalysis('dataset2', batch_size = 5)
        self.assertAlmostEqual((1.0 ** 2.0) * 8.0 / 10.0, test.performance['dataset2']['error1']['mse'], places=6)
        self.assertAlmostEqual(((2.0 ** 2) * 6.0 + (1.0 ** 2) * 2.0) / 10.0, test.performance['dataset2']['error2']['mse'], places=6)
        self.assertAlmostEqual(((1.0 ** 2) * 8.0 / 10.0 + ((2.0 ** 2) * 6.0 + (1.0 ** 2) * 2.0) / 10.0 )/2.0, test.performance['dataset2']['total']['mean_error'], places=6)

        test.resultAnalysis('dataset2', batch_size = 6)
        self.assertEqual({'A': [[[2.0]],[[2.0]],[[2.0]],[[2.0]],[[2.0]],[[2.0]]],
                               'B': [[[1.0]],[[1.0]],[[1.0]],[[1.0]],[[1.0]],[[1.0]]]},
                         test.prediction['dataset2']['error1'])
        self.assertEqual((1.0 ** 2) * 6.0 / 6.0, test.performance['dataset2']['error1']['mse'])
        self.assertEqual((2.0 ** 2) * 6.0 / 6.0, test.performance['dataset2']['error2']['mse'])
        self.assertEqual((1+4)/2.0, test.performance['dataset2']['total']['mean_error'])

        test.resultAnalysis('dataset2', minimize_gain={'error1': 0.5, 'error2': 0.0})
        self.assertAlmostEqual((1.0 ** 2.0) * 8.0 / 10.0 * 0.5, test.performance['dataset2']['error1']['mse'], places=6)
        self.assertAlmostEqual(0.0, test.performance['dataset2']['error2']['mse'], places=6)
        self.assertAlmostEqual(((1.0 ** 2) * 8.0 / 10.0 * 0.5 + 0.0)/2.0, test.performance['dataset2']['total']['mean_error'], places=6)

    def test_analysis_results_closed_loop_state(self):
        input1 = State('in1')
        target1 = Input('out1')
        target2 = Input('out2')
        a = Parameter('a', values=[[2]])
        output1 = Output('out', Fir(parameter=a)(input1.last()))

        test = Neu4mes(visualizer=None,seed=42)
        test.addModel('model', output1)
        test.addClosedLoop(output1,input1)
        test.addMinimize('error1', target1.last(), output1)
        test.addMinimize('error2', target2.last(), output1)
        test.neuralizeModel()

        #Prediction samples = None
        dataset = {'in1': [1,1,1,1,1,1,1,1,1,1], 'out1': [2,2,2,2,2,2,2,2,2,2], 'out2': [3,3,3,3,3,3,3,3,3,3]}
        test.loadData(name='dataset', source=dataset)

        # Test prediction
        test.resultAnalysis('dataset')
        self.assertEqual({'A': [[[2.0]],[[2.0]],[[2.0]],[[2.0]],[[2.0]],[[2.0]],[[2.0]],[[2.0]],[[2.0]],[[2.0]]],
                               'B': [[[2.0]],[[2.0]],[[2.0]],[[2.0]],[[2.0]],[[2.0]],[[2.0]],[[2.0]],[[2.0]],[[2.0]]]},
                         test.prediction['dataset']['error1'])
        self.assertEqual((0.0 ** 2) * 10.0 / 10.0, test.performance['dataset']['error1']['mse'])
        self.assertEqual((1.0 ** 2) * 10.0 / 10.0, test.performance['dataset']['error2']['mse'])
        self.assertEqual((0+1)/2.0, test.performance['dataset']['total']['mean_error'])

        dataset = {'in1': [1,1,1,1,1,1,2,2,3,3], 'out1': [2,2,2,2,2,2,2,2,2,2], 'out2': [3,3,3,3,3,3,3,3,3,3]}
        test.loadData(name='dataset2', source=dataset)

        test.resultAnalysis('dataset2', batch_size = 5)
        self.assertEqual((0.0 ** 2) * 10.0 / 10.0, test.performance['dataset']['error1']['mse'])
        self.assertEqual((1.0 ** 2) * 10.0 / 10.0, test.performance['dataset']['error2']['mse'])
        self.assertEqual((0+1)/2.0, test.performance['dataset']['total']['mean_error'])

        # Prediction samples = 5
        dataset = {'in1': [1,2,3,4,5,6,7,8,9,10], 'out1': [11,12,13,14,15,16,17,18,19,20], 'out2': [10,20,30,40,50,60,70,80,90,100]}
        test.loadData(name='dataset', source=dataset)

        test.resultAnalysis('dataset', prediction_samples=5, batch_size=2)
        A =[[[[11.0]], [[12.0]], [[13.0]], [[14.0]]],
                                   [[[12.0]], [[13.0]], [[14.0]], [[15.0]]],
                                   [[[13.0]], [[14.0]], [[15.0]], [[16.0]]],
                                   [[[14.0]], [[15.0]], [[16.0]], [[17.0]]],
                                   [[[15.0]], [[16.0]], [[17.0]], [[18.0]]],
                                   [[[16.0]], [[17.0]], [[18.0]], [[19.0]]]]
        B =[[[[2.0]], [[4.0]], [[6.0]], [[8.0]]],
                                   [[[4.0]], [[8.0]], [[12.0]], [[16.0]]],
                                   [[[8.0]], [[16.0]], [[24.0]], [[32.0]]],
                                   [[[16.0]], [[32.0]], [[48.0]], [[64.0]]],
                                   [[[32.0]], [[64.0]], [[96.0]], [[128.0]]],
                                   [[[64.0]], [[128.0]], [[192.0]], [[256.0]]]]
        C = [[[[10.0]], [[20.0]], [[30.0]], [[40.0]]],
                                   [[[20.0]], [[30.0]], [[40.0]], [[50.0]]],
                                   [[[30.0]], [[40.0]], [[50.0]], [[60.0]]],
                                   [[[40.0]], [[50.0]], [[60.0]], [[70.0]]],
                                   [[[50.0]], [[60.0]], [[70.0]], [[80.0]]],
                                   [[[60.0]], [[70.0]], [[80.0]], [[90.0]]]]
        self.assertEqual({'A': A, 'B': B}, test.prediction['dataset']['error1'])
        self.assertEqual({'A': C, 'B': B}, test.prediction['dataset']['error2'])
        self.assertAlmostEqual(np.sum((np.array(A).flatten()-np.array(B).flatten())**2)/24.0, test.performance['dataset']['error1']['mse'], places=3)
        self.assertAlmostEqual(np.sum((np.array(C).flatten()-np.array(B).flatten())**2)/24.0, test.performance['dataset']['error2']['mse'], places=3)
        self.assertAlmostEqual((np.sum((np.array(A).flatten()-np.array(B).flatten())**2)/24.0+np.sum((np.array(C).flatten()-np.array(B).flatten())**2)/24.0)/2.0, test.performance['dataset']['total']['mean_error'], places=3)

        test.resultAnalysis('dataset', prediction_samples=5, batch_size=4)
        A =[[[[11.0]], [[12.0]], [[13.0]], [[14.0]]],
                                   [[[12.0]], [[13.0]], [[14.0]], [[15.0]]],
                                   [[[13.0]], [[14.0]], [[15.0]], [[16.0]]],
                                   [[[14.0]], [[15.0]], [[16.0]], [[17.0]]],
                                   [[[15.0]], [[16.0]], [[17.0]], [[18.0]]],
                                   [[[16.0]], [[17.0]], [[18.0]], [[19.0]]]]
        B =[[[[2.0]], [[4.0]], [[6.0]], [[8.0]]],
                                   [[[4.0]], [[8.0]], [[12.0]], [[16.0]]],
                                   [[[8.0]], [[16.0]], [[24.0]], [[32.0]]],
                                   [[[16.0]], [[32.0]], [[48.0]], [[64.0]]],
                                   [[[32.0]], [[64.0]], [[96.0]], [[128.0]]],
                                   [[[64.0]], [[128.0]], [[192.0]], [[256.0]]]]
        C = [[[[10.0]], [[20.0]], [[30.0]], [[40.0]]],
                                   [[[20.0]], [[30.0]], [[40.0]], [[50.0]]],
                                   [[[30.0]], [[40.0]], [[50.0]], [[60.0]]],
                                   [[[40.0]], [[50.0]], [[60.0]], [[70.0]]],
                                   [[[50.0]], [[60.0]], [[70.0]], [[80.0]]],
                                   [[[60.0]], [[70.0]], [[80.0]], [[90.0]]]]
        self.assertEqual({'A': A, 'B': B}, test.prediction['dataset']['error1'])
        self.assertEqual({'A': C, 'B': B}, test.prediction['dataset']['error2'])
        self.assertAlmostEqual(np.sum((np.array(A).flatten()-np.array(B).flatten())**2)/24.0, test.performance['dataset']['error1']['mse'], places=3)
        self.assertAlmostEqual(np.sum((np.array(C).flatten()-np.array(B).flatten())**2)/24.0, test.performance['dataset']['error2']['mse'], places=3)
        self.assertAlmostEqual((np.sum((np.array(A).flatten()-np.array(B).flatten())**2)/24.0+np.sum((np.array(C).flatten()-np.array(B).flatten())**2)/24.0)/2.0, test.performance['dataset']['total']['mean_error'], places=3)

        test.resultAnalysis('dataset', prediction_samples=4, batch_size=6)
        A =[[[[11.0]], [[12.0]], [[13.0]], [[14.0]], [[15.0]], [[16.0]]],
                                   [[[12.0]], [[13.0]], [[14.0]], [[15.0]], [[16.0]], [[17.0]]],
                                   [[[13.0]], [[14.0]], [[15.0]], [[16.0]], [[17.0]], [[18.0]]],
                                   [[[14.0]], [[15.0]], [[16.0]], [[17.0]], [[18.0]], [[19.0]]],
                                   [[[15.0]], [[16.0]], [[17.0]], [[18.0]], [[19.0]], [[20.0]]]]
        B =[[[[2.0]], [[4.0]], [[6.0]], [[8.0]], [[10.0]], [[12.0]]],
                                   [[[4.0]], [[8.0]], [[12.0]], [[16.0]], [[20.0]], [[24.0]]],
                                   [[[8.0]], [[16.0]], [[24.0]], [[32.0]], [[40.0]], [[48.0]]],
                                   [[[16.0]], [[32.0]], [[48.0]], [[64.0]], [[80.0]], [[96.0]]],
                                   [[[32.0]], [[64.0]], [[96.0]], [[128.0]], [[160.0]], [[192.0]]]]
        C = [[[[10.0]], [[20.0]], [[30.0]], [[40.0]], [[50.0]], [[60.0]]],
                                   [[[20.0]], [[30.0]], [[40.0]], [[50.0]], [[60.0]], [[70.0]]],
                                   [[[30.0]], [[40.0]], [[50.0]], [[60.0]], [[70.0]], [[80.0]]],
                                   [[[40.0]], [[50.0]], [[60.0]], [[70.0]], [[80.0]], [[90.0]]],
                                   [[[50.0]], [[60.0]], [[70.0]], [[80.0]], [[90.0]], [[100.0]]]]
        self.assertEqual({'A': A, 'B': B}, test.prediction['dataset']['error1'])
        self.assertEqual({'A': C, 'B': B}, test.prediction['dataset']['error2'])
        self.assertAlmostEqual(np.sum((np.array(A).flatten()-np.array(B).flatten())**2)/30.0, test.performance['dataset']['error1']['mse'], places=3)
        self.assertAlmostEqual(np.sum((np.array(C).flatten()-np.array(B).flatten())**2)/30.0, test.performance['dataset']['error2']['mse'], places=3)
        self.assertAlmostEqual((np.sum((np.array(A).flatten()-np.array(B).flatten())**2)/30.0+np.sum((np.array(C).flatten()-np.array(B).flatten())**2)/30.0)/2.0, test.performance['dataset']['total']['mean_error'], places=3)


    def test_analysis_results_closed_loop(self):
        input1 = Input('in1')
        target1 = Input('out1')
        target2 = Input('out2')
        a = Parameter('a', values=[[2]])
        output1 = Output('out', Fir(parameter=a)(input1.last()))

        test = Neu4mes(visualizer=None, seed=42)
        test.addModel('model', output1)
        test.addMinimize('error1', target1.last(), output1)
        test.addMinimize('error2', target2.last(), output1)
        test.neuralizeModel()

        dataset = {'in1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 'out1': [11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
                   'out2': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]}
        test.loadData(name='dataset', source=dataset)

        test.resultAnalysis('dataset', prediction_samples=5, batch_size=2, closed_loop={'in1':'out'})
        A = [[[[11.0]], [[12.0]], [[13.0]], [[14.0]]],
             [[[12.0]], [[13.0]], [[14.0]], [[15.0]]],
             [[[13.0]], [[14.0]], [[15.0]], [[16.0]]],
             [[[14.0]], [[15.0]], [[16.0]], [[17.0]]],
             [[[15.0]], [[16.0]], [[17.0]], [[18.0]]],
             [[[16.0]], [[17.0]], [[18.0]], [[19.0]]]]
        B = [[[[2.0]], [[4.0]], [[6.0]], [[8.0]]],
             [[[4.0]], [[8.0]], [[12.0]], [[16.0]]],
             [[[8.0]], [[16.0]], [[24.0]], [[32.0]]],
             [[[16.0]], [[32.0]], [[48.0]], [[64.0]]],
             [[[32.0]], [[64.0]], [[96.0]], [[128.0]]],
             [[[64.0]], [[128.0]], [[192.0]], [[256.0]]]]
        C = [[[[10.0]], [[20.0]], [[30.0]], [[40.0]]],
             [[[20.0]], [[30.0]], [[40.0]], [[50.0]]],
             [[[30.0]], [[40.0]], [[50.0]], [[60.0]]],
             [[[40.0]], [[50.0]], [[60.0]], [[70.0]]],
             [[[50.0]], [[60.0]], [[70.0]], [[80.0]]],
             [[[60.0]], [[70.0]], [[80.0]], [[90.0]]]]
        self.assertEqual({'A': A, 'B': B}, test.prediction['dataset']['error1'])
        self.assertEqual({'A': C, 'B': B}, test.prediction['dataset']['error2'])
        self.assertAlmostEqual(np.sum((np.array(A).flatten() - np.array(B).flatten()) ** 2) / 24.0,
                               test.performance['dataset']['error1']['mse'], places=3)
        self.assertAlmostEqual(np.sum((np.array(C).flatten() - np.array(B).flatten()) ** 2) / 24.0,
                               test.performance['dataset']['error2']['mse'], places=3)
        self.assertAlmostEqual((np.sum((np.array(A).flatten() - np.array(B).flatten()) ** 2) / 24.0 + np.sum(
            (np.array(C).flatten() - np.array(B).flatten()) ** 2) / 24.0) / 2.0,
                               test.performance['dataset']['total']['mean_error'], places=3)

        test.resultAnalysis('dataset', prediction_samples=5, batch_size=4, closed_loop={'in1':'out'})
        A = [[[[11.0]], [[12.0]], [[13.0]], [[14.0]]],
             [[[12.0]], [[13.0]], [[14.0]], [[15.0]]],
             [[[13.0]], [[14.0]], [[15.0]], [[16.0]]],
             [[[14.0]], [[15.0]], [[16.0]], [[17.0]]],
             [[[15.0]], [[16.0]], [[17.0]], [[18.0]]],
             [[[16.0]], [[17.0]], [[18.0]], [[19.0]]]]
        B = [[[[2.0]], [[4.0]], [[6.0]], [[8.0]]],
             [[[4.0]], [[8.0]], [[12.0]], [[16.0]]],
             [[[8.0]], [[16.0]], [[24.0]], [[32.0]]],
             [[[16.0]], [[32.0]], [[48.0]], [[64.0]]],
             [[[32.0]], [[64.0]], [[96.0]], [[128.0]]],
             [[[64.0]], [[128.0]], [[192.0]], [[256.0]]]]
        C = [[[[10.0]], [[20.0]], [[30.0]], [[40.0]]],
             [[[20.0]], [[30.0]], [[40.0]], [[50.0]]],
             [[[30.0]], [[40.0]], [[50.0]], [[60.0]]],
             [[[40.0]], [[50.0]], [[60.0]], [[70.0]]],
             [[[50.0]], [[60.0]], [[70.0]], [[80.0]]],
             [[[60.0]], [[70.0]], [[80.0]], [[90.0]]]]
        self.assertEqual({'A': A, 'B': B}, test.prediction['dataset']['error1'])
        self.assertEqual({'A': C, 'B': B}, test.prediction['dataset']['error2'])
        self.assertAlmostEqual(np.sum((np.array(A).flatten() - np.array(B).flatten()) ** 2) / 24.0,
                               test.performance['dataset']['error1']['mse'], places=3)
        self.assertAlmostEqual(np.sum((np.array(C).flatten() - np.array(B).flatten()) ** 2) / 24.0,
                               test.performance['dataset']['error2']['mse'], places=3)
        self.assertAlmostEqual((np.sum((np.array(A).flatten() - np.array(B).flatten()) ** 2) / 24.0 + np.sum(
            (np.array(C).flatten() - np.array(B).flatten()) ** 2) / 24.0) / 2.0,
                               test.performance['dataset']['total']['mean_error'], places=3)

        test.resultAnalysis('dataset', prediction_samples=4, batch_size=6, closed_loop={'in1':'out'})
        A = [[[[11.0]], [[12.0]], [[13.0]], [[14.0]], [[15.0]], [[16.0]]],
             [[[12.0]], [[13.0]], [[14.0]], [[15.0]], [[16.0]], [[17.0]]],
             [[[13.0]], [[14.0]], [[15.0]], [[16.0]], [[17.0]], [[18.0]]],
             [[[14.0]], [[15.0]], [[16.0]], [[17.0]], [[18.0]], [[19.0]]],
             [[[15.0]], [[16.0]], [[17.0]], [[18.0]], [[19.0]], [[20.0]]]]
        B = [[[[2.0]], [[4.0]], [[6.0]], [[8.0]], [[10.0]], [[12.0]]],
             [[[4.0]], [[8.0]], [[12.0]], [[16.0]], [[20.0]], [[24.0]]],
             [[[8.0]], [[16.0]], [[24.0]], [[32.0]], [[40.0]], [[48.0]]],
             [[[16.0]], [[32.0]], [[48.0]], [[64.0]], [[80.0]], [[96.0]]],
             [[[32.0]], [[64.0]], [[96.0]], [[128.0]], [[160.0]], [[192.0]]]]
        C = [[[[10.0]], [[20.0]], [[30.0]], [[40.0]], [[50.0]], [[60.0]]],
             [[[20.0]], [[30.0]], [[40.0]], [[50.0]], [[60.0]], [[70.0]]],
             [[[30.0]], [[40.0]], [[50.0]], [[60.0]], [[70.0]], [[80.0]]],
             [[[40.0]], [[50.0]], [[60.0]], [[70.0]], [[80.0]], [[90.0]]],
             [[[50.0]], [[60.0]], [[70.0]], [[80.0]], [[90.0]], [[100.0]]]]
        self.assertEqual({'A': A, 'B': B}, test.prediction['dataset']['error1'])
        self.assertEqual({'A': C, 'B': B}, test.prediction['dataset']['error2'])
        self.assertAlmostEqual(np.sum((np.array(A).flatten() - np.array(B).flatten()) ** 2) / 30.0,
                               test.performance['dataset']['error1']['mse'], places=3)
        self.assertAlmostEqual(np.sum((np.array(C).flatten() - np.array(B).flatten()) ** 2) / 30.0,
                               test.performance['dataset']['error2']['mse'], places=3)
        self.assertAlmostEqual((np.sum((np.array(A).flatten() - np.array(B).flatten()) ** 2) / 30.0 + np.sum(
            (np.array(C).flatten() - np.array(B).flatten()) ** 2) / 30.0) / 2.0,
                               test.performance['dataset']['total']['mean_error'], places=3)

    def test_analysis_results_connect(self):
        pass

    def test_analysis_results_connect_state(self):
        pass

if __name__ == '__main__':
    unittest.main()