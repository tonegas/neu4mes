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

    def test_losses_compare(self):
        input1 = Input('in1')
        target1 = Input('out1')
        target2 = Input('out2')
        a = Parameter('a', values=[[1]])
        output1 = Output('out', Fir(parameter=a)(input1.last()))

        test = Neu4mes(visualizer=None, seed=42)
        test.addModel('model', output1)
        test.addMinimize('error1', target1.last(), output1)
        test.addMinimize('error2', target2.last(), output1)
        test.neuralizeModel()

        dataset = {'in1': [1,1,1,1,1,1,1,1,1,1], 'out1': [2,2,2,2,2,2,2,2,2,2], 'out2': [5,5,5,5,5,5,5,5,5,5]}
        test.loadData(name='dataset', source=dataset)
        test.trainModel(optimizer='SGD', num_of_epochs=5, lr=0.5)
        self.TestAlmostEqual( [[[2.0]], [[2.0]], [[2.0]], [[2.0]], [[2.0]], [[2.0]], [[2.0]], [[2.0]], [[2.0]], [[2.0]]], test.prediction['train_dataset_0.70']['error1']['A'])
        self.TestAlmostEqual([[[6.0]],[[6.0]],[[6.0]],[[6.0]],[[6.0]],[[6.0]],[[6.0]],[[6.0]],[[6.0]],[[6.0]]] ,test.prediction['train_dataset_0.70']['error1']['B'])
        self.TestAlmostEqual( [[[5.0]], [[5.0]], [[5.0]], [[5.0]], [[5.0]], [[5.0]], [[5.0]], [[5.0]], [[5.0]], [[5.0]]], test.prediction['train_dataset_0.70']['error2']['A'])
        self.TestAlmostEqual([1.0, 16.0, 1.0, 16.0, 1.0], test.training['error1']['train'])
        self.TestAlmostEqual([16.0, 1.0, 16.0, 1.0, 16.0], test.training['error1']['val'])
        self.TestAlmostEqual([16.0, 1.0, 16.0, 1.0, 16.0], test.training['error2']['train'])
        self.TestAlmostEqual([1.0, 16.0, 1.0, 16.0, 1.0], test.training['error2']['val'])
        self.TestAlmostEqual(test.performance['validation_dataset_0.20']['error1']['mse'], test.training['error1']['val'][-1])
        self.TestAlmostEqual(test.performance['validation_dataset_0.20']['error2']['mse'], test.training['error2']['val'][-1])
        self.TestAlmostEqual(test.performance['validation_dataset_0.20']['total']['mean_error'],
                             (test.training['error1']['val'][-1] + test.training['error2']['val'][-1]) / 2.0)
        self.TestAlmostEqual(test.performance['test_dataset_0.10']['error1']['mse'], test.training['error1']['val'][-1])
        self.TestAlmostEqual(test.performance['test_dataset_0.10']['error2']['mse'], test.training['error2']['val'][-1])
        self.TestAlmostEqual(test.performance['test_dataset_0.10']['total']['mean_error'], (test.training['error1']['val'][-1]+test.training['error2']['val'][-1])/2.0)

        test.neuralizeModel(clear_model=True)
        test.trainModel(optimizer='SGD', splits=[60,20,20], num_of_epochs=5, lr=0.5, train_batch_size=2)
        self.TestAlmostEqual( [[[2.0]], [[2.0]], [[2.0]], [[2.0]], [[2.0]], [[2.0]], [[2.0]], [[2.0]], [[2.0]], [[2.0]]], test.prediction['train_dataset_0.60']['error1']['A'])
        self.TestAlmostEqual([[[6.0]],[[6.0]],[[6.0]],[[6.0]],[[6.0]],[[6.0]],[[6.0]],[[6.0]],[[6.0]],[[6.0]]] ,test.prediction['train_dataset_0.60']['error1']['B'])
        self.TestAlmostEqual( [[[5.0]], [[5.0]], [[5.0]], [[5.0]], [[5.0]], [[5.0]], [[5.0]], [[5.0]], [[5.0]], [[5.0]]], test.prediction['train_dataset_0.70']['error2']['A'])
        self.TestAlmostEqual([6.0, 11.0, 6.0, 11.0, 6.0], test.training['error1']['train'])
        self.TestAlmostEqual([16.0, 1.0, 16.0, 1.0, 16.0], test.training['error1']['val'])
        self.TestAlmostEqual([11.0, 6.0, 11.0, 6.0, 11.0], test.training['error2']['train'])
        self.TestAlmostEqual([1.0, 16.0, 1.0, 16.0, 1.0], test.training['error2']['val'])
        self.TestAlmostEqual(test.performance['validation_dataset_0.20']['error1']['mse'], test.training['error1']['val'][-1])
        self.TestAlmostEqual(test.performance['validation_dataset_0.20']['error2']['mse'], test.training['error2']['val'][-1])
        self.TestAlmostEqual(test.performance['test_dataset_0.10']['error1']['mse'], test.training['error1']['val'][-1])
        self.TestAlmostEqual(test.performance['test_dataset_0.10']['error2']['mse'], test.training['error2']['val'][-1])

    def test_losses_compare_closed_loop_state(self):
        input1 = State('in1')
        target1 = Input('out1')
        target2 = Input('out2')
        a = Parameter('a', values=[[1]])
        output1 = Output('out', Fir(parameter=a)(input1.last()))

        test = Neu4mes(visualizer=TextVisualizer(5),seed=42)
        test.addModel('model', output1)
        test.addClosedLoop(output1, input1)
        test.addMinimize('error1', target1.last(), output1)
        test.addMinimize('error2', target2.last(), output1)
        test.neuralizeModel()

        dataset = {'in1': [1,1,1,1,1,1,1,1,1,1], 'out1': [2,2,2,2,2,2,2,2,2,2], 'out2': [5,5,5,5,5,5,5,5,5,5]}
        test.loadData(name='dataset', source=dataset)
        test.trainModel(optimizer='SGD', num_of_epochs=5, lr=0.5)
        self.TestAlmostEqual([[[[2.0]], [[2.0]], [[2.0]], [[2.0]], [[2.0]], [[2.0]], [[2.0]], [[2.0]], [[2.0]], [[2.0]]]], test.prediction['train_dataset_0.70']['error1']['A'])
        self.TestAlmostEqual([[[[6.0]], [[6.0]], [[6.0]], [[6.0]], [[6.0]], [[6.0]], [[6.0]], [[6.0]], [[6.0]], [[6.0]]]] ,test.prediction['train_dataset_0.70']['error1']['B'])
        self.TestAlmostEqual([[[[5.0]], [[5.0]], [[5.0]], [[5.0]], [[5.0]], [[5.0]], [[5.0]], [[5.0]], [[5.0]], [[5.0]]]], test.prediction['train_dataset_0.70']['error2']['A'])
        self.TestAlmostEqual([1.0, 16.0, 1.0, 16.0, 1.0], test.training['error1']['train'])
        self.TestAlmostEqual([16.0, 1.0, 16.0, 1.0, 16.0], test.training['error1']['val'])
        self.TestAlmostEqual([16.0, 1.0, 16.0, 1.0, 16.0], test.training['error2']['train'])
        self.TestAlmostEqual([1.0, 16.0, 1.0, 16.0, 1.0], test.training['error2']['val'])
        self.TestAlmostEqual(test.performance['validation_dataset_0.20']['error1']['mse'], test.training['error1']['val'][-1])
        self.TestAlmostEqual(test.performance['validation_dataset_0.20']['error2']['mse'], test.training['error2']['val'][-1])
        self.TestAlmostEqual(test.performance['validation_dataset_0.20']['total']['mean_error'],
                             (test.training['error1']['val'][-1] + test.training['error2']['val'][-1]) / 2.0)
        self.TestAlmostEqual(test.performance['test_dataset_0.10']['error1']['mse'], test.training['error1']['val'][-1])
        self.TestAlmostEqual(test.performance['test_dataset_0.10']['error2']['mse'], test.training['error2']['val'][-1])
        self.TestAlmostEqual(test.performance['test_dataset_0.10']['total']['mean_error'], (test.training['error1']['val'][-1]+test.training['error2']['val'][-1])/2.0)

        test.neuralizeModel(clear_model=True)
        test.trainModel(optimizer='SGD', splits=[60,20,20], num_of_epochs=5, lr=0.5, train_batch_size=2)
        self.TestAlmostEqual([[[[2.0]], [[2.0]], [[2.0]], [[2.0]], [[2.0]], [[2.0]], [[2.0]], [[2.0]], [[2.0]], [[2.0]]]], test.prediction['train_dataset_0.60']['error1']['A'])
        self.TestAlmostEqual([[[[6.0]], [[6.0]], [[6.0]], [[6.0]], [[6.0]], [[6.0]], [[6.0]], [[6.0]], [[6.0]], [[6.0]]]] ,test.prediction['train_dataset_0.60']['error1']['B'])
        self.TestAlmostEqual([[[[5.0]], [[5.0]], [[5.0]], [[5.0]], [[5.0]], [[5.0]], [[5.0]], [[5.0]], [[5.0]], [[5.0]]]], test.prediction['train_dataset_0.70']['error2']['A'])
        self.TestAlmostEqual([6.0, 11.0, 6.0, 11.0, 6.0], test.training['error1']['train'])
        self.TestAlmostEqual([16.0, 1.0, 16.0, 1.0, 16.0], test.training['error1']['val'])
        self.TestAlmostEqual([11.0, 6.0, 11.0, 6.0, 11.0], test.training['error2']['train'])
        self.TestAlmostEqual([1.0, 16.0, 1.0, 16.0, 1.0], test.training['error2']['val'])
        self.TestAlmostEqual(test.performance['validation_dataset_0.20']['error1']['mse'], test.training['error1']['val'][-1])
        self.TestAlmostEqual(test.performance['validation_dataset_0.20']['error2']['mse'], test.training['error2']['val'][-1])
        self.TestAlmostEqual(test.performance['test_dataset_0.10']['error1']['mse'], test.training['error1']['val'][-1])
        self.TestAlmostEqual(test.performance['test_dataset_0.10']['error2']['mse'], test.training['error2']['val'][-1])

        test.neuralizeModel(clear_model=True)
        with self.assertRaises(ValueError):
            test.trainModel(optimizer='SGD', splits=[60,20,20], num_of_epochs=5, lr=0.5, train_batch_size=2, prediction_samples=4)

        test.neuralizeModel(clear_model=True)
        test.trainModel(optimizer='SGD', splits=[50, 50, 0], num_of_epochs=5, lr=0.001, train_batch_size=2, prediction_samples=3)

        self.TestAlmostEqual([[[[2.0]], [[2.0]]], [[[2.0]], [[2.0]]], [[[2.0]], [[2.0]]], [[[2.0]], [[2.0]]]], test.prediction['train_dataset_0.50']['error1']['A'])
        self.TestAlmostEqual([[[[1.1285]], [[1.1285]]], [[[1.2735]], [[1.2735]]], [[[1.4371]], [[1.4371]]], [[[1.6217]], [[1.6217]]]] ,test.prediction['train_dataset_0.50']['error1']['B'])
        self.TestAlmostEqual([[[[5.0]], [[5.0]]], [[[5.0]], [[5.0]]], [[[5.0]], [[5.0]]], [[[5.0]], [[5.0]]]], test.prediction['train_dataset_0.50']['error2']['A'])
        self.TestAlmostEqual([1.0, 0.8768, 0.75615, 0.64057, 0.533], test.training['error1']['train'])
        self.TestAlmostEqual([0.8768, 0.75615, 0.64057, 0.533, 0.4368], test.training['error1']['val'])
        self.TestAlmostEqual([16.0, 15.4923, 14.9602, 14.4059, 13.8328], test.training['error2']['train'])
        self.TestAlmostEqual([15.4923, 14.9602, 14.4059, 13.8328, 13.2457], test.training['error2']['val'])
        self.TestAlmostEqual(test.performance['validation_dataset_0.50']['error1']['mse'], test.training['error1']['val'][-1])
        self.TestAlmostEqual(test.performance['validation_dataset_0.50']['error2']['mse'], test.training['error2']['val'][-1])

    def test_losses_compare_closed_loop(self):
        input1 = Input('in1')
        target1 = Input('out1')
        target2 = Input('out2')
        a = Parameter('a', values=[[1]])
        output1 = Output('out', Fir(parameter=a)(input1.last()))

        test = Neu4mes(visualizer=TextVisualizer(5),seed=42)
        test.addModel('model', output1)
        test.addMinimize('error1', target1.last(), output1)
        test.addMinimize('error2', target2.last(), output1)
        test.neuralizeModel()

        dataset = {'in1': [1,1,1,1,1,1,1,1,1,1], 'out1': [2,2,2,2,2,2,2,2,2,2], 'out2': [5,5,5,5,5,5,5,5,5,5]}
        test.loadData(name='dataset', source=dataset)
        test.trainModel(optimizer='SGD', splits=[50, 50, 0], num_of_epochs=5, lr=0.001, train_batch_size=2, prediction_samples=3, closed_loop={'in1': 'out'})

        self.TestAlmostEqual([[[[2.0]], [[2.0]]], [[[2.0]], [[2.0]]], [[[2.0]], [[2.0]]], [[[2.0]], [[2.0]]]], test.prediction['train_dataset_0.50']['error1']['A'])
        self.TestAlmostEqual([[[[1.1285]], [[1.1285]]], [[[1.2735]], [[1.2735]]], [[[1.4371]], [[1.4371]]], [[[1.6217]], [[1.6217]]]] ,test.prediction['train_dataset_0.50']['error1']['B'])
        self.TestAlmostEqual([[[[5.0]], [[5.0]]], [[[5.0]], [[5.0]]], [[[5.0]], [[5.0]]], [[[5.0]], [[5.0]]]], test.prediction['train_dataset_0.50']['error2']['A'])
        self.TestAlmostEqual([1.0, 0.8768, 0.75615, 0.64057, 0.533], test.training['error1']['train'])
        self.TestAlmostEqual([0.8768, 0.75615, 0.64057, 0.533, 0.4368], test.training['error1']['val'])
        self.TestAlmostEqual([16.0, 15.4923, 14.9602, 14.4059, 13.8328], test.training['error2']['train'])
        self.TestAlmostEqual([15.4923, 14.9602, 14.4059, 13.8328, 13.2457], test.training['error2']['val'])
        self.TestAlmostEqual(test.performance['validation_dataset_0.50']['error1']['mse'], test.training['error1']['val'][-1])
        self.TestAlmostEqual(test.performance['validation_dataset_0.50']['error2']['mse'], test.training['error2']['val'][-1])

    # def test_analysis_results_connect(self):
    #     pass

    # def test_analysis_results_connect_state(self):
    #     pass

if __name__ == '__main__':
    unittest.main()