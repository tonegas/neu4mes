import sys, os, torch
sys.path.append(os.getcwd())
import unittest
from neu4mes import *
from neu4mes import relation
relation.CHECK_NAMES = False

from neu4mes.logger import logging, Neu4MesLogger
log = Neu4MesLogger(__name__, logging.CRITICAL)
log.setAllLevel(logging.CRITICAL)

# 14 Tests
# This file test the model prediction when closed loop or connect are present in particular the output value
# Test the state variables and the connect_values

# Dimensions
# The first dimension must indicate the time dimension i.e. how many time samples I asked for
# The second dimension indicates the output time dimension for each sample.
# The third is the size of the signal

def myfun(x, P):
    out = x*P
    return out[:,1:,:]

def myfunsum(x, P):
    out = x + P
    return out

def matmul(x,y):
    import torch
    return torch.matmul(torch.transpose(x,1,2),y)

# def myfun2(a, b ,c):
#     import torch
#     return torch.sin(a + b) * c
#
# def myfun3(a, b, p1, p2):
#     import torch
#     at = torch.transpose(a[:, :, 0:2],1,2)
#     bt = torch.transpose(b, 1, 2)
#     return torch.matmul(p1,at+bt)+p2.t()

class MyTestCase(unittest.TestCase):
    
    def TestAlmostEqual(self, data1, data2, precision=4):
        assert np.asarray(data1, dtype=np.float32).ndim == np.asarray(data2, dtype=np.float32).ndim, f'Inputs must have the same dimension! Received {type(data1)} and {type(data2)}'
        if type(data1) == type(data2) == list:  
            for pred, label in zip(data1, data2):
                self.TestAlmostEqual(pred, label, precision=precision)
        else:
            self.assertAlmostEqual(data1, data2, places=precision)

    def test_predict_and_states_values_fir_simple_closed_loop(self):
        x = Input('x')
        x_state = State('x_state')
        p = Parameter('p', dimensions=1, sw=1, values=[[1.0]])
        rel_x = Fir(parameter=p)(x_state.last())
        rel_x = ClosedLoop(rel_x, x_state)
        out = Output('out', rel_x)

        test = Neu4mes(visualizer=None, seed=42)
        test.addModel('out',out)
        test.addMinimize('pos_x', x.next(), out)
        test.neuralizeModel(0.01)

        result = test(inputs={'x': [2], 'x_state':[1]})
        self.assertEqual(test.model.states['x_state'], torch.tensor(result['out']))
        self.assertEqual({'out': [1]}, result)
        result = test(inputs={'x': [2]})
        self.assertEqual(test.model.states['x_state'], torch.tensor(1.0))
        self.assertEqual({'out': [1.0]}, result)
        test.resetStates()
        result = test(inputs={'x': [2]})
        self.assertEqual(test.model.states['x_state'], torch.tensor(0.0))
        self.assertEqual({'out': [0.0]}, result)

    def test_predict_values_fir_simple_closed_loop_predict(self):
        x = Input('x')
        x_in = Input('x_in')
        p = Parameter('p', dimensions=1, sw=1, values=[[1.0]])
        rel_x = Fir(parameter=p)(x_in.last())
        #rel_x = ClosedLoop(rel_x, x_state)
        out = Output('out', rel_x)

        test = Neu4mes(visualizer=None, seed=42)
        test.addModel('out',out)
        test.addMinimize('pos_x', x.next(), out)
        test.neuralizeModel(0.01)

        result = test(inputs={'x': [2], 'x_in':[1]},closed_loop={'x_in':'out'})
        self.assertEqual({'out':[1]}, result)
        result = test(inputs={'x': [2]}, closed_loop={'x_in':'out'})
        self.assertEqual({'out': [0.0]}, result)

    def test_predict_values_fir_closed_loop(self):
        ## the memory is not shared between different calls
        x = Input('x')
        F = State('F')
        p = Parameter('p', tw=0.5, dimensions=1, values=[[1.0],[1.0],[1.0],[1.0],[1.0]])
        x_out = Fir(parameter=p)(x.tw(0.5))+F.last()
        out = Output('out',x_out)
        test = Neu4mes(visualizer=None, seed=42)
        test.addClosedLoop(x_out,F)
        test.addModel('out',out)
        test.neuralizeModel(0.1)

        ## one sample prediction with F initialized with zeros
        test.resetStates()
        result = test(inputs={'x':[1,2,3,4,5]})
        self.assertEqual(result['out'], [15.0])
        ## 5 samples prediction with F initialized with zero only the first time
        test.resetStates()
        result = test(inputs={'x':[1,2,3,4,5,6,7,8,9]})
        self.assertEqual(result['out'], [15.0, 35.0, 60.0, 90.0, 125.0])
        ## one sample prediction with F initialized with [1]
        test.resetStates()
        result = test(inputs={'x':[1,2,3,4,5], 'F':[1]})
        self.assertEqual(result['out'], [16.0])
        ## 5 samples prediction with F initialized with [1] only the first time
        test.resetStates()
        result = test(inputs={'x':[1,2,3,4,5,6,7,8,9], 'F':[1]})
        self.assertEqual(result['out'], [16.0, 36.0, 61.0, 91.0, 126.0])
        ## 5 samples prediction with F initialized with [1] the first time, [2] the second time and [3] the third time
        test.resetStates()
        result = test(inputs={'x':[1,2,3,4,5,6,7,8,9], 'F':[1,2,3]})
        self.assertEqual(result['out'], [16.0, 22.0, 28.0, 58.0, 93.0])
        ## one sample prediction with F initialized with [1] (the other values are ignored)
        test.resetStates()
        result = test(inputs={'x':[1,2,3,4,5], 'F':[1,2,3]})
        self.assertEqual(result['out'], [16.0])

    def test_predict_values_fir_closed_loop_predict(self):
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
        result = test(inputs={'x':[1,2,3,4,5]}, closed_loop={'F':'out'})
        self.assertEqual(result['out'], [15.0])
        ## 5 samples prediction with F initialized with zero only the first time
        result = test(inputs={'x':[1,2,3,4,5,6,7,8,9]}, closed_loop={'F':'out'})
        self.assertEqual(result['out'], [15.0, 35.0, 60.0, 90.0, 125.0])
        ## one sample prediction with F initialized with [1]
        result = test(inputs={'x':[1,2,3,4,5], 'F':[1]}, closed_loop={'F':'out'})
        self.assertEqual(result['out'], [16.0])
        ## 5 samples prediction with F initialized with [1] only the first time
        result = test(inputs={'x':[1,2,3,4,5,6,7,8,9], 'F':[1]}, closed_loop={'F':'out'})
        self.assertEqual(result['out'], [16.0, 36.0, 61.0, 91.0, 126.0])
        ## 5 samples prediction with F initialized with [1] the first time, [2] the second time and [3] the third time
        result = test(inputs={'x':[1,2,3,4,5,6,7,8,9], 'F':[1,2,3]}, closed_loop={'F':'out'})
        self.assertEqual(result['out'], [16.0, 22.0, 28.0, 58.0, 93.0])
        ## one sample prediction with F initialized with [1] (the other values are ignored)
        result = test(inputs={'x':[1,2,3,4,5], 'F':[1,2,3]}, closed_loop={'F':'out'})
        self.assertEqual(result['out'], [16.0])

    def test_predict_values_2fir_closed_loop(self):
        ## the memory is not shared between different calls
        x = State('x')
        y = State('y')
        p = Parameter('p', tw=0.5, dimensions=1, values=[[1.0],[1.0],[1.0],[1.0],[1.0]])
        n = Parameter('n', tw=0.5, dimensions=1, values=[[-1.0],[-1.0],[-1.0],[-1.0],[-1.0]])
        fir_pos = Fir(parameter=p)(x.tw(0.5))
        fir_neg = Fir(parameter=n)(y.tw(0.5))
        out_pos = Output('out_pos', fir_pos)
        out_neg = Output('out_neg', fir_neg)
        out = Output('out',fir_neg+fir_pos)
        test = Neu4mes(visualizer=None, seed=42)
        test.addClosedLoop(fir_pos, x)
        test.addClosedLoop(fir_neg, y)
        test.addModel('out', out)
        test.addModel('out_pos',out_pos)
        test.addModel('out_neg',out_neg)
        test.neuralizeModel(0.1)

        ## one sample prediction with both close loops
        result = test(inputs={'x':[1,2,3,4,5], 'y':[1,2,3,4,5]})
        self.assertEqual(result['out'], [0.0])
        self.assertEqual(result['out_pos'], [15.0])
        self.assertEqual(result['out_neg'], [-15.0])
        ## three sample prediction due to the max dimensions of inputs + prediction_samples
        result = test(inputs={'x':[1,2,3,4,5], 'y':[1,2,3,4,5]}, prediction_samples=2, num_of_samples=3)
        self.assertEqual(result['out'], [0.0, 30.0, 58.0])
        self.assertEqual(result['out_pos'], [15.0, 29.0, 56.0])
        self.assertEqual(result['out_neg'], [-15.0, 1.0, 2.0])
        ## three sample prediction with both close loops but y gets initialized for 3 steps
        ## (!! since all the inputs are recurrent we must specify the prediction horizon (defualt=1))
        result = test(inputs={'x':[1,2,3,4,5], 'y':[1,2,3,4,5,6,7]}, prediction_samples=2, num_of_samples=3)
        self.assertEqual(result['out'], [0.0, 30.0, 58.0])
        self.assertEqual(result['out_pos'], [15.0, 29.0, 56.0])
        self.assertEqual(result['out_neg'], [-15.0, 1.0, 2.0])

        test.resetStates()
        result = test(inputs={'x': [1, 2, 3, 4, 5], 'y': [1, 2, 3, 4, 5, 6, 7]}, num_of_samples=3)
        self.assertEqual(result['out'], [0.0, 9.0, 31.0])
        self.assertEqual(result['out_pos'], [15.0, 29.0, 56.0])
        self.assertEqual(result['out_neg'], [-15.0, -20.0, -25.0])

    def test_predict_values_2fir_closed_loop_predict(self):
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

        ## two sample one prediction for x in close loop
        result = test(inputs={'x':[1,2,3,4,5], 'y':[1,2,3,4,5,6]}, closed_loop={'x':'out_pos'})
        self.assertEqual(result['out'], [0.0, 9.0])
        self.assertEqual(result['out_pos'], [15.0, 29.0])
        self.assertEqual(result['out_neg'], [-15.0, -20.0])
        ## two sample one prediction for y in close loop
        result = test(inputs={'x':[1,2,3,4,5,6], 'y':[1,2,3,4,5]}, closed_loop={'y':'out_pos'})
        self.assertEqual(result['out'], [0.0, -9.0])
        self.assertEqual(result['out_pos'], [15.0, 20.0])
        self.assertEqual(result['out_neg'], [-15.0, -29.0])
        ## one sample prediction with both close loops
        result = test(inputs={'x':[1,2,3,4,5], 'y':[1,2,3,4,5]}, closed_loop={'x':'out_pos', 'y':'out_neg'})
        self.assertEqual(result['out'], [0.0])
        self.assertEqual(result['out_pos'], [15.0])
        self.assertEqual(result['out_neg'], [-15.0])
        ## three sample prediction due to the max dimensions of inputs + prediction_samples
        result = test(inputs={'x':[1,2,3,4,5], 'y':[1,2,3,4,5]}, closed_loop={'x':'out_pos', 'y':'out_neg'}, prediction_samples=2, num_of_samples=3)
        self.assertEqual(result['out'], [0.0, 30.0, 58.0])
        self.assertEqual(result['out_pos'], [15.0, 29.0, 56.0])
        self.assertEqual(result['out_neg'], [-15.0, 1.0, 2.0])
        ## three sample prediction with both close loops but y gets initialized for 3 steps
        ## (!! since all the inputs are recurrent we must specify the prediction horizon (defualt=1))
        result = test(inputs={'x':[1,2,3,4,5], 'y':[1,2,3,4,5,6,7]}, closed_loop={'x':'out_pos', 'y':'out_neg'}, prediction_samples=2, num_of_samples=3)
        ## 1+2+3+4+5 -1-2-3-4-5 2+3+4+5+15 -2-3-4-5+15
        #self.assertEqual(result['out'], [0.0, 9.0, 31.0])
        self.assertEqual(result['out_pos'], [15.0, 29.0, 56.0])
        self.assertEqual(result['out_neg'], [-15.0, 1.0, 2.0])

    def test_predict_values_3states_closed_loop(self):
        ## the state is saved inside the model so the memory is shared between different calls
        x = Input('x') 
        F_state = State('F')
        y_state = State('y')
        z_state = State('z')
        p = Parameter('p', tw=0.5, dimensions=1, values=[[1.0],[1.0],[1.0],[1.0],[1.0]])
        x_out = Fir(parameter=p)(x.tw(0.5))+F_state.last()+y_state.last()+z_state.last()
        x_out = ClosedLoop(x_out, F_state)
        x_out = ClosedLoop(x_out, y_state)
        x_out = ClosedLoop(x_out, z_state)
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
        test.resetStates()
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
        # 2 samples prediction with state variables inizialized only at %prediction_samples
        result = test(inputs={'F': [1,2,3,4], 'y': [1,2], 'z': [1,2,3,4,5]}, prediction_samples=2, num_of_samples=4)
        # 1+1+1 = 3, 3+3+3 = 9, 9+9+9 = 27, 4+0+4 = 8, 8+8+8 = 24
        self.assertEqual(result['out'], [3.0, 9.0, 27.0, 8.0])
        #self.assertEqual(result['out'], [3.0,9.0,27.0,8.0])
        #self.assertEqual(result['out'], [3.0, 6.0, 12.0, 20.0])

    def test_predict_values_3states_closed_loop_predict(self):
        ## the state is saved inside the model so the memory is shared between different calls
        x = Input('x')
        F_state = Input('F')
        y_state = Input('y')
        z_state = Input('z')
        p = Parameter('p', tw=0.5, dimensions=1, values=[[1.0],[1.0],[1.0],[1.0],[1.0]])
        x_out = Fir(parameter=p)(x.tw(0.5))+F_state.last()+y_state.last()+z_state.last()
        out = Output('out',x_out)

        test = Neu4mes(visualizer=None, seed=42)
        test.addModel('out',out)
        test.neuralizeModel(0.1)

        ## one sample prediction with state variables not initialized
        ## (they will have the last valid state)
        result = test(inputs={'x':[1,2,3,4,5]},closed_loop={'F':'out','x':'out','z':'out'})
        self.assertEqual(result['out'], [15.0])
        ## 5 sample prediction with state variables not initialized
        ## (the first prediction will preserve the state of the previous test [15.0])
        result = test(inputs={'x':[1,2,3,4,5,6,7,8,9]}, closed_loop={'F':'out','x':'out','z':'out'})
        #self.assertEqual(result['out'], [60.0, 200.0, 625.0, 1905.0, 5750.0]) #TODO
        ## one sample prediction with state variables initialized with zero
        test.resetStates()
        result = test(inputs={'x':[1,2,3,4,5]}, closed_loop={'F':'out','x':'out','z':'out'})
        self.assertEqual(result['out'], [15.0])
        ## one sample prediction with F initialized with [1] and the others not initialized (so they will have 15.0 in the memory)
        result = test(inputs={'x':[1,2,3,4,5], 'F':[1]}, closed_loop={'F':'out','x':'out','z':'out'})
        self.assertEqual(result['out'], [16.0])
        ## one sample prediction with all the state variables initialized
        result = test(inputs={'x':[1,2,3,4,5], 'F':[1], 'y':[2], 'z':[3]}, closed_loop={'F':'out','x':'out','z':'out'})
        self.assertEqual(result['out'], [21.0])
        ## 5 samples prediction with state variables initialized as many times as they have values to take
        result = test(inputs={'x':[1,2,3,4,5,6,7,8,9], 'F':[1,2,3], 'y':[2,3], 'z':[3]}, closed_loop={'F':'out','x':'out','z':'out'})
        #self.assertEqual(result['out'], [21.0, 46.0, 120.0, 390.0, 1205.0])
        # 2 samples prediction with state variables inizialized only at %prediction_samples
        result = test(inputs={'F': [1,2,3,4], 'y': [1,2], 'z': [1,2,3,4,5]}, closed_loop={'F':'out','x':'out','z':'out'}, prediction_samples=2)
        # 1+1+1 = 3, 3+3+3 = 9, 9+9+9 = 27, 4+0+4 = 8, 8+8+8 = 24
        #self.assertEqual(result['out'], [3.0,9.0,27.0,8.0])
        #self.assertEqual(result['out'], [3.0, 6.0, 12.0, 20.0]) #TODO

    def test_predict_values_and_states_3states_more_window_closed_loop(self):
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
        y_fir = ClosedLoop(y_fir, y_state)
        z_fir = ClosedLoop(z_fir, z_state)
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
        test.resetStates({'y'})
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
        test.resetStates()
        self.assertEqual(test.model.states['y'].numpy().tolist(), [[[0.0], [0.0], [0.0], [0.0], [0.0]]])
        self.assertEqual(test.model.states['z'].numpy().tolist(), [[[0.0], [0.0], [0.0], [0.0], [0.0]]])

    def test_predict_values_and_states_3states_more_window_closed_loop_predict(self):
        ## the state is saved inside the model so the memory is shared between different calls
        x = Input('x')
        y_state = Input('y')
        z_state = Input('z')
        x_p = Parameter('x_p', tw=0.5, dimensions=1, values=[[1.0],[1.0],[1.0],[1.0],[1.0]])
        y_p = Parameter('y_p', tw=0.5, dimensions=1, values=[[2.0],[2.0],[2.0],[2.0],[2.0]])
        z_p = Parameter('z_p', tw=0.5, dimensions=1, values=[[3.0],[3.0],[3.0],[3.0],[3.0]])
        x_fir = Fir(parameter=x_p)(x.tw(0.5))
        y_fir = Fir(parameter=y_p)(y_state.tw(0.5))
        z_fir = Fir(parameter=z_p)(z_state.tw(0.5))
        out_x = Output('out_x', x_fir)
        out_y = Output('out_y', y_fir)
        out_z = Output('out_z', z_fir)
        out = Output('out',x_fir+y_fir+z_fir)

        test = Neu4mes(visualizer=None, seed=42)
        test.addModel('out_all',[out, out_x, out_y, out_z])
        test.neuralizeModel(0.1)

        ## one sample prediction with state variables not initialized
        ## (they will have the last valid state)
        result = test(inputs={'x':[1,2,3,4,5]}, closed_loop={'y':'out_y', 'z':'out_z'})
        self.assertEqual(result['out'], [15.0])
        self.assertEqual(result['out_x'], [15.0])
        self.assertEqual(result['out_y'], [0.0])
        self.assertEqual(result['out_z'], [0.0])
        ## 1 sample prediction with state variables all initialized
        result = test(inputs={'x':[1,2,3,4,5], 'y':[1,2,3,4,5], 'z':[1,2,3,4,5]}, closed_loop={'y':'out_y', 'z':'out_z'})
        self.assertEqual(result['out'], [90.0])
        self.assertEqual(result['out_x'], [15.0])
        self.assertEqual(result['out_y'], [30.0])
        self.assertEqual(result['out_z'], [45.0])
        ## multi-sample prediction with states initialized as many times as they have values
        result = test(inputs={'x':[1,2,3,4,5,6,7,8,9], 'y':[1,2,3,4,5,6,7], 'z':[1,2,3,4,5,6]}, closed_loop={'y':'out_y', 'z':'out_z'})
        self.assertEqual(result['out'], [90.0, 120.0, 309.0, 1101.0, 4155.0])
        self.assertEqual(result['out_x'], [15.0, 20.0, 25.0, 30.0, 35.0])
        self.assertEqual(result['out_y'], [30.0, 40.0, 50.0, 144.0, 424.0])
        self.assertEqual(result['out_z'], [45.0, 60.0, 234.0, 927.0, 3696.0])

    def test_predict_values_and_states_2states_more_window_connect(self):
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
        x_fir = Connect(x_fir, y_state)
        x_fir = Connect(x_fir, z_state)
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
        self.assertEqual(result['out'], [90.0])
        self.assertEqual(result['out_x'], [15.0])
        self.assertEqual(result['out_y'], [30.0])
        self.assertEqual(result['out_z'], [45.0])
        self.assertEqual(test.model.states['y'].numpy().tolist(), [[[0.0], [0.0], [0.0], [0.0], [15.0]]])
        self.assertEqual(test.model.states['z'].numpy().tolist(), [[[0.0], [0.0], [0.0], [0.0], [15.0]]])
        # Replace insead of rolling
        # self.assertEqual(test.model.states['y'].numpy().tolist(), [[[0.0], [0.0], [0.0], [15.0], [0.0]]])
        # self.assertEqual(test.model.states['z'].numpy().tolist(), [[[0.0], [0.0], [0.0], [15.0], [0.0]]])
        ## 1 sample prediction with state variables all initialized
        result = test(inputs={'x':[1,2,3,4,5], 'y':[1,2,3,4,5], 'z':[1,2,3,4,5]})
        self.assertEqual(result['out_x'], [15.0])
        #(1+2+3+4+5)+(2+3+4+5+(1+2+3+4+5))*2+(2+3+4+5+(1+2+3+4+5))*3
        self.assertEqual(result['out'], [160.0])
        self.assertEqual(result['out_y'], [58.0])
        self.assertEqual(result['out_z'], [87.0])
        self.assertEqual(test.model.states['y'].numpy().tolist(), [[[2.0], [3.0], [4.0], [5.0], [15.0]]])
        self.assertEqual(test.model.states['z'].numpy().tolist(), [[[2.0], [3.0], [4.0], [5.0], [15.0]]])
        # Replace instead of rolling
        #(1+2+3+4+5)+(1+2+3+4+(1+2+3+4+5))*2+(1+2+3+4+(1+2+3+4+5))*3
        # self.assertEqual(result['out'], [140.0])
        # self.assertEqual(result['out_y'], [50.0])
        # self.assertEqual(result['out_z'], [75.0])
        # self.assertEqual(test.model.states['y'].numpy().tolist(), [[[2.0], [3.0], [4.0], [15.0], [1.0]]])
        # self.assertEqual(test.model.states['z'].numpy().tolist(), [[[2.0], [3.0], [4.0], [15.0], [1.0]]])
        ## clear state of y
        test.resetStates({'y'})
        self.assertEqual(test.model.states['y'].numpy().tolist(), [[[0.0], [0.0], [0.0], [0.0], [0.0]]])
        self.assertEqual(test.model.states['z'].numpy().tolist(), [[[2.0], [3.0], [4.0], [5.0], [15.0]]])
        # # Replace insead of rolling
        # ## clear state of y
        # test.resetStates({'y'})
        # self.assertEqual(test.model.states['y'].numpy().tolist(), [[[0.0], [0.0], [0.0], [0.0], [0.0]]])
        # self.assertEqual(test.model.states['z'].numpy().tolist(), [[[2.0], [3.0], [4.0], [15.0], [1.0]]])
        ## multi-sample prediction with states initialized as many times as they have values
        result = test(inputs={'x':[1,2,3,4,5,6,7,8,9], 'y':[1,2,3,4,5,6,7], 'z':[1,2,3,4,5,6]})
        self.assertEqual(result['out_x'], [15.0, 20.0, 25.0, 30.0, 35.0])
        self.assertEqual(result['out_y'], [2*(2+3+4+5+15), 2*(3+4+5+6+20), 2*(4+5+6+7+25), 2*(5+6+7+25+30), 2*(6+7+25+30+35)])
        self.assertEqual(result['out_z'], [3*(2+3+4+5+15), 3*(3+4+5+6+20), 3*(4+5+6+20+25), 3*(5+6+20+25+30), 3*(6+20+25+30+35)])
        self.assertEqual(result['out'], [sum(x) for x in zip(result['out_x'],result['out_y'],result['out_z'])])
        self.assertEqual(test.model.states['y'].numpy().tolist(), [[[6.0], [7.0], [25.0], [30.0], [35.0]]])
        self.assertEqual(test.model.states['z'].numpy().tolist(), [[[6.0], [20.0], [25.0], [30.0], [35.0]]])
        # Replace instead of rolling
        # self.assertEqual(result['out_y'], [2*(1+2+3+4+15), 2*(2+3+4+5+20), 2*(3+4+5+6+25), 2*(4+5+6+25+30), 2*(5+6+25+30+35)])
        # self.assertEqual(result['out_z'], [3*(1+2+3+4+15), 3*(2+3+4+5+20), 3*(3+4+5+20+25), 3*(4+5+20+25+30), 3*(5+20+25+30+35)])
        # self.assertEqual(result['out'], [sum(x) for x in zip(result['out_x'],result['out_y'],result['out_z'])])
        # self.assertEqual(test.model.states['y'].numpy().tolist(), [[[6.0], [25.0], [30.0], [35.0], [5.0]]])
        # self.assertEqual(test.model.states['z'].numpy().tolist(), [[[20.0], [25.0], [30.0], [35.0], [5.0]]])
        ## Clear all states
        test.resetStates()
        self.assertEqual(test.model.states['y'].numpy().tolist(), [[[0.0], [0.0], [0.0], [0.0], [0.0]]])
        self.assertEqual(test.model.states['z'].numpy().tolist(), [[[0.0], [0.0], [0.0], [0.0], [0.0]]])

    def test_predict_values_and_states_2states_more_window_connect_predict(self):
        ## the state is saved inside the model so the memory is shared between different calls
        x = Input('x')
        y_state = Input('y')
        z_state = Input('z')
        x_p = Parameter('x_p', tw=0.5, dimensions=1, values=[[1.0],[1.0],[1.0],[1.0],[1.0]])
        y_p = Parameter('y_p', tw=0.5, dimensions=1, values=[[2.0],[2.0],[2.0],[2.0],[2.0]])
        z_p = Parameter('z_p', tw=0.5, dimensions=1, values=[[3.0],[3.0],[3.0],[3.0],[3.0]])
        x_fir = Fir(parameter=x_p)(x.tw(0.5))
        y_fir = Fir(parameter=y_p)(y_state.tw(0.5))
        z_fir = Fir(parameter=z_p)(z_state.tw(0.5))
        out_x = Output('out_x', x_fir)
        out_y = Output('out_y', y_fir)
        out_z = Output('out_z', z_fir)
        out = Output('out',x_fir+y_fir+z_fir)

        test = Neu4mes(visualizer=None, seed=42)
        test.addModel('out_all',[out, out_x, out_y, out_z])
        test.neuralizeModel(0.1)

        ## one sample prediction with state variables not initialized
        ## (they will have the last valid state)
        result = test(inputs={'x':[1,2,3,4,5]}, connect={'y':'out_x','z':'out_x'})
        self.assertEqual(result['out'], [90.0])
        self.assertEqual(result['out_x'], [15.0])
        self.assertEqual(result['out_y'], [30.0])
        self.assertEqual(result['out_z'], [45.0])
        ## 1 sample prediction with state variables all initialized
        result = test(inputs={'x':[1,2,3,4,5], 'y':[1,2,3,4,5], 'z':[1,2,3,4,5]}, connect={'y':'out_x','z':'out_x'})
        self.assertEqual(result['out_x'], [15.0])
        #(1+2+3+4+5)+(2+3+4+5+(1+2+3+4+5))*2+(2+3+4+5+(1+2+3+4+5))*3
        self.assertEqual(result['out'], [160.0])
        self.assertEqual(result['out_y'], [58.0])
        self.assertEqual(result['out_z'], [87.0])
        # Replace instead of rolling
        #(1+2+3+4+5)+(1+2+3+4+(1+2+3+4+5))*2+(1+2+3+4+(1+2+3+4+5))*3
        # self.assertEqual(result['out'], [140.0])
        # self.assertEqual(result['out_y'], [50.0])
        # self.assertEqual(result['out_z'], [75.0])
        ## multi-sample prediction with states initialized as many times as they have values
        result = test(inputs={'x':[1,2,3,4,5,6,7,8,9], 'y':[1,2,3,4,5,6,7], 'z':[1,2,3,4,5,6]}, connect={'y':'out_x','z':'out_x'})
        self.assertEqual(result['out_x'], [15.0, 20.0, 25.0, 30.0, 35.0])
        self.assertEqual(result['out_y'], [2*(2+3+4+5+15), 2*(3+4+5+6+20), 2*(4+5+6+7+25), 2*(5+6+7+25+30), 2*(6+7+25+30+35)])
        self.assertEqual(result['out_z'], [3*(2+3+4+5+15), 3*(3+4+5+6+20), 3*(4+5+6+20+25), 3*(5+6+20+25+30), 3*(6+20+25+30+35)])
        # Reaplce instead of rolling
        # self.assertEqual(result['out_y'], [2*(1+2+3+4+15), 2*(2+3+4+5+20), 2*(3+4+5+6+25), 2*(4+5+6+25+30), 2*(5+6+25+30+35)])
        # self.assertEqual(result['out_z'], [3*(1+2+3+4+15), 3*(2+3+4+5+20), 3*(3+4+5+20+25), 3*(4+5+20+25+30), 3*(5+20+25+30+35)])
        self.assertEqual(result['out'], [sum(x) for x in zip(result['out_x'],result['out_y'],result['out_z'])])

    def test_predict_values_and_connect_variables_2models_more_window_connect(self):
        ## Model1
        input1 = Input('in1')
        a = Parameter('a', dimensions=1, tw=0.05, values=[[1],[1],[1],[1],[1]])
        output1 = Output('out1', Fir(parameter=a)(input1.tw(0.05)))

        test = Neu4mes(visualizer=None, seed=42)
        test.addModel('model1', output1)
        test.addMinimize('error1', input1.next(), output1)
        test.neuralizeModel(0.01)

        ## Model2
        input2 = Input('in2')
        input3 = State('in3')
        b = Parameter('b', dimensions=1, tw=0.05, values=[[1],[1],[1],[1],[1]])
        c = Parameter('c', dimensions=1, tw=0.03, values=[[1],[1],[1]])
        output2 = Output('out2', Fir(parameter=b)(input2.tw(0.05))+Fir(parameter=c)(input3.tw(0.03)))

        test.addModel('model2', output2)
        test.addConnect(output1,input3)
        test.addMinimize('error2', input2.next(), output2)
        test.neuralizeModel(0.01)

        ## Without connect
        results = test(inputs={'in1':[[1],[2],[3],[4],[5],[6],[7],[8],[9]], 'in2':[[1],[2],[3],[4],[5],[6],[7],[8],[9]], 'in3':[[1],[2],[3],[4],[5],[6]]}, prediction_samples=None)
        self.assertEqual(results['out1'], [15.0, 20.0, 25.0, 30.0])
        self.assertEqual(results['out2'], [21.0, 29.0, 37.0, 45.0])

        ## connect out1 to in3 for 4 samples
        test.resetStates()
        results = test(inputs={'in1':[[1],[2],[3],[4],[5],[6],[7],[8],[9]], 'in2':[[1],[2],[3],[4],[5],[6],[7],[8],[9]]}, prediction_samples=3)
        self.assertEqual(results['out1'], [15.0, 20.0, 25.0, 30.0])
        self.assertEqual(results['out2'], [30.0, 55.0, 85.0, 105.0])
        self.assertEqual(test.model.states['in3'].detach().numpy().tolist(), [[[20.], [25.], [30.]]])
        # Replace insead of rolling
        # self.assertEqual(test.model.states['in3'].detach().numpy().tolist(), [[[25.], [30.], [20.]]])

        ## connect out1 to in3 for 3 samples
        test.resetStates()
        results = test(inputs={'in1':[[1],[2],[3],[4],[5],[6],[7],[8],[9]], 'in2':[[1],[2],[3],[4],[5],[6],[7],[8],[9]]}, prediction_samples=2)
        self.assertEqual(results['out1'], [15.0, 20.0, 25.0, 30.0])
        self.assertEqual(results['out2'], [30.0, 55.0, 85.0, 60.0])
        self.assertEqual(test.model.states['in3'].detach().numpy().tolist(), [[[0.0], [0.], [30.]]])
        # Replace insead of rolling
        # self.assertEqual(test.model.states['in3'].detach().numpy().tolist(), [[[0.], [30.], [0.]]])

        ## connect out1 to in3 for 4 samples (initialize in3 with data)
        test.resetStates()
        results = test(inputs={'in1':[[1],[2],[3],[4],[5],[6],[7],[8],[9]], 'in2':[[1],[2],[3],[4],[5],[6],[7],[8],[9]], 'in3':[[1],[2],[3],[4],[5],[6]]}, prediction_samples=3)
        self.assertEqual(results['out1'], [15.0, 20.0, 25.0, 30.0])
        #(1+2+3+4+5)+(2+3+15)
        #(2+3+4+5+6)+(3+15+20)
        self.assertEqual(results['out2'], [35.0, 58.0, 85.0, 105.0])
        self.assertEqual(test.model.states['in3'].detach().numpy().tolist(), [[[20.], [25.], [30.]]])
        # Replace insead of rolling
        #(1+2+3+4+5)+(1+2+15)
        #(2+3+4+5+6)+(2+15+20)
        # self.assertEqual(results['out2'], [33.0, 57.0, 85.0, 105.0])
        # self.assertEqual(test.model.states['in3'].detach().numpy().tolist(), [[[25.], [30.], [20.]]])

        ## connect out1 to in3 for 3 samples (initialize in3 with data)
        test.resetStates()
        results = test(inputs={'in1':[[1],[2],[3],[4],[5],[6],[7],[8],[9]], 'in2':[[1],[2],[3],[4],[5],[6],[7],[8],[9]], 'in3':[[1],[2],[3],[4],[5],[6]]}, prediction_samples=2)
        self.assertEqual(results['out1'], [15.0, 20.0, 25.0, 30.0])
        # (4+5+6+7+8)+(5+6+30)
        self.assertEqual(results['out2'], [35.0, 58.0, 85.0, 71.0])
        self.assertEqual(test.model.states['in3'].detach().numpy().tolist(), [[[5.], [6.], [30.]]])
        # Replace insead of rolling
        # (4+5+6+7+8)+(4+5+30)
        # self.assertEqual(results['out2'], [33.0, 57.0, 85.0, 69.0])
        # self.assertEqual(test.model.states['in3'].detach().numpy().tolist(), [[[5.], [30.], [4.]]])

    def test_predict_values_and_connect_variables_2models_more_window_connect_predict(self):
        ## Model1
        input1 = Input('in1')
        a = Parameter('a', dimensions=1, tw=0.05, values=[[1],[1],[1],[1],[1]])
        output1 = Output('out1', Fir(parameter=a)(input1.tw(0.05)))

        test = Neu4mes(visualizer=None, seed=42)
        test.addModel('model1', output1)
        test.addMinimize('error1', input1.next(), output1)
        test.neuralizeModel(0.01)

        ## Model2
        input2 = Input('in2')
        input3 = Input('in3')
        b = Parameter('b', dimensions=1, tw=0.05, values=[[1],[1],[1],[1],[1]])
        c = Parameter('c', dimensions=1, tw=0.03, values=[[1],[1],[1]])
        output2 = Output('out2', Fir(parameter=b)(input2.tw(0.05))+Fir(parameter=c)(input3.tw(0.03)))

        test.addModel('model2', output2)
        test.addMinimize('error2', input2.next(), output2)
        test.neuralizeModel(0.01)

        ## Without connect
        results = test(inputs={'in1':[[1],[2],[3],[4],[5],[6],[7],[8],[9]], 'in2':[[1],[2],[3],[4],[5],[6],[7],[8],[9]], 'in3':[[1],[2],[3],[4],[5],[6]]})
        self.assertEqual(results['out1'], [15.0, 20.0, 25.0, 30.0])
        self.assertEqual(results['out2'], [21.0, 29.0, 37.0, 45.0])

        ## connect out1 to in3 for 4 samples
        results = test(inputs={'in1':[[1],[2],[3],[4],[5],[6],[7],[8],[9]], 'in2':[[1],[2],[3],[4],[5],[6],[7],[8],[9]]}, prediction_samples=3, connect={'in3':'out1'})
        self.assertEqual(results['out1'], [15.0, 20.0, 25.0, 30.0])
        self.assertEqual(results['out2'], [30.0, 55.0, 85.0, 105.0])
        self.assertEqual(test.model.connect_variables['in3'].detach().numpy().tolist(), [[[20.], [25.], [30.]]])
        # Replace insead of rolling
        # self.assertEqual(test.model.connect_variables['in3'].detach().numpy().tolist(), [[[25.], [30.], [20.]]])

        ## connect out1 to in3 for 3 samples
        results = test(inputs={'in1':[[1],[2],[3],[4],[5],[6],[7],[8],[9]], 'in2':[[1],[2],[3],[4],[5],[6],[7],[8],[9]]}, prediction_samples=2, connect={'in3':'out1'})
        self.assertEqual(results['out1'], [15.0, 20.0, 25.0, 30.0])
        self.assertEqual(results['out2'], [30.0, 55.0, 85.0, 60.0])
        self.assertEqual(test.model.connect_variables['in3'].detach().numpy().tolist(), [[[0.0], [0.], [30.]]])
        # Replace insead of rolling
        # self.assertEqual(test.model.connect_variables['in3'].detach().numpy().tolist(), [[[0.], [30.], [0.]]])

        ## connect out1 to in3 for 4 samples (initialize in3 with data)
        results = test(inputs={'in1':[[1],[2],[3],[4],[5],[6],[7],[8],[9]], 'in2':[[1],[2],[3],[4],[5],[6],[7],[8],[9]], 'in3':[[1],[2],[3],[4],[5],[6]]}, prediction_samples=3, connect={'in3':'out1'})
        self.assertEqual(results['out1'], [15.0, 20.0, 25.0, 30.0])
        #(1+2+3+4+5)+(2+3+15)
        #(2+3+4+5+6)+(3+15+20)
        self.assertEqual(results['out2'], [35.0, 58.0, 85.0, 105.0])
        self.assertEqual(test.model.connect_variables['in3'].detach().numpy().tolist(), [[[20.], [25.], [30.]]])
        # Replace insead of rolling
        #(1+2+3+4+5)+(1+2+15)
        #(2+3+4+5+6)+(2+15+20)
        # self.assertEqual(results['out2'], [33.0, 57.0, 85.0, 105.0])
        # self.assertEqual(test.model.connect_variables['in3'].detach().numpy().tolist(), [[[25.], [30.], [20.]]])

        ## connect out1 to in3 for 3 samples (initialize in3 with data)
        results = test(inputs={'in1':[[1],[2],[3],[4],[5],[6],[7],[8],[9]], 'in2':[[1],[2],[3],[4],[5],[6],[7],[8],[9]], 'in3':[[1],[2],[3],[4],[5],[6]]}, prediction_samples=2, connect={'in3':'out1'})
        self.assertEqual(results['out1'], [15.0, 20.0, 25.0, 30.0])
        # (4+5+6+7+8)+(5+6+30)
        self.assertEqual(results['out2'], [35.0, 58.0, 85.0, 71.0])
        self.assertEqual(test.model.connect_variables['in3'].detach().numpy().tolist(), [[[5.], [6.], [30.]]])
        # Replace insead of rolling
        # (4+5+6+7+8)+(4+5+30)
        # self.assertEqual(results['out2'], [33.0, 57.0, 85.0, 69.0])
        # self.assertEqual(test.model.connect_variables['in3'].detach().numpy().tolist(), [[[5.], [30.], [4.]]])

    def test_predict_values_and_states_only_state_variables_more_window_closed_loop(self):
        x_state = State('x_state')
        p = Parameter('p', dimensions=1, tw=0.03, values=[[1.0], [1.0], [1.0]])
        rel_x = Fir(parameter=p)(x_state.tw(0.03))
        rel_x = ClosedLoop(rel_x, x_state)
        out = Output('out', rel_x)

        test = Neu4mes(visualizer = None, seed=42)
        test.addModel('out',out)
        test.neuralizeModel(0.01)

        result = test(inputs={'x_state':[1, 2, 3]})
        self.assertEqual(test.model.states['x_state'].numpy().tolist(), [[[2.],[3.],[6.]]])
        result = test()
        self.assertEqual(test.model.states['x_state'].numpy().tolist(), [[[3.],[6.],[11.]]])

    def test_predict_values_linear_and_fir_2models_same_window_connect(self):
        NeuObj.reset_count()
        input1 = Input('in1',dimensions=2)
        W = Parameter('W', values=[[[-1],[-5]]])
        b = Parameter('b', values=[[1]])
        lin_out = Linear(W=W, b=b)(input1.sw(2))
        output1 = Output('out1', lin_out)

        inout = State('inout')
        a = Parameter('a', values=[[4],[5]])
        output2 = Output('out2', Fir(parameter=a)(inout.sw(2)))
        output3 = Output('out3', Fir(parameter=a)(lin_out))

        test = Neu4mes(visualizer=None, seed=42)
        test.addModel('model', [output1,output2,output3])
        test.addConnect(output1,inout)
        test.neuralizeModel()
        # [[1,2],[2,3]]*[-1,-5] = [[1*-1+2*-5=-11],[2*-1+3*-5=-17]]+[1] = [-10,-16] -> [-10,-16]*[4,5] -> [-16*5+-10*4=-120] <------
        self.assertEqual({'out1': [[-10.0, -16.0]], 'out2': [-120.0], 'out3':[-120.0]}, test({'in1': [[1.0, 2.0], [2.0, 3.0]],'inout':[-10,-16]}))
        self.assertEqual({'out1': [[-10.0,-16.0]], 'out2': [-120.0], 'out3':[-120.0]}, test({'in1': [[1.0,2.0],[2.0,3.0]]}))

    def test_predict_values_linear_and_fir_2models_same_window_connect_predict(self):
        NeuObj.reset_count()
        input1 = Input('in1',dimensions=2)
        W = Parameter('W', values=[[[-1],[-5]]])
        b = Parameter('b', values=[[1]])
        lin_out = Linear(W=W, b=b)(input1.sw(2))
        output1 = Output('out1', lin_out)

        inout = Input('inout')
        a = Parameter('a', values=[[4],[5]])
        output2 = Output('out2', Fir(parameter=a)(inout.sw(2)))
        output3 = Output('out3', Fir(parameter=a)(lin_out))

        test = Neu4mes(visualizer=None, seed=42)
        test.addModel('model', [output1,output2,output3])
        test.neuralizeModel()
        # [[1,2],[2,3]]*[-1,-5] = [[1*-1+2*-5=-11],[2*-1+3*-5=-17]]+[1] = [-10,-16] -> [-10,-16]*[4,5] -> [-16*5+-10*4=-120] <------
        self.assertEqual({'out1': [[-10.0, -16.0]], 'out2': [-120.0], 'out3':[-120.0]}, test({'in1': [[1.0, 2.0], [2.0, 3.0]],'inout':[-10,-16]}))
        self.assertEqual({'out1': [[-10.0,-16.0]], 'out2': [-120.0], 'out3':[-120.0]}, test({'in1': [[1.0,2.0],[2.0,3.0]]},connect={'inout': 'out1'}))
        self.assertEqual({'out1': [[-10.0, -16.0]], 'out2': [-120.0], 'out3': [-120.0]},
                         test({'in1': [[1.0, 2.0], [2.0, 3.0]],'inout':[-30,-30]}, connect={'inout': 'out1'}))

    def test_predict_values_linear_and_fir_2models_more_window_connect(self):
        NeuObj.reset_count()
        input1 = Input('in1',dimensions=2)
        W = Parameter('W', values=[[[-1],[-5]]])
        b = Parameter('b', values=[[1]])
        lin_out = Linear(W=W, b=b)(input1.sw(2))
        output1 = Output('out1', lin_out)

        inout = State('inout')
        a = Parameter('a', values=[[4], [5]])
        a_big = Parameter('ab', values=[[1], [2], [3], [4], [5]])
        output2 = Output('out2', Fir(parameter=a)(inout.sw(2)))
        output3 = Output('out3', Fir(parameter=a_big)(inout.sw(5)))
        output4 = Output('out4', Fir(parameter=a)(lin_out))

        test = Neu4mes(visualizer=None, seed=42)
        test.addModel('model', [output1,output2,output3,output4])
        test.addConnect(output1, inout)
        test.neuralizeModel()
        # [[1,2],[2,3]]*[-1,-5] = [[1*-1+2*-5=-11],[2*-1+3*-5=-17]]+[1] = [-10,-16] -> [-10,-16]*[4,5] -> [-16*5+-10*4=-120] <------
        self.assertEqual({'out1': [[-10.0, -16.0]], 'out2': [-120.0], 'out3': [-120.0], 'out4': [-120.0]}, test({'in1': [[1.0, 2.0], [2.0, 3.0]]}))
        test.resetStates()
        # out2 # = [[-10,-16]] -> 1) [-10,-16]*[4,5] -> [-16*5+-10*4=-120]
        # out3 # = [[-10,-16]] -> 1) [0,0,-10,-10,-16]*[1,2,3,4,5] -> [-10*3+-16*5+-10*4=-150]
        self.assertEqual({'out1': [[-10.0, -16.0]], 'out2': [-120.0], 'out3': [-150.0], 'out4': [-120.0]}, test({'in1': [[1.0, 2.0], [2.0, 3.0]],'inout':[0,0,0,-10,-16]}))
        # Replace instead of rolling
        # self.assertEqual({'out1': [[-10.0, -16.0]], 'out2': [-120.0], 'out3': [-120.0], 'out4': [-120.0]},
        #                  test({'in1': [[1.0, 2.0], [2.0, 3.0]], 'inout': [0, 0, 0, -10, -16]}))
        test.resetStates()

        # out2 # = [[-10,-16],[-16,-10]] -> 1) [-10,-16]*[4,5] -> [-16*5+-10*4=-120]             2) [-16,-10]*[4,5] -> [-16*4+-10*5=-114] -> [-120,-114]
        # out3 # = [[-10,-16],[-16,-10]] -> 1) [0,0,0,-10,-16]*[1,2,3,4,5] -> [-16*5+-10*4=-120] 2) [0,0,-10,-16,-10]*[1,2,3,4,5] -> [-10*3+-16*4+-10*5 = -144] -> [-120,-144]
        self.assertEqual({'out1': [[-10.0, -16.0],[-16.0, -10.0]], 'out2': [-120.0,-114.0], 'out3': [-120.0,-144], 'out4': [-120.0,-114.0]},
                         test({'in1': [[1.0, 2.0], [2.0, 3.0], [1.0,2.0]]}))

    def test_predict_values_linear_and_fir_2models_more_window_connect_predict(self):
        NeuObj.reset_count()
        input1 = Input('in1',dimensions=2)
        W = Parameter('W', values=[[[-1],[-5]]])
        b = Parameter('b', values=[[1]])
        lin_out = Linear(W=W, b=b)(input1.sw(2))
        output1 = Output('out1', lin_out)

        inout = Input('inout')
        a = Parameter('a', values=[[4], [5]])
        output2 = Output('out2', Fir(parameter=a)(inout.sw(2)))
        a_big = Parameter('ab', values=[[1], [2], [3], [4], [5]])
        output3 = Output('out3', Fir(parameter=a_big)(inout.sw(5)))
        output4 = Output('out4', Fir(parameter=a)(lin_out))

        test = Neu4mes(visualizer=None, seed=42)
        test.addModel('model', [output1,output2,output3,output4])
        test.neuralizeModel()
        # [[1,2],[2,3]]*[-1,-5] = [[1*-1+2*-5=-11],[2*-1+3*-5=-17]]+[1] = [-10,-16] -> [-10,-16]*[4,5] -> [-16*5+-10*4=-120] <------
        self.assertEqual({'out1': [[-10.0, -16.0]], 'out2': [-120.0], 'out3': [-120.0], 'out4': [-120.0]}, test({'in1': [[1.0, 2.0], [2.0, 3.0]],'inout':[0,0,0,-10,-16]}))
        self.assertEqual({'out1': [[-10.0, -16.0]], 'out2': [-120.0], 'out3': [-120.0], 'out4': [-120.0]},
                        test({'in1': [[1.0, 2.0], [2.0, 3.0]]}, connect={'inout': 'out1'}))
        self.assertEqual({'out1': [[-10.0, -16.0]], 'out2': [-120.0], 'out3': [-150.0], 'out4': [-120.0]},
                        test({'in1': [[1.0, 2.0], [2.0, 3.0]],'inout':[0,0,0,-10,-16]}, connect={'inout': 'out1'}))
        with self.assertRaises(StopIteration):
            self.assertEqual({}, test())
        with self.assertRaises(StopIteration):
            self.assertEqual({}, test(prediction_samples=0))
        with self.assertRaises(StopIteration):
            self.assertEqual({}, test(prediction_samples=4))

        self.assertEqual({'out1': [[1.0, 1.0]], 'out2': [9.0], 'out3': [9.0], 'out4': [9.0]},
                        test(connect={'inout': 'out1'}))
        self.assertEqual({'out1': [[1.0, 1.0]], 'out2': [9.0], 'out3': [9.0], 'out4': [9.0]},
                        test(connect={'inout': 'out1'}, prediction_samples=0))
        self.assertEqual({'out1': [[1.0, 1.0],[1.0, 1.0]], 'out2': [9.0,9.0], 'out3': [9.0,12.0], 'out4': [9.0,9.0]},
                        test(connect={'inout': 'out1'}, prediction_samples=1, num_of_samples=2))

        # [[1,2],[2,3]]*[-1,-5] = [[1*-1+2*-5=-11],[2*-1+3*-5=-17]]+[1]
        # [[2,3],[1,2]]*[-1,-5] = [[2*-1+3*-5=-17],[1*-1+2*-5=-11]]+[1]
        # out2 # = [[-10,-16],[-16,-10]] -> 1) [-10,-16]*[4,5] -> [-16*5+-10*4=-120]             2) [-16,-10]*[4,5] -> [-16*4+-10*5=-114] -> [-120,-114]
        # out3 # = [[-10,-16],[-16,-10]] -> 1) [0,0,0,-10,-16]*[1,2,3,4,5] -> [-16*5+-10*4=-120] 2) [0,0,-10,-16,-10]*[1,2,3,4,5] -> [-10*3+-16*4+-10*5 = -144] -> [-120,-144]
        self.assertEqual({'out1': [[-10.0, -16.0],[-16.0, -10.0]], 'out2': [-120.0,-114.0], 'out3': [-120.0,-144], 'out4': [-120.0,-114.0]},
                         test({'in1': [[1.0, 2.0], [2.0, 3.0], [1.0,2.0]]},
                              connect={'inout': 'out1'}))

    def test_predict_values_linear_and_fir_2models_more_window_closed_loop(self):
        NeuObj.reset_count()
        input1 = State('in1',dimensions=2)
        W = Parameter('W', values=[[[-1],[-5]]])
        b = Parameter('b', values=[[1]])
        output1 = Output('out1', Linear(W=W, b=b)(input1.sw(2)))

        # input2 = State('inout') #TODO loop forever
        # test.addConnect(output1, input1) # With this
        input2 = State('in2')
        a = Parameter('a', values=[[1,3],[2,4],[3,5],[4,6],[5,7]])
        output2 = Output('out2', Fir(output_dimension=2,parameter=a)(input2.sw(5)))

        test = Neu4mes(visualizer=None, seed=42)
        test.addModel('model', [output1,output2])
        test.addClosedLoop(output1, input2)
        test.addClosedLoop(output2, input1)
        test.neuralizeModel()
        self.assertEqual({'out1': [[-10.0, -16.0]], 'out2': [[[-34.0, -86.0]]]},
                         test({'in1': [[1.0, 2.0], [2.0, 3.0]], 'in2': [-10, -16, -5, 2, 3]}))
        self.assertEqual({'out1': [[-10.0, -16.0]], 'out2': [[[-34.0, -86.0]]]},
                         test({'in1': [[1.0, 2.0], [2.0, 3.0]], 'in2': [-10, -16, -5, 2, 3]},prediction_samples=0))

        self.assertEqual({'out1': [[-10.0, -16.0],[-16.0,465.0]], 'out2': [[[-34.0, -86.0]],[[-140.0,-230.0]]]},
                         test({'in1': [[1.0, 2.0], [2.0, 3.0]], 'in2': [-10, -16, -5, 2, 3]}, prediction_samples=1, num_of_samples=2))
        self.assertEqual({'out1': [[465.0,1291.0]], 'out2': [[[2230.0, 3102.0]]]}, test())
        test.resetStates()
        self.assertEqual({'out1': [[-10.0, -16.0],[-16.0,465.0],[465.0,1291.0]], 'out2': [[[-34.0, -86.0]],[[-140.0,-230.0]],[[2230.0, 3102.0]]]},
                         test({'in1': [[1.0, 2.0], [2.0, 3.0]], 'in2': [-10, -16, -5, 2, 3]}, prediction_samples=2, num_of_samples=3))

    def test_predict_values_linear_and_fir_2models_more_window_closed_loop_predict(self):
        NeuObj.reset_count()
        input1 = Input('in1',dimensions=2)
        W = Parameter('W', values=[[[-1],[-5]]])
        b = Parameter('b', values=[[1]])
        output1 = Output('out1', Linear(W=W, b=b)(input1.sw(2)))

        # input2 = State('inout') #TODO loop forever
        # test.addConnect(output1, input1) # With this
        input2 = Input('in2')
        a = Parameter('a', values=[[1,3],[2,4],[3,5],[4,6],[5,7]])
        output2 = Output('out2', Fir(output_dimension=2,parameter=a)(input2.sw(5)))

        test = Neu4mes(visualizer=None, seed=42)
        test.addModel('model', [output1,output2])
        test.neuralizeModel()
        # 1*-1+2*-5+1 = -10 2*-1+3*-5+1 = -16 -10*1+-16*2+-5*3+2*4+3*5 = -34 -16*2+-10*3+-5*4+2*5+3*6 = -86
        self.assertEqual({'out1': [[-10.0, -16.0]], 'out2': [[[-34.0, -86.0]]]},
                         test({'in1': [[1.0, 2.0], [2.0, 3.0]], 'in2': [-10, -16, -5, 2, 3]}, closed_loop={'in1':'out2', 'in2':'out1'}))
        self.assertEqual({'out1': [[-10.0, -16.0]], 'out2': [[[-34.0, -86.0]]]},
                         test({'in1': [[1.0, 2.0], [2.0, 3.0]], 'in2': [-10, -16, -5, 2, 3]}, prediction_samples=0, closed_loop={'in1':'out2', 'in2':'out1'}))
        self.assertEqual({'out1': [[-10.0, -16.0], [-16.0,465.0]], 'out2': [[[-34.0, -86.0]],[[-140.0,-230.0]]]},
                         test({'in1': [[1.0, 2.0], [2.0, 3.0]], 'in2': [-10, -16, -5, 2, 3]}, num_of_samples=2, prediction_samples=2, closed_loop={'in1':'out2', 'in2':'out1'}))

        with self.assertRaises(StopIteration):
             self.assertEqual({'out1': [[465.0, 1291.0]], 'out2': [[[2230.0, 3102.0]]]}, test())
        with self.assertRaises(StopIteration):
             self.assertEqual({'out1': [[465.0, 1291.0]], 'out2': [[[2230.0, 3102.0]]]}, test(prediction_samples=0))
        with self.assertRaises(StopIteration):
             self.assertEqual({'out1': [[465.0, 1291.0]], 'out2': [[[2230.0, 3102.0]]]}, test(prediction_samples=3))
        self.assertEqual({'out1': [[1.0, 1.0]], 'out2': [[[0.0, 0.0]]]},
                          test(closed_loop={'in1': 'out2', 'in2': 'out1'}))
        self.assertEqual({'out1': [[1.0, 1.0]], 'out2': [[[0.0, 0.0]]]},
                          test(closed_loop={'in1': 'out2', 'in2': 'out1'},prediction_samples=0))
        self.assertEqual({'out1': [[1.0, 1.0],[1.0,1.0]], 'out2': [[[0.0, 0.0]],[[9.0,13.0]]]},
                          test(closed_loop={'in1': 'out2', 'in2': 'out1'},prediction_samples=1, num_of_samples=2))
        self.assertEqual({'out1': [[1.0, 1.0],[1.0,1.0],[1.0,-73.0]], 'out2': [[[0.0, 0.0]],[[9.0,13.0]],[[12.0,18.0]]]},
                          test(closed_loop={'in1': 'out2', 'in2': 'out1'},prediction_samples=2, num_of_samples=3))

        self.assertEqual({'out1': [[-10.0, -16.0], [-16.0,1.0], [1.0,1.0], [1.0,1.0], [1.0,1.0]],
                          'out2': [[[-34.0, -86.0]],[[-8.0,-40.0]],[[8.0,8.0]],[[8.0,18.0]],[[3.0,9.0]]]},
                         test({'in1': [[1.0, 2.0], [2.0, 3.0]], 'in2': [-10, -16, -5, 2, 3]},  num_of_samples=5))
        self.assertEqual({'out1': [[-10.0, -16.0], [-16.0,1.0], [1.0,1.0], [1.0,1.0], [1.0,1.0]],
                          'out2': [[[-34.0, -86.0]],[[-8.0,-40.0]],[[8.0,8.0]],[[8.0,18.0]],[[3.0,9.0]]]},
                         test({'in1': [[1.0, 2.0], [2.0, 3.0]], 'in2': [-10, -16, -5, 2, 3]},  prediction_samples=None, num_of_samples=5))

        #-34*-1+ -86*-5+1 = 465.0
        #-140*-1+ -230*-5+1 = 465.0
        #8*-1 + 18*-5+1 = -97.0
        #[[1, 3], [2, 4], [3, 5], [4, 6], [5, 7]] * [465.0, 1291.0]
        #-16*1+-5*2+2*3+-10.0*4-16.0*5 = 140
        #-5*1+2*2-10*3+-16*4+465*5 = 2230 , -5*3+2*4-10*5+-16*6+465*7 = 3102
        #2*1+3*2 = 8, 2*3+3*4 = 18
        #3*1+1*4+1*5 = 12.0, 3*3+1*6+1*7 = 22.0
        self.assertEqual({'out1': [[-10.0, -16.0], [-16.0, 465.0], [465.0, 1291.0], [1.0, 1.0], [1.0, -97.0]],
                          'out2': [[[-34.0, -86.0]], [[-140.0, -230.0]],[[2230.0, 3102.0]], [[8.0, 18.0]], [[12.0, 22.0]]]},
                         test({'in1': [[1.0, 2.0], [2.0, 3.0]], 'in2': [-10, -16, -5, 2, 3]}, num_of_samples=5,
                              prediction_samples=2, closed_loop={'in1': 'out2', 'in2': 'out1'}))

    def test_predict_parameters(self):
        NeuObj.reset_count()
        input1 = Input('in1')
        cl1 = State('cl1')
        co1 = State('co1')
        W = Parameter('W', values=[[1], [2], [3]])
        parfun = ParamFun(myfunsum, parameters=[W])
        matmulfun = ParamFun(matmul)
        parfun_out = parfun(input1.sw(3))
        output = Output('out', parfun_out)
        matmul_outcl = matmulfun(parfun_out, cl1.sw(3))+1.0
        matmul_outcl.connect(co1)
        matmul_outcl.closedLoop(cl1)
        outputCl = Output('outCl', matmul_outcl)
        outputCo = Output('outCo', matmulfun(parfun_out, co1.sw(3)))


        test = Neu4mes(visualizer=None, seed=42)
        test.addModel('model', [output,outputCl,outputCo])
        test.neuralizeModel()

        # Test only one input
        result = test({'in1':[1,2,3]})
        self.assertEqual((1,3), np.array(result['out']).shape)
        self.assertEqual((1,), np.array(result['outCl']).shape)
        self.assertEqual((1,), np.array(result['outCo']).shape)
        self.assertEqual([[2.0,4.0,6.0]], result['out'])
        self.assertEqual([1.0], result['outCl'])
        self.assertEqual([6.0],result['outCo'])
        self.assertEqual(test.model.states['cl1'].detach().numpy().tolist(), [[[0.], [0.], [1.]]])
        self.assertEqual(test.model.states['co1'].detach().numpy().tolist(), [[[0.], [0.], [1.]]])

        # Test two input
        test.resetStates()
        result = test({'in1':[1,2,3,4]})
        self.assertEqual((2,3), np.array(result['out']).shape)
        self.assertEqual((2,), np.array(result['outCl']).shape)
        self.assertEqual((2,), np.array(result['outCo']).shape)
        self.assertEqual([[2.0,4.0,6.0],[3.0,5.0,7.0]], result['out'])
        self.assertEqual([1.0,1*7+1], result['outCl'])
        self.assertEqual([1 * 6.0, 1. * 5 + 7. * 8.], result['outCo'])
        self.assertEqual(test.model.states['co1'].detach().numpy().tolist(), [[[0.], [1.], [8.]]])
        self.assertEqual(test.model.states['cl1'].detach().numpy().tolist(), [[[0.], [1.], [8.]]])

        # Test two input
        test.resetStates()
        result = test({'in1':[1,2,3,4], 'cl1':[2,2,2,2,2,2]})
        self.assertEqual((2,3), np.array(result['out']).shape)
        self.assertEqual((2,), np.array(result['outCl']).shape)
        self.assertEqual((2,), np.array(result['outCo']).shape)
        self.assertEqual([[2.0,4.0,6.0],[3.0,5.0,7.0]], result['out'])
        # 2*2+4*2+6*2+1, 2*3+5*2+7*2+1
        self.assertEqual([25.0, 31.], result['outCl'])
        self.assertEqual([150.0, 342.0], result['outCo'])
        self.assertEqual(test.model.states['cl1'].detach().numpy().tolist(), [[[2.], [2.], [31.]]])
        self.assertEqual(test.model.states['co1'].detach().numpy().tolist(), [[[0.], [25.], [31.]]])

        # Test two input
        test.resetStates()
        result = test({'in1':[1,2,3,4], 'co1':[2,2,2,2,2,2]})
        self.assertEqual((2,3), np.array(result['out']).shape)
        self.assertEqual((2,), np.array(result['outCl']).shape)
        self.assertEqual((2,), np.array(result['outCo']).shape)
        self.assertEqual([[2.0,4.0,6.0],[3.0,5.0,7.0]], result['out'])
        # 2*0+4*0+6*0+1
        self.assertEqual([1.0, 7*1+1.], result['outCl'])
        # 2*2+4*2+6*1, 2*3+2*5+7*8
        self.assertEqual([18.0, 72.0], result['outCo'])
        self.assertEqual(test.model.states['cl1'].detach().numpy().tolist(), [[[0.], [1.], [8.]]])
        self.assertEqual(test.model.states['co1'].detach().numpy().tolist(), [[[2.], [2.], [8.]]])

        test.resetStates()
        result = test({'co1':[2,2,2,2,2,2]})
        self.assertEqual((4,3), np.array(result['out']).shape)
        self.assertEqual((4,), np.array(result['outCl']).shape)
        self.assertEqual((4,), np.array(result['outCo']).shape)
        self.assertEqual([[1.0,2.0,3.0],[1.0,2.0,3.0],[1.0,2.0,3.0],[1.0,2.0,3.0]], result['out'])
        self.assertEqual(test.model.states['cl1'].detach().numpy().tolist(), [[[4.], [15.], [55.]]])
        self.assertEqual(test.model.states['co1'].detach().numpy().tolist(), [[[2.], [2.], [55.]]])

        test.resetStates()
        result = test({'co1':[2,2,2,2,2,2]}, prediction_samples = 2)
        self.assertEqual((4,3), np.array(result['out']).shape)
        self.assertEqual((4,), np.array(result['outCl']).shape)
        self.assertEqual((4,), np.array(result['outCo']).shape)


        # Test output recurrent

        # prediction_samples set the dimension of the prediction samples every prediction_samples the recurrent variable is read from input (as in train)
        # default value = 'auto' means the network is connected and is closed loop for all the available samples
        # None means that the network is not anymore closed loop or connected
        # 0 means that every sample the state is reset or with zero or with the input but the connect works
        # 1 means that for 1 sample the network use the inside value

        # num_of_samples is the numer of sample return from the call if the input is missing is filled with zero
        # default value = 'auto' means that the network choose the best number of samples to generate
        # at least one sample is generate if the network is called without values
        # or the maximum number of samples if it is called without aligned_input = True
        # otherwise generate the same number of the dataset if it is called with aligned_inputs = True

        # aligned_inputs is a flag for align the inputs as the raw of the dataset
        # default value = False

        # Cases with aligned_input = False
        # Case predict 1 step using the states or (fill the input with zero with a warning)
        #   test()
        #
        # Case prediction using partial inputs. The states and inputs are filled using a rolling window reaching the correct dimension.
        # The network predict the maximum number of samples (considering the bigger input) uses a rolling window.
        # in this way a partial initialization of the state is possible. (if some inputs are missing for generate an input return a warning).
        #   test(data)
        #
        # Case predict using the dataset or inputs without aligned_inputs the network predict the maximum number of samples (considering the bigger input) uses a rolling window
        # The states are reset only if the input is present. it is used a rolling window for the input.
        #   test(dataset)
        #
        # Case predict using the dataset or inputs without aligned_inputs the network predict the maximum number of samples (considering the bigger input) uses a rolling window
        # The states are reset every prediction_sample using inputs or zeros
        #   test(dataset, prediction_sample=N)
        #
        # Case predict M samples using the dataset or inputs
        # the states are reset only if the input is present. if the input are not present are fill with zero. it is used a rolling window for the input
        #   test(dataset, num_of_samples=M)
        #
        # Case predict M samples using the dataset or inputs
        # the states are reset every prediction_sample using inputs or zeros. if the input are not present are fill with zero. it is used a rolling window for the input
        #   test(dataset, prediction_sample=N, num_of_samples=M)
        #
        # Cases with aligned_input=True
        # If the input are messing the network return an arror.
        # The number of the input or states max be equal.
        # The num_of_samples must be less than the number of sample of the data.
        # The states are reset only if the input is present.
        # The network predict the same sample of training.
        # Case with no inputs test(aligned_input=True) the output is the same of before.
        # Case partial inputs test(data, aligned_input=True) the output is the same of before.
        # Case with a data or dataset test(dataset, aligned_input=True) the output is the same of before.
        # Case predict using the dataset as the training does the network predict the same sample of training
        # The states are reset every prediction_sample using inputs or zeros
        #   test(dataset, prediction_sample=N, aligned_input=True)

        # Case predict fewer samples using the dataset
        # The states are reset only if the input is present.
        #   test(dataset, num_of_samples=M, aligned_input=True)

        # Case predict fewer samples and reset every prediction_sample
        # The states are reset only if the input is present.
        #   test(dataset, prediction_sample=N, num_of_samples=M, aligned_input=True)

    def test_parameters_predict_closed_loop_perdict(self):
        NeuObj.reset_count()
        input1 = Input('in1')
        W = Parameter('W', values=[[1], [2], [3]])
        out = Output('out',Fir(parameter=W)(input1.sw(3)))

        test = Neu4mes(visualizer=None, seed=42)
        test.addModel('model', [out])
        test.neuralizeModel()

        result = test({'in1':[1,2,3]})
        self.assertEqual((1,), np.array(result['out']).shape)
        self.assertEqual([14.0], result['out'])

        result = test({'in1': [1, 2, 3]},closed_loop={'in1':'out'})
        self.assertEqual((1,), np.array(result['out']).shape)
        self.assertEqual([14.0], result['out'])

        result = test({'in1': [1, 2, 3, 4, 5]},closed_loop={'in1':'out'})
        self.assertEqual((3,), np.array(result['out']).shape)
        self.assertEqual([14.0,3*4+2*3+2*1,5*3+4*2+3*1], result['out'])

        result = test({'in1': [1, 2, 3, 4, 5]}, closed_loop = {'in1':'out'}, num_of_samples = 5)
        self.assertEqual((5,), np.array(result['out']).shape)
        self.assertEqual([14.0, 3*4+2*3+2*1, 5*3+4*2+3*1, 26*3+5*2+4*1, 92*3+26*2+1*5], result['out'])

        result = test({'in1': [1, 2, 3, 4, 5]},closed_loop={'in1':'out'}, prediction_samples=None)
        self.assertEqual((3,), np.array(result['out']).shape)
        self.assertEqual([14.0,3*4+2*3+2*1,5*3+4*2+3*1], result['out'])

        result = test({'in1': [1, 2, 3, 4, 5]},closed_loop={'in1':'out'}, prediction_samples=0)
        self.assertEqual((3,), np.array(result['out']).shape)
        self.assertEqual([14.0,3*4+2*3+2*1,5*3+4*2+3*1], result['out'])

        result = test({'in1': [1, 2, 3, 4, 5]},closed_loop={'in1':'out'}, prediction_samples=1)
        self.assertEqual((3,), np.array(result['out']).shape)
        self.assertEqual([14.0,3*14+2*3+2*1,5*3+4*2+3*1], result['out'])

        result = test({'in1': [1, 2, 3, 4, 5]},closed_loop={'in1':'out'}, prediction_samples=2)
        self.assertEqual((3,), np.array(result['out']).shape)
        self.assertEqual([14.0,3*14+2*3+2*1,50*3+14*2+3*1], result['out'])

        result = test({'in1': [1, 2, 3, 4, 5]},closed_loop={'in1':'out'}, prediction_samples=3)
        self.assertEqual((3,), np.array(result['out']).shape)
        self.assertEqual([14.0,3*14+2*3+2*1,50*3+14*2+3*1], result['out'])

        result = test({'in1': [1, 2, 3, 4, 5]},closed_loop={'in1':'out'}, prediction_samples=None, num_of_samples = 5)
        self.assertEqual((5,), np.array(result['out']).shape)
        self.assertEqual([14.0,3*4+2*3+2*1,5*3+4*2+3*1,0*3+5*2+4*1,0*3+0*2+5*1], result['out'])

        result = test({'in1': [1, 2, 3, 4, 5]},closed_loop={'in1':'out'}, prediction_samples=0, num_of_samples = 5)
        self.assertEqual((5,), np.array(result['out']).shape)
        self.assertEqual([14.0,3*4+2*3+2*1,5*3+4*2+3*1,0*3+5*2+4*1,0*3+0*2+5*1], result['out'])

        result = test({'in1': [1, 2, 3, 4, 5]},closed_loop={'in1':'out'}, prediction_samples=1, num_of_samples = 5)
        self.assertEqual((5,), np.array(result['out']).shape)
        self.assertEqual([14.0,3*14+2*3+2*1,5*3+4*2+3*1,26*3+5*2+4*1,0*3+0*2+5*1], result['out'])

        result = test({'in1': [1, 2, 3, 4, 5]},closed_loop={'in1':'out'}, prediction_samples=2, num_of_samples = 5)
        self.assertEqual((5,), np.array(result['out']).shape)
        self.assertEqual([14.0,3*14+2*3+2*1,50*3+14*2+3*1,0*3+5*2+4*1,14*3+0*2+5*1], result['out'])

        result = test({'in1': [1, 2, 3, 4, 5]},closed_loop={'in1':'out'}, prediction_samples=3, num_of_samples = 5)
        self.assertEqual((5,), np.array(result['out']).shape)
        self.assertEqual([14.0,3*14+2*3+2*1,50*3+14*2+3*1,181*3+50*2+14*1,0*3+0*2+5*1], result['out'])

    def test_parameters_predict_closed_loop(self):
        NeuObj.reset_count()
        input1 = State('in1')
        W = Parameter('W', values=[[1], [2], [3]])
        out = Output('out',Fir(parameter=W)(input1.sw(3)))

        test = Neu4mes(visualizer=None, seed=42)
        test.addModel('model', [out])
        test.addClosedLoop(out, input1)
        test.neuralizeModel()

        result = test({'in1':[1,2,3]})
        self.assertEqual((1,), np.array(result['out']).shape)
        self.assertEqual([14.0], result['out'])

        test.resetStates()
        result = test({'in1': [1, 2, 3]})
        self.assertEqual((1,), np.array(result['out']).shape)
        self.assertEqual([14.0], result['out'])

        test.resetStates()
        result = test({'in1': [1, 2, 3, 4, 5]})
        self.assertEqual((3,), np.array(result['out']).shape)
        self.assertEqual([14.0,3*4+2*3+2*1,5*3+4*2+3*1], result['out'])

        test.resetStates()
        result = test({'in1': [1, 2, 3, 4, 5]}, num_of_samples = 5)
        self.assertEqual((5,), np.array(result['out']).shape)
        self.assertEqual([14.0, 3*4+2*3+2*1, 5*3+4*2+3*1, 26*3+5*2+4*1, 92*3+26*2+1*5], result['out'])

        test.resetStates()
        result = test({'in1': [1, 2, 3, 4, 5]}, prediction_samples=None)
        self.assertEqual((3,), np.array(result['out']).shape)
        self.assertEqual([14.0,3*4+2*3+2*1,5*3+4*2+3*1], result['out'])

        test.resetStates()
        result = test({'in1': [1, 2, 3, 4, 5]}, prediction_samples=0)
        self.assertEqual((3,), np.array(result['out']).shape)
        self.assertEqual([14.0,3*4+2*3+2*1,5*3+4*2+3*1], result['out'])

        test.resetStates()
        result = test({'in1': [1, 2, 3, 4, 5]}, prediction_samples=1)
        self.assertEqual((3,), np.array(result['out']).shape)
        self.assertEqual([14.0,3*14+2*3+2*1,5*3+4*2+3*1], result['out'])

        test.resetStates()
        result = test({'in1': [1, 2, 3, 4, 5]}, prediction_samples=2)
        self.assertEqual((3,), np.array(result['out']).shape)
        self.assertEqual([14.0,3*14+2*3+2*1,50*3+14*2+3*1], result['out'])

        test.resetStates()
        result = test({'in1': [1, 2, 3, 4, 5]}, prediction_samples=3)
        self.assertEqual((3,), np.array(result['out']).shape)
        self.assertEqual([14.0,3*14+2*3+2*1,50*3+14*2+3*1], result['out'])

        test.resetStates()
        result = test({'in1': [1, 2, 3, 4, 5]}, prediction_samples=None, num_of_samples = 5)
        self.assertEqual((5,), np.array(result['out']).shape)
        self.assertEqual([14.0,3*4+2*3+2*1,5*3+4*2+3*1,0*3+5*2+4*1,0*3+0*2+5*1], result['out'])

        test.resetStates()
        result = test({'in1': [1, 2, 3, 4, 5]}, prediction_samples=0, num_of_samples = 5)
        self.assertEqual((5,), np.array(result['out']).shape)
        self.assertEqual([14.0,3*4+2*3+2*1,5*3+4*2+3*1,0*3+5*2+4*1,0*3+0*2+5*1], result['out'])

        test.resetStates()
        result = test({'in1': [1, 2, 3, 4, 5]}, prediction_samples=1, num_of_samples = 5)
        self.assertEqual((5,), np.array(result['out']).shape)
        self.assertEqual([14.0,3*14+2*3+2*1,5*3+4*2+3*1,26*3+5*2+4*1,0*3+0*2+5*1], result['out'])

        test.resetStates()
        result = test({'in1': [1, 2, 3, 4, 5]}, prediction_samples=2, num_of_samples = 5)
        self.assertEqual((5,), np.array(result['out']).shape)
        self.assertEqual([14.0,3*14+2*3+2*1,50*3+14*2+3*1,0*3+5*2+4*1,14*3+0*2+5*1], result['out'])

        test.resetStates()
        result = test({'in1': [1, 2, 3, 4, 5]}, prediction_samples=3, num_of_samples = 5)
        self.assertEqual((5,), np.array(result['out']).shape)
        self.assertEqual([14.0,3*14+2*3+2*1,50*3+14*2+3*1,181*3+50*2+14*1,0*3+0*2+5*1], result['out'])


if __name__ == '__main__':
    unittest.main()

