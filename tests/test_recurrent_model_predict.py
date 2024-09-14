import sys
import os
# append a new directory to sys.path
sys.path.append(os.getcwd())

import unittest
from neu4mes import *
from neu4mes import relation
relation.CHECK_NAMES = False

import torch

# 14 Tests
# This file test the model prediction when closed loop or connect are present in particular the output value

# Dimensions
# The first dimension must indicate the time dimension i.e. how many time samples I asked for
# The second dimension indicates the output time dimension for each sample.
# The third is the size of the signal

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

class MyTestCase(unittest.TestCase):
    
    def TestAlmostEqual(self, data1, data2, precision=4):
        assert np.asarray(data1, dtype=np.float32).ndim == np.asarray(data2, dtype=np.float32).ndim, f'Inputs must have the same dimension! Received {type(data1)} and {type(data2)}'
        if type(data1) == type(data2) == list:  
            for pred, label in zip(data1, data2):
                self.TestAlmostEqual(pred, label, precision=precision)
        else:
            self.assertAlmostEqual(data1, data2, places=precision)

    def test_closed_loop(self):
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

    def test_closed_loop_complex(self):
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
        result = test(inputs={'x':[1,2,3,4,5], 'y':[1,2,3,4,5,6]}, closed_loop={'x':'out_pos'})
        self.assertEqual(result['out'], [0.0, 9.0])
        self.assertEqual(result['out_pos'], [15.0, 29.0])
        self.assertEqual(result['out_neg'], [-15.0, -20.0])
        ## two sample prediction with y in close loop
        result = test(inputs={'x':[1,2,3,4,5,6], 'y':[1,2,3,4,5]}, closed_loop={'y':'out_pos'})
        self.assertEqual(result['out'], [0.0, -9.0])
        self.assertEqual(result['out_pos'], [15.0, 20.0])
        self.assertEqual(result['out_neg'], [-15.0, -29.0])
        ## one sample prediction with both close loops
        ## (!! since all the inputs are recurrent we must specify the prediction horizon (defualt=1))
        result = test(inputs={'x':[1,2,3,4,5], 'y':[1,2,3,4,5]}, closed_loop={'x':'out_pos', 'y':'out_neg'})
        self.assertEqual(result['out'], [0.0])
        self.assertEqual(result['out_pos'], [15.0])
        self.assertEqual(result['out_neg'], [-15.0])
        ## three sample prediction with both close loops
        ## (!! since all the inputs are recurrent we must specify the prediction horizon (defualt=1))
        result = test(inputs={'x':[1,2,3,4,5], 'y':[1,2,3,4,5]}, closed_loop={'x':'out_pos', 'y':'out_neg'}, prediction_samples=2)
        self.assertEqual(result['out'], [0.0, 30.0, 58.0])
        self.assertEqual(result['out_pos'], [15.0, 29.0, 56.0])
        self.assertEqual(result['out_neg'], [-15.0, 1.0, 2.0])
        ## three sample prediction with both close loops but y gets initialized for 3 steps
        ## (!! since all the inputs are recurrent we must specify the prediction horizon (defualt=1))
        result = test(inputs={'x':[1,2,3,4,5], 'y':[1,2,3,4,5,6,7]}, closed_loop={'x':'out_pos', 'y':'out_neg'}, prediction_samples=2)
        self.assertEqual(result['out'], [0.0, 9.0, 31.0])
        self.assertEqual(result['out_pos'], [15.0, 29.0, 56.0])
        self.assertEqual(result['out_neg'], [-15.0, -20.0, -25.0])
    
    def test_state_closed_loop(self):
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
    
    def test_state_closed_loop_complex(self):
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

    def test_state_connect(self):
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
        ## 1 sample prediction with state variables all initialized
        result = test(inputs={'x':[1,2,3,4,5], 'y':[1,2,3,4,5], 'z':[1,2,3,4,5]})
        self.assertEqual(result['out'], [160.0])
        self.assertEqual(result['out_x'], [15.0])
        self.assertEqual(result['out_y'], [58.0])
        self.assertEqual(result['out_z'], [87.0])
        self.assertEqual(test.model.states['y'].numpy().tolist(), [[[2.0], [3.0], [4.0], [5.0], [15.0]]])
        self.assertEqual(test.model.states['z'].numpy().tolist(), [[[2.0], [3.0], [4.0], [5.0], [15.0]]])
        ## clear state of y
        test.clear_state(state='y')
        self.assertEqual(test.model.states['y'].numpy().tolist(), [[[0.0], [0.0], [0.0], [0.0], [0.0]]])
        self.assertEqual(test.model.states['z'].numpy().tolist(), [[[2.0], [3.0], [4.0], [5.0], [15.0]]])
        ## multi-sample prediction with states initialized as many times as they have values
        result = test(inputs={'x':[1,2,3,4,5,6,7,8,9], 'y':[1,2,3,4,5,6,7], 'z':[1,2,3,4,5,6]})
        self.assertEqual(result['out_x'], [15.0, 20.0, 25.0, 30.0, 35.0])
        self.assertEqual(result['out_y'], [2*(2+3+4+5+15), 2*(3+4+5+6+20), 2*(4+5+6+7+25), 2*(5+6+7+25+30), 2*(6+7+25+30+35)])
        self.assertEqual(result['out_z'], [3*(2+3+4+5+15), 3*(3+4+5+6+20), 3*(4+5+6+20+25), 3*(5+6+20+25+30), 3*(6+20+25+30+35)])
        self.assertEqual(result['out'], [sum(x) for x in zip(result['out_x'],result['out_y'],result['out_z'])])
        self.assertEqual(test.model.states['y'].numpy().tolist(), [[[6.0], [7.0], [25.0], [30.0], [35.0]]])
        self.assertEqual(test.model.states['z'].numpy().tolist(), [[[6.0], [20.0], [25.0], [30.0], [35.0]]])
        ## Clear all states
        test.clear_state()
        self.assertEqual(test.model.states['y'].numpy().tolist(), [[[0.0], [0.0], [0.0], [0.0], [0.0]]])
        self.assertEqual(test.model.states['z'].numpy().tolist(), [[[0.0], [0.0], [0.0], [0.0], [0.0]]])
    
    def test_state_connect_complex(self):
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

        ## connect out1 to in3 for 3 samples
        results = test(inputs={'in1':[[1],[2],[3],[4],[5],[6],[7],[8],[9]], 'in2':[[1],[2],[3],[4],[5],[6],[7],[8],[9]]}, prediction_samples=2, connect={'in3':'out1'})
        self.assertEqual(results['out1'], [15.0, 20.0, 25.0, 30.0])
        self.assertEqual(results['out2'], [30.0, 55.0, 85.0, 60.0])
        self.assertEqual(test.model.connect_variables['in3'].detach().numpy().tolist(), [[[0.], [0.], [30.]]])

        ## connect out1 to in3 for 4 samples (initialize in3 with data)
        results = test(inputs={'in1':[[1],[2],[3],[4],[5],[6],[7],[8],[9]], 'in2':[[1],[2],[3],[4],[5],[6],[7],[8],[9]], 'in3':[[1],[2],[3],[4],[5],[6]]}, prediction_samples=3, connect={'in3':'out1'})
        self.assertEqual(results['out1'], [15.0, 20.0, 25.0, 30.0])
        self.assertEqual(results['out2'], [33.0, 57.0, 85.0, 105.0])
        self.assertEqual(test.model.connect_variables['in3'].detach().numpy().tolist(), [[[20.], [25.], [30.]]])

        ## connect out1 to in3 for 3 samples (initialize in3 with data)
        results = test(inputs={'in1':[[1],[2],[3],[4],[5],[6],[7],[8],[9]], 'in2':[[1],[2],[3],[4],[5],[6],[7],[8],[9]], 'in3':[[1],[2],[3],[4],[5],[6]]}, prediction_samples=2, connect={'in3':'out1'})
        self.assertEqual(results['out1'], [15.0, 20.0, 25.0, 30.0])
        self.assertEqual(results['out2'], [33.0, 57.0, 85.0, 69.0])
        self.assertEqual(test.model.connect_variables['in3'].detach().numpy().tolist(), [[[4.], [5.], [30.]]])

    def test_recurrent_one_state_variable(self):
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
        result = test(inputs={'x': [2]})
        self.assertEqual(test.model.states['x_state'], torch.tensor(1.0))

    def test_recurrent_only_state_variables(self):
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

    def test_recurrent_connect_predict_values_same_window(self):
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

    def test_recurrent_connect_predict_values_bigger_window(self):
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
        self.assertEqual({'out1': [[-10.0, -16.0]], 'out2': [-120.0], 'out3': [-120.0], 'out4': [-120.0]},
                        test({'in1': [[1.0, 2.0], [2.0, 3.0]],'inout':[0,0,0,-30,-30]}, connect={'inout': 'out1'}))
        with self.assertRaises(StopIteration):
            self.assertEqual({}, test())
        with self.assertRaises(StopIteration):
            self.assertEqual({}, test(prediction_samples=0))
        with self.assertRaises(StopIteration):
            self.assertEqual({}, test(prediction_samples=4))

        self.assertEqual({'out1': [[1.0, 1.0]], 'out2': [9.0], 'out3': [9.0], 'out4': [9.0]},
                        test(connect={'inout': 'out1'}))
        self.assertEqual({'out1': [[1.0, 1.0]], 'out2': [9.0], 'out3': [9.0], 'out4': [9.0]},
                        test(connect={'inout': 'out1'},prediction_samples=0))
        self.assertEqual({'out1': [[1.0, 1.0],[1.0, 1.0]], 'out2': [9.0,9.0], 'out3': [9.0,12.0], 'out4': [9.0,9.0]},
                        test(connect={'inout': 'out1'},prediction_samples=1))

        # [[1,2],[2,3]]*[-1,-5] = [[1*-1+2*-5=-11],[2*-1+3*-5=-17]]+[1]
        # [[2,3],[1,2]]*[-1,-5] = [[2*-1+3*-5=-17],[1*-1+2*-5=-11]]+[1]
        # out2 # = [[-10,-16],[-16,-10]] -> 1) [-10,-16]*[4,5] -> [-16*5+-10*4=-120]             2) [-16,-10]*[4,5] -> [-16*4+-10*5=-114] -> [-120,-114]
        # out3 # = [[-10,-16],[-16,-10]] -> 1) [0,0,0,-10,-16]*[1,2,3,4,5] -> [-16*5+-10*4=-120] 2) [0,0,-10,-16,-10]*[1,2,3,4,5] -> [-10*3+-16*4+-10*5 = -144] -> [-120,-144]
        self.assertEqual({'out1': [[-10.0, -16.0],[-16.0, -10.0]], 'out2': [-120.0,-114.0], 'out3': [-120.0,-144], 'out4': [-120.0,-114.0]},
                         test({'in1': [[1.0, 2.0], [2.0, 3.0], [1.0,2.0]]},
                              connect={'inout': 'out1'}))

    def test_recurrent_connect_values_same_window(self):
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

    def test_recurrent_connect_values_bigger_window(self):
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
        test.clear_state()
        # out2 # = [[-10,-16]] -> 1) [-10,-16]*[4,5] -> [-16*5+-10*4=-120]
        # out3 # = [[-10,-16]] -> 1) [0,0,-10,-10,-16]*[1,2,3,4,5] -> [-10*3+-16*5+-10*4=-150]
        self.assertEqual({'out1': [[-10.0, -16.0]], 'out2': [-120.0], 'out3': [-150.0], 'out4': [-120.0]}, test({'in1': [[1.0, 2.0], [2.0, 3.0]],'inout':[0,0,0,-10,-16]}))
        test.clear_state()

        # out2 # = [[-10,-16],[-16,-10]] -> 1) [-10,-16]*[4,5] -> [-16*5+-10*4=-120]             2) [-16,-10]*[4,5] -> [-16*4+-10*5=-114] -> [-120,-114]
        # out3 # = [[-10,-16],[-16,-10]] -> 1) [0,0,0,-10,-16]*[1,2,3,4,5] -> [-16*5+-10*4=-120] 2) [0,0,-10,-16,-10]*[1,2,3,4,5] -> [-10*3+-16*4+-10*5 = -144] -> [-120,-144]
        self.assertEqual({'out1': [[-10.0, -16.0],[-16.0, -10.0]], 'out2': [-120.0,-114.0], 'out3': [-120.0,-144], 'out4': [-120.0,-114.0]},
                         test({'in1': [[1.0, 2.0], [2.0, 3.0], [1.0,2.0]]}))

    def test_recurrent_closed_loop_values_bigger_window(self):
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
                         test({'in1': [[1.0, 2.0], [2.0, 3.0]], 'in2': [-10, -16, -5, 2, 3]},prediction_samples=1))
        self.assertEqual({'out1': [[465.0,1291.0]], 'out2': [[[2230.0, 3102.0]]]}, test())
        test.clear_state()
        self.assertEqual({'out1': [[-10.0, -16.0],[-16.0,465.0],[465.0,1291.0]], 'out2': [[[-34.0, -86.0]],[[-140.0,-230.0]],[[2230.0, 3102.0]]]},
                         test({'in1': [[1.0, 2.0], [2.0, 3.0]], 'in2': [-10, -16, -5, 2, 3]},prediction_samples=2))

    def test_recurrent_closed_loop_predict_values_bigger_window(self):
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
        self.assertEqual({'out1': [[-10.0, -16.0]], 'out2': [[[-34.0, -86.0]]]},
                         test({'in1': [[1.0, 2.0], [2.0, 3.0]], 'in2': [-10, -16, -5, 2, 3]}, closed_loop={'in1':'out2', 'in2':'out1'}))
        self.assertEqual({'out1': [[-10.0, -16.0]], 'out2': [[[-34.0, -86.0]]]},
                         test({'in1': [[1.0, 2.0], [2.0, 3.0]], 'in2': [-10, -16, -5, 2, 3]},prediction_samples=0, closed_loop={'in1':'out2', 'in2':'out1'}))
        self.assertEqual({'out1': [[-10.0, -16.0],[-16.0,465.0]], 'out2': [[[-34.0, -86.0]],[[-140.0,-230.0]]]},
                         test({'in1': [[1.0, 2.0], [2.0, 3.0]], 'in2': [-10, -16, -5, 2, 3]},prediction_samples=1, closed_loop={'in1':'out2', 'in2':'out1'}))
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
                          test(closed_loop={'in1': 'out2', 'in2': 'out1'},prediction_samples=1))
        self.assertEqual({'out1': [[1.0, 1.0],[1.0,1.0],[1.0,-73.0]], 'out2': [[[0.0, 0.0]],[[9.0,13.0]],[[12.0,18.0]]]},
                          test(closed_loop={'in1': 'out2', 'in2': 'out1'},prediction_samples=2))

if __name__ == '__main__':
    unittest.main()

