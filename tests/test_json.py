import sys
import os
# append a new directory to sys.path
sys.path.append(os.getcwd())

import unittest, logging
from neu4mes import *

# This test file tests the json, in particular
# the dimensions that are propagated through the relations
# and the structure of the json itself

def myFun(K1,K2,p1,p2):
    import torch
    return p1*K1+p2*torch.sin(K2)

def myFun_out5(K1,p1):
    import torch
    return torch.stack([K1,K1,K1,K1,K1],dim=2).squeeze(-1)*p1


NeuObj.count = 0

class Neu4mesJson(unittest.TestCase):
    def test_input(self):
        input = Input('in')
        self.assertEqual({'Inputs': {'in': {'dim': 1, 'tw': [0,0], 'sw': [0, 0]}}, 'Functions' : {}, 'Parameters' : {}, 'Outputs': {}, 'Relations': {}, 'SampleTime': 0},input.json)
        
        input = Input('in', values=[2,3,4])
        self.assertEqual({'Inputs': {'in': {'dim': 1, 'discrete': [2,3,4], 'tw': [0,0], 'sw': [0, 0]}},'Functions' : {}, 'Parameters' : {}, 'Outputs': {}, 'Relations': {}, 'SampleTime': 0},input.json)

    def test_aritmetic(self):
        Stream.reset_count()
        NeuObj.reset_count()
        input = Input('in')
        inlast = input.last()
        out = inlast+inlast
        self.assertEqual({'Inputs': {'in': {'dim': 1, 'tw': [0,0], 'sw': [-1, 0]}},'Functions' : {}, 'Parameters' : {}, 'Outputs': {}, 'Relations': {'Add3': ['Add', ['SamplePart2', 'SamplePart2']],
               'SamplePart2': ['SamplePart', ['in'], [-1, 0]]}, 'SampleTime': 0},out.json)
        out = input.tw(1) + input.tw(1)
        self.assertEqual({'Inputs': {'in': {'dim': 1, 'tw': [-1,0], 'sw': [0, 0]}},'Functions' : {}, 'Parameters' : {}, 'Outputs': {}, 'Relations': {'Add8': ['Add', ['TimePart5', 'TimePart7']],
               'TimePart5': ['TimePart', ['in'], [-1, 0]],
               'TimePart7': ['TimePart', ['in'], [-1, 0]]}, 'SampleTime': 0},out.json)
        out = input.tw(1) * input.tw(1)
        self.assertEqual({'Inputs': {'in': {'dim': 1, 'tw': [-1,0], 'sw': [0, 0]}},'Functions' : {}, 'Parameters' : {}, 'Outputs': {}, 'Relations': {'Mul13': ['Mul', ['TimePart10', 'TimePart12']],
               'TimePart10': ['TimePart', ['in'], [-1, 0]],
               'TimePart12': ['TimePart', ['in'], [-1, 0]]}, 'SampleTime': 0},out.json)
        out = input.tw(1) - input.tw(1)
        self.assertEqual({'Inputs': {'in': {'dim': 1, 'tw': [-1,0], 'sw': [0, 0]}},'Functions' : {}, 'Parameters' : {}, 'Outputs': {}, 'Relations': {'Sub18': ['Sub', ['TimePart15', 'TimePart17']],
               'TimePart15': ['TimePart', ['in'], [-1, 0]],
               'TimePart17': ['TimePart', ['in'], [-1, 0]]}, 'SampleTime': 0},out.json)
        input = Input('in', dimensions = 5)
        inlast = input.last()
        out = inlast + inlast
        self.assertEqual({'Inputs': {'in': {'dim': 5, 'tw': [0,0], 'sw': [-1, 0]}},'Functions' : {}, 'Parameters' : {}, 'Outputs': {}, 'Relations': {'Add22': ['Add', ['SamplePart21', 'SamplePart21']],
               'SamplePart21': ['SamplePart', ['in'], [-1, 0]]}, 'SampleTime': 0},out.json)
        out = input.tw(1) + input.tw(1)
        self.assertEqual({'Inputs': {'in': {'dim': 5, 'tw': [-1, 0], 'sw': [0, 0]}}, 'Functions': {}, 'Parameters': {},'Outputs': {}, 'Relations': {'Add27': ['Add', ['TimePart24', 'TimePart26']],
               'TimePart24': ['TimePart', ['in'], [-1, 0]],
               'TimePart26': ['TimePart', ['in'], [-1, 0]]},'SampleTime': 0}, out.json)
        out = input.tw([2,5]) + input.tw([3,6])
        self.assertEqual({'Inputs': {'in': {'dim': 5, 'tw': [2, 6], 'sw': [0, 0]}}, 'Functions': {}, 'Parameters': {},'Outputs': {}, 'Relations': {'Add32': ['Add', ['TimePart29', 'TimePart31']],
               'TimePart29': ['TimePart', ['in'], [2, 5]],
               'TimePart31': ['TimePart', ['in'], [3, 6]]},'SampleTime': 0}, out.json)
        out = input.tw([-5,-2]) + input.tw([-6,-3])
        self.assertEqual({'Inputs': {'in': {'dim': 5, 'tw': [-6, -2], 'sw': [0, 0]}}, 'Functions': {}, 'Parameters': {},'Outputs': {}, 'Relations': {'Add37': ['Add', ['TimePart34', 'TimePart36']],
               'TimePart34': ['TimePart', ['in'], [-5, -2]],
               'TimePart36': ['TimePart', ['in'], [-6, -3]]},'SampleTime': 0}, out.json)

    def test_scalar_input_dimensions(self):
        input = Input('in').last()
        out = input+input
        self.assertEqual({'dim': 1,'sw': 1}, out.dim)
        out = Fir(input)
        self.assertEqual({'dim': 1,'sw': 1}, out.dim)
        out = Fir(7)(input)
        self.assertEqual({'dim': 7,'sw': 1}, out.dim)
        out = Fuzzify(5)(input)
        self.assertEqual({'dim': 5,'sw': 1}, out.dim)
        out = ParamFun(myFun)(input)
        self.assertEqual({'dim': 1,'sw': 1}, out.dim)
        out = ParamFun(myFun_out5)(input)
        self.assertEqual({'dim': 5, 'sw': 1}, out.dim)
        with self.assertRaises(ValueError):
            out = Fir(Fir(7)(input))
        #
        with self.assertRaises(IndexError):
            out = Part(input,0,4)
        inpart = ParamFun(myFun_out5)(input)
        out = Part(inpart,0,4)
        self.assertEqual({'dim': 4, 'sw': 1}, out.dim)
        out = Part(inpart,0,1)
        self.assertEqual({'dim': 1, 'sw': 1}, out.dim)
        out = Part(inpart,1,3)
        self.assertEqual({'dim': 2, 'sw': 1}, out.dim)
        out = Select(inpart,0)
        self.assertEqual({'dim': 1, 'sw': 1}, out.dim)
        with self.assertRaises(IndexError):
            out = Select(inpart,5)
        with self.assertRaises(IndexError):
            out = Select(inpart,-1)
        with self.assertRaises(KeyError):
            out = TimePart(inpart,-1,0)
        with self.assertRaises(KeyError):
            out = TimeSelect(inpart,-1)

    def test_scalar_input_tw_dimensions(self):
        input = Input('in')
        out = input.tw(1) + input.tw(1)
        self.assertEqual({'dim': 1, 'tw': 1}, out.dim)
        out = Fir(input.tw(1))
        self.assertEqual({'dim': 1, 'sw': 1}, out.dim)
        out = Fir(5)(input.tw(1))
        self.assertEqual({'dim': 5, 'sw': 1}, out.dim)
        out = Fuzzify(5)(input.tw(2))
        self.assertEqual({'dim': 5, 'tw': 2}, out.dim)
        out = Fuzzify(5,range=[-1,5])(input.tw(2))
        self.assertEqual({'dim': 5, 'tw': 2}, out.dim)
        out = Fuzzify(2,centers=[-1,5])(input.tw(2))
        self.assertEqual({'dim': 2, 'tw': 2}, out.dim)
        out = ParamFun(myFun)(input.tw(1))
        self.assertEqual({'dim': 1, 'tw' : 1}, out.dim)
        out = ParamFun(myFun_out5)(input.tw(2))
        self.assertEqual({'dim': 5, 'tw': 2}, out.dim)
        with self.assertRaises(RuntimeError):
            out = ParamFun(myFun_out5)(input.tw(2),input.tw(1))
        inpart = ParamFun(myFun_out5)(input.tw(2))
        out = Part(inpart,0,4)
        self.assertEqual({'dim': 4,'tw': 2}, out.dim)
        out = Part(inpart,0,1)
        self.assertEqual({'dim': 1,'tw': 2}, out.dim)
        out = Part(inpart,1,3)
        self.assertEqual({'dim': 2,'tw': 2}, out.dim)
        out = Select(inpart,0)
        self.assertEqual({'dim': 1,'tw': 2}, out.dim)
        with self.assertRaises(IndexError):
            out = Select(inpart,5)
        with self.assertRaises(IndexError):
            out = Select(inpart,-1)
        out = TimePart(inpart, 0,1)
        self.assertEqual({'dim': 5, 'tw': 1}, out.dim)
        out = TimeSelect(inpart,0)
        self.assertEqual({'dim': 5}, out.dim)
        with self.assertRaises(ValueError):
            out = TimeSelect(inpart,-3)
        twinput = input.tw([-2,4])
        out = TimePart(twinput, 0, 1)
        self.assertEqual({'dim': 1, 'tw': 1}, out.dim)

    def test_scalar_input_tw2_dimensions(self):
        input = Input('in')
        out = input.tw([-1,1])+input.tw([-2,0])
        self.assertEqual({'dim': 1, 'tw': 2}, out.dim)
        out = input.tw(1)+input.tw([-1,0])
        self.assertEqual({'dim': 1, 'tw': 1}, out.dim)
        out = Fir(input.tw(1) + input.tw([-1, 0]))
        self.assertEqual({'dim': 1, 'sw': 1}, out.dim)
        out = input.tw([-1,0])+input.tw([-4,-3])+input.tw(1)
        self.assertEqual({'dim': 1,'tw': 1}, out.dim)
        with self.assertRaises(ValueError):
             out = input.tw([-2,0])-input.tw([-1,0])
        with self.assertRaises(ValueError):
             out = input.tw([-2,0])+input.tw([-1,0])

    def test_scalar_input_sw_dimensions(self):
        input = Input('in')
        out = input.sw([-1,1])+input.sw([-2,0])
        self.assertEqual({'dim': 1, 'sw': 2}, out.dim)
        out = input.sw(1)+input.sw([-1,0])
        self.assertEqual({'dim': 1, 'sw': 1}, out.dim)
        out = Fir(input.sw(1) + input.sw([-1, 0]))
        self.assertEqual({'dim': 1, 'sw': 1}, out.dim)
        out = input.sw([-1,0])+input.sw([-4,-3])+input.sw(1)
        self.assertEqual({'dim': 1,'sw': 1}, out.dim)
        with self.assertRaises(ValueError):
            out = input.sw([-2,0])-input.sw([-1,0])
        with self.assertRaises(ValueError):
            out = input.sw([-2,0])+input.sw([-1,0])
        with self.assertRaises(ValueError):
            out = input.sw(1) + input.tw([-1, 0])
        with self.assertRaises(TypeError):
            out = input.sw(1.2)
        with self.assertRaises(TypeError):
            out = input.sw([-1.2,0.05])

    def test_vector_input_dimensions(self):
        input = Input('in', dimensions = 5)
        self.assertEqual({'dim': 5}, input.dim)
        self.assertEqual({'dim': 5, 'tw' : 2}, input.tw(2).dim)
        out = input.tw(1) + input.tw(1)
        self.assertEqual({'dim': 5, 'tw': 1}, out.dim)
        out = Relu(input.tw(1))
        self.assertEqual({'dim': 5, 'tw': 1}, out.dim)
        with self.assertRaises(TypeError):
            Fir(7)(input)
        with self.assertRaises(TypeError):
            Fuzzify(7)(input)
        out = ParamFun(myFun)(input.tw(1))
        self.assertEqual({'dim': 5, 'tw' : 1}, out.dim)
        out = ParamFun(myFun)(input.tw(2))
        self.assertEqual({'dim': 5, 'tw': 2}, out.dim)
        with self.assertRaises(RuntimeError):
            out = ParamFun(myFun)(input.tw(2),input.tw(1))

    def test_parameter_and_linear(self):
        input = Input('in').last()
        W15 = Parameter('W15', dimensions=(1, 5))
        b15 = Parameter('b15', dimensions=5)
        input4 = Input('in4',dimensions=4).last()
        W45 = Parameter('W45', dimensions=(4, 5))
        b45 = Parameter('b15', dimensions=5)

        out = Linear(input) + Linear(input4)
        out3 = Linear(3)(input) + Linear(3)(input4)
        outW = Linear(W = W15)(input) + Linear(W = W45)(input4)
        outWb = Linear(W = W15,b = b15)(input) + Linear(W = W45, b = b45)(input4)
        self.assertEqual({'dim': 1, 'sw': 1}, out.dim)
        self.assertEqual({'dim': 3, 'sw': 1}, out3.dim)
        self.assertEqual({'dim': 5, 'sw': 1}, outW.dim)
        self.assertEqual({'dim': 5, 'sw': 1}, outWb.dim)

        input2 = Input('in').sw([-1,1])
        W15 = Parameter('W15', dimensions=(1, 5))
        b15 = Parameter('b15', dimensions=5)
        input42 = Input('in4', dimensions=4).sw([-1,1])
        W45 = Parameter('W45', dimensions=(4, 5))
        b45 = Parameter('b45', dimensions=5)

        out = Linear(input2) + Linear(input42)
        out3 = Linear(3)(input2) + Linear(3)(input42)
        outW = Linear(W = W15)(input2) + Linear(W = W45)(input42)
        outWb = Linear(W = W15,b = b15)(input2) + Linear(W = W45, b = b45)(input42)
        self.assertEqual({'dim': 1, 'sw': 2}, out.dim)
        self.assertEqual({'dim': 3, 'sw': 2}, out3.dim)
        self.assertEqual({'dim': 5, 'sw': 2}, outW.dim)
        self.assertEqual({'dim': 5, 'sw': 2}, outWb.dim)

        with self.assertRaises(ValueError):
            Linear(input) + Linear(input42)
        with self.assertRaises(ValueError):
            Linear(3)(input2) + Linear(3)(input4)
        with self.assertRaises(ValueError):
            Linear(W = W15)(input) + Linear(W = W45)(input42)
        with self.assertRaises(ValueError):
            Linear(W = W15,b = b15)(input2) + Linear(W = W45, b = b45)(input4)

    def test_output(self):
        pass

if __name__ == '__main__':
    unittest.main()
