import sys
import os
# append a new directory to sys.path
sys.path.append(os.getcwd())

import unittest, logging
from neu4mes import *

def myFun(K1,K2,p1,p2):
    return p1*K1+p2*np.sin(K2)

class Neu4mesJson(unittest.TestCase):
    def test_input(self):
        input = Input('in')
        self.assertEqual({'Inputs': {'in': {'dim': 1, 'tw': [0,0]}}, 'Functions' : {}, 'Parameters' : {}, 'Outputs': {}, 'Relations': {}, 'SampleTime': 0},input.json)
        
        input = Input('in', values=[2,3,4])
        self.assertEqual({'Inputs': {'in': {'dim': 1, 'discrete': [2,3,4], 'tw': [0,0]}},'Functions' : {}, 'Parameters' : {}, 'Outputs': {}, 'Relations': {}, 'SampleTime': 0},input.json)

    def test_aritmetic(self):
        input = Input('in')
        out = input+input
        self.assertEqual({'Inputs': {'in': {'dim': 1, 'tw': [0,0]}},'Functions' : {}, 'Parameters' : {}, 'Outputs': {}, 'Relations': {'Add1': ['Add', ['in', 'in']]}, 'SampleTime': 0},out.json)
        out = input.tw(1) + input.tw(1)
        self.assertEqual({'Inputs': {'in': {'dim': 1, 'tw': [-1,0]}},'Functions' : {}, 'Parameters' : {}, 'Outputs': {}, 'Relations': {'Add4': ['Add', [('in', [-1,0]), ('in', [-1,0])]]}, 'SampleTime': 0},out.json)

    def test_scalar_input_dimensions(self):
        input = Input('in')
        out = input+input
        self.assertEqual({'dim': 1}, out.dim)
        out = Fir(input)
        self.assertEqual({'dim': 1}, out.dim)
        out = Fir(7)(input)
        self.assertEqual({'dim': 7}, out.dim)
        out = Fuzzify(5)(input)
        self.assertEqual({'dim': 5}, out.dim)
        out = ParamFun(myFun)(input)
        self.assertEqual({'dim': 1}, out.dim)
        out = ParamFun(myFun,5)(input)
        self.assertEqual({'dim': 5}, out.dim)
        with self.assertRaises(AssertionError):
            out = Fir(Fir(7)(input))
        #
        with self.assertRaises(AssertionError):
            out = Part(input,0,4)
        inpart = ParamFun(myFun, 5)(input)
        out = Part(inpart,0,4)
        self.assertEqual({'dim': 4}, out.dim)
        out = Part(inpart,0,1)
        self.assertEqual({'dim': 1}, out.dim)
        out = Part(inpart,1,3)
        self.assertEqual({'dim': 2}, out.dim)
        out = Select(inpart,0)
        self.assertEqual({'dim': 1}, out.dim)
        with self.assertRaises(AssertionError):
            out = Select(inpart,5)
        with self.assertRaises(AssertionError):
            out = Select(inpart,-1)
        with self.assertRaises(AssertionError):
            out = TimePart(inpart,-1,0)
        with self.assertRaises(AssertionError):
            out = TimeSelect(inpart,-1)

    def test_scalar_input_tw_dimensions(self):
        input = Input('in')
        out = input.tw(1) + input.tw(1)
        self.assertEqual({'dim': 1, 'tw': 1}, out.dim)
        out = Fir(input.tw(1))
        self.assertEqual({'dim': 1}, out.dim)
        out = Fir(5)(input.tw(1))
        self.assertEqual({'dim': 5}, out.dim)
        out = Fuzzify(5)(input.tw(2))
        self.assertEqual({'dim': 5, 'tw': 2}, out.dim)
        out = Fuzzify(5,range=[-1,5])(input.tw(2))
        self.assertEqual({'dim': 5, 'tw': 2}, out.dim)
        out = Fuzzify(2,centers=[-1,5])(input.tw(2))
        self.assertEqual({'dim': 2, 'tw': 2}, out.dim)
        out = ParamFun(myFun)(input.tw(1))
        self.assertEqual({'dim': 1, 'tw' : 1}, out.dim)
        out = ParamFun(myFun,5)(input.tw(2))
        self.assertEqual({'dim': 5, 'tw': 2}, out.dim)
        with self.assertRaises(AssertionError):
            out = ParamFun(myFun,5)(input.tw(2),input.tw(1))

        inpart = ParamFun(myFun, 5)(input.tw(2))
        out = Part(inpart,0,4)
        self.assertEqual({'dim': 4,'tw': 2}, out.dim)
        out = Part(inpart,0,1)
        self.assertEqual({'dim': 1,'tw': 2}, out.dim)
        out = Part(inpart,1,3)
        self.assertEqual({'dim': 2,'tw': 2}, out.dim)
        out = Select(inpart,0)
        self.assertEqual({'dim': 1,'tw': 2}, out.dim)
        with self.assertRaises(AssertionError):
            out = Select(inpart,5)
        with self.assertRaises(AssertionError):
            out = Select(inpart,-1)
        out = TimePart(inpart,-1,0)
        self.assertEqual({'dim': 5, 'tw': 1}, out.dim)
        out = TimeSelect(inpart,-1)
        self.assertEqual({'dim': 5}, out.dim)
        with self.assertRaises(AssertionError):
            out = TimeSelect(inpart,-3)
        twinput = input.tw([-2,4])
        out = TimePart(twinput, -1, 0)
        self.assertEqual({'dim': 1, 'tw': 1}, out.dim)

    def test_vector_input_dimensions(self):
        input = Input('in', dimensions = 5)
        self.assertEqual({'dim': 5}, input.dim)
        self.assertEqual({'dim': 5, 'tw' : 2}, input.tw(2).dim)
        out = input.tw(1) + input.tw(1)
        self.assertEqual({'dim': 5, 'tw': 1}, out.dim)
        out = Relu(input.tw(1))
        self.assertEqual({'dim': 5, 'tw': 1}, out.dim)
        with self.assertRaises(AssertionError):
            Fir(7)(input)
        with self.assertRaises(AssertionError):
            Fuzzify(7)(input)
        out = ParamFun(myFun)(input.tw(1))
        self.assertEqual({'dim': 1, 'tw' : 1}, out.dim)
        out = ParamFun(myFun,5)(input.tw(2))
        self.assertEqual({'dim': 5, 'tw': 2}, out.dim)
        with self.assertRaises(AssertionError):
            out = ParamFun(myFun,5)(input.tw(2),input.tw(1))





    def test_output(self):
        pass

if __name__ == '__main__':
    unittest.main()
