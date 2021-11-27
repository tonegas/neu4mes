import unittest, logging
from neu4mes import *

class Neu4mesBasicElementTest(unittest.TestCase):
    def test_input(self):
        input = Input('in')
        self.assertEqual({'Inputs': {'in': {}}, 'Outputs': {}, 'Relations': {}, 'SampleTime': 0},input.json)
        
        input = Input('in', values=[2,3,4])
        self.assertEqual({'Inputs': {'in': {'Discrete': [2,3,4]}},'Outputs': {}, 'Relations': {}, 'SampleTime': 0},input.json)
    
    def test_relation(self):
        pass

    def test_output(self):
        pass
