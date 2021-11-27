import unittest, logging
from neu4mes import *

class Neu4mesNetworkBuildingTest(unittest.TestCase):
    def test_network_building_simple(self):
        input1 = Input('in1')
        input2 = Input('in2')
        output = Input('out')
        rel1 = Linear(input1.tw(0.05))
        rel2 = Linear(input1.tw(0.01))
        fun = Output(output.z(-1),rel1+rel2)

        test = Neu4mes()
        test.addModel(fun)
        test.neuralizeModel(0.01)

        self.assertEqual(test.max_n_samples, 5) # 5 samples
        self.assertEqual({'in1': 5} , test.input_n_samples)
        
        self.assertEqual([None,5] ,list(test.inputs_for_model['in1'].shape))
        self.assertEqual([None,None,5] ,list(test.inputs_for_rnn_model['in1'].shape))

    def test_network_building_complex(self):
        input1 = Input('in1')
        input2 = Input('in2')
        output = Input('out')
        rel1 = Linear(input1.tw(0.05))
        rel2 = Linear(input1.tw(0.01))
        rel3 = Linear(input2.tw(0.05))
        rel4 = Linear(input2.tw([0.02,-0.02]))
        fun = Output(output.z(-1),rel1+rel2+rel3+rel4)

        test = Neu4mes()
        test.addModel(fun)
        test.neuralizeModel(0.01)

        self.assertEqual(test.max_n_samples, 7) # 5 samples + 2 samples of the horizon
        self.assertEqual({'in1': 5, 'in2': 7} , test.input_n_samples)
        
        self.assertEqual([None,5] ,list(test.inputs_for_model['in1'].shape))
        self.assertEqual([None,7] ,list(test.inputs_for_model['in2'].shape))
        self.assertEqual([None,None,5] ,list(test.inputs_for_rnn_model['in1'].shape))
        self.assertEqual([None,None,7] ,list(test.inputs_for_rnn_model['in2'].shape))
        self.assertEqual([None,5] ,list(test.inputs['in1'].shape))
        self.assertEqual([None,7] ,list(test.inputs['in2'].shape))    
        self.assertEqual([None,5] ,list(test.inputs[('in1',5)].shape))
        self.assertEqual([None,1] ,list(test.inputs[('in1',1)].shape))
        self.assertEqual([None,5] ,list(test.inputs[('in2',5)].shape))
        self.assertEqual([None,4] ,list(test.inputs[('in2',2,2)].shape))      
