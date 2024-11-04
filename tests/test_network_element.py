import unittest, sys, os, torch

from neu4mes import *
from neu4mes import relation
relation.CHECK_NAMES = False

from neu4mes.logger import logging, Neu4MesLogger
log = Neu4MesLogger(__name__, logging.CRITICAL)
log.setAllLevel(logging.CRITICAL)

# append a new directory to sys.path
sys.path.append(os.getcwd())

# 11 Tests
# This file tests the dimensions and the of the element created in the pytorch environment

class Neu4mesNetworkBuildingTest(unittest.TestCase):

    def test_network_building_very_simple(self):

        input1 = Input('in1').last()
        rel1 = Fir(input1)
        fun = Output('out', rel1)

        test = Neu4mes(visualizer=None)
        test.addModel('fun',fun)
        test.neuralizeModel(0.01)

        list_of_dimensions = [[1, 1], [1, 1]]
        for ind, (key, value) in enumerate({k: v for k, v in test.model.relation_forward.items() if 'Fir' in k}.items()):
            self.assertEqual(list_of_dimensions[ind],list(value.weights.shape))
      
    def test_network_building_simple(self):
        Stream.reset_count()
        input1 = Input('in1')
        rel1 = Fir(input1.tw(0.05))
        rel2 = Fir(input1.tw(0.01))
        fun = Output('out',rel1+rel2)

        test = Neu4mes(visualizer=None)
        test.addModel('fun',fun)
        test.neuralizeModel(0.01)
        
        list_of_dimensions = {'Fir3':[5,1],'Fir6':[1,1]}
        for key, value in {k:v for k,v in test.model.relation_forward.items() if 'Fir' in k}.items():
            self.assertEqual(list_of_dimensions[key],list(value.weights.shape))

    def test_network_building_tw(self):
        Stream.reset_count()
        input1 = Input('in1')
        input2 = Input('in2')
        rel1 = Fir(input1.tw(0.05))
        rel2 = Fir(input1.tw(0.01))
        rel3 = Fir(input2.tw(0.05))
        rel4 = Fir(input2.tw([-0.02,0.02]))
        fun = Output('out',rel1+rel2+rel3+rel4)

        test = Neu4mes(visualizer=None)
        test.addModel('fun',fun)
        test.neuralizeModel(0.01)
        
        list_of_dimensions = {'Fir4':[5,1], 'Fir7':[1,1], 'Fir10':[5,1], 'Fir13':[4,1]}
        for key, value in {k:v for k,v in test.model.relation_forward.items() if 'Fir' in k}.items():
            self.assertEqual(list_of_dimensions[key],list(value.weights.shape))
    
    def test_network_building_tw2(self):
        Stream.reset_count()
        input2 = Input('in2')
        rel3 = Fir(input2.tw(0.05))
        rel4 = Fir(input2.tw([-0.02,0.02]))
        rel5 = Fir(input2.tw([-0.03,0.03]))
        rel6 = Fir(input2.tw([-0.03, 0]))
        rel7 = Fir(input2.tw(0.03))
        fun = Output('out',rel3+rel4+rel5+rel6+rel7)

        test = Neu4mes(visualizer=None)
        test.addModel('fun',fun)
        test.neuralizeModel(0.01)

        self.assertEqual(test.max_n_samples, 8) # 5 samples + 3 samples of the horizon
        self.assertEqual({'in2': 8} , test.input_n_samples)
        
        list_of_dimensions = {'Fir3':[5,1], 'Fir6':[4,1], 'Fir9':[6,1], 'Fir12':[3,1], 'Fir15':[3,1]}
        for  key, value in {k:v for k,v in test.model.relation_forward.items() if 'Fir' in k}.items():
            self.assertEqual(list_of_dimensions[key],list(value.weights.shape))

    def test_network_building_tw3(self):
        input2 = Input('in2')
        rel3 = Fir(input2.tw(0.05))
        rel4 = Fir(input2.tw([-0.01,0.03]))
        rel5 = Fir(input2.tw([-0.04,0.01]))
        fun = Output('out',rel3+rel4+rel5)

        test = Neu4mes(visualizer=None)
        test.addModel('fun',fun)
        test.neuralizeModel(0.01)

        list_of_dimensions = [[5,1], [4,1], [5,1]]
        for ind, (key, value) in enumerate({k:v for k,v in test.model.relation_forward.items() if 'Fir' in k}.items()):
            self.assertEqual(list_of_dimensions[ind],list(value.weights.shape))

    def test_network_building_tw_with_offest(self):
        Stream.reset_count()
        input2 = Input('in2')
        rel3 = Fir(input2.tw(0.05))
        rel4 = Fir(input2.tw([-0.04,0.02]))
        rel5 = Fir(input2.tw([-0.04,0.02],offset=0))
        rel6 = Fir(input2.tw([-0.04,0.02],offset=0.01))
        fun = Output('out',rel3+rel4+rel5+rel6)

        test = Neu4mes(visualizer=None)
        test.addModel('fun',fun)
        test.neuralizeModel(0.01)

        list_of_dimensions = {'Fir3':[5,1], 'Fir6':[6,1], 'Fir9':[6,1], 'Fir12':[6,1]}
        for key, value in {k:v for k,v in test.model.relation_forward.items() if 'Fir' in k}.items():
            self.assertEqual(list_of_dimensions[key],list(value.weights.shape))

    def test_network_building_tw_negative(self):
        input2 = Input('in2')
        rel1 = Fir(input2.tw([-0.04,-0.01]))
        rel2 = Fir(input2.tw([-0.06,-0.03]))
        fun = Output('out',rel1+rel2)

        test = Neu4mes(visualizer=None)
        test.addModel('fun',fun)
        test.neuralizeModel(0.01)

        list_of_dimensions = [[3,1], [3,1]]
        for ind, (key, value) in enumerate({k:v for k,v in test.model.relation_forward.items() if 'Fir' in k}.items()):
            self.assertEqual(list_of_dimensions[ind],list(value.weights.shape))

    def test_network_building_tw_positive(self):
        input2 = Input('in2')
        rel1 = Fir(input2.tw([0.01,0.04]))
        rel2 = Fir(input2.tw([0.03,0.06]))
        fun = Output('out',rel1+rel2)

        test = Neu4mes(visualizer=None)
        test.addModel('fun',fun)
        test.neuralizeModel(0.01)

        list_of_dimensions = [[3,1], [3,1]]
        for ind, (key, value) in enumerate({k:v for k,v in test.model.relation_forward.items() if 'Fir' in k}.items()):
            self.assertEqual(list_of_dimensions[ind],list(value.weights.shape))

    def test_network_building_sw_with_offset(self):
        Stream.reset_count()
        input2 = Input('in2')
        rel3 = Fir(input2.sw(5))
        rel4 = Fir(input2.sw([-4,2]))
        rel5 = Fir(input2.sw([-4,2],offset=0))
        rel6 = Fir(input2.sw([-4,2],offset=1))
        fun = Output('out',rel3+rel4+rel5+rel6)

        test = Neu4mes(visualizer=None, seed=1)
        test.addModel('fun',fun)
        test.neuralizeModel(0.01)

        list_of_dimensions = {'Fir3':[5,1], 'Fir6':[6,1], 'Fir9':[6,1], 'Fir12':[6,1]}
        for key, value in {k:v for k,v in test.model.relation_forward.items() if 'Fir' in k}.items():
            self.assertEqual(list_of_dimensions[key],list(value.weights.shape))

    def test_network_building_sw_and_tw(self):
        input2 = Input('in2')
        with self.assertRaises(ValueError):
            input2.sw(5)+input2.tw(0.05)

        rel1 = Fir(input2.sw([-4,2]))+Fir(input2.tw([-0.01,0]))
        fun = Output('out',rel1)

        test = Neu4mes(visualizer=None)
        test.addModel('fun',fun)
        test.neuralizeModel(0.01)

        list_of_dimensions = [[6,1], [1,1]]
        for ind, (key, value) in enumerate({k:v for k,v in test.model.relation_forward.items() if 'Fir' in k}.items()):
            self.assertEqual(list_of_dimensions[ind],list(value.weights.shape))

    def test_network_linear(self):
        torch.manual_seed(1)
        input = Input('in')
        rel1 = Linear(input.sw([-4,2]))
        rel2 = Linear(5)(input.sw([-1, 2]))
        fun1 = Output('out1',rel1)
        fun2 = Output('out2', rel2)

        input5 = Input('in5', dimensions=3)
        rel15 = Linear(input5.sw([-4,2]))
        rel25 = Linear(5)(input5.last())
        fun15 = Output('out51',rel15)
        fun25 = Output('out52', rel25)

        test = Neu4mes(visualizer=None)
        test.addModel('fun',[fun1,fun2,fun15,fun25])
        test.neuralizeModel(0.01)

        list_of_dimensions = [[1,1,1],[1,1,5],[1,3,1],[1,3,5]]
        for ind, (key, value) in enumerate({k:v for k,v in test.model.relation_forward.items() if 'Linear' in k}.items()):
            self.assertEqual(list_of_dimensions[ind],list(value.weights.shape))

if __name__ == '__main__':
    unittest.main()