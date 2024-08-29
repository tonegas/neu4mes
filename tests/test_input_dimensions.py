import unittest

import sys
import os
# append a new directory to sys.path
sys.path.append(os.getcwd())
from neu4mes import *
relation.CHECK_NAMES = False

logging.getLogger("neu4mes.neu4mes").setLevel(logging.CRITICAL)

# This file tests the dimensions of the inputs in particular:
# The dimensions for each input
# input_tw_backward, input_tw_forward
# input_ns_backward, input_ns_forward, and input_n_samples
# The total maximum dimensions:
# max_samples_backward, max_samples_forward, and max_n_samples
# And finally the dimensions for each relation
# relation_samples

class Neu4mesNetworkBuildingTest(unittest.TestCase):

    def test_network_building_very_simple(self):

        input1 = Input('in1')
        rel1 = Fir(input1.last())
        fun = Output('out', rel1)

        test = Neu4mes(visualizer=None)
        test.addModel('fun',fun)
        test.neuralizeModel(0.01)

        self.assertEqual(0,test.input_tw_backward['in1'])
        self.assertEqual(0,test.input_tw_forward['in1'])
        self.assertEqual(1,test.input_ns_backward['in1'])
        self.assertEqual(0,test.input_ns_forward['in1'])
        self.assertEqual(1,test.input_n_samples['in1'])

        self.assertEqual(1,test.max_samples_backward)
        self.assertEqual(0,test.max_samples_forward)
        self.assertEqual(1,test.max_n_samples)  # 5 samples

    def test_network_building_simple(self):
        input1 = Input('in1')
        rel1 = Fir(input1.tw(0.05))
        rel2 = Fir(input1.tw(0.01))
        fun = Output('out',rel1+rel2)

        test = Neu4mes(visualizer=None)
        test.addModel('fun',fun)
        test.neuralizeModel(0.01)

        self.assertEqual(0.05,test.input_tw_backward['in1'])
        self.assertEqual(0,test.input_tw_forward['in1'])
        self.assertEqual(5,test.input_ns_backward['in1'])
        self.assertEqual(0,test.input_ns_forward['in1'])
        self.assertEqual(5,test.input_n_samples['in1'])

        self.assertEqual(5,test.max_samples_backward)
        self.assertEqual(0,test.max_samples_forward)
        self.assertEqual(5,test.max_n_samples)  # 5 samples

    def test_network_building_tw(self):
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

        self.assertEqual({'in1': 0.05, 'in2': 0.05}, test.input_tw_backward)
        self.assertEqual({'in1': 0, 'in2': 0.02},test.input_tw_forward )
        self.assertEqual({'in1': 5, 'in2': 5},test.input_ns_backward)
        self.assertEqual({'in1': 0, 'in2': 2},test.input_ns_forward)
        self.assertEqual({'in1': 5, 'in2': 7},test.input_n_samples)

        self.assertEqual(5,test.max_samples_backward)
        self.assertEqual(2,test.max_samples_forward)
        self.assertEqual(7,test.max_n_samples)  # 5 samples + 2 samples of the horizon

    def test_network_building_tw2(self):
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

        self.assertEqual(0.05,test.input_tw_backward['in2'])
        self.assertEqual(0.03,test.input_tw_forward['in2'])
        self.assertEqual(5,test.input_ns_backward['in2'])
        self.assertEqual(3,test.input_ns_forward['in2'])
        self.assertEqual(8,test.input_n_samples['in2']) # 5 samples + 3 samples of the horizon

        self.assertEqual(5,test.max_samples_backward)
        self.assertEqual(3,test.max_samples_forward)
        self.assertEqual(8,test.max_n_samples)  # 5 samples

    def test_network_building_tw3(self):
        input2 = Input('in2')
        rel3 = Fir(input2.tw(0.05))
        rel4 = Fir(input2.tw([-0.01,0.03]))
        rel5 = Fir(input2.tw([-0.04,0.01]))
        fun = Output('out',rel3+rel4+rel5)

        test = Neu4mes(visualizer=None)
        test.addModel('fun',fun)
        test.neuralizeModel(0.01)

        self.assertEqual(0.05, test.input_tw_backward['in2'])
        self.assertEqual(0.03, test.input_tw_forward['in2'])
        self.assertEqual(5, test.input_ns_backward['in2'],)
        self.assertEqual(3, test.input_ns_forward['in2'])
        self.assertEqual(8, test.input_n_samples['in2']) # 5 samples + 3 samples of the horizon

        self.assertEqual(5, test.max_samples_backward)
        self.assertEqual(3, test.max_samples_forward)
        self.assertEqual(8, test.max_n_samples)  # 5 samples

    def test_network_building_tw_with_offest(self):
        input2 = Input('in2')
        rel3 = Fir(input2.tw(0.05))
        rel4 = Fir(input2.tw([-0.04,0.02]))
        rel5 = Fir(input2.tw([-0.04, 0.02], offset=-0.04))
        rel6 = Fir(input2.tw([-0.04, 0.02], offset=-0.01))
        rel7 = Fir(input2.tw([-0.04, 0.02], offset=0.01))
        fun = Output('out',rel3+rel4+rel5+rel6+rel7)

        test = Neu4mes(visualizer=None)
        test.addModel('fun',fun)
        test.neuralizeModel(0.01)

        self.assertEqual(0.05, test.input_tw_backward['in2'])
        self.assertEqual(0.02, test.input_tw_forward['in2'])
        self.assertEqual(5, test.input_ns_backward['in2'])
        self.assertEqual(2, test.input_ns_forward['in2'] )
        self.assertEqual(7, test.input_n_samples['in2']) # 5 samples + 2 samples of the horizon

        self.assertEqual(5, test.max_samples_backward)
        self.assertEqual(2, test.max_samples_forward)
        self.assertEqual(7,test.max_n_samples)  # 5 samples

    def test_network_building_tw_negative(self):
        input2 = Input('in2')
        rel1 = Fir(input2.tw([-0.05,-0.01]))
        rel2 = Fir(input2.tw([-0.06,-0.03]))
        fun = Output('out',rel1+rel2)

        test = Neu4mes(visualizer=None)
        test.addModel('fun',fun)
        test.neuralizeModel(0.01)

        self.assertEqual(0.06,test.input_tw_backward['in2'])
        self.assertEqual( -0.01, test.input_tw_forward['in2'])
        self.assertEqual(6, test.input_ns_backward['in2'])
        self.assertEqual(-1, test.input_ns_forward['in2'])
        self.assertEqual(5, test.input_n_samples['in2']) # 6 samples - 1 samples of the horizon

        self.assertEqual(6, test.max_samples_backward)
        self.assertEqual(-1, test.max_samples_forward)
        self.assertEqual(5, test.max_n_samples)  # 5 samples

    def test_network_building_tw_negative_with_offset(self):
        input2 = Input('in2')
        rel1 = Fir(input2.tw([-0.05, -0.01], offset=-0.05))
        rel2 = Fir(input2.tw([-0.02, -0.01], offset=-0.02))
        rel3 = Fir(input2.tw([-0.06, -0.03], offset=-0.06))
        rel4 = Fir(input2.tw([-0.06, -0.03], offset=-0.05))
        with self.assertRaises(ValueError):
            input2.tw([-0.01, -0.01], offset=-0.02)
        with self.assertRaises(IndexError):
            input2.tw([-0.06, -0.03], offset=-0.07)
        with self.assertRaises(IndexError):
            input2.tw([-0.06, -0.01], offset=-0.01)
        fun = Output('out', rel1 + rel2 + rel3 + rel4)

        test = Neu4mes(visualizer=None)
        test.addModel('fun',fun)
        test.neuralizeModel(0.01)

        self.assertEqual(0.06,test.input_tw_backward['in2'])
        self.assertEqual( -0.01, test.input_tw_forward['in2'])
        self.assertEqual(6, test.input_ns_backward['in2'])
        self.assertEqual(-1, test.input_ns_forward['in2'])
        self.assertEqual(5, test.input_n_samples['in2']) # 6 samples - 1 samples of the horizon

        self.assertEqual(6, test.max_samples_backward)
        self.assertEqual(-1, test.max_samples_forward)
        self.assertEqual(5, test.max_n_samples)  # 5 samples


    def test_network_building_tw_positive(self):
        input1 = Input('in1')
        rel = Fir(input1.tw([0.03,0.04]))
        fun = Output('out', rel)
        test = Neu4mes(visualizer=None)
        test.addModel('fun',fun)
        test.neuralizeModel(0.01)

        input2 = Input('in2')
        rel1 = Fir(input2.tw([0.01,0.04]))
        rel2 = Fir(input2.tw([0.03,0.07]))
        fun = Output('out',rel1+rel2)

        test = Neu4mes(visualizer=None)
        test.addModel('fun',fun)
        test.neuralizeModel(0.01)

        self.assertEqual(-0.01, test.input_tw_backward['in2'])
        self.assertEqual(0.07, test.input_tw_forward['in2'])
        self.assertEqual(-1, test.input_ns_backward['in2'])
        self.assertEqual(7, test.input_ns_forward['in2'])
        self.assertEqual(6, test.input_n_samples['in2']) # -1 samples + 6 samples of the horizon

        self.assertEqual(-1, test.max_samples_backward)
        self.assertEqual(7, test.max_samples_forward)
        self.assertEqual(6, test.max_n_samples)  # 5 samples

    def test_network_building_tw_positive_with_offset(self):
        input2 = Input('in2')
        rel1 = Fir(input2.tw([0.01,0.04],offset=0.02))
        rel2 = Fir(input2.tw([0.03,0.07],offset=0.04))
        with self.assertRaises(ValueError):
            input2.tw([0.03, 0.02])
        with self.assertRaises(IndexError):
            input2.tw([0.03, 0.07], offset=0.08)
        with self.assertRaises(IndexError):
            input2.tw([0.03, 0.07], offset=0)

        fun = Output('out', rel1 + rel2)

        test = Neu4mes(visualizer=None)
        test.addModel('fun',fun)
        test.neuralizeModel(0.01)

        self.assertEqual(-0.01,test.input_tw_backward['in2'])
        self.assertEqual( 0.07, test.input_tw_forward['in2'])
        self.assertEqual(-1, test.input_ns_backward['in2'])
        self.assertEqual(7, test.input_ns_forward['in2'])
        self.assertEqual(6, test.input_n_samples['in2']) # 6 samples - 1 samples of the horizon

        self.assertEqual(-1, test.max_samples_backward)
        self.assertEqual(7, test.max_samples_forward)
        self.assertEqual(6, test.max_n_samples)  # 5 samples

    def test_network_building_sw(self):
        input1 = Input('in1')
        rel3 = Fir(input1.sw(2))
        rel4 = Fir(input1.sw([-2,2]))
        rel5 = Fir(input1.sw([-3,3]))
        rel6 = Fir(input1.sw([-3, 0]))
        rel7 = Fir(input1.sw(3))
        fun = Output('out',rel3+rel4+rel5+rel6+rel7)

        test = Neu4mes(visualizer=None)
        test.addModel('fun',fun)
        test.neuralizeModel(0.01)

        self.assertEqual(0,test.input_tw_backward['in1'])
        self.assertEqual(0,test.input_tw_forward['in1'])
        self.assertEqual(3,test.input_ns_backward['in1'])
        self.assertEqual(3,test.input_ns_forward['in1'])
        self.assertEqual(6,test.input_n_samples['in1']) # 6 samples - 1 samples of the horizon

        self.assertEqual(3,test.max_samples_backward)
        self.assertEqual(3,test.max_samples_forward)
        self.assertEqual(6,test.max_n_samples)  # 5 samples

    def test_network_building_sw_with_offset(self):
        input2 = Input('in2')
        rel3 = Fir(input2.sw(5))
        rel4 = Fir(input2.sw([-4,2]))
        rel5 = Fir(input2.sw([-4, 2], offset=0))
        rel6 = Fir(input2.sw([-4, 2], offset=1))
        rel7 = Fir(input2.sw([-2, 2], offset=1))
        rel8 = Fir(input2.sw([-4, 2], offset=-3))
        fun = Output('out',rel3+rel4+rel5+rel6+rel7+rel8)

        test = Neu4mes(visualizer=None)
        test.addModel('fun',fun)
        test.neuralizeModel(0.01)

        self.assertEqual(0, test.input_tw_backward['in2'])
        self.assertEqual(0, test.input_tw_forward['in2'])
        self.assertEqual(5, test.input_ns_backward['in2'])
        self.assertEqual(2, test.input_ns_forward['in2'])
        self.assertEqual(7, test.input_n_samples['in2'])

        self.assertEqual(5, test.max_samples_backward)
        self.assertEqual(2, test.max_samples_forward)
        self.assertEqual(7, test.max_n_samples)

    def test_network_building_sw_and_tw(self):
        input2 = Input('in2')
        with self.assertRaises(ValueError):
            input2.sw(5)+input2.tw(0.05)

        rel1 = Fir(input2.sw([-4,2]))+Fir(input2.tw([-0.01,0]))
        fun = Output('out',rel1)

        test = Neu4mes(visualizer=None)
        test.addModel('fun',fun)
        test.neuralizeModel(0.01)

        self.assertEqual(0.01,test.input_tw_backward['in2'])
        self.assertEqual(0,test.input_tw_forward['in2'])
        self.assertEqual(4,test.input_ns_backward['in2'])
        self.assertEqual(2,test.input_ns_forward['in2'])
        self.assertEqual(6,test.input_n_samples['in2'])

        self.assertEqual(4,test.max_samples_backward)
        self.assertEqual(2,test.max_samples_forward)
        self.assertEqual(6,test.max_n_samples)


if __name__ == '__main__':
    unittest.main()