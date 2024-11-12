import unittest, sys, os, torch
sys.path.append(os.getcwd())
from neu4mes import *
from neu4mes import relation
relation.CHECK_NAMES = False

from neu4mes.logger import logging, Neu4MesLogger
log = Neu4MesLogger(__name__, logging.CRITICAL)
log.setAllLevel(logging.CRITICAL)

# 13 Tests
# This file tests the dimensions of the inputs in particular:
# The dimensions for each input
# input_tw_backward, input_tw_forward
# test.model_def['Inputs'][KEY]['ns'], and test.model_def['Inputs'][KEY]['ntot']
# The total maximum dimensions:
# model_def['Info']['ns'][0], model_def['Info']['ns'][1], and model_def['Info']['ntot']
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

        #self.assertEqual(0,test.input_tw_backward['in1'])
        #self.assertEqual(0,test.input_tw_forward['in1'])
        self.assertEqual(1,test.model_def['Inputs']['in1']['ns'][0])
        self.assertEqual(0,test.model_def['Inputs']['in1']['ns'][1])
        self.assertEqual(1,test.model_def['Inputs']['in1']['ntot'])

        self.assertEqual(1,test.model_def['Info']['ns'][0])
        self.assertEqual(0,test.model_def['Info']['ns'][1])
        self.assertEqual(1,test.model_def['Info']['ntot'])  # 5 samples

    def test_network_building_simple(self):
        input1 = Input('in1')
        rel1 = Fir(input1.tw(0.05))
        rel2 = Fir(input1.tw(0.01))
        fun = Output('out',rel1+rel2)

        test = Neu4mes(visualizer=None)
        test.addModel('fun',fun)
        test.neuralizeModel(0.01)

        #self.assertEqual(0.05,test.input_tw_backward['in1'])
        #self.assertEqual(0,test.input_tw_forward['in1'])
        self.assertEqual(5,test.model_def['Inputs']['in1']['ns'][0])
        self.assertEqual(0,test.model_def['Inputs']['in1']['ns'][1])
        self.assertEqual(5,test.model_def['Inputs']['in1']['ntot'])

        self.assertEqual(5,test.model_def['Info']['ns'][0])
        self.assertEqual(0,test.model_def['Info']['ns'][1])
        self.assertEqual(5,test.model_def['Info']['ntot'])  # 5 samples

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

        # self.assertEqual({'in1': 0.05, 'in2': 0.05}, test.input_tw_backward)
        # self.assertEqual({'in1': 0, 'in2': 0.02},test.input_tw_forward)
        # self.assertEqual({'in1': 5, 'in2': 5},test.input_ns_backward)
        # self.assertEqual({'in1': 0, 'in2': 2},test.input_ns_forward)
        # self.assertEqual({'in1': 5, 'in2': 7},test.input_n_samples)
        self.assertEqual([5,0] ,test.model_def['Inputs']['in1']['ns'])
        self.assertEqual([5,2],test.model_def['Inputs']['in2']['ns'])
        self.assertEqual(5,test.model_def['Inputs']['in1']['ntot'])
        self.assertEqual(7,test.model_def['Inputs']['in2']['ntot'])

        self.assertEqual(5,test.model_def['Info']['ns'][0])
        self.assertEqual(2,test.model_def['Info']['ns'][1])
        self.assertEqual(7,test.model_def['Info']['ntot'])  # 5 samples + 2 samples of the horizon

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

        #self.assertEqual(0.05,test.input_tw_backward['in2'])
        #self.assertEqual(0.03,test.input_tw_forward['in2'])
        self.assertEqual(5,test.model_def['Inputs']['in2']['ns'][0])
        self.assertEqual(3,test.model_def['Inputs']['in2']['ns'][1])
        self.assertEqual(8,test.model_def['Inputs']['in2']['ntot']) # 5 samples + 3 samples of the horizon

        self.assertEqual(5,test.model_def['Info']['ns'][0])
        self.assertEqual(3,test.model_def['Info']['ns'][1])
        self.assertEqual(8,test.model_def['Info']['ntot'])  # 5 samples

    def test_network_building_tw3(self):
        input2 = Input('in2')
        rel3 = Fir(input2.tw(0.05))
        rel4 = Fir(input2.tw([-0.01,0.03]))
        rel5 = Fir(input2.tw([-0.04,0.01]))
        fun = Output('out',rel3+rel4+rel5)

        test = Neu4mes(visualizer=None)
        test.addModel('fun',fun)
        test.neuralizeModel(0.01)

        #self.assertEqual(0.05, test.input_tw_backward['in2'])
        #self.assertEqual(0.03, test.input_tw_forward['in2'])
        self.assertEqual(5, test.model_def['Inputs']['in2']['ns'][0],)
        self.assertEqual(3, test.model_def['Inputs']['in2']['ns'][1])
        self.assertEqual(8, test.model_def['Inputs']['in2']['ntot']) # 5 samples + 3 samples of the horizon

        self.assertEqual(5, test.model_def['Info']['ns'][0])
        self.assertEqual(3, test.model_def['Info']['ns'][1])
        self.assertEqual(8, test.model_def['Info']['ntot'])  # 5 samples

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

        #self.assertEqual(0.05, test.input_tw_backward['in2'])
        #self.assertEqual(0.02, test.input_tw_forward['in2'])
        self.assertEqual(5, test.model_def['Inputs']['in2']['ns'][0])
        self.assertEqual(2, test.model_def['Inputs']['in2']['ns'][1] )
        self.assertEqual(7, test.model_def['Inputs']['in2']['ntot']) # 5 samples + 2 samples of the horizon

        self.assertEqual(5, test.model_def['Info']['ns'][0])
        self.assertEqual(2, test.model_def['Info']['ns'][1])
        self.assertEqual(7,test.model_def['Info']['ntot'])  # 5 samples

    def test_network_building_tw_negative(self):
        input2 = Input('in2')
        rel1 = Fir(input2.tw([-0.05,-0.01]))
        rel2 = Fir(input2.tw([-0.06,-0.03]))
        fun = Output('out',rel1+rel2)

        test = Neu4mes(visualizer=None)
        test.addModel('fun',fun)
        test.neuralizeModel(0.01)

        #self.assertEqual(0.06,test.input_tw_backward['in2'])
        #self.assertEqual( -0.01, test.input_tw_forward['in2'])
        self.assertEqual(6, test.model_def['Inputs']['in2']['ns'][0])
        self.assertEqual(-1, test.model_def['Inputs']['in2']['ns'][1])
        self.assertEqual(5, test.model_def['Inputs']['in2']['ntot']) # 6 samples - 1 samples of the horizon

        self.assertEqual(6, test.model_def['Info']['ns'][0])
        self.assertEqual(-1, test.model_def['Info']['ns'][1])
        self.assertEqual(5, test.model_def['Info']['ntot'])  # 5 samples

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

        #self.assertEqual(0.06,test.input_tw_backward['in2'])
        #self.assertEqual( -0.01, test.input_tw_forward['in2'])
        self.assertEqual(6, test.model_def['Inputs']['in2']['ns'][0])
        self.assertEqual(-1, test.model_def['Inputs']['in2']['ns'][1])
        self.assertEqual(5, test.model_def['Inputs']['in2']['ntot']) # 6 samples - 1 samples of the horizon

        self.assertEqual(6, test.model_def['Info']['ns'][0])
        self.assertEqual(-1, test.model_def['Info']['ns'][1])
        self.assertEqual(5, test.model_def['Info']['ntot'])  # 5 samples

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

        #self.assertEqual(-0.01, test.input_tw_backward['in2'])
        #self.assertEqual(0.07, test.input_tw_forward['in2'])
        self.assertEqual(-1, test.model_def['Inputs']['in2']['ns'][0])
        self.assertEqual(7, test.model_def['Inputs']['in2']['ns'][1])
        self.assertEqual(6, test.model_def['Inputs']['in2']['ntot']) # -1 samples + 6 samples of the horizon

        self.assertEqual(-1, test.model_def['Info']['ns'][0])
        self.assertEqual(7, test.model_def['Info']['ns'][1])
        self.assertEqual(6, test.model_def['Info']['ntot'])  # 5 samples

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

        #self.assertEqual(-0.01,test.input_tw_backward['in2'])
        #self.assertEqual( 0.07, test.input_tw_forward['in2'])
        self.assertEqual(-1, test.model_def['Inputs']['in2']['ns'][0])
        self.assertEqual(7, test.model_def['Inputs']['in2']['ns'][1])
        self.assertEqual(6, test.model_def['Inputs']['in2']['ntot']) # 6 samples - 1 samples of the horizon

        self.assertEqual(-1, test.model_def['Info']['ns'][0])
        self.assertEqual(7, test.model_def['Info']['ns'][1])
        self.assertEqual(6, test.model_def['Info']['ntot'])  # 5 samples

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

        #self.assertEqual(0,test.input_tw_backward['in1'])
        #self.assertEqual(0,test.input_tw_forward['in1'])
        self.assertEqual(3,test.model_def['Inputs']['in1']['ns'][0])
        self.assertEqual(3,test.model_def['Inputs']['in1']['ns'][1])
        self.assertEqual(6,test.model_def['Inputs']['in1']['ntot']) # 6 samples - 1 samples of the horizon

        self.assertEqual(3,test.model_def['Info']['ns'][0])
        self.assertEqual(3,test.model_def['Info']['ns'][1])
        self.assertEqual(6,test.model_def['Info']['ntot'])  # 5 samples

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

        #self.assertEqual(0, test.input_tw_backward['in2'])
        #self.assertEqual(0, test.input_tw_forward['in2'])
        self.assertEqual(5, test.model_def['Inputs']['in2']['ns'][0])
        self.assertEqual(2, test.model_def['Inputs']['in2']['ns'][1])
        self.assertEqual(7, test.model_def['Inputs']['in2']['ntot'])

        self.assertEqual(5, test.model_def['Info']['ns'][0])
        self.assertEqual(2, test.model_def['Info']['ns'][1])
        self.assertEqual(7, test.model_def['Info']['ntot'])

    def test_network_building_sw_and_tw(self):
        input2 = Input('in2')
        with self.assertRaises(ValueError):
            input2.sw(5)+input2.tw(0.05)

        rel1 = Fir(input2.sw([-4,2]))+Fir(input2.tw([-0.01,0]))
        fun = Output('out',rel1)

        test = Neu4mes(visualizer=None)
        test.addModel('fun',fun)
        test.neuralizeModel(0.01)

        #self.assertEqual(0.01,test.input_tw_backward['in2'])
        #self.assertEqual(0,test.input_tw_forward['in2'])
        self.assertEqual(4,test.model_def['Inputs']['in2']['ns'][0])
        self.assertEqual(2,test.model_def['Inputs']['in2']['ns'][1])
        self.assertEqual(6,test.model_def['Inputs']['in2']['ntot'])

        self.assertEqual(4,test.model_def['Info']['ns'][0])
        self.assertEqual(2,test.model_def['Info']['ns'][1])
        self.assertEqual(6,test.model_def['Info']['ntot'])

    def test_example_rotto(self):
        test = Neu4mes(visualizer=None, seed=42)
        x = Input('x')
        y = Input('y')
        z = Input('z')

        ## create the relations
        def myFun(K1, p1, p2):
            return K1 * p1 * p2

        K_x = Parameter('k_x', dimensions=1, tw=1)
        K_y = Parameter('k_y', dimensions=1, tw=1)
        w = Parameter('w', dimensions=1, tw=1)
        t = Parameter('t', dimensions=1, tw=1)
        c_v = Constant('c_v', tw=1, values=[[1], [2]])
        c = 5
        w_5 = Parameter('w_5', dimensions=1, tw=5)
        t_5 = Parameter('t_5', dimensions=1, tw=5)
        c_5 = [[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]]
        parfun_x = ParamFun(myFun, parameters=[K_x], constants=[c_v])
        parfun_y = ParamFun(myFun, parameters=[K_y])
        parfun_z = ParamFun(myFun)
        fir_w = Fir(parameter=w_5)(x.tw(5))
        fir_t = Fir(parameter=t_5)(y.tw(5))
        time_part = TimePart(x.tw(5), i=1, j=3)
        sample_select = SampleSelect(x.sw(5), i=1)

        def fuzzyfun(x):
            return torch.tan(x)

        fuzzy = Fuzzify(output_dimension=4, range=[0, 4], functions=fuzzyfun)(x.tw(1))

        out = Output('out', Fir(parfun_x(x.tw(1)) + parfun_y(y.tw(1), c_v)))
        # out = Output('out', Fir(parfun_x(x.tw(1))+parfun_y(y.tw(1),c_v)+parfun_z(x.tw(5),t_5,c_5)))
        out2 = Output('out2', Add(w, x.tw(1)) + Add(t, y.tw(1)) + Add(w, c))
        out3 = Output('out3', Add(fir_w, fir_t))
        out4 = Output('out4', Linear(output_dimension=1)(fuzzy))
        out5 = Output('out5', Fir(time_part) + Fir(sample_select))
        out6 = Output('out6', LocalModel(output_function=Fir())(x.tw(1), fuzzy))

        test.addModel('modelA', out)
        test.addModel('modelB', [out2, out3, out4])
        test.addModel('modelC', [out4, out5, out6])
        test.addMinimize('error1', x.last(), out)
        test.addMinimize('error2', y.last(), out3, loss_function='rmse')
        test.addMinimize('error3', z.last(), out6, loss_function='rmse')
        test.neuralizeModel(0.5)

        self.assertEqual([10,0],test.model_def['Inputs']['x']['ns'])
        self.assertEqual([10,0],test.model_def['Inputs']['y']['ns'])
        self.assertEqual([1,0],test.model_def['Inputs']['z']['ns'])
        #
        # self.assertEqual(4,test.model_def['Info']['ns'][0])
        # self.assertEqual(2,test.model_def['Info']['ns'][1])
        # self.assertEqual(6,test.model_def['Info']['ntot'])


if __name__ == '__main__':
    unittest.main()