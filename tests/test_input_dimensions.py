import unittest

import sys
import os
# append a new directory to sys.path
sys.path.append(os.getcwd())
from neu4mes import *
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
        rel1 = Fir(input1)
        fun = Output('out', rel1)

        test = Neu4mes(visualizer=None)
        test.addModel(fun)
        test.neuralizeModel(0.01)

        self.assertEqual(test.max_n_samples, 1)  # 5 samples
        self.assertEqual({'in1': 1}, test.input_n_samples)

        list_of_dimensions = [[1, 1], [1, 1]]
        for ind, (key, value) in enumerate({k: v for k, v in test.model.relation_forward.items() if 'Fir' in k}.items()):
            self.assertEqual([value.in_features, value.out_features], list_of_dimensions[ind])
    def test_network_building_simple(self):
        input1 = Input('in1')
        rel1 = Fir(input1.tw(0.05))
        rel2 = Fir(input1.tw(0.01))
        fun = Output('out',rel1+rel2)

        test = Neu4mes(visualizer=None)
        test.addModel(fun)
        test.neuralizeModel(0.01)

        self.assertEqual(test.max_n_samples, 5) # 5 samples
        self.assertEqual({'in1': 5} , test.input_n_samples)
        
        list_of_dimensions = [[5,1], [1,1]]
        for ind, (key, value) in enumerate({k:v for k,v in test.model.relation_forward.items() if 'Fir' in k}.items()):
            self.assertEqual([value.in_features, value.out_features], list_of_dimensions[ind])

    def test_network_building_tw(self):
        input1 = Input('in1')
        input2 = Input('in2')
        rel1 = Fir(input1.tw(0.05))
        rel2 = Fir(input1.tw(0.01))
        rel3 = Fir(input2.tw(0.05))
        rel4 = Fir(input2.tw([-0.02,0.02]))
        fun = Output('out',rel1+rel2+rel3+rel4)

        test = Neu4mes(visualizer=None)
        test.addModel(fun)
        test.neuralizeModel(0.01)

        self.assertEqual(test.max_n_samples, 7) # 5 samples + 2 samples of the horizon
        self.assertEqual({'in1': 5, 'in2': 7} , test.input_n_samples)
        
        list_of_dimensions = [[5,1], [1,1], [5,1], [4,1]]
        for ind, (key, value) in enumerate({k:v for k,v in test.model.relation_forward.items() if 'Fir' in k}.items()):
            self.assertEqual([value.in_features, value.out_features], list_of_dimensions[ind])

        in1 = [0,1,2,3,4]
        in2 = [0,1,2,3,4,5,6]
        list_of_windows = [[3,4,5,6], [0,1,2,3,4], [4], [0,1,2,3,4]]
        for ind, (key, value) in enumerate(test.relation_samples.items()):
            if 'Fir' in key:
                for k, v in value.items():
                    if k == 'in1':
                        self.assertEqual(in1[v['start_idx']:v['end_idx']], list_of_windows[ind])
                    elif k == 'in2':
                        self.assertEqual(in2[v['start_idx']:v['end_idx']], list_of_windows[ind])

    def test_network_building_tw2(self):
        input2 = Input('in2')
        rel3 = Fir(input2.tw(0.05))
        rel4 = Fir(input2.tw([-0.02,0.02]))
        rel5 = Fir(input2.tw([-0.03,0.03]))
        rel6 = Fir(input2.tw([-0.03, 0]))
        rel7 = Fir(input2.tw(0.03))
        fun = Output('out',rel3+rel4+rel5+rel6+rel7)

        test = Neu4mes(visualizer=None)
        test.addModel(fun)
        test.neuralizeModel(0.01)

        self.assertEqual(test.max_n_samples, 8) # 5 samples + 3 samples of the horizon
        self.assertEqual({'in2': 8} , test.input_n_samples)
        
        list_of_dimensions = [[5,1], [4,1], [6,1], [3,1], [3,1]]
        for ind, (key, value) in enumerate({k:v for k,v in test.model.relation_forward.items() if 'Fir' in k}.items()):
            self.assertEqual([value.in_features, value.out_features], list_of_dimensions[ind])

        in2 = [-5,-4,-3,-2,-1,0,1,2]
        list_of_windows = [[-3,-2,-1], [-3,-2,-1], [-3,-2,-1,0,1,2], [-2,-1,0,1], [-5,-4,-3,-2,-1]]
        for ind, (key, value) in enumerate(test.relation_samples.items()):
            if 'Fir' in key:
                for k, v in value.items():
                    if k == 'in2':
                        self.assertEqual(in2[v['start_idx']:v['end_idx']], list_of_windows[ind])

    def test_network_building_tw3(self):
        input2 = Input('in2')
        rel3 = Fir(input2.tw(0.05))
        rel4 = Fir(input2.tw([-0.01,0.03]))
        rel5 = Fir(input2.tw([-0.04,0.01]))
        fun = Output('out',rel3+rel4+rel5)

        test = Neu4mes(visualizer=None)
        test.addModel(fun)
        test.neuralizeModel(0.01)

        self.assertEqual(test.max_n_samples, 8) # 5 samples + 3 samples of the horizon
        self.assertEqual({'in2': 8} , test.input_n_samples)

        list_of_dimensions = [[5,1], [4,1], [5,1]]
        for ind, (key, value) in enumerate({k:v for k,v in test.model.relation_forward.items() if 'Fir' in k}.items()):
            self.assertEqual([value.in_features, value.out_features], list_of_dimensions[ind])

        in2 = [0,1,2,3,4,5,6,7]
        list_of_windows = [[1,2,3,4,5], [4,5,6,7], [0,1,2,3,4]]
        for ind, (key, value) in enumerate(test.relation_samples.items()):
            if 'Fir' in key:
                for k, v in value.items():
                    if k == 'in2':
                        self.assertEqual(in2[v['start_idx']:v['end_idx']], list_of_windows[ind])

    def test_network_building_tw_with_offest(self):
        input2 = Input('in2')
        rel3 = Fir(input2.tw(0.05))
        rel4 = Fir(input2.tw([-0.04,0.02]))
        rel5 = Fir(input2.tw([-0.04,0.02],offset=0))
        rel6 = Fir(input2.tw([-0.04,0.02],offset=0.02))
        fun = Output('out',rel3+rel4+rel5+rel6)

        test = Neu4mes(visualizer=None)
        test.addModel(fun)
        test.neuralizeModel(0.01)

        self.assertEqual(test.max_n_samples, 7) # 5 samples + 3 samples of the horizon
        self.assertEqual({'in2': 7} , test.input_n_samples)

        list_of_dimensions = [[5,1], [6,1], [6,1], [6,1]]
        for ind, (key, value) in enumerate({k:v for k,v in test.model.relation_forward.items() if 'Fir' in k}.items()):
            self.assertEqual([value.in_features, value.out_features], list_of_dimensions[ind])

        in2 = [0,1,2,3,4,5,6]
        list_of_windows = [[-5,-4,-3,-2,-1,0], [-3,-2,-1,0,1,2], [1,2,3,4,5,6], [0,1,2,3,4]]
        for ind, (key, value) in enumerate(test.relation_samples.items()):
            if 'Fir' in key:
                for k, v in value.items():
                    if k == 'in2':
                        offset = 0
                        if 'offset_idx' in v:
                            offset = v['offset_idx']
                        self.assertEqual([a-offset for a in in2[v['start_idx']:v['end_idx']]], list_of_windows[ind])

    def test_network_building_tw_negative(self):
        input2 = Input('in2')
        rel1 = Fir(input2.tw([-0.04,-0.01]))
        rel2 = Fir(input2.tw([-0.06,-0.03]))
        fun = Output('out',rel1+rel2)

        test = Neu4mes(visualizer=None)
        test.addModel(fun)
        test.neuralizeModel(0.01)

        self.assertEqual(test.max_n_samples, 5) # 5 samples + 3 samples of the horizon
        self.assertEqual({'in2': 5} , test.input_n_samples)

        list_of_dimensions = [[3,1], [3,1]]
        for ind, (key, value) in enumerate({k:v for k,v in test.model.relation_forward.items() if 'Fir' in k}.items()):
            self.assertEqual([value.in_features, value.out_features], list_of_dimensions[ind])

        in2 = [0,1,2,3,4]
        list_of_windows = [[2,3,4],[0,1,2]]
        for ind, (key, value) in enumerate(test.relation_samples.items()):
            if 'Fir' in key:
                for k, v in value.items():
                    if k == 'in2':
                        offset = 0
                        if 'offset_idx' in v:
                            offset = v['offset_idx']
                        self.assertEqual([a-offset for a in in2[v['start_idx']:v['end_idx']]], list_of_windows[ind])

    def test_network_building_tw_positive(self):
        input2 = Input('in2')
        rel1 = Fir(input2.tw([0.01,0.04]))
        rel2 = Fir(input2.tw([0.03,0.06]))
        fun = Output('out',rel1+rel2)

        test = Neu4mes(visualizer=None)
        test.addModel(fun)
        test.neuralizeModel(0.01)

        self.assertEqual(test.max_n_samples, 5) # 5 samples + 3 samples of the horizon
        self.assertEqual({'in2': 5} , test.input_n_samples)

        list_of_dimensions = [[3,1], [3,1]]
        for ind, (key, value) in enumerate({k:v for k,v in test.model.relation_forward.items() if 'Fir' in k}.items()):
            self.assertEqual([value.in_features, value.out_features], list_of_dimensions[ind])

        in2 = [0,1,2,3,4]
        list_of_windows = [[0,1,2],[2,3,4]]
        for ind, (key, value) in enumerate(test.relation_samples.items()):
            if 'Fir' in key:
                for k, v in value.items():
                    if k == 'in2':
                        offset = 0
                        if 'offset_idx' in v:
                            offset = v['offset_idx']
                        self.assertEqual([a-offset for a in in2[v['start_idx']:v['end_idx']]], list_of_windows[ind])

    def test_network_building_sw(self):
        input1 = Input('in1')
        rel3 = Fir(input1.sw(2))
        rel4 = Fir(input1.sw([-2,2]))
        rel5 = Fir(input1.sw([-3,3]))
        rel6 = Fir(input1.sw([-3, 0]))
        rel7 = Fir(input1.sw(3))
        fun = Output('out',rel3+rel4+rel5+rel6+rel7)

        test = Neu4mes(visualizer=None)
        test.addModel(fun)
        test.neuralizeModel(0.01)

        in2 = [0,1,2,3,4,5,6]
        list_of_windows = [[0,1,2], [0,1,2], [0,1,2,3,4,5], [1,2,3,4], [1,2]]
        for ind, (key, value) in enumerate(test.relation_samples.items()):
            if 'Fir' in key:
                for k, v in value.items():
                    if k == 'in1':
                        offset = 0
                        if 'offset_idx' in v:
                            offset = v['offset_idx']
                        self.assertEqual([a-offset for a in in2[v['start_idx']:v['end_idx']]], list_of_windows[ind])

    def test_network_building_sw_with_offset(self):
        input2 = Input('in2')
        rel3 = Fir(input2.sw(5))
        rel4 = Fir(input2.sw([-4,2]))
        rel5 = Fir(input2.sw([-4,2],offset=0))
        rel6 = Fir(input2.sw([-4,2],offset=2))
        fun = Output('out',rel3+rel4+rel5+rel6)

        test = Neu4mes(visualizer=None)
        test.addModel(fun)
        test.neuralizeModel(0.01)

        self.assertEqual(test.max_n_samples, 7) # 5 samples + 3 samples of the horizon
        self.assertEqual({'in2': 7} , test.input_n_samples)

        list_of_dimensions = [[5,1], [6,1], [6,1], [6,1]]
        for ind, (key, value) in enumerate({k:v for k,v in test.model.relation_forward.items() if 'Fir' in k}.items()):
            self.assertEqual([value.in_features, value.out_features], list_of_dimensions[ind])

        in2 = [0,1,2,3,4,5,6]
        list_of_windows = [[-5,-4,-3,-2,-1,0], [-3,-2,-1,0,1,2], [1,2,3,4,5,6], [0,1,2,3,4]]
        for ind, (key, value) in enumerate(test.relation_samples.items()):
            if 'Fir' in key:
                for k, v in value.items():
                    if k == 'in2':
                        offset = 0
                        if 'offset_idx' in v:
                            offset = v['offset_idx']
                        self.assertEqual([a-offset for a in in2[v['start_idx']:v['end_idx']]], list_of_windows[ind])

    def test_network_building_sw_and_tw(self):
        input2 = Input('in2')
        with self.assertRaises(AssertionError):
            input2.sw(5)+input2.tw(0.05)

        rel1 = Fir(input2.sw([-4,2]))+Fir(input2.tw([-0.01,0]))
        fun = Output('out',rel1)

        test = Neu4mes(visualizer=None)
        test.addModel(fun)
        test.neuralizeModel(0.01)

        self.assertEqual(test.max_n_samples, 6) # 5 samples + 3 samples of the horizon
        self.assertEqual({'in2': 6} , test.input_n_samples)

        list_of_dimensions = [[6,1], [1,1]]
        for ind, (key, value) in enumerate({k:v for k,v in test.model.relation_forward.items() if 'Fir' in k}.items()):
            self.assertEqual([value.in_features, value.out_features], list_of_dimensions[ind])

        in2 = [0,1,2,3,4,5]
        list_of_windows = [[3], [0,1,2,3,4,5]]
        for ind, (key, value) in enumerate(test.relation_samples.items()):
            if 'Fir' in key:
                for k, v in value.items():
                    if k == 'in2':
                        offset = 0
                        if 'offset_idx' in v:
                            offset = v['offset_idx']
                        self.assertEqual([a-offset for a in in2[v['start_idx']:v['end_idx']]], list_of_windows[ind])

    '''
    def test_network_building_discrete_input_and_local_model(self):
        in1 = Input('in1', values=[2,3,4])
        in2 = Input('in2')
        rel = LocalModel(in2.tw(1), in1)
        fun = Output(in2.z(-1),rel)

        test = Neu4mes()
        test.addModel(fun)
        test.neuralizeModel(0.5)

        test_layer = Model(inputs=[test.inputs_for_model['in1']], outputs=test.inputs[('in1', 1)])
        self.assertEqual([[1.,0.,0.]],test_layer.predict([[2]]).tolist())
        self.assertEqual([[0.,1.,0.]],test_layer.predict([[3]]).tolist())
        self.assertEqual([[0.,0.,1.]],test_layer.predict([[4]]).tolist())

        test_layer = Model(inputs=[test.inputs_for_model['in2'],test.inputs_for_model['in1']], outputs=test.outputs['in2__-z1'])
        weights = test_layer.get_weights()
        self.assertEqual((2, 3),weights[0].shape) 
        self.assertEqual([[weights[0][1][0]]],test_layer.predict([np.array([[0,1]]),np.array([[2]])]).tolist())
        self.assertEqual([[weights[0][1][1]]],test_layer.predict([np.array([[0,1]]),np.array([[3]])]).tolist())
        self.assertEqual([[weights[0][1][2]]],test_layer.predict([np.array([[0,1]]),np.array([[4]])]).tolist())
        self.assertEqual([[weights[0][0][0]]],test_layer.predict([np.array([[1,0]]),np.array([[2]])]).tolist())
        self.assertEqual([[weights[0][0][1]]],test_layer.predict([np.array([[1,0]]),np.array([[3]])]).tolist())
        self.assertEqual([[weights[0][0][2]]],test_layer.predict([np.array([[1,0]]),np.array([[4]])]).tolist())
    '''

if __name__ == '__main__':
    unittest.main()