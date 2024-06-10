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

        self.assertEqual(test.input_tw_backward['in1'],0.01)
        self.assertEqual(test.input_tw_forward['in1'], 0)
        self.assertEqual(test.input_ns_backward['in1'],1)
        self.assertEqual(test.input_ns_forward['in1'], 0)
        self.assertEqual(test.input_n_samples['in1'],1)

        self.assertEqual(test.max_samples_backward, 1)
        self.assertEqual(test.max_samples_forward, 0)
        self.assertEqual(test.max_n_samples, 1)  # 5 samples

    def test_network_building_simple(self):
        input1 = Input('in1')
        rel1 = Fir(input1.tw(0.05))
        rel2 = Fir(input1.tw(0.01))
        fun = Output('out',rel1+rel2)

        test = Neu4mes(visualizer=None)
        test.addModel(fun)
        test.neuralizeModel(0.01)

        self.assertEqual(test.input_tw_backward['in1'],0.05)
        self.assertEqual(test.input_tw_forward['in1'], 0)
        self.assertEqual(test.input_ns_backward['in1'],5)
        self.assertEqual(test.input_ns_forward['in1'], 0)
        self.assertEqual(test.input_n_samples['in1'],5)

        self.assertEqual(test.max_samples_backward, 5)
        self.assertEqual(test.max_samples_forward, 0)
        self.assertEqual(test.max_n_samples, 5)  # 5 samples

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

        self.assertEqual(test.input_tw_backward,{'in1': 0.05, 'in2': 0.05})
        self.assertEqual(test.input_tw_forward, {'in1': 0, 'in2': 0.02})
        self.assertEqual(test.input_ns_backward,{'in1': 5, 'in2': 5})
        self.assertEqual(test.input_ns_forward, {'in1': 0, 'in2': 2})
        self.assertEqual(test.input_n_samples,{'in1': 5, 'in2': 7})

        self.assertEqual(test.max_samples_backward, 5)
        self.assertEqual(test.max_samples_forward, 2)
        self.assertEqual(test.max_n_samples, 7)  # 5 samples + 2 samples of the horizon

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

        self.assertEqual(test.input_tw_backward['in2'],0.05)
        self.assertEqual(test.input_tw_forward['in2'], 0.03)
        self.assertEqual(test.input_ns_backward['in2'],5)
        self.assertEqual(test.input_ns_forward['in2'], 3)
        self.assertEqual(test.input_n_samples['in2'],8) # 5 samples + 3 samples of the horizon

        self.assertEqual(test.max_samples_backward, 5)
        self.assertEqual(test.max_samples_forward, 3)
        self.assertEqual(test.max_n_samples, 8)  # 5 samples

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

        self.assertEqual(test.input_tw_backward['in2'],0.05)
        self.assertEqual(test.input_tw_forward['in2'], 0.03)
        self.assertEqual(test.input_ns_backward['in2'],5)
        self.assertEqual(test.input_ns_forward['in2'], 3)
        self.assertEqual(test.input_n_samples['in2'],8) # 5 samples + 3 samples of the horizon

        self.assertEqual(test.max_samples_backward, 5)
        self.assertEqual(test.max_samples_forward, 3)
        self.assertEqual(test.max_n_samples, 8)  # 5 samples

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

        self.assertEqual(test.input_tw_backward['in2'],0.05)
        self.assertEqual(test.input_tw_forward['in2'], 0.02)
        self.assertEqual(test.input_ns_backward['in2'],5)
        self.assertEqual(test.input_ns_forward['in2'], 2)
        self.assertEqual(test.input_n_samples['in2'],7) # 5 samples + 2 samples of the horizon

        self.assertEqual(test.max_samples_backward, 5)
        self.assertEqual(test.max_samples_forward, 2)
        self.assertEqual(test.max_n_samples, 7)  # 5 samples

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

        self.assertEqual(0.06,test.input_tw_backward['in2'])
        self.assertEqual( -0.01, test.input_tw_forward['in2'])
        self.assertEqual(6, test.input_ns_backward['in2'])
        self.assertEqual(-1, test.input_ns_forward['in2'])
        self.assertEqual(5, test.input_n_samples['in2']) # 6 samples - 1 samples of the horizon

        self.assertEqual(6, test.max_samples_backward)
        self.assertEqual(-1, test.max_samples_forward)
        self.assertEqual(5, test.max_n_samples)  # 5 samples

        #TODO In questo caso la rete deve lanciare un warning per dire che non è compreso l'istante 0

        # in2 = [0,1,2,3,4]
        # list_of_windows = [[2,3,4],[0,1,2]]
        # for ind, (key, value) in enumerate(test.relation_samples.items()):
        #     if 'Fir' in key:
        #         for k, v in value.items():
        #             if k == 'in2':
        #                 offset = 0
        #                 if 'offset_idx' in v:
        #                     offset = v['offset_idx']
        #                 self.assertEqual(list_of_windows[ind],[a-offset for a in in2[v['start_idx']:v['end_idx']]])

    def test_network_building_tw_positive(self):
        input2 = Input('in2')
        rel1 = Fir(input2.tw([0.01,0.04]))
        rel2 = Fir(input2.tw([0.03,0.06]))
        fun = Output('out',rel1+rel2)

        test = Neu4mes(visualizer=None)
        test.addModel(fun)
        test.neuralizeModel(0.01)

        self.assertEqual(-0.01, test.input_tw_backward['in2'])
        self.assertEqual(0.06, test.input_tw_forward['in2'])
        self.assertEqual(-1, test.input_ns_backward['in2'])
        self.assertEqual(6, test.input_ns_forward['in2'])
        self.assertEqual(5, test.input_n_samples['in2']) # -1 samples + 6 samples of the horizon

        self.assertEqual(-1, test.max_samples_backward)
        self.assertEqual(6, test.max_samples_forward)
        self.assertEqual(5, test.max_n_samples)  # 5 samples

        #TODO In questo caso la rete deve lanciare un warning per dire che non è compreso l'istante 0

        # in2 = [0,1,2,3,4]
        # list_of_windows = [[0,1,2],[2,3,4]]
        # for ind, (key, value) in enumerate(test.relation_samples.items()):
        #     if 'Fir' in key:
        #         for k, v in value.items():
        #             if k == 'in2':
        #                 offset = 0
        #                 if 'offset_idx' in v:
        #                     offset = v['offset_idx']
        #                 self.assertEqual(list_of_windows[ind],[a-offset for a in in2[v['start_idx']:v['end_idx']]])

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

        self.assertEqual(test.input_tw_backward['in1'],0)
        self.assertEqual(test.input_tw_forward['in1'], 0)
        self.assertEqual(test.input_ns_backward['in1'],3)
        self.assertEqual(test.input_ns_forward['in1'], 3)
        self.assertEqual(test.input_n_samples['in1'],6) # 6 samples - 1 samples of the horizon

        self.assertEqual(test.max_samples_backward, 3)
        self.assertEqual(test.max_samples_forward, 3)
        self.assertEqual(test.max_n_samples, 6)  # 5 samples

        in2 = [0,1,2,3,4,5]
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

        self.assertEqual(test.input_tw_backward['in2'],0)
        self.assertEqual(test.input_tw_forward['in2'], 0)
        self.assertEqual(test.input_ns_backward['in2'],5)
        self.assertEqual(test.input_ns_forward['in2'], 2)
        self.assertEqual(test.input_n_samples['in2'],7)

        self.assertEqual(test.max_samples_backward, 5)
        self.assertEqual(test.max_samples_forward, 2)
        self.assertEqual(test.max_n_samples, 7)

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

        self.assertEqual(test.input_tw_backward['in2'],0.01)
        self.assertEqual(test.input_tw_forward['in2'], 0)
        self.assertEqual(test.input_ns_backward['in2'],4)
        self.assertEqual(test.input_ns_forward['in2'], 2)
        self.assertEqual(test.input_n_samples['in2'],6)

        self.assertEqual(test.max_samples_backward, 4)
        self.assertEqual(test.max_samples_forward, 2)
        self.assertEqual(test.max_n_samples, 6)

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