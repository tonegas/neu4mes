import unittest, logging

import sys
import os
# append a new directory to sys.path
sys.path.append(os.getcwd())
from neu4mes import *

class Neu4mesNetworkBuildingTest(unittest.TestCase):
    def test_network_building_simple(self):
        print('start')
        input1 = Input('in1')
        input2 = Input('in2')
        output = Input('out')
        rel1 = Fir(input1.tw(0.05))
        rel2 = Fir(input1.tw(0.01))
        fun = Output(output.z(-1),rel1+rel2)

        test = Neu4mes()
        test.addModel(fun)
        test.neuralizeModel(0.01)

        self.assertEqual(test.max_n_samples, 5) # 5 samples
        self.assertEqual({'in1': 5} , test.input_n_samples)
        
        list_of_dimensions = [[5,1], [1,1]]
        for key, value in {k:v for k,v in test.model.params.items() if 'Fir' in k}.items():
            self.assertIn([value.in_features, value.out_features], list_of_dimensions)

    def test_network_building_complex1(self):
        input1 = Input('in1')
        input2 = Input('in2')
        output = Input('out')
        rel1 = Fir(input1.tw(0.05))
        rel2 = Fir(input1.tw(0.01))
        rel3 = Fir(input2.tw(0.05))
        rel4 = Fir(input2.tw([-0.02,0.02]))
        fun = Output(output.z(-1),rel1+rel2+rel3+rel4)

        test = Neu4mes()
        test.addModel(fun)
        test.neuralizeModel(0.01)

        self.assertEqual(test.max_n_samples, 7) # 5 samples + 2 samples of the horizon
        self.assertEqual({'in1': 5, 'in2': 7} , test.input_n_samples)
        
        list_of_dimensions = [[5,1], [4,1], [1,1]]
        for key, value in {k:v for k,v in test.model.params.items() if 'Fir' in k}.items():
            self.assertIn([value.in_features, value.out_features], list_of_dimensions)

        in1 = [0,1,2,3,4]
        in2 = [0,1,2,3,4,5,6]
        list_of_windows = [[0,1,2,3,4], [4], [0,1,2,3,4], [3,4,5,6]]
        for key, value in test.relation_samples.items():
            if 'Fir' in key:
                for k, v in value.items():
                    if k == 'in1':
                        self.assertIn(in1[v['backward']:v['forward']], list_of_windows)
                    elif k == 'in2':
                        self.assertIn(in2[v['backward']:v['forward']], list_of_windows)

    def test_network_building_complex2(self):
        input2 = Input('in2')
        output = Input('out')
        rel3 = Fir(input2.tw(0.05))
        rel4 = Fir(input2.tw([-0.02,0.02]))
        rel5 = Fir(input2.tw([-0.03,0.03]))
        fun = Output(output.z(-1),rel3+rel4+rel5)

        test = Neu4mes()
        test.addModel(fun)
        test.neuralizeModel(0.01)

        self.assertEqual(test.max_n_samples, 8) # 5 samples + 3 samples of the horizon
        self.assertEqual({'in2': 8} , test.input_n_samples)
        
        list_of_dimensions = [[8,1], [5,1], [4,1], [6,1]]
        for key, value in {k:v for k,v in test.model.params.items() if 'Fir' in k}.items():
            self.assertIn([value.in_features, value.out_features], list_of_dimensions)

        in2 = [0,1,2,3,4,5,6,7]
        list_of_windows = [[0,1,2,3,4], [2,3,4,5,6,7], [3,4,5,6]]
        for key, value in test.relation_samples.items():
            if 'Fir' in key:
                for k, v in value.items():
                    if k == 'in2':
                        self.assertIn(in2[v['backward']:v['forward']], list_of_windows)

    def test_network_building_complex3(self):
        input2 = Input('in2')
        output = Input('out')
        rel3 = Fir(input2.tw(0.05))
        rel4 = Fir(input2.tw([-0.01,0.03]))
        rel5 = Fir(input2.tw([-0.04,0.01]))
        fun = Output(output.z(-1),rel3+rel4+rel5)

        test = Neu4mes()
        test.addModel(fun)
        test.neuralizeModel(0.01)

        self.assertEqual(test.max_n_samples, 8) # 5 samples + 3 samples of the horizon
        self.assertEqual({'in2': 8} , test.input_n_samples)

        list_of_dimensions = [[8,1], [5,1], [4,1]]
        for key, value in {k:v for k,v in test.model.params.items() if 'Fir' in k}.items():
            self.assertIn([value.in_features, value.out_features], list_of_dimensions)

        in2 = [0,1,2,3,4,5,6,7]
        list_of_windows = [[0,1,2,3,4], [1,2,3,4,5], [4,5,6,7]]
        for key, value in test.relation_samples.items():
            if 'Fir' in key:
                for k, v in value.items():
                    if k == 'in2':
                        self.assertIn(in2[v['backward']:v['forward']], list_of_windows)

    def test_network_building_tw_with_offest(self):
        input2 = Input('in2')
        output = Input('out')
        rel3 = Fir(input2.tw(0.05))
        rel4 = Fir(input2.tw([-0.04,0.02]))
        rel5 = Fir(input2.tw([-0.04,0.02],offset=0))
        rel6 = Fir(input2.tw([-0.04,0.02],offset=0.02))
        fun = Output(output.z(-1),rel3+rel4+rel5+rel6)

        test = Neu4mes(verbose=True)
        test.addModel(fun)
        test.neuralizeModel(0.01)

        in2 = [0,1,2,3,4,5,6]
        list_of_windows = [[0,1,2,3,4],[1,2,3,4,5,6], [-3,-2,-1,0,1,2], [-5,-4,-3,-2,-1,0]]
        for key, value in test.relation_samples.items():
            if 'Fir' in key:
                for k, v in value.items():
                    if k == 'in2':
                        self.assertIn(in2[v['backward']:v['forward']], list_of_windows)
        
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