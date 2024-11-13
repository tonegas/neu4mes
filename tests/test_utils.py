import unittest, os, sys
import torch
sys.path.append(os.getcwd())
from neu4mes import *

from neu4mes.logger import logging, Neu4MesLogger
log = Neu4MesLogger(__name__, logging.CRITICAL)
log.setAllLevel(logging.CRITICAL)

class Neu4mesTrainingTest(unittest.TestCase):
    # test the linear interpolation function with batches of input data with shape torch.Size([N, 1, 1])
    def test_linear_interp_with_batched_input_1(self):
        # x is a tensor of query points, and is a tensor of shape torch.Size([N, 1, 1])
        x = torch.tensor([[[-1.0]],[[0.15]],[[0.25]],[[0.35]],[[1.3]]])
        # x_data and y_data are tensors of shape torch.Size([Q, 1])
        x_data = torch.tensor([[0.0],[0.1],[0.2],[0.3],[0.4],[0.5],[0.6],[0.7],[0.8],[0.9]])
        y_data = torch.tensor([[0.5],[0.6],[0.7],[0.8],[0.9],[1.0],[1.1],[1.2],[1.3],[1.4]])

        y = linear_interp(x,x_data,y_data)
        print('Output y of linear_interp with batches:\n', y)
        self.assertEqual(y.shape, x.shape)  # check that the output has the same shape as the input

if __name__ == '__main__':
    unittest.main()