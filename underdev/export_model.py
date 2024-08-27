import torch.nn as nn
import torch

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.relation_forward = {}
        self.relation_forward['Fir12'] = nn.Parameter() 
        self.relation_
        self.relation_forward = nn.ParameterDict(self.relation_forward)

    def forward(self, kwargs):
        getitem = kwargs['torque']
        getitem_1 = getitem[(slice(None, None, None), slice(0, 1, None))];  getitem = None
        size = getitem_1.size(0)
        relation_forward_fir12_weights = self.relation_forward.Fir12.weights
        size_1 = relation_forward_fir12_weights.size(1)
        squeeze = getitem_1.squeeze(-1);  getitem_1 = None
        matmul = torch.matmul(squeeze, relation_forward_fir12_weights);  squeeze = relation_forward_fir12_weights = None
        view = matmul.view(size, 1, size_1);  matmul = size = size_1 = None
        getitem_2 = kwargs['theta']
        getitem_3 = getitem_2[(slice(None, None, None), slice(0, 10, None))];  getitem_2 = None
        size_2 = getitem_3.size(0)
        relation_forward_fir9_weights = self.relation_forward.Fir9.weights
        size_3 = relation_forward_fir9_weights.size(1)
        squeeze_1 = getitem_3.squeeze(-1);  getitem_3 = None
        matmul_1 = torch.matmul(squeeze_1, relation_forward_fir9_weights);  squeeze_1 = relation_forward_fir9_weights = None
        view_1 = matmul_1.view(size_2, 1, size_3);  matmul_1 = size_2 = size_3 = None
        getitem_4 = kwargs['theta']
        getitem_5 = getitem_4[(slice(None, None, None), slice(0, 10, None))];  getitem_4 = None
        sin = torch.sin(getitem_5);  getitem_5 = None
        size_4 = sin.size(0)
        relation_forward_fir6_weights = self.relation_forward.Fir6.weights
        size_5 = relation_forward_fir6_weights.size(1)
        squeeze_2 = sin.squeeze(-1);  sin = None
        squeeze_2 = sin.squeeze(-1);  sin = None
        matmul_2 = torch.matmul(squeeze_2, relation_forward_fir6_weights);  squeeze_2 = relation_forward_fir6_weights = None
        view_2 = matmul_2.view(size_4, 1, size_5);  matmul_2 = size_4 = size_5 = None
        squeeze_2 = sin.squeeze(-1);  sin = None
        matmul_2 = torch.matmul(squeeze_2, relation_forward_fir6_weights);  squeeze_2 = relation_forward_fir6_weights = None
        squeeze_2 = sin.squeeze(-1);  sin = None
        squeeze_2 = sin.squeeze(-1);  sin = None
        squeeze_2 = sin.squeeze(-1);  sin = None
        squeeze_2 = sin.squeeze(-1);  sin = None
        squeeze_2 = sin.squeeze(-1);  sin = None
        matmul_2 = torch.matmul(squeeze_2, relation_forward_fir6_weights);  squeeze_2 = relation_forward_fir6_weights = None
        view_2 = matmul_2.view(size_4, 1, size_5);  matmul_2 = size_4 = size_5 = None
        add = torch.add(view_2, view_1);  view_2 = view_1 = None
        add_1 = torch.add(add, view);  add = view = None
        getitem_6 = kwargs['omega_target'];  kwargs = None
        getitem_7 = getitem_6[(slice(None, None, None), slice(0, 1, None))];  getitem_6 = None
        return ({'omega': add_1}, {'SamplePart17': getitem_7, 'omega': add_1})
    
    def forward(self, torque, theta, omega_target):
        getitem = torque
        getitem_1 = getitem[(slice(None, None, None), slice(0, 1, None))];  getitem = None
        size = getitem_1.size(0)
        relation_forward_fir12_weights = self.relation_forward.Fir12.weights
        size_1 = relation_forward_fir12_weights.size(1)
        squeeze = getitem_1.squeeze(-1);  getitem_1 = None
        matmul = torch.matmul(squeeze, relation_forward_fir12_weights);  squeeze = relation_forward_fir12_weights = None
        view = matmul.view(size, 1, size_1);  matmul = size = size_1 = None
        getitem_2 = theta
        getitem_3 = getitem_2[(slice(None, None, None), slice(0, 10, None))];  getitem_2 = None
        size_2 = getitem_3.size(0)
        relation_forward_fir9_weights = self.relation_forward.Fir9.weights
        size_3 = relation_forward_fir9_weights.size(1)
        squeeze_1 = getitem_3.squeeze(-1);  getitem_3 = None
        matmul_1 = torch.matmul(squeeze_1, relation_forward_fir9_weights);  squeeze_1 = relation_forward_fir9_weights = None
        view_1 = matmul_1.view(size_2, 1, size_3);  matmul_1 = size_2 = size_3 = None
        getitem_4 = theta
        getitem_5 = getitem_4[(slice(None, None, None), slice(0, 10, None))];  getitem_4 = None
        sin = torch.sin(getitem_5);  getitem_5 = None
        size_4 = sin.size(0)
        relation_forward_fir6_weights = self.relation_forward.Fir6.weights
        size_5 = relation_forward_fir6_weights.size(1)
        squeeze_2 = sin.squeeze(-1);  sin = None
        squeeze_2 = sin.squeeze(-1);  sin = None
        matmul_2 = torch.matmul(squeeze_2, relation_forward_fir6_weights);  squeeze_2 = relation_forward_fir6_weights = None
        view_2 = matmul_2.view(size_4, 1, size_5);  matmul_2 = size_4 = size_5 = None
        squeeze_2 = sin.squeeze(-1);  sin = None
        matmul_2 = torch.matmul(squeeze_2, relation_forward_fir6_weights);  squeeze_2 = relation_forward_fir6_weights = None
        squeeze_2 = sin.squeeze(-1);  sin = None
        squeeze_2 = sin.squeeze(-1);  sin = None
        squeeze_2 = sin.squeeze(-1);  sin = None
        squeeze_2 = sin.squeeze(-1);  sin = None
        squeeze_2 = sin.squeeze(-1);  sin = None
        matmul_2 = torch.matmul(squeeze_2, relation_forward_fir6_weights);  squeeze_2 = relation_forward_fir6_weights = None
        view_2 = matmul_2.view(size_4, 1, size_5);  matmul_2 = size_4 = size_5 = None
        add = torch.add(view_2, view_1);  view_2 = view_1 = None
        add_1 = torch.add(add, view);  add = view = None
        getitem_6 = omega_target;  kwargs = None
        getitem_7 = getitem_6[(slice(None, None, None), slice(0, 1, None))];  getitem_6 = None
        return (add_1, getitem_7, add_1)