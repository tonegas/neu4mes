import sys
import os
# append a new directory to sys.path
sys.path.append(os.getcwd())

from neu4mes import *
import torch
import torch.nn as nn

x = Input('x')
F = Input('F')

def rk4(fun, y0, t, dt, *args, **kwargs):
    dt2 = dt / 2.0
    k1 = fun(y0, t, *args, **kwargs)
    k2 = fun(y0 + dt2 * k1, t + dt2, *args, **kwargs)
    k3 = fun(y0 + dt2 * k2, t + dt2, *args, **kwargs)
    k4 = fun(y0 + dt * k3, t + dt, *args, **kwargs)
    dy = dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
    return dy

class DifferentiableModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, activation='tanh'):
        super(DifferentiableModel, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, output_size, bias=None)

        ## Orthogonal Initialization
        nn.init.orthogonal_(self.linear1.weight)
        nn.init.orthogonal_(self.linear2.weight)
        nn.init.orthogonal_(self.linear3.weight)

        if activation == 'tanh':
            self.activation = torch.tanh
        elif activation == 'relu':
            self.activation = torch.relu
        elif activation == 'sigmoid':
            self.activation = torch.sigmoid
        else:
            raise ValueError("activation not valid")

    def forward(self, x):
        x = self.activation(self.linear1(x))
        x = self.activation(self.linear2(x))
        x = self.linear3(x)
        return x
    
class HNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim=2, field_type='solenoidal', activation='tanh', canonical_coords=True):
        super(HNN, self).__init__()
        self.canonical_coords = canonical_coords
        self.differentiable_model = DifferentiableModel(input_size=input_dim, hidden_size=hidden_dim, output_size=output_dim, activation=activation)
        self.field_type = field_type
        self.M = self.permutation_tensor(input_dim)

    def forward(self, x):
        y = self.differentiable_model(x)
        return y.split(1,1)

    def permutation_tensor(self, n):
        M = None
        if self.canonical_coords:
            M = torch.eye(n)
            M = torch.cat([M[n//2:], -M[:n//2]])
        else:
            '''Constructs the Levi-Civita permutation tensor'''
            M = torch.ones(n,n) # matrix of ones
            M *= 1 - torch.eye(n) # clear diagonals
            M[::2] *= -1 # pattern of signs
            M[:,::2] *= -1

            for i in range(n): # make asymmetric
                for j in range(i+1, n):
                    M[i,j] *= -1
        return M

    def time_derivative(self, x, t=None, separate_fields=False):
        F1, F2 = self.forward(x)

        conservative_field = torch.zeros_like(x)
        solenoidal_field = torch.zeros_like(x)

        if self.field_type != 'solenoidal':
            dF1 = torch.autograd.grad(F1.sum(), x, create_graph=True)[0]
            conservative_field = dF1 @ torch.eye(*self.M.shape)

        if self.field_type != 'conservative':
            dF2 = torch.autograd.grad(F2.sum(), x, create_graph=True)[0]
            solenoidal_field = dF2 @ self.M.t()

        if separate_fields:
            return [solenoidal_field, conservative_field]

        return solenoidal_field + conservative_field

    def rk4_time_derivative(self, x, dt):
        return rk4(fun=self.time_derivative, y0=x, t=0, dt=dt)

print("------------------------EXAMPLE 1------------------------")
# Example 1
# Create an Hamiltonian Neural Network
hnn = Hamiltonian()
out = Output('out',hnn(x.last()))
example = Neu4mes()
example.addModel(out)
example.neuralizeModel(0.05)
print(example({'x':[2]}))