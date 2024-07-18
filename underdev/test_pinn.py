import torch


import os
import torch.nn as nn
import torch
import time

import numpy as np
torch.manual_seed(2)


# import torch
# a = torch.tensor([0.0],requires_grad=True)
# b = torch.tensor([8.])
# k = nn.Parameter(b,requires_grad=True)
# c = torch.sin(a)*k
# loss = torch.autograd.grad(c, [a], create_graph=True, retain_graph=True, allow_unused= False)
# print(loss)
# optimizer = torch.optim.Adam([k], lr=0.01)
# optimizer.zero_grad()
# sum(loss).backward(create_graph=True)
# optimizer.step()
#
# #print(a)
# #print(k)
# exit(1)

class U(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin_in = nn.Linear(2,20)
        self.lin = nn.ParameterList([nn.Linear(20, 20) for i in range(7)])
        self.lin_out = nn.Linear(20, 1)

    def forward(self, x, t):
        if len(x.shape) == 0:
            x = torch.unsqueeze(x, 0)
        if len(t.shape) == 0:
            t = torch.unsqueeze(t, 0)
        input = torch.transpose(torch.stack((x,t)),0,1)
        out = nn.Tanh()(self.lin_in(input))
        for i in range(7):
            out = nn.Tanh()(self.lin[i](out))
        out = self.lin_out(out)
        return out

class F(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, u, x, t):
        u_t = torch.autograd.grad([u], [t], grad_outputs=torch.ones_like(u), create_graph=True, retain_graph=True, allow_unused= False)[0]
        u_x = torch.autograd.grad([u], [x], grad_outputs=torch.ones_like(u), create_graph=True, retain_graph=True, allow_unused= False)[0]
        u_xx = torch.autograd.grad([u_x], [x], grad_outputs=torch.ones_like(u_x), create_graph=True, retain_graph=True, allow_unused= False)[0]
        return u_t + u * u_x - (0.01 / np.pi) * u_xx

learning_rate = 0.001
num_of_epochs = 5000

model_u = U()
model_f = F()
print(model_u)
print(sum(p.numel() for p in model_u.parameters() if p.requires_grad))
optimizer = torch.optim.Adam(model_u.parameters(), lr=learning_rate)
loss_fun = torch.nn.MSELoss()


#tt_0 = torch.zeros(50, dtype=torch.float32, requires_grad=True)
#xx_0 = 2 * torch.rand(50, dtype=torch.float32, requires_grad=True) - 1
#uu = model_u(xx_0,tt_0)
# torch.autograd.grad([uu], [tt_0], grad_outputs=torch.ones_like(uu), create_graph=True, retain_graph=True, allow_unused=False)[0]
# out_f = model_f(uu, xx_0, tt_0)
# zero = torch.zeros_like(out_f)
# mse_f = loss_fun(out_f,zero)
# mse_u = loss_fun(uu,uu_0)
# #
#

# Nu
tt_0 = torch.zeros(50, dtype=torch.float32)
xx_0 = 2 * torch.rand(50, dtype=torch.float32) - 1
uu_0 = -torch.sin(torch.pi * xx_0)
#
tt_1 = torch.rand(25, dtype=torch.float32)
xx_1 = torch.ones(25, dtype=torch.float32)
uu_1 = torch.zeros(25, dtype=torch.float32)
#
tt_2 = torch.rand(25, dtype=torch.float32)
xx_2 = -torch.ones(25, dtype=torch.float32)
uu_2 = torch.zeros(25, dtype=torch.float32)
#
tt_3 = torch.cat((tt_0, tt_1, tt_2))
xx_3 = torch.cat((xx_0, xx_1, xx_2))
uu_3 = torch.cat((uu_0, uu_1, uu_2))
ok_u = torch.ones(100, dtype=torch.float32)
data_u = torch.transpose(torch.vstack((uu_3, xx_3, tt_3, ok_u)),0,1)


# Nf
tt_4 = torch.rand(2900, dtype=torch.float32)
xx_4 = 2 * torch.rand(2900, dtype=torch.float32) - 1
uu_4 = torch.zeros(2900, dtype=torch.float32)
ok_u = torch.zeros(2900, dtype=torch.float32)
data_f = torch.transpose(torch.vstack((uu_4, xx_4, tt_4, ok_u)),0,1)

data = torch.vstack((data_u,data_f))

model_u.train()
#model_f.train()
for iter in range(num_of_epochs):
    data = data[torch.randperm(data.size()[0])]
    #data_u = data_u[torch.randperm(data_u.size()[0])]
    #data_f = data_f[torch.randperm(data_f.size()[0])]
    optimizer.zero_grad()
    loss = 0
    # mse = 0
    # mse_u = 0
    # mse_f = 0
    for i in range(30):
        uu = data[i * 100:(i + 1) * 100, 0].clone().detach().requires_grad_(True)
        xx = data[i * 100:(i + 1) * 100, 1].clone().detach().requires_grad_(True)
        tt = data[i * 100:(i + 1) * 100, 2].clone().detach().requires_grad_(True)
        ok = data[i * 100:(i + 1) * 100, 3].clone().detach().requires_grad_(True)
        uu_out = model_u(xx, tt)
        out_f = model_f(torch.squeeze(uu_out), xx, tt)
        zero = torch.zeros_like(out_f)
        mse_f = loss_fun(out_f, zero)
        mse_u = loss_fun(torch.squeeze(uu_out) * ok, uu * ok)
        mse = mse_u + mse_f
        loss += mse.detach().item()
        mse.backward()
        optimizer.step()

    print(f"Epoch {iter} Loss: {loss}")

    # for i in range(4):
    #     uu = data_u[i * 25:(i + 1) * 25, 0].clone().detach().requires_grad_(True)
    #     xx = data_u[i * 25:(i + 1) * 25, 1].clone().detach().requires_grad_(True)
    #     tt = data_u[i * 25:(i + 1) * 25, 2].clone().detach().requires_grad_(True)
    #     uu_out = model_u(xx, tt)
    #     mse_u = loss_fun(torch.squeeze(uu_out), uu)
    #     mse_u.backward()
    #     optimizer.step()
    #
    # for i in range(20):
    #     xx = data_f[i * 100:(i + 1) * 100, 0].clone().detach().requires_grad_(True)
    #     tt = data_f[i * 100:(i + 1) * 100, 1].clone().detach().requires_grad_(True)
    #     uu_out = model_u(xx, tt)
    #     out_f = model_f(torch.squeeze(uu_out), xx, tt)
    #     zero = torch.zeros_like(out_f)
    #     mse_f = loss_fun(out_f, zero)
    #     mse_f.backward()
    #     optimizer.step()
    # mse = mse_u + mse_f
    # #mse.backward()
    # #optimizer.step()
    # loss += mse.detach().item()
    # print(loss)

import matplotlib.pyplot as plt
x = torch.linspace(-1, 1, steps=100, dtype=torch.float32)
t = torch.zeros(100, dtype=torch.float32)
u = model_u(x,t)
plt.plot(x.detach().numpy(),u.detach().numpy())
plt.plot(xx_0.detach().numpy(),uu_0.detach().numpy(),'o')
plt.show()