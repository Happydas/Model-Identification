import os
import numpy as np
import pandas as pd
import scipy.io as sio
import matplotlib.pyplot as plt
import torch
import tensorflow as tf

from torch.autograd import grad
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ExponentialLR

from pdefind import *
from Model_Identification.PDE_Equation import pde_matrix_mul, sparse_coeff, normalized_xi_threshold, pde_Recover
from Model_Identification.build_Library import construct_Dictonary_2D
from datetime import datetime

start_time = datetime.now()
print('start_time:', start_time)

# Prepare dataset
data = sio.loadmat(os.path.join(os.getcwd(), "../data", "Advection_diffusion.mat"))
usol = np.real(data['Expression1'])
print('u.shape', usol.shape)
usol= usol.reshape((51,51,61,4))
#u = data["usol"]
x = usol[:,:,:,0]
y = usol[:,:,:,1]
t = usol[:,:,:,2]
u = usol[:,:,:,3]
print('u.shape', u.shape)
print('x.shape', x.shape)
print('t.shape', t.shape)
print('y.shape', y.shape)

X = np.transpose((t.flatten(),x.flatten(), y.flatten()))
Y = u.reshape((u.size, 1))
print(X.shape, Y.shape)
print('X:', X)

# Add noise
np.random.seed(0)
noise_level = 0.01
y = Y + noise_level * np.std(Y) * np.random.randn(Y.size, 1)

idxs = np.random.choice(y.size, 2000, replace=False)
X_train = torch.tensor(X[idxs], dtype=torch.float32, requires_grad=True)
y_train = torch.tensor(y[idxs], dtype=torch.float32)
print("X_train shape", X_train.shape)
print("y_train shape", y_train.shape)

# Setup Network
net = PINN(sizes=[3,20,20,20,20,20,1], activation=torch.nn.Tanh())
print(net)

polynm = ['1', 'u']
spa_der = ['1', 'u_{x}', 'u_{y}','u_{xx}', 'u_{yy}','u_{xy}']
library_coeffs = pde_matrix_mul(polynm, spa_der)
print('library_coeffs:', library_coeffs)

tot_items = len(library_coeffs)
print('tot_items:', tot_items)

mask = torch.ones(tot_items, 1)
epochs = 10000
xi = nn.Parameter(torch.randn((tot_items, 1), requires_grad=True, device="cpu", dtype=torch.float32))
#params = [{'params': net.parameters(), 'lr': 3e-3}, {'params': xi, 'lr': 3e-2}]
params = [{'params': net.parameters(), 'lr': 1e-3}, {'params': xi, 'lr': 1e-2}]

optimizer = Adam(params)
scheduler = ExponentialLR(optimizer, .9998)


def model_identification(features, label, mask, poly_order, deriv_order):
    lamb = 0
    tolerance = 1e-6
    mask = torch.ones(tot_items, 1)
    print('xi', xi)
    print('mask:', mask.shape)
    lambd = 1e-5

    L1_loss = []
    MSE_loss = []
    Reg_loss = []
    Total_loss = []

    for epoch in range(epochs):
        optimizer.zero_grad()
        uhat = net(features)

        if epoch == 1000:
            lamb = 1

        dudt, theta = construct_Dictonary_2D(features, uhat, poly_order=1, deriv_order=2)
        # print('dudt:', dudt.shape)
        dudt_norm = torch.norm(dudt, dim=0)
        # print('dudt_norm:', dudt_norm.shape)

        theta_scaling = (torch.norm(theta, dim=0))
        # print('theta_scaling:', theta_scaling.shape)
        # Returns a new tensor with a dimension of size one inserted at the specified position. from 9 it will be 9,1
        theta_norm = torch.unsqueeze(theta_scaling, dim=1)
        # print('theta_norm:', theta_norm.shape)
        xi_normalized = xi * (theta_norm / dudt_norm)
        L1 = lambd * torch.sum(torch.abs(xi_normalized[1:, :]))

        l_u = nn.MSELoss()(uhat, label)
        l_reg = lamb * torch.mean((dudt - theta @ xi) ** 2)
        # l_reg = torch.mean((dudt - theta @ xi)**2)

        loss = l_u + l_reg + L1
        # print('loss', loss)

        L1_loss.append(L1.item())
        MSE_loss.append(l_u.item())
        Reg_loss.append(l_reg.item())
        Total_loss.append(loss.item())

        losses = {"L1_loss": L1_loss,
                  "MSE_loss": MSE_loss,
                  "Reg_loss": Reg_loss,
                  "Total_loss": Total_loss}

        gradient_loss = torch.max(torch.abs(grad(outputs=loss, inputs=xi,
                                                 grad_outputs=torch.ones_like(loss), create_graph=True)[0]) / (
                                              theta_norm / dudt_norm))

        loss.backward(retain_graph=True)
        optimizer.step()

        # print("epoch {}/{}, loss={:.10f}".format(epoch+1, epochs, loss.item()), end="\r")

        if epoch % 1000 == 0:
            print('loss:', epoch, loss)
            if gradient_loss < tolerance:
                print('Optimizer converged.')
                break

    # print('xi_normalized:', xi_normalized)
    xi_list = sparse_coeff(mask, xi.detach().numpy())
    xi_normalized = sparse_coeff(mask, xi_normalized.detach().numpy())
    print('xi_normalized:', xi_normalized)

    sparsity = normalized_xi_threshold(xi_normalized, mode='auto')
    print('sparsity:', sparsity)

    xi_thresholded = np.expand_dims(xi_list[sparsity], axis=1)
    print('xi_thresholded:', xi_thresholded)
    # Printing current sparse vector
    print('Coefficient xi:')
    xi_updated = sparse_coeff(sparsity, xi_thresholded)
    print(xi_updated)
    print('Finished')

    return xi_updated, losses

mask = torch.ones(tot_items, 1)
uhat = net(X_train)
xi_updated, losses = model_identification(X_train, y_train, mask, poly_order=1, deriv_order=2)

print(uhat.shape)
pde_Recover(xi_updated, library_coeffs, equation_form='u_t')