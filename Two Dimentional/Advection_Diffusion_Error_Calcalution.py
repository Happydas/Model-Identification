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
    xi1 = []
    xi2 = []
    xi3 = []
    xi4 = []
    xi1_error_app = []
    xi2_error_app = []
    xi3_error_app = []
    xi4_error_app = []
    xi_update_app = []

    numbers = [0, 5, 26, 57, 73, 80, 95, 104, 129, 151]

    f1 = open("Results/Average_errors.txt", "w+", encoding="utf-8")
    f2 = open("Results/Average_xi.txt", "w+", encoding="utf-8")

    correct_prediction = 0
    # Using for loop
    for i in numbers:
        np.random.seed(i)
        print('seed:', i)
        idxs = np.random.choice(y.size, 2000, replace=False)
        X_train = torch.tensor(features[idxs], dtype=torch.float32, requires_grad=True)
        y_train = torch.tensor(label[idxs], dtype=torch.float32)

        lamb = 0
        tolerance = 1e-6
        mask = torch.ones(tot_items, 1)
        # print('xi', xi)
        print('mask:', mask.shape)
        lambd = 1e-6

        for epoch in range(epochs):
            optimizer.zero_grad()
            uhat = net(X_train)

            if epoch == 1000:
                lamb = 1

            dudt, theta = construct_Dictonary_2D(X_train, uhat, poly_order=1, deriv_order=2)
            # print('dudt:', dudt.shape)
            dudt_norm = torch.norm(dudt, dim=0)

            theta_scaling = (torch.norm(theta, dim=0))
            # Returns a new tensor with a dimension of size one inserted at the specified position. from 9 it will be 9,1
            theta_norm = torch.unsqueeze(theta_scaling, dim=1)
            xi_normalized = xi * (theta_norm / dudt_norm)
            L1 = lambd * torch.sum(torch.abs(xi_normalized[1:, :]))

            l_u = nn.MSELoss()(uhat, y_train)
            l_reg = lamb * torch.mean((dudt - theta @ xi) ** 2)
            # l_reg = torch.mean((dudt - theta @ xi)**2)

            loss = l_u + l_reg + L1

            gradient_loss = torch.max(torch.abs(grad(outputs=loss, inputs=xi,
                                                     grad_outputs=torch.ones_like(loss), create_graph=True)[0]) / (
                                                  theta_norm / dudt_norm))

            loss.backward(retain_graph=True)
            optimizer.step()

            if epoch % 1000 == 0:
                print('loss:', epoch, loss)
                if gradient_loss < tolerance:
                    print('Optimizer converged.')
                    break

        xi_list = sparse_coeff(mask, xi.detach().numpy())
        xi_normalized = sparse_coeff(mask, xi_normalized.detach().numpy())
        print('xi_normalized:', xi_normalized)

        sparsity = normalized_xi_threshold(xi_normalized, mode='auto')
        print('sparsity:', sparsity)

        xi_thresholded = np.expand_dims(xi_list[sparsity], axis=1)
        print('xi_thresholded:', xi_thresholded)

        print('Coefficient xi:')
        xi_update = sparse_coeff(sparsity, xi_thresholded)
        print('xi_update:', xi_update)

        # Calculate Error in xi
        if xi_update[1] != 0 and xi_update[2] != 0 and xi_update[3] != 0 and xi_update[4] != 0 and len(
                xi_thresholded) < 5:
            xi_update_app.append(xi_update)
            # print('xi_update_app:', xi_update_app)
            print('xi_update:', xi_update[1], xi_update[2], xi_update[3], xi_update[4])
            error_1 = np.subtract(np.array([0.25000]), xi_update[1])
            xi1_error = np.append(xi1, error_1)
            print('xi1_error', xi1_error)
            error_2 = np.subtract(np.array([0.50000]), xi_update[2])
            xi2_error = np.append(xi2, error_2)
            print('xi2_error', xi2_error)
            error_3 = np.subtract(np.array([0.50000]), xi_update[3])
            xi3_error = np.append(xi3, error_3)
            print('xi3_error', xi3_error)
            error_4 = np.subtract(np.array([0.50000]), xi_update[4])
            xi4_error = np.append(xi4, error_4)
            print('xi4_error', xi4_error)
            xi1_error_app.append(xi1_error)
            print('xi1_error_app:', xi1_error_app)
            xi2_error_app.append(xi2_error)
            print('xi2_error_app:', xi2_error_app)
            xi3_error_app.append(xi3_error)
            print('xi3_error_app:', xi3_error_app)
            xi4_error_app.append(xi4_error)
            print('xi4_error_app:', xi4_error_app)
            correct_prediction += 1

        else:
            print('PDE prediction not correct')

    print('correct_prediction:', correct_prediction)
    Accuracy = correct_prediction / len(numbers) * 100
    print('Accuracy:', Accuracy)
    xi1_error_avg = np.sum(xi1_error_app, axis=0) / correct_prediction
    print('xi1_error_avg:', xi1_error_avg)
    xi2_error_avg = np.sum(xi2_error_app, axis=0) / correct_prediction
    print('xi2_error_avg:', xi2_error_avg)
    xi3_error_avg = np.sum(xi3_error_app, axis=0) / correct_prediction
    print('xi3_error_avg:', xi3_error_avg)
    xi4_error_avg = np.sum(xi4_error_app, axis=0) / correct_prediction
    print('xi4_error_avg:', xi4_error_avg)
    f1.write(str(xi1_error_avg) + " " + str(xi2_error_avg) + " " + str(xi3_error_avg) + " " + str(xi4_error_avg) + "\n")
    xi_updated = np.sum(xi_update_app, axis=0) / correct_prediction
    print('xi_updated:', xi_updated)
    f2.write(str(xi_updated[1]) + " " + str(xi_updated[2]) + " " + str(xi_updated[3]) + " " + str(xi_updated[4]) + "\n")
    print('Finished')
    return xi_updated

xi_updated = model_identification(X, y, mask, poly_order=1, deriv_order=2)

pde_Recover(xi_updated, library_coeffs, equation_form='u_t')
end_time = datetime.now()
print('end_time:', end_time)
print('Duration: {}'.format(end_time - start_time))