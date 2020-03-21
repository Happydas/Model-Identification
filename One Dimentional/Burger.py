import os
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import torch
import tensorflow as tf

import csv
from torch.autograd import grad
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ExponentialLR
from pylab import figure, text, scatter, show
from pdefind import *

from Model_Identification.PDE_Equation import pde_matrix_mul, sparse_coeff, normalized_xi_threshold, pde_Recover
from Model_Identification.build_Library import construct_Dictonary

# Data setup
data = sio.loadmat(os.path.join(os.getcwd(), "../data", "burgers.mat"))
u = data["usol"]
x = data["x"][0]
t = np.squeeze(data["t"], axis=1)

print("u shape", u.shape)
print("x shape", x.shape)
print("t shape", t.shape)

im = plt.imshow(u.real, cmap="seismic", aspect=0.4)
plt.colorbar(im)
plt.title("Burgers Equation")
plt.xlabel("t")
plt.ylabel("x")


xpos  = np.arange(len(t), step=25)
ypos  = np.where(x%4==0)[0].tolist()
ytick = x[ypos].tolist()
ypos  += [len(x)-1]
ytick += [8.0]

plt.xticks(xpos, t[xpos])
plt.yticks(ypos, ytick)
plt.gcf().set_size_inches(6,5)
plt.show()

# Add noise
np.random.seed(0)
noise_level = 0.1
u = u.real + noise_level*np.std(u.real)*np.random.randn(u.shape[0],u.shape[1])

im = plt.imshow(u.real, cmap="seismic", aspect=0.4)
plt.colorbar(im)
plt.title("Noisy Data")
plt.xlabel("t")
plt.ylabel("x")

plt.xticks(xpos, t[xpos])
plt.yticks(ypos, ytick)
plt.gcf().set_size_inches(6,5)
plt.show()

# Prepare Training Data
xx, tt = np.meshgrid(x,t)
X = np.vstack([xx.ravel(), tt.ravel()]).T
print("X shape", X.shape)
print(X)

y = np.zeros((u.size, 1), dtype=np.float)
for i, _x in enumerate(u.real.T):
    y[i * len(x):(i + 1) * len(x)] = _x.reshape(len(x), 1)

print("y shape", y.shape)

plt.plot(y[:256], label="t=0")
plt.plot(y[50 * 256:51 * 256], label="t=50")
plt.plot(y[100 * 256:101 * 256], label="t=100")
plt.xlabel("x")
plt.ylabel("u")
plt.xticks(ypos, ytick)
plt.legend()
plt.show()

idxs = np.random.choice(y.size, 1000, replace=False)

X_train = torch.tensor(X[idxs], dtype=torch.float32, requires_grad=True)
y_train = torch.tensor(y[idxs], dtype=torch.float32)
print("X_train shape", X_train.shape)
print("y_train shape", y_train.shape)
print("X shape", X.shape)
print("y shape", y.shape)

plt.subplot(1,2,1)
im = plt.imshow(u, cmap="seismic", aspect=0.4)
plt.colorbar(im)
plt.xlabel("t")
plt.ylabel("x")
plt.title("Noisy")

plt.xticks(xpos, t[xpos])
plt.yticks(ypos, ytick)

usamp = np.full(y.shape,np.nan)
usamp[idxs] = y[idxs]
usamp = usamp.reshape(u.T.shape).T

plt.subplot(1,2,2)
im = plt.imshow(usamp, cmap="seismic", aspect=.4)
plt.colorbar(im)
plt.xlabel("t")
plt.ylabel("x")
plt.title("Sampled")

plt.xticks(xpos, t[xpos])
plt.yticks(ypos, ytick)

plt.gcf().set_size_inches(10,4)
plt.show()

# Setup Network
net = PINN(sizes=[2,20,20,20,20,20,1], activation=torch.nn.Tanh())
print(net)

polynm = ['1', 'u', 'uˆ2']
spa_der = ['1', 'u_{x}', 'u_{xx}']
library_coeffs = pde_matrix_mul(polynm, spa_der)
print('library_coeffs:', library_coeffs)

tot_items = len(library_coeffs)
print('tot_items:', tot_items)

mask = torch.ones(tot_items, 1)
epochs = 10000
xi = nn.Parameter(torch.randn((tot_items, 1), requires_grad=True, device="cpu", dtype=torch.float32))
params = [{'params': net.parameters(), 'lr': 3e-3}, {'params': xi, 'lr': 3e-2}]

optimizer = Adam(params)
scheduler = ExponentialLR(optimizer, .9998)


def model_identification(features, label, mask, poly_order, deriv_order):
    lamb = 0
    tolerance = 1e-6
    mask = torch.ones(tot_items, 1)
    print('xi', xi)
    print('mask:', mask.shape)
    lambd = 1e-6

    L1_loss = []
    MSE_loss = []
    Reg_loss = []
    Total_loss = []

    for epoch in range(epochs):
        optimizer.zero_grad()
        uhat = net(features)

        if epoch == 1000:
            lamb = 1

        dudt, theta = construct_Dictonary(features, uhat, poly_order=2, deriv_order=2)
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

    np.savetxt('xi1.txt', xi_thresholded[0], fmt="%.8f")
    np.savetxt('xi2.txt', xi_thresholded[1], fmt="%.8f")
    # Calculate Error in xi
    xi1_error = np.subtract(np.array([0.10000]), xi_thresholded[0])
    print('xi1_error', xi1_error)
    xi2_error = np.subtract(np.array([1.00000]), xi_thresholded[1])
    print('xi2_error', xi2_error)
    print('Coefficient xi:')
    xi_updated = sparse_coeff(sparsity, xi_thresholded)
    print(xi_updated)

    print('Finished')

    return xi_updated, losses

uhat = net(X_train)
xi_updated, losses= model_identification(X_train, y_train, mask, poly_order=2, deriv_order=2)

print(uhat.shape)
pde_Recover(xi_updated, library_coeffs, equation_form='u_t')

uhat = net(torch.FloatTensor(X))
print('uhat.shape:', uhat.shape)
print("MSE loss", nn.MSELoss()(uhat, torch.FloatTensor(y)).item())

uhat = uhat.data.reshape(u.T.shape)

plt.subplot(1,3,1)
im = plt.imshow(data["usol"].real, cmap="seismic", aspect=0.4)
plt.colorbar(im)
plt.xlabel("t")
plt.ylabel("x")
plt.title("Original")

plt.xticks(xpos, t[xpos])
plt.yticks(ypos, ytick)

plt.subplot(1,3,2)
im = plt.imshow(u, cmap="seismic", aspect=0.4)
plt.colorbar(im)
plt.xlabel("t")
plt.ylabel("x")
plt.title("Noisy")

plt.xticks(xpos, t[xpos])
plt.yticks(ypos, ytick)

plt.subplot(1,3,3)
im = plt.imshow(uhat.numpy().T, cmap="seismic", aspect=.4)
plt.colorbar(im)
plt.xlabel("t")
plt.ylabel("x")
plt.title("Reconstructed")

plt.xticks(xpos, t[xpos])
plt.yticks(ypos, ytick)

plt.gcf().set_size_inches(18,4)
plt.show()


plt.imshow(np.abs(uhat.numpy().T.data - data["usol"].real), cmap="seismic", aspect=.4)
plt.colorbar()
plt.show()

usamp = np.full(y.shape,np.nan)
usamp[idxs] = y_train
usamp = usamp.reshape(u.T.shape).T

plt.subplot(1,2,2)
im = plt.imshow(usamp, cmap="seismic", aspect=.4)
plt.colorbar(im)
plt.xlabel("t")
plt.ylabel("x")
plt.title("Sampled")

#Ploting loss function
Total_loss = losses['Total_loss']
#print('Total Loss:', Total_loss)
L1_loss = losses['L1_loss']
MSE_loss = losses['MSE_loss']
Reg_loss = losses['Reg_loss']

plt.figure(figsize=(10, 5))
#plt.title('Losses against epochs (noise level = 10%, $\lambda$ = $10^{\mathrm{-6}}$ and tolerance = $10^{\mathrm{-6}}$)', fontsize=14)
plt.suptitle('Losses against epochs (noise level = 10%, $\lambda$ = $10^{\mathrm{-6}}$ and tolerance = $10^{\mathrm{-6}}$)', fontsize=14)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.plot(L1_loss, label="L1_loss")
plt.plot(MSE_loss, label="MSE_loss")
plt.plot(Reg_loss, label="Reg_loss")
plt.plot(Total_loss, label="Total_loss")
plt.legend(loc='upper right')
plt.show()