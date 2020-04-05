import time
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from torch.autograd import grad
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ExponentialLR

# Construct Library
def construct_Dictonary(data, uhat, poly_order, deriv_order):
    # build polynomials
    poly = torch.ones_like(uhat)

    # concatinate different orders
    for o in np.arange(1, poly_order + 1):
        poly_o = poly[:, o - 1:o] * uhat
        poly = torch.cat((poly, poly_o), dim=1)

    # build derivatives
    # returns gradient of uhat w.r.t. data (id0=spatial, id1=temporal)
    du = grad(outputs=uhat, inputs=data,
              grad_outputs=torch.ones_like(uhat), create_graph=True)[0]

    # time derivative
    dudt = du[:, 1:2]

    # spatial derivatives
    dudx = torch.cat((torch.ones_like(dudt), du[:, 0:1]), dim=1)

    # concatinate different orders
    for o in np.arange(1, deriv_order):
        du = grad(outputs=dudx[:, o:o + 1], inputs=data,
                  grad_outputs=torch.ones_like(uhat), create_graph=True)[0]
        dudx = torch.cat((dudx, du[:, 0:1]), dim=1)

    # build all possible combinations of poly and dudx vectors
    theta = None
    for i in range(poly.shape[1]):
        for j in range(dudx.shape[1]):
            comb = poly[:, i:i + 1] * dudx[:, j:j + 1]

            if theta is None:
                theta = comb
            else:
                theta = torch.cat((theta, comb), dim=1)

    return dudt, theta


#Construct Library for two dimentional data
def construct_Dictonary_2D(data, uhat, poly_order, deriv_order):
    # build polynomials
    poly = torch.ones_like(uhat)
    
    # concatinate different orders
    for o in np.arange(1, poly_order+1):
        poly_o = poly[:,o-1:o]*uhat
        poly = torch.cat((poly, poly_o), dim=1)
        #print('poly.shape', poly.shape)
        
    # build derivatives
    du = grad(outputs=uhat, inputs=data, 
              grad_outputs=torch.ones_like(uhat), create_graph=True)[0]
    
    dudt = du[:, 0:1]
    dudx = du[:, 1:2]
    dudy = du[:, 2:3]
    dudu = grad(outputs=dudx, inputs=data, 
              grad_outputs=torch.ones_like(uhat), create_graph=True)[0]
    dudxx = dudu[:, 1:2]
    dudxy = dudu[:, 2:3]
    dudyy = grad(outputs=dudy, inputs=data, 
              grad_outputs=torch.ones_like(dudxx), create_graph=True)[0]
    
    dudyy = dudyy[:, 2:3]
   
    #dudu = torch.cat((torch.ones_like(dudx), dudx, dudy, dudxx, dudyy, dudxy), dim=1)
    #dudu = torch.cat((torch.ones_like(dudx), dudx, dudy, dudxx, dudyy[:, 2:3], dudxy), dim=1)

    for o in np.arange(1, deriv_order):
        dudu = torch.cat((torch.ones_like(dudx), dudx, dudy, dudxx, dudyy, dudxy), dim=1)

    

    # build all possible combinations of poly and dudu vectors
    theta = None
    for i in range(poly.shape[1]):
        # print('i:', i)
        for j in range(dudu.shape[1]):
            # print('j:', j)
            comb = poly[:, i:i + 1] * dudu[:, j:j + 1]

            if theta is None:
                theta = comb
            else:
                theta = torch.cat((theta, comb), dim=1)
                
    return dudt, theta