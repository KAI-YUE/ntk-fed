from copy import copy
import logging
import numpy as np
import time
import operator
from collections import OrderedDict

# PyTorch libraries
import torch
import torch.nn as nn
from opt_einsum import contract
from fedlearning import quantizer

# My libraries
from fedlearning.quantizer import *
from fedlearning.evolve import jacobian, empirical_kernel
from fedlearning.evolve import WeightMod

class LocalUpdater(object):
    def __init__(self, config, user_resource):
        """Construct a local updater for a user.

        Args:
            user_resources(dict):   
            a dictionary containing images and labels listed as follows. 
                - images (ndarray): training images of the user.
                - labels (ndarray): training labels of the user.

            config (class):         
            global configuration containing info listed as follows:
                - lr (float):       learning rate for the user.
                - batch_size (int): batch size for the user. 
                - mode (int):       the mode indicating the local model type.

            f: 
            neural network apply fun.
        """
        try:
            self.lr = user_resource["lr"]
            self.batch_size = user_resource["batch_size"]

            self.xs = torch.from_numpy(user_resource["images"]).to(config.device)
            self.ys = torch.from_numpy(user_resource["labels"]).to(config.device)
            self.label_size = user_resource["labels"].shape[-1]

        except KeyError:
            logging.error("LocalUpdater initialization failure! Input should include `lr`, `batch_size`!") 

        if config.loss == "ce":
            self.loss = lambda fx, y: -config.label_size*torch.mean(torch.log(torch.softmax(fx, dim=1))*y)
        elif config.loss == "mse":
            self.loss =  lambda yhat, y: 0.5*torch.mean((yhat - y)**2) 
        else:
            raise NotImplementedError

        self.taus = config.taus
        self.loss_type = config.loss
        self.debug = config.debug

    def _get_omegas(self, t, jac_mats, fx_t, state_dict):
        
        def jac_col(jac_mats, col_idx):
            jac = OrderedDict()
            for w_name, jac_mat in jac_mats.items():
                jac[w_name] = jac_mat[:,col_idx,...]
                                
            return jac
        
        if t[-1] == 0:
            return WeightMod(state_dict, "zeros")
        
        if self.loss_type == "ce":
            residuals = torch.sum(torch.softmax(fx_t[1:], dim=-1), dim=0) - (t[-1])*self.ys
        elif self.loss_type == "mse":
            residuals = torch.sum(fx_t[1:], dim=0) - (t[-1])*self.ys
        
        acc_omega = WeightMod(jac_col(jac_mats, 0))
        acc_omega.mat_mul(residuals[:,0], "ij...,i->j...")
        for col in range(1, self.ys.shape[-1]):
            omega = WeightMod(jac_col(jac_mats, col))
            omega.mat_mul(residuals[:,col], "ij...,i->j...")
            acc_omega.add(omega)

        if self.loss_type == "ce":
            acc_omega.mul(-self.lr*1/self.ys.shape[0])
        elif self.loss_type == "mse":
            acc_omega.mul(-self.lr*1/self.ys.shape[0]**1/self.ys.shape[1])

        return acc_omega

    def local_step(self, model, tau=None):
        self.jac_mats = jacobian(model, self.xs)

    def uplink_transmit(self):
        """Simulate the transmission of local weights to the central server.
        """ 
        local_package = self.jac_mats
        return local_package

class GlobalUpdater(object):
    def __init__(self, config, **kwargs):
        """Construct a global updater for a server.

        Args:
            config (class):              global configuration containing info listed as follows:
                - quantizer (str):       quantizer type.

        """
        self.num_users = int(config.users*config.part_rate)
        self.device = config.device       

    def init_aggregator(self, model):
        self.global_weight = WeightMod(model.state_dict())
        self.omegas_agg = WeightMod(model.state_dict(), mode="zeros")
    
    def receive(self, local_package):
        local_package.to(self.device)
        self.omegas_agg.add(local_package)

    def global_step(self):
        self.omegas_agg.div(self.num_users)
        self.global_weight.add(self.omegas_agg)
    
    def agg_weight(self):
        return self.global_weight.state_dict()

def get_omegas(t, lr, jac_mats, 
               ys, fx_t, 
               loss_type, state_dict):
    
    def jac_col(jac_mats, col_idx, device="cpu"):
        jac = OrderedDict()
        for w_name, jac_mat in jac_mats.items():
            jac[w_name] = jac_mat[:,col_idx,...].to(device)
                            
        return jac
    
    if t[-1] == 0:
        return WeightMod(state_dict, "zeros")
    
    if loss_type == "ce":
        residuals = torch.sum(torch.softmax(fx_t[:-1], dim=-1), dim=0) - (t[-1])*ys
    elif loss_type == "mse":
        residuals = torch.sum(fx_t[1:], dim=0) - (t[-1])*ys

    acc_omega = WeightMod(jac_col(jac_mats, 0, "cuda"))
    acc_omega.mat_mul(residuals[:,0].cuda(), "ij...,i->j...")
    for col in range(1, ys.shape[-1]):
        omega = WeightMod(jac_col(jac_mats, col, "cuda"))
        omega.mat_mul(residuals[:,col].cuda(), "ij...,i->j...")
        acc_omega.add(omega)

    if loss_type == "ce":
        acc_omega.mul(-lr*1/ys.shape[0])
    elif loss_type == "mse":
        acc_omega.mul(-lr*1/ys.shape[0]**1/ys.shape[1])

    return acc_omega