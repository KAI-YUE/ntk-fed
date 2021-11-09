import copy
import numpy as np
import os
from collections import OrderedDict
import time
from typing import List, Tuple

import torch
import torch.nn as nn
Tensor = torch.Tensor
from torchdiffeq import odeint

def gradient_descent_ce(k_train_train, y_train, learning_rate):
    num_datpts = y_train.shape[0]
    odf_fn = lambda t, fx: -1/num_datpts*k_train_train @ (torch.softmax(fx, dim=-1)-y_train)
    
    def pred_fn(t, f_0): 
        t = t*learning_rate
        f_t = odeint(odf_fn, f_0, t)
        return f_t

    return pred_fn

def _make_expm1_fn(normalization):

    def expm1_fn(evals, t):
        return torch.expm1(-torch.max(evals, torch.tensor(0.)) * t / normalization)

    return expm1_fn

def _make_inv_expm1_fn(normalization):
    expm1_fn = _make_expm1_fn(normalization)

    def _inv_expm1_fn(evals, t):
        return expm1_fn(evals, t) / torch.abs(evals)

    return _inv_expm1_fn

def _get_fns_in_eigenbasis(k_train_train, fns):
    """Build functions of a matrix in its eigenbasis.
    
    Args:
      k_train_train:
        an n x n matrix.
      fns:
        a sequence of functions that add on the eigenvalues (evals, dt) ->
        modified_evals.
    
    Returns:
      A tuple of functions that act as functions of the matrix mat
      acting on vectors: `transform(vec, dt) = fn(mat, dt) @ vec`
    """
    evals, evecs = torch.linalg.eigh(k_train_train)
    evals = torch.unsqueeze(evals, 0)

    def to_eigenbasis(fn):
        """Generates a transform given a function on the eigenvalues."""
        def new_fn(y_train, t):
          return torch.einsum('ji,ti,ki,k...->tj...',
                           evecs, fn(evals, t), evecs, y_train)

        return new_fn

    return (to_eigenbasis(fn) for fn in fns)

def gradient_descent_mse(k_train_train, y_train, learning_rate):
    expm1_fn, inv_expm1_fn = _get_fns_in_eigenbasis(
        k_train_train,
        (_make_expm1_fn(y_train.numel()),
         _make_inv_expm1_fn(y_train.numel())))

    def predict_fn_finite(t, fx_train_0):
        t = t * learning_rate
        t = t.reshape((-1, 1))
        rhs = -y_train if fx_train_0 is None else fx_train_0 - y_train

        dfx_train = expm1_fn(rhs, t)
        fx_train_t = fx_train_0 + dfx_train
        return fx_train_t

    return predict_fn_finite

def _del_nested_attr(obj: nn.Module, names: List[str]) -> None:
	"""
	Deletes the attribute specified by the given list of names.
	For example, to delete the attribute obj.conv.weight,
	use _del_nested_attr(obj, ['conv', 'weight'])
	"""
	if len(names) == 1:
		delattr(obj, names[0])
	else:
		_del_nested_attr(getattr(obj, names[0]), names[1:])

def extract_weights(mod: nn.Module) -> Tuple[Tuple[Tensor, ...], List[str]]:
	"""
	This function removes all the Parameters from the model and
	return them as a tuple as well as their original attribute names.
	The weights must be re-loaded with `load_weights` before the model
	can be used again.
	Note that this function modifies the model in place and after this
	call, mod.parameters() will be empty.
	"""
	orig_params = tuple(mod.parameters())
	# Remove all the parameters in the model
	names = []
	for name, p in list(mod.named_parameters()):
		_del_nested_attr(mod, name.split("."))
		names.append(name)

	'''
		Make params regular Tensors instead of nn.Parameter
	'''
	params = tuple(p.detach().requires_grad_() for p in orig_params)
	return params, names

def _set_nested_attr(obj: nn.Module, names: List[str], value: torch.Tensor) -> None:
	"""
	Set the attribute specified by the given list of names to value.
	For example, to set the attribute obj.conv.weight,
	use _del_nested_attr(obj, ['conv', 'weight'], value)
	"""
	if len(names) == 1:
		setattr(obj, names[0], value)
	else:
		_set_nested_attr(getattr(obj, names[0]), names[1:], value)

def load_weights(mod: nn.Module, names: List[str], params: Tuple[Tensor, ...]) -> None:
	"""
	Reload a set of weights so that `mod` can be used again to perform a forward pass.
	Note that the `params` are regular Tensors (that can have history) and so are left
	as Tensors. This means that mod.parameters() will still be empty after this call.
	"""
	for name, p in zip(names, params):
		_set_nested_attr(mod, name.split("."), p)

def jacobian(model, x):
    """
    Args:
	model: model with vector output (not scalar output!) the parameters of which we want to compute the Jacobian for
	@param x: input since any gradients requires some input
	@return: either store jac directly in parameters or store them differently
    """
    jac_model = copy.deepcopy(model) # because we're messing around with parameters (deleting, reinstating etc)
    all_params, all_names = extract_weights(jac_model) # "deparameterize weights"
    load_weights(jac_model, all_names, all_params) # reinstate all weights as plain tensors
    jacs = OrderedDict()

    def param_as_input_func(model, x, param):
        load_weights(model, [name], [param]) # name is from the outer scope
        out = model(x)
        return out
    
    for i, (name, param) in enumerate(zip(all_names, all_params)):
        jac = torch.autograd.functional.jacobian(lambda param: param_as_input_func(jac_model, x, param), param, 
                                                 strict=True if i==0 else False, vectorize=False if i==0 else True)							 
        jacs[name] = jac.to("cpu")
        # jacs[name] = jac

    return jacs


def global_jacobian(model, x):
    """
    To avoid out of memory error, we calculate mini-batch jacobians and put it on cpu by default.  
    """
    batch_size = 100 
    num_datapoints = x.shape[0]
    iterations = num_datapoints//batch_size

    if num_datapoints % batch_size != 0:
        iterations += 1

    jac_template = jacobian(model, x[0].view(1,x.shape[1]))
    global_jacs = OrderedDict()

    for w_name, grad in jac_template.items():
        shape = torch.tensor(grad.shape)
        shape[0] = num_datapoints
        global_jacs[w_name] = torch.zeros(shape.tolist())

    for i in range(iterations):
        data = x[i*batch_size:(i+1)*batch_size]
        jac_template = jacobian(model, data)
        for w_name, grad in jac_template.items():
            global_jacs[w_name][i*batch_size:(i+1)*batch_size] = grad.cpu()
    
    return global_jacs


def combine_local_jacobians(local_packages):
    global_jacs = OrderedDict()
    num_datapoints = [0]
    w_names = list(local_packages[0].keys())
    w_name = w_names[0]
    for local_package in local_packages:
        num_datapoints.append(local_package[w_name].shape[0])

    for i in range(len(num_datapoints)-1):
        num_datapoints[i+1] += num_datapoints[i]

    sum_num_datapoints = num_datapoints[-1]

    for i, w_name in enumerate(w_names):
        shape = local_package[w_name].shape 
        shape = torch.tensor(shape)
        shape[0] = sum_num_datapoints
        global_jacs[w_name] = torch.zeros(shape.tolist())

    for i, local_package in enumerate(local_packages):
        for w_name in w_names:
            global_jacs[w_name][num_datapoints[i]:num_datapoints[i+1]] = local_package[w_name]

    return global_jacs

def empirical_kernel(jac_mats):
    keys = list(jac_mats.keys())
    num_datapts = jac_mats[keys[0]].shape[0]
    out_dim = jac_mats[keys[0]].shape[1]
    ker = torch.zeros((num_datapts, num_datapts), device="cuda")

    for w_name, jac_mat in jac_mats.items():
        jac_mat = jac_mat.view((jac_mat.shape[0], -1)).cuda()
        ker += jac_mat @ jac_mat.T

    ker /= out_dim
    
    return ker

def diag_fill(global_kernel, local_kernels):
    num_datapoints = [0]
    for local_kernel in local_kernels:
        num_datapoints.append(local_kernel.shape[0])

    for i in range(len(num_datapoints)-1):
        num_datapoints[i+1] += num_datapoints[i]

    for i, local_kernel in enumerate(local_kernels):
        global_kernel[num_datapoints[i]:num_datapoints[i+1], num_datapoints[i]:num_datapoints[i+1]] = local_kernel

    return global_kernel

class WeightMod(object):
    def __init__(self, weight_dict, mode="copy"):
        self._weight_dict = copy.deepcopy(weight_dict)
        
        if mode == "zeros":
            for w_name, w_value in self._weight_dict.items():
                self._weight_dict[w_name].data = torch.zeros_like(w_value)
        
        for w_name, w_value in self._weight_dict.items():
            self._weight_dict[w_name].data = self._weight_dict[w_name].data.to(weight_dict[w_name])
        
    def __add__(self, weight_buffer):
        weight_dict = copy.deepcopy(self._weight_dict)
        for w_name, w_value in weight_dict.items():
            weight_dict[w_name].data = self._weight_dict[w_name].data + weight_buffer._weight_dict[w_name].data

        return WeightMod(weight_dict)

    def __sub__(self, weight_buffer):
        weight_dict = copy.deepcopy(self._weight_dict)
        for w_name, w_value in weight_dict.items():
            weight_dict[w_name].data = self._weight_dict[w_name].data - weight_buffer._weight_dict[w_name].data

        return WeightMod(weight_dict)

    def __mul__(self, rhs):
        weight_dict = copy.deepcopy(self._weight_dict)
        for w_name, w_value in weight_dict.items():
            weight_dict[w_name].data = rhs*self._weight_dict[w_name].data

        return WeightMod(weight_dict)
    
    def add(self, params):
        if type(self) == type(params):
            rhs = params.state_dict()
            for w_name, w_value in self._weight_dict.items():
                self._weight_dict[w_name] += rhs[w_name].to(w_value) 
        else:
            for w_name, w_value in self._weight_dict.items():
                self._weight_dict[w_name].data += params[w_name].to(w_value)
    
    def mul(self, rhs):
        for w_name, w_value in self._weight_dict.items():
            self._weight_dict[w_name].data *= rhs
    
    def div(self, rhs):
        for i, w_name in enumerate(self._weight_dict):
            self._weight_dict[w_name].data /= rhs

    def mat_mul(self, rhs, subscripts):
        for w_name, w_value in self._weight_dict.items():
            self._weight_dict[w_name].data = torch.einsum(subscripts, w_value, rhs.to(w_value))
            # self._weight_dict[w_name].data = contract(subscripts, w_value, rhs.to(w_value))

    def push(self, weight_dict):
        self._weight_dict = copy.deepcopy(weight_dict)

    def apply_quant(self, quantizer):
        bits = 0
        total_codewords = quantizer.quant_level
        for w_name, w_value in self._weight_dict.items():
            quant_set = quantizer.quantize(w_value)
            bits += quant_set["quantized_arr"].numel()*self._entropy(quant_set["quantized_arr"], total_codewords)
            dequant_w = quantizer.dequantize(quant_set)
            self._weight_dict[w_name].data = dequant_w

        return bits

    def state_dict(self):
        return self._weight_dict

    def cpu(self):
        for w_name, w_value in self._weight_dict.items():
            self._weight_dict[w_name].data = w_value.cpu()

    def to(self, device):
        for w_name, w_value in self._weight_dict.items():
            self._weight_dict[w_name].data = w_value.to(device)

    def _entropy(self, seq, total_codewords):
        histogram = torch.histc(seq, bins=total_codewords, min=0, max=total_codewords-1) 
        total_symbols = seq.numel()

        histogram = histogram.detach().cpu().numpy().astype("float")
        histogram /= total_symbols

        entropy = 0
        for i, prob in enumerate(histogram):
            if prob == 0:
                continue
            entropy += -prob * np.log2(prob)

        return entropy