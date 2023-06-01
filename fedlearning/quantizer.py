import numpy as onp

import torch
from fedlearning import Quantizer

class UniformPosQuantizer(Quantizer):
    def __init__(self, quant_level):
        self._epsilon = 0.05
        self.quant_level = quant_level 
        self.quantbound = (quant_level - 1)/2

        if self.quant_level % 2 == 0:   # even mid-riser quant, not mid-tread quant 
            self.mid_tread = False
        else:
            self.mid_tread = True

    def quantize(self, arr):
        """
        quantize a given arr array with unifrom quantization.
        """
        max_val = torch.max(arr.abs())
        
        quant_step = 2*max_val/self.quant_level

        if self.mid_tread:
            quantized_arr = torch.floor(arr/quant_step + 0.5)
        else:
            quantized_arr = torch.floor(arr/quant_step)
        
        quantized_arr = torch.where(quantized_arr>0, 2*quantized_arr-1, -2*quantized_arr)
        quantized_set = dict(norm=max_val, quantized_arr=quantized_arr)
        return quantized_set
    
    def dequantize(self, quantized_set):
        """
        dequantize a given array which is uniformed quantized.
        """
        quant_arr = quantized_set["quantized_arr"]
        dequant_arr = torch.where(quant_arr%2==0, -0.5*quant_arr, 0.5*(quant_arr+1))
        dequant_arr = dequant_arr/self.quantbound
        dequant_arr = quantized_set["norm"] * dequant_arr

        return dequant_arr


class UniformQuantizer(Quantizer):
    def __init__(self, quantization_level):
        self.quantbound = quantization_level - 1     

    def quantize(self, arr):
        """
        quantize a given arr array with unifrom quantization.
        """
        max_val = torch.max(torch.abs(arr))
        sign_arr = torch.sign(arr)
        quantized_arr = (arr/max_val)*self.quantbound
        quantized_arr = torch.abs(quantized_arr)
        quantized_arr = torch.round(quantized_arr).astype(int)
        
        quantized_set = dict(max_val=max_val, signs=sign_arr, quantized_arr=quantized_arr)
        
        return quantized_set
    
    def dequantize(self, quantized_set):
        """
        dequantize a given array which is uniformed quantized.
        """
        coefficients = quantized_set["max_val"]/self.quantbound  * quantized_set["signs"] 
        dequant_arr =  coefficients * quantized_set["quantized_arr"]

        return dequant_arr

class NormPosQuantizer(Quantizer):

    def __init__(self, quant_level):
        
        self.quant_level = quant_level 
        self.quantbound = (quant_level - 1)/2
        if self.quant_level % 2 == 0:   # even mid-riser quant, not mid-tread quant 
            self.mid_tread = False
        else:
            self.mid_tread = True

    def quantize(self, arr):
        # norm = 0.1*arr.norm()
        norm = torch.max(torch.abs(arr))
        abs_arr = arr.abs()

        level_float = abs_arr / norm * self.quantbound 
        lower_level = level_float.floor()
        rand_variable = torch.empty_like(arr).uniform_() 
        is_upper_level = rand_variable < (level_float - lower_level)
        new_level = (lower_level + is_upper_level)

        sign = arr.sign()
        quantized_arr = sign * torch.round(new_level).to(torch.int)

        quantized_arr = torch.where(quantized_arr>0, 2*quantized_arr-1, -2*quantized_arr)

        quantized_set = dict(norm=norm, quantized_arr=quantized_arr)

        return quantized_set

    def dequantize(self, quantized_set):
        quant_arr = quantized_set["quantized_arr"]
        dequant_arr = torch.where(quant_arr%2==0, -0.5*quant_arr, 0.5*(quant_arr+1))
        dequant_arr = dequant_arr/self.quantbound
        dequant_arr = quantized_set["norm"] * dequant_arr

        return dequant_arr

class QsgdQuantizer(Quantizer):

    def __init__(self, quantization_level):
        self.quantlevel = quantization_level
        self.quantbound = self.quantlevel - 1
    
    def quantize(self, arr):
        abs_arr = torch.abs(arr)
        norm = torch.max(abs_arr)

        level_float = abs_arr / norm * self.quantbound 
        lower_level = torch.floor(level_float)
        rand_variable = onp.random.uniform(0,1, arr.shape) 
        is_upper_level = rand_variable < (level_float - lower_level)
        new_level = (lower_level + is_upper_level)
        quantized_arr = torch.round(new_level)

        sign = arr.sign()
        quantized_set = dict(norm=norm, signs=sign, quantized_arr=quantized_arr)

        return quantized_set

    def dequantize(self, quantized_set):
        coefficients = quantized_set["norm"]/self.quantbound * quantized_set["signs"]
        dequant_arr = coefficients * quantized_set["quantized_arr"]

        return dequant_arr


class SqcCompressor:

    def __init__(self, params):
        super().__init__()
        self.sparsity = params["sparsity"]
        self.quant_level = params["quant_level"]

        self.epsilon = 1e-6
        
        if self.quant_level % 2 == 0:   # even mi-riser quant, not mid-tread quant 
            self.mid_tread = False
        else:
            self.mid_tread = True

    def quantize(self, tensor, **kwargs):
        """
        Compress the input tensor with stc.

        Args,
            tensor (torch.tensor): the input tensor.
        """
        signs = tensor.sign()
        k = onp.ceil(tensor.numel()*self.sparsity).astype(int)
        top_k_element, top_k_index = torch.kthvalue(-tensor.abs().flatten(), k)
        mask = tensor.abs() > -top_k_element
        tensor_masked = mask * tensor

        tensor_masked_abs = tensor_masked.abs()
        max_abs_val = tensor_masked_abs.max()
        min_abs_val = tensor_masked_abs.min()
        
        range_ = max_abs_val - min_abs_val + 1.e-8

        normalized_tensor = torch.where(tensor>0, tensor-min_abs_val, tensor+min_abs_val)
        quant_step = 2*range_/self.quant_level

        if self.mid_tread:
            quantized_tensor = torch.floor(normalized_tensor/quant_step + 0.5)
        else:
            quantized_tensor = torch.floor(normalized_tensor/quant_step)

        quantized_tensor = quantized_tensor*mask

        coded_set = dict(min_abs_val=min_abs_val,
                         range_=range_,
                         signs=signs,
                         quantized_arr=quantized_tensor,
                         mask=mask)
        
        return coded_set 

    def dequantize(self, coded_set):
        quantized_arr = coded_set["quantized_arr"]
        range_ = coded_set["range_"]
        min_abs_val = coded_set["min_abs_val"]
        
        quant_step = 2*range_/self.quant_level
        if self.mid_tread:
            dequantized_arr = quant_step*quantized_arr
        else:
            dequantized_arr = quant_step*(quantized_arr+0.5)
        
        dequantized_arr = (dequantized_arr.abs() + min_abs_val)*coded_set["signs"]
        dequantized_arr = coded_set["mask"]*dequantized_arr

        return dequantized_arr


def sparsity_based_on_rate(coding_rate, codingrate_params_pair):
    ptr = 0
    while ptr < len(codingrate_params_pair["coding_rate"]):
        ptr = ptr
        if coding_rate >= codingrate_params_pair["coding_rate"][ptr] and coding_rate < codingrate_params_pair["coding_rate"][ptr+1]:
            break
        else:
            ptr += 1
    
    hyperparams = dict(sparsity=codingrate_params_pair["sparsity"][ptr], 
                       quant_level=codingrate_params_pair["quant_level"][ptr])

    return hyperparams