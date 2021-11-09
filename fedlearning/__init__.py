from abc import ABC, abstractmethod

from .model import *

class Quantizer(ABC):
    """Interface for quantizing and dequantizing a given tensor."""

    def __init__(self):
        pass

    @abstractmethod
    def quantize(self, seq):
        """Compresses a tensor with the given compression context, 
        and then returns it with the context needed to decompress it."""

    @abstractmethod
    def dequantize(self, quantized_set):
        """Decompress the tensor with the given decompression context."""


nn_registry = {
    "mlp": NaiveMLP
}
