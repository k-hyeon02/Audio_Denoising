import numpy as np
from tools import Module

class sigmoid(Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return 1 / (1 + np.exp(x))
    
    def backward(self, dout):
        
        return 