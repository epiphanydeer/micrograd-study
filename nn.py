import random

from engine import *


class Neuron:
    # number of inputs : nin
    def __init__(self, nin):
        self.w = [Value(random.uniform(-1,1)) for _ in range(nin)]
        self.b = Value(random.uniform(-1,1))

    def __call__(self, x):
        # w * x + b
        # 使用zip函数将w和输入x关联起来
        act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        out = act.relu()
        return out
    
    def parameters(self):
        return self.w + [self.b]
    
class Layer:

    def __init__(self, nin, nout):
        # 每个nout都要和nin关联
        self.neurons = [Neuron(nin) for _ in range(nout)]

    def __call__(self, x):
        # 把输出n整合成列表
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs
    
    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]

class MLP:

    def __init__(self, nin, nouts):
        # nin输入维度，nouts是列表，后续每一层的输出维度nin=3, nouts=[4, 4, 1]
        sz = [nin] + nouts
        # i = 0：Layer(sz[0], sz[1]) -> Layer(3, 4)输入3维，输出4维
        self.layers = [Layer(sz[i], sz[i+1]) for i in range(len(nouts))]

    def __call__(self, x):
        # 串联流水线：前一层的输出作为下一层的输入，逐层传递
        for layer in self.layers:
            x = layer(x)
        return x    
    
    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]
    
