import torch

class Module:
    def __init__(self):
        self._train_mode = True

        self.params = {}    # 가중치 저장
        self.grads = {}     # 가중치 미분값 저장
        self.cache = None   # forward때 입력값 저장
    
    def forward(self, x):
        raise NotImplementedError
    
    def backward(self, dout):
        raise NotImplementedError
    
    def zero_grad(self):    # 한 루프마다 역전파 전 초기화를 해준다.
        for key in self.grads:
            self.grads[key] = torch.zeros_like(self.grads[key])
    
    def parameters(self):
        """
        모든 가중치 출력
        """
        params_list = []

        for val in self.params.values():
            params_list.append(val)
        
        for key, value in self.__dict__.items():
            if isinstance(value, Module):
                params_list.extend(value.parameters())
        
        return params_list
    
    def train(self):
        """
        학습 모드로 전환 (Dropout, BatchNormal 등에 영향)
        """
        self._train_mode = True
        for key, value in self.__dict__.items():
            if isinstance(value, Module):
                value.train() # 자식들도 훈련 모드로!
    
    def eval(self):
        """평가 모드로 전환"""
        self._train_mode = False
        for key, value in self.__dict__.items():
            if isinstance(value, Module):
                value.eval() # 자식들도 평가 모드로!
            

class Sequential(Module):
    def __init__(self, *layers):
        """
        Sequential(layer1, layer2, ...)
        """
        super().__init__()
        self.layers = []

        for layer in layers:
            self.layers.append(layer) 

        
    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        
        return x
    
    def backward(self, dout):
        for layer in reversed(self.layers):
            dout =  layer.backward(dout)
        
        return dout

class ModuleList(Module):
    def __init__(self, modules=None):
        super().__init__()
        self.modules = []
        if modules:
            for m in modules:
                self.modules.append(m)
    
    def __getitem__(self, idx):
        return self.modules[idx]
    