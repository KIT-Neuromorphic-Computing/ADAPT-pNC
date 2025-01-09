import torch
import numpy as np

class Evaluator(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.sensing_margin = 0.01
        self.performance = self.nominal
        self.args = args
        
    def nominal(self, nn, x, label):
        prediction = nn(x)
        act, idx = torch.max(prediction, dim=1)
        corrects = (label.view(-1) == idx)
        return corrects.float().sum().item() / label.numel()
    
    def maa(self, nn, x, label):
        prediction = nn(x)
        act, idx = torch.topk(prediction, k=2, dim=1)
        corrects = (act[:,0] >= self.sensing_margin) & (act[:,1]<=0) & (label.view(-1)==idx[:,0])
        return corrects.float().sum().item() / label.numel()
    
    def nominal_snn(self, nn, x, label):
        spk, mem = nn(x)
        act, idx = spk.sum(2).max(dim=1)
        corrects = (label.view(-1) == idx)
        return corrects.float().sum().item() / label.numel()
    
    def nominal_temporal(self, nn, x, label):
        output = nn(x)
        act, idx = output.sum(2).max(dim=1)
        corrects = (label.view(-1) == idx)
        return corrects.float().sum().item() / label.numel()
    
    def nominal_temporized(self, nn, x, label):
        Corrects = []
        T = x.shape[2]
        for t in range(T):
            act, idx = nn(x[:,:,t]).max(dim=1)
            corrects = (label.view(-1) == idx)
            Corrects.append(corrects.float().sum().item() / label.numel())
        return np.mean(Corrects)
        
    def forward(self, nn, x, label):
        if self.args.metric == 'acc':
            self.performance = self.nominal
        elif self.args.metric == 'maa':
            self.performance = self.maa
        elif self.args.metric == 'temporal_acc':
            self.performance = self.nominal_temporal
        elif self.args.metric == 'temporized_acc':
            self.performance = self.nominal_temporized
        elif self.args.metric == 'nominal_snn':
            self.performance = self.nominal_snn
        return self.performance(nn, x, label)
    
  