import numpy as np
import torch

# ================================================================================================================================================
# =====================================================  Learnable Negative Weight Circuit  ======================================================
# ================================================================================================================================================

class InvRT(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.N = args.N_train
        self.epsilon = args.e_train
        # R1, R2, R3, W1, L1, W2, L2, W3, L3
        self.rt_ = torch.nn.Parameter(torch.tensor([args.NEG_R1n, args.NEG_R2n, args.NEG_R3n, args.NEG_W1n, args.NEG_L1n, args.NEG_W2n, args.NEG_L2n, args.NEG_W3n, args.NEG_L3n]).to(args.DEVICE), requires_grad=True)
        # model
        package = torch.load('./utils/neg_param.package')
        self.eta_estimator = package['eta_estimator'].to(self.DEVICE)
        self.eta_estimator.train(False)
        for name, param in self.eta_estimator.named_parameters():
            param.requires_grad = False
        self.X_max = package['X_max'].to(self.DEVICE)
        self.X_min = package['X_min'].to(self.DEVICE)
        self.Y_max = package['Y_max'].to(self.DEVICE)
        self.Y_min = package['Y_min'].to(self.DEVICE)
        # load power model
        package = torch.load('./utils/neg_power.package')
        self.power_estimator = package['power_estimator'].to(self.DEVICE)
        for name, param in self.power_estimator.named_parameters():
            param.requires_grad = False
        self.power_estimator.train(False)
        self.pow_X_max = package['X_max'].to(self.DEVICE)
        self.pow_X_min = package['X_min'].to(self.DEVICE)
        self.pow_Y_max = package['Y_max'].to(self.DEVICE)
        self.pow_Y_min = package['Y_min'].to(self.DEVICE)

    @property
    def DEVICE(self):
        return self.args.DEVICE
    
    @property
    def RT(self):
        # keep values in (0,1)
        rt_temp = torch.sigmoid(self.rt_)
        RTn = torch.zeros([12]).to(self.DEVICE)
        RTn[:9] = rt_temp
        # denormalization
        RT = RTn * (self.X_max - self.X_min) + self.X_min
        return RT
    
    @property
    def RT_noisy(self):
        RT_mean = self.RT.repeat(self.N, 1).to(self.DEVICE)
        noise = (((torch.rand(RT_mean.shape) * 2. - 1.) * self.epsilon) + 1.).to(self.DEVICE)
        RT_variation = RT_mean * noise
        return RT_variation

    @property
    def RTn_extend(self):
        RT_extend = torch.stack([self.RT_noisy[:,0], self.RT_noisy[:,1], self.RT_noisy[:,2], self.RT_noisy[:,3],
                                 self.RT_noisy[:,4], self.RT_noisy[:,5], self.RT_noisy[:,6], self.RT_noisy[:,7],
                                 self.RT_noisy[:,8], self.RT_noisy[:,3]/self.RT_noisy[:,4], self.RT_noisy[:,5]/self.RT_noisy[:,6],
                                 self.RT_noisy[:,7]/self.RT_noisy[:,8]], dim=1)
        return (RT_extend - self.X_min) / (self.X_max - self.X_min)

    @property
    def eta(self):
        # calculate eta
        eta_n = self.eta_estimator(self.RTn_extend)
        eta = eta_n * (self.Y_max - self.Y_min) + self.Y_min
        return eta

    @property
    def power(self):
        # calculate power
        power_n = self.power_estimator(self.RTn_extend)
        power = power_n * (self.pow_Y_max - self.pow_Y_min) + self.pow_Y_min
        return power.mean()

    def forward(self, z):
        eta_noisy = self.eta
        a = torch.zeros_like(z)
        for i in range(self.N):
            a[i,:,:] = - (self.eta[i,0] + self.eta[i,1] * torch.tanh((z[i,:,:] - self.eta[i,2]) * self.eta[i,3]))
        return a
    
    def UpdateArgs(self, args):
        self.args = args
    

# ================================================================================================================================================
# ========================================================  Learnable Activation Circuit  ========================================================
# ================================================================================================================================================

class TanhRT(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.N = args.N_train
        self.epsilon = args.e_train
        # R1n, R2n, W1n, L1n, W2n, L2n
        self.rt_ = torch.nn.Parameter(
            torch.tensor([args.ACT_R1n, args.ACT_R2n, args.ACT_W1n, args.ACT_L1n, args.ACT_W2n, args.ACT_L2n]), requires_grad=True)

        # model
        package = torch.load('./utils/act_model_package')
        self.eta_estimator = package['eta_estimator'].to(self.DEVICE)
        self.eta_estimator.train(False)
        for n, p in self.eta_estimator.named_parameters():
            p.requires_grad = False
        self.X_max = package['X_max'].to(self.DEVICE)
        self.X_min = package['X_min'].to(self.DEVICE)
        self.Y_max = package['Y_max'].to(self.DEVICE)
        self.Y_min = package['Y_min'].to(self.DEVICE)
        # load power model
        package = torch.load('./utils/act_power_model_package')
        self.power_estimator = package['power_estimator'].to(self.DEVICE)
        self.power_estimator.train(False)
        for n, p in self.power_estimator.named_parameters():
            p.requires_grad = False
        self.pow_X_max = package['X_max'].to(self.DEVICE)
        self.pow_X_min = package['X_min'].to(self.DEVICE)
        self.pow_Y_max = package['Y_max'].to(self.DEVICE)
        self.pow_Y_min = package['Y_min'].to(self.DEVICE)

    @property
    def DEVICE(self):
        return self.args.DEVICE
    
    @property
    def RT(self):
        # keep values in (0,1)
        rt_temp = torch.sigmoid(self.rt_).to(self.DEVICE)
        RTn = torch.zeros([9]).to(self.DEVICE)
        RTn[:6] = rt_temp.to(self.DEVICE)
        # denormalization
        RT = RTn * (self.X_max - self.X_min) + self.X_min
        return RT
    
    @property
    def RT_noisy(self):
        RT_mean = self.RT.repeat(self.N, 1).to(self.DEVICE)
        noise = (((torch.rand(RT_mean.shape) * 2. - 1.) * self.epsilon) + 1.).to(self.DEVICE)
        RT_variation = RT_mean * noise
        return RT_variation
    
    @property
    def RTn_extend(self):
        RT_extend = torch.stack([self.RT_noisy[:,0], self.RT_noisy[:,1], self.RT_noisy[:,2], self.RT_noisy[:,3],
                                 self.RT_noisy[:,4], self.RT_noisy[:,5], self.RT_noisy[:,1]/self.RT_noisy[:,0],
                                 self.RT_noisy[:,3]/self.RT_noisy[:,2], self.RT_noisy[:,5]/self.RT_noisy[:,4]], dim=1)
        return (RT_extend - self.X_min) / (self.X_max - self.X_min)

    @property
    def eta(self):
        # calculate eta
        eta_n = self.eta_estimator(self.RTn_extend)
        eta = eta_n * (self.Y_max - self.Y_min) + self.Y_min
        return eta

    @property
    def power(self):
        # calculate power
        power_n = self.power_estimator(self.RTn_extend)
        power = power_n * (self.pow_Y_max - self.pow_Y_min) + self.pow_Y_min
        return power.mean()

    def forward(self, z):
        eta_noisy = self.eta
        a = torch.zeros_like(z)
        for i in range(self.N):
            a[i,:,:] = self.eta[i,0] + self.eta[i,1] * torch.tanh((z[i,:,:] - self.eta[i,2]) * self.eta[i,3])
        return a
    
    def UpdateArgs(self, args):
        self.args = args
    

#================================================================================================================================================
#======================================================  Learnable Hard Sigmoid Activation  =====================================================
#================================================================================================================================================


class HardSigmoidRT(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.N = args.N_train
        self.epsilon = args.e_train
        # R, W, L
        self.rt_ = torch.nn.Parameter(torch.tensor([args.HS_Rn, args.HS_Wn, args.HS_Ln]).to(args.DEVICE), requires_grad=True)
        # model
        package = torch.load('./utils/hard_sigmoid_param.package')
        self.eta_estimator = package['eta_estimator'].to(self.DEVICE)
        self.eta_estimator.train(False)
        for name, param in self.eta_estimator.named_parameters():
            param.requires_grad = False
        self.X_max = package['X_max'].to(self.DEVICE)
        self.X_min = package['X_min'].to(self.DEVICE)
        self.Y_max = package['Y_max'].to(self.DEVICE)
        self.Y_min = package['Y_min'].to(self.DEVICE)
        # load power model
        package = torch.load('./utils/hard_sigmoid_power.package')
        self.power_estimator = package['power_estimator'].to(self.DEVICE)
        for name, param in self.power_estimator.named_parameters():
            param.requires_grad = False
        self.power_estimator.train(False)
        self.pow_X_max = package['X_max'].to(self.DEVICE)
        self.pow_X_min = package['X_min'].to(self.DEVICE)
        self.pow_Y_max = package['Y_max'].to(self.DEVICE)
        self.pow_Y_min = package['Y_min'].to(self.DEVICE)

    @property
    def DEVICE(self):
        return self.args.DEVICE
    
    @property
    def RT(self):
        # keep values in (0,1)
        rt_temp = torch.sigmoid(self.rt_)
        RTn = torch.zeros([4]).to(self.DEVICE)
        RTn[:3] = rt_temp
        # denormalization
        RT = RTn * (self.X_max - self.X_min) + self.X_min
        return RT
    
    @property
    def RT_noisy(self):
        RT_mean = self.RT.repeat(self.N, 1)
        noise = ((torch.rand(RT_mean.shape) * 2. - 1.) * self.epsilon) + 1.
        RT_variation = RT_mean * noise
        return RT_variation
    
    @property
    def RTn_extend(self):
        RT_extend = torch.stack([self.RT_noisy[:,0], self.RT_noisy[:,1], self.RT_noisy[:,2], self.RT_noisy[:,1]/self.RT_noisy[:,2]], dim=1)
        return (RT_extend - self.X_min) / (self.X_max - self.X_min)

    @property
    def eta(self):
        # calculate eta
        eta_n = self.eta_estimator(self.RTn_extend)
        eta = eta_n * (self.Y_max - self.Y_min) + self.Y_min
        return eta

    @property
    def power(self):
        # calculate power
        power_n = self.power_estimator(self.RTn_extend)
        power = power_n * (self.pow_Y_max - self.pow_Y_min) + self.pow_Y_min
        return power.mean()

    def forward(self, z):
        eta_noisy = self.eta
        a = torch.zeros_like(z)
        for i in range(self.N):
            linear_segment = self.eta[i,0] + (self.eta[i,1] - self.eta[i,0]) / (self.eta[i,3] - self.eta[i,2]) * (z[i,:,:] - self.eta[i,2])
            a[i,:,:] = torch.where(z[i,:,:] < self.eta[i,2], self.eta[i,0], torch.where(z[i,:,:] <= self.eta[i,3], linear_segment, self.eta[i,1]))
        return a
    
    def UpdateArgs(self, args):
        self.args = args
    

#================================================================================================================================================
#=======================================================  Learnable Soft pReLU Activation  ======================================================
#================================================================================================================================================


class pReLURT(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.N = args.N_train
        self.epsilon = args.e_train
        # RH, RL, RD, RB, W, L
        self.rt_ = torch.nn.Parameter(torch.tensor([args.ReLU_RHn, args.ReLU_RLn, args.ReLU_RDn, args.ReLU_RBn, args.ReLU_Wn, args.ReLU_Ln]).to(args.DEVICE), requires_grad=True)
        # model
        package = torch.load('./utils/ReLU_param.package')
        self.eta_estimator = package['eta_estimator'].to(self.DEVICE)
        self.eta_estimator.train(False)
        for name, param in self.eta_estimator.named_parameters():
            param.requires_grad = False
        self.X_max = package['X_max'].to(self.DEVICE)
        self.X_min = package['X_min'].to(self.DEVICE)
        self.Y_max = package['Y_max'].to(self.DEVICE)
        self.Y_min = package['Y_min'].to(self.DEVICE)
        # load power model
        package = torch.load('./utils/ReLU_power.package')
        self.power_estimator = package['power_estimator'].to(self.DEVICE)
        for name, param in self.power_estimator.named_parameters():
            param.requires_grad = False
        self.power_estimator.train(False)
        self.pow_X_max = package['X_max'].to(self.DEVICE)
        self.pow_X_min = package['X_min'].to(self.DEVICE)
        self.pow_Y_max = package['Y_max'].to(self.DEVICE)
        self.pow_Y_min = package['Y_min'].to(self.DEVICE)

    @property
    def DEVICE(self):
        return self.args.DEVICE
    
    @property
    def RT(self):
        # keep values in (0,1)
        rt_temp = torch.sigmoid(self.rt_)
        RTn = torch.zeros([8]).to(self.DEVICE)
        RTn[:6] = rt_temp
        # denormalization
        RT = RTn * (self.X_max - self.X_min) + self.X_min
        return RT
    
    @property
    def RT_noisy(self):
        RT_mean = self.RT.repeat(self.N, 1)
        noise = ((torch.rand(RT_mean.shape) * 2. - 1.) * self.epsilon) + 1.
        RT_variation = RT_mean * noise
        return RT_variation
    
    @property
    def RTn_extend(self):
        RT_extend = torch.stack([self.RT_noisy[:,0], self.RT_noisy[:,1], self.RT_noisy[:,2], self.RT_noisy[:,3],
                                 self.RT_noisy[:,4], self.RT_noisy[:,5], self.RT_noisy[:,0]/self.RT_noisy[:,1], self.RT_noisy[:,4]/self.RT_noisy[:,5]], dim=1)
        return (RT_extend - self.X_min) / (self.X_max - self.X_min)

    @property
    def eta(self):
        # calculate eta
        eta_n = self.eta_estimator(self.RTn_extend)
        eta = eta_n * (self.Y_max - self.Y_min) + self.Y_min
        return eta

    @property
    def power(self):
        # calculate power
        power_n = self.power_estimator(self.RTn_extend)
        power = power_n * (self.pow_Y_max - self.pow_Y_min) + self.pow_Y_min
        return power.mean()

    def forward(self, z):
        def softplus(x, beta):
            return (1.0 / beta) * torch.log(1 + torch.exp(beta * x))
        eta_noisy = self.eta
        a = torch.zeros_like(z)
        for i in range(self.N):
            a[i,:,:] = self.eta[i,0] * (z[i,:,:] - self.eta[i,2]) + self.eta[i,1] * softplus(z[i,:,:] - self.eta[i,2], self.eta[i,4]) + self.eta[i,3]
        return a
    
    def UpdateArgs(self, args):
        self.args = args
    
# ================================================================================================================================================
# ========================================================  Learnable Sigmoid Activation  ========================================================
# ================================================================================================================================================

class SigmoidRT(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.N = args.N_train
        self.epsilon = args.e_train
        # R1, R2, W1, L1, W2, L2
        self.rt_ = torch.nn.Parameter(torch.tensor([args.S_R1n, args.S_R2n, args.S_W1n, args.S_L1n, args.S_W2n, args.S_L2n]).to(args.DEVICE), requires_grad=True)
        # model
        package = torch.load('./utils/sigmoid_param.package')
        self.eta_estimator = package['eta_estimator'].to(self.DEVICE)
        self.eta_estimator.train(False)
        for name, param in self.eta_estimator.named_parameters():
            param.requires_grad = False
        self.X_max = package['X_max'].to(self.DEVICE)
        self.X_min = package['X_min'].to(self.DEVICE)
        self.Y_max = package['Y_max'].to(self.DEVICE)
        self.Y_min = package['Y_min'].to(self.DEVICE)
        # load power model
        package = torch.load('./utils/sigmoid_power.package')
        self.power_estimator = package['power_estimator'].to(self.DEVICE)
        for name, param in self.power_estimator.named_parameters():
            param.requires_grad = False
        self.power_estimator.train(False)
        self.pow_X_max = package['X_max'].to(self.DEVICE)
        self.pow_X_min = package['X_min'].to(self.DEVICE)
        self.pow_Y_max = package['Y_max'].to(self.DEVICE)
        self.pow_Y_min = package['Y_min'].to(self.DEVICE)

    @property
    def DEVICE(self):
        return self.args.DEVICE

    @property
    def RT(self):
        # keep values in (0,1)
        rt_temp = torch.sigmoid(self.rt_)
        RTn = torch.zeros([8]).to(self.DEVICE)
        RTn[:6] = rt_temp
        # denormalization
        RT = RTn * (self.X_max - self.X_min) + self.X_min
        return RT

    @property
    def RT_noisy(self):
        RT_mean = self.RT.repeat(self.N, 1)
        noise = ((torch.rand(RT_mean.shape) * 2. - 1.) * self.epsilon) + 1.
        RT_variation = RT_mean * noise
        return RT_variation

    @property
    def RTn_extend(self):
        RT_extend = torch.stack([self.RT_noisy[:,0], self.RT_noisy[:,1], self.RT_noisy[:,2], self.RT_noisy[:,3],
                                 self.RT_noisy[:,4], self.RT_noisy[:,5], self.RT_noisy[:,2]/self.RT_noisy[:,3], self.RT_noisy[:,4]/self.RT_noisy[:,5]], dim=1)
        return (RT_extend - self.X_min) / (self.X_max - self.X_min)    

    @property
    def eta(self):
        # calculate eta
        eta_n = self.eta_estimator(self.RTn_extend)
        eta = eta_n * (self.Y_max - self.Y_min) + self.Y_min
        return eta

    @property
    def power(self):
        # calculate power
        power_n = self.power_estimator(self.RTn_extend)
        power = power_n * (self.pow_Y_max - self.pow_Y_min) + self.pow_Y_min
        return power.mean()

    def forward(self, z):
        eta_noisy = self.eta
        a = torch.zeros_like(z)
        for i in range(self.N):
            a[i,:,:] = self.eta[i,0] + self.eta[i,1] * torch.sigmoid((z[i,:,:] - self.eta[i,2]) * self.eta[i,3])
        return a
    
    def UpdateArgs(self, args):
        self.args = args