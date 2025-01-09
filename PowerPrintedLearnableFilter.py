import torch
from pLNC import *

# ===============================================================================
# =========================== Single Learnable Filter ===========================
# ===============================================================================


class LearnableFilter(torch.nn.Module):
    def __init__(self, args, beta_init=None, random_state=True):
        super().__init__()
        self.args = args
        # variation
        self.N = args.N_train
        self.epsilon = args.e_train
        # learnable hardware
        # time discretization for simulation
        self.dT = torch.tensor(0.1).to(self.DEVICE)
        self.Rmin = torch.tensor(1e5).to(self.DEVICE)       # 100kOhm
        self.Rmax = torch.tensor(1e7).to(self.DEVICE)       # 10MOhm
        self.Cmin = torch.tensor(1e-7).to(self.DEVICE)      # 100nF
        self.Cmax = torch.tensor(1e-4).to(self.DEVICE)      # 100uF
        if beta_init is None:
            self.R_ = torch.nn.Parameter(
                torch.rand([])*10.-10., requires_grad=True)
            self.C_ = torch.nn.Parameter(
                torch.rand([])*10.-10., requires_grad=True)
        else:
            self.C_ = torch.nn.Parameter(
                torch.tensor(-10.), requires_grad=True)
            R_true = beta_init * self.dT / (1 - beta_init) / self.C.mean()
            Rn = (R_true - self.Rmin) / (self.Rmax - self.Rmin)
            self.R_ = torch.nn.Parameter(
                torch.log(Rn / (1 - Rn)), requires_grad=True)
        # removing dependency on the initial state
        self.random_state = random_state

    @property
    def DEVICE(self):
        return self.args.DEVICE

    @property
    def R(self):
        R_true = torch.sigmoid(self.R_) * (self.Rmax - self.Rmin) + self.Rmin
        mean = R_true.repeat(self.N, 1)
        noise = (torch.rand(mean.shape).to(self.DEVICE)
                 * 2. - 1.) * self.epsilon + 1.
        return mean * noise

    @property
    def C(self):
        C_true = torch.sigmoid(self.C_) * (self.Cmax - self.Cmin) + self.Cmin
        mean = C_true.repeat(self.N, 1)
        noise = (torch.rand(mean.shape).to(self.DEVICE)
                 * 2. - 1.) * self.epsilon + 1.
        return mean * noise

    @property
    def beta(self):
        mu = torch.rand(self.N, 1).to(self.DEVICE) * \
            self.args.coupling_factor + 1.
        R = self.R
        C = self.C
        beta = mu * R * C / (mu * R * C + self.dT)
        return beta

    def StateUpdate(self, x):
        return self.beta * self.memory + (1 - self.beta) * x

    def SingleStepForward(self, x):
        self.memory = self.StateUpdate(x)
        return self.memory

    def forward(self, x):
        _, N_batch, T = x.shape
        if self.random_state:
            self.memory = torch.rand(self.N, N_batch).to(self.DEVICE)
        else:
            self.memory = torch.zeros(self.N, N_batch).to(self.DEVICE)
        memories = [self.memory]
        for t in range(T):
            memory = self.SingleStepForward(x[:, :, t])
            memories.append(memory)
        memories.pop()
        return torch.stack(memories, dim=2)

    def UpdateArgs(self, args):
        self.args = args

    def UpdateVariation(self, N, epsilon):
        self.N = N
        self.epsilon = epsilon


class SecOLearnableFilter(torch.nn.Module):
    def __init__(self, args, beta_init=None, random_state=True):
        super().__init__()
        self.args = args

        self.LearnableFilters = torch.nn.ModuleList()
        for n in range(2):
            self.LearnableFilters.append(
                LearnableFilter(args, beta_init[n], random_state))

    @property
    def DEVICE(self):
        return self.args.DEVICE

    def forward(self, x):
        for filter in self.LearnableFilters:
            x = filter(x)
        return x

    def UpdateArgs(self, args):
        self.args = args

    def UpdateVariation(self, N, epsilon):
        self.N = N
        self.epsilon = epsilon
        for filter in self.LearnableFilters:
            filter.UpdateVariation(N, epsilon)

# ===============================================================================
# ====================== A Group of Filters for One Input =======================
# ===============================================================================


class FilterGroup(torch.nn.Module):
    def __init__(self, args, N_filters, N_feature, random_state=True):
        super().__init__()
        self.args = args
        self.N = args.N_train
        self.epsilon = args.e_train

        self.FilterGroup = torch.nn.ModuleList()
        betas1 = torch.linspace(0.1, 0.9, N_filters)
        betas2 = torch.linspace(0.1, 0.9, N_filters)
        for n in range(N_feature):
            self.FilterGroup.append(
                SecOLearnableFilter(args, [betas1[n], betas2[-(n+1)]], random_state))

    @property
    def DEVICE(self):
        return self.args.DEVICE

    def forward(self, x):
        memories = [x]
        for filter in self.FilterGroup:
            memory = filter(x)
            memories.append(memory)
        return torch.stack(memories, dim=2)

    def UpdateArgs(self, args):
        self.args = args

    def UpdateVariation(self, N, epsilon):
        self.N = N
        self.epsilon = epsilon
        for filter in self.FilterGroup:
            filter.UpdateVariation(N, epsilon)


# ===============================================================================
# ================== Filter Layers consist of Multiple Groups ===================
# ===============================================================================

class FilterLayer(torch.nn.Module):
    def __init__(self, args, N_channel, N_feature, random_state=True):
        super().__init__()
        self.args = args

        self.FilterGroups = torch.nn.ModuleList()
        for n in range(N_channel):
            self.FilterGroups.append(
                FilterGroup(args, N_channel, N_feature, random_state))

    @property
    def DEVICE(self):
        return self.args.DEVICE

    def forward(self, x):
        memories = []
        for c in range(x.shape[2]):
            memory = self.FilterGroups[c](x[:, :, c, :])
            memories.append(memory)
        return torch.cat(memories, dim=2)

    def UpdateArgs(self, args):
        self.args = args
        for g in self.FilterGroups:
            g.UpdateArgs(args)

    def UpdateVariation(self, N, epsilon):
        for g in self.FilterGroups:
            g.UpdateVariation(N, epsilon)

# ================================================================================================================================================
# ===============================================================  Printed Layer  ================================================================
# ================================================================================================================================================


class pLayer(torch.nn.Module):
    def __init__(self, n_in, n_out, args, ACT, INV):
        super().__init__()
        self.args = args
        self.N = args.N_train
        self.epsilon = args.e_train
        # define nonlinear circuits
        self.ACT = ACT
        self.INV = INV
        # initialize conductances for weights
        theta_temp = torch.nn.init.xavier_uniform_(
            torch.empty([n_in + 2, n_out]))
        theta = (theta_temp - theta_temp.min()) / \
            (theta_temp.max() - theta_temp.min())
        # theta = torch.rand([n_in + 2, n_out])/100. + args.gmin

        eta_2 = ACT.eta.mean(0)[2].detach().item()
        theta[-1, :] = theta[-1, :] + args.gmax
        theta[-2, :] = eta_2 / (1.-eta_2) * \
            (torch.sum(theta[:-2, :], axis=0)+theta[-1, :])
        self.theta_ = torch.nn.Parameter(theta, requires_grad=True)

    @property
    def device(self):
        return self.args.DEVICE

    @property
    def theta(self):
        self.theta_.data.clamp_(-self.args.gmax, self.args.gmax)
        theta_temp = self.theta_.clone()
        theta_temp[theta_temp.abs() < self.args.gmin] = 0.
        return theta_temp.detach() + self.theta_ - self.theta_.detach()

    @property
    def theta_noisy(self):
        mean = (self.theta.repeat(self.N, 1, 1)).to(self.device)
        nosie = (((torch.rand(mean.shape) * 2.) - 1.)
                 * self.epsilon + 1.).to(self.device)
        return mean * nosie

    @property
    def W(self):
        # to deal with case that the whole colume of theta is 0
        G = torch.sum(self.theta_noisy.abs(), axis=1, keepdim=True)
        W = self.theta_noisy.abs() / (G + 1e-10)
        return W.to(self.device)

    def MAC(self, a):
        # a.shape: (num_batch, B, D, T)
        # 0 and positive thetas are corresponding to no negative weight circuit
        positive = self.theta_noisy.clone().to(self.device)
        positive[positive >= 0] = 1.
        positive[positive < 0] = 0.
        negative = 1. - positive

        a_reshaped = a.permute(0, 1, 3, 2)
        ones_tensor = torch.ones(
            [a_reshaped.shape[0], a_reshaped.shape[1], a_reshaped.shape[2], 1]).to(self.device)
        zeros_tensor = torch.zeros_like(ones_tensor).to(self.device)
        a_extend = torch.cat([a_reshaped, ones_tensor, zeros_tensor], dim=3)

        batch_size = a_extend.shape[1]

        a_neg = self.INV(a_extend)
        a_neg[:, :, :, -1] = torch.tensor(0.).to(self.device)

        positive_w = (self.W * positive).unsqueeze(1).expand(-1,
                                                             batch_size, -1, -1)
        negative_w = (self.W * negative).unsqueeze(1).expand(-1,
                                                             batch_size, -1, -1)

        z = torch.matmul(a_extend, positive_w) + \
            torch.matmul(a_neg, negative_w)

        return z

    def MAC_power(self, x, y):
        # dimensions of x: (num_batch, B, D, T)
        # the features are in last dimension
        x = x.permute(0, 1, 3, 2)
        x_extend = torch.cat([x,
                              torch.ones([x.shape[0], x.shape[1], x.shape[2], 1]).to(
                                  self.device),
                              torch.zeros([x.shape[0], x.shape[1], x.shape[2], 1]).to(self.device)], dim=3)

        x_neg = self.INV(x_extend)
        x_neg[:, :, :, -1] = 0.

        F = x_extend.shape[0]
        V = x_extend.shape[1]
        E = x_extend.shape[2]
        M = x_extend.shape[3]
        N = y.shape[3]

        positive = self.theta_noisy.clone().detach().to(self.device)
        positive[positive >= 0] = 1.
        positive[positive < 0] = 0.
        negative = 1. - positive

        Power = torch.tensor(0.).to(self.device)
        for f in range(F):
            for v in range(V):
                for m in range(M):
                    for n in range(N):
                        Power += self.g_tilde[m, n] * ((x_extend[f, v, :, m]*positive[f, v, m, n] +
                                                       x_neg[f, v, :, m]*negative[f, v, m, n])-y[f, v, :, n]).pow(2.).sum()
        Power = Power / E / V / F
        return Power

    # @property
    # def soft_num_theta(self):
    #     # forward pass: number of theta
    #     nonzero = self.theta.clone().detach().abs()
    #     nonzero[nonzero > 0] = 1.
    #     N_theta = nonzero.sum()
    #     # backward pass: pvalue of the minimal negative weights
    #     soft_count = torch.sigmoid(self.theta.abs())
    #     soft_count = soft_count * nonzero
    #     soft_count = soft_count.sum()
    #     return N_theta.detach() + soft_count - soft_count.detach()

    # @property
    # def soft_num_act(self):
    #     # forward pass: number of act
    #     nonzero = self.theta.clone().detach().abs()[:-2, :]
    #     nonzero[nonzero > 0] = 1.
    #     N_act = nonzero.max(0)[0].sum()
    #     # backward pass: pvalue of the minimal negative weights
    #     soft_count = torch.sigmoid(self.theta.abs()[:-2, :])
    #     soft_count = soft_count * nonzero
    #     soft_count = soft_count.max(0)[0].sum()
    #     return N_act.detach() + soft_count - soft_count.detach()

    # @property
    # def soft_num_neg(self):
    #     # forward pass: number of negative weights
    #     positive = self.theta.clone().detach()[:-2, :]
    #     positive[positive >= 0] = 1.
    #     positive[positive < 0] = 0.
    #     negative = 1. - positive
    #     N_neg = negative.max(1)[0].sum()
    #     # backward pass: pvalue of the minimal negative weights
    #     soft_count = 1 - torch.sigmoid(self.theta[:-2, :])
    #     soft_count = soft_count * negative
    #     soft_count = soft_count.max(1)[0].sum()
    #     return N_neg.detach() + soft_count - soft_count.detach()

    # @property
    # def NEG_power(self):
    #     # convert uW to W: *1e-6
    #     return self.INV.power * self.soft_count_neg

    # @property
    # def ACT_power(self):
    #     # convert uW to W
    #     return self.ACT.power * self.soft_count_act

    def forward(self, a_previous):
        mac = self.MAC(a_previous)
        result = self.ACT(mac)
        a_new = result.permute(0, 1, 3, 2)
        return a_new

    def UpdateArgs(self, args):
        self.args = args

    def UpdateVariation(self, N, epsilon):
        self.N = N
        self.epsilon = epsilon
        self.INV.N = N
        self.INV.epsilon = epsilon
        self.ACT.N = N
        self.ACT.epsilon = epsilon


# ================================================================================================================================================
# ==========================================================  Printed Recurrent Layer ============================================================
# ================================================================================================================================================

class pRecurrentLayer(torch.nn.Module):
    def __init__(self, args, N_in, N_out, ACT, INV, N_feature=None):
        super().__init__()
        self.args = args
        N_feature = N_out if N_feature == None else N_feature
        self.model = torch.nn.Sequential()
        self.model.add_module('0_MAC', pLayer(N_in, N_out, args, ACT, INV))
        self.model.add_module('1_LF', FilterLayer(args, N_out, N_feature))
        self.model.add_module('2_MAC', pLayer(
            N_out * (N_feature + 1), N_out, args, ACT, INV))

    @property
    def device(self):
        return self.args.DEVICE

    def forward(self, x):
        return self.model(x).to(self.device)

    # @property
    # def power_mac(self):
    #     power_mac = torch.tensor([0.]).to(self.device)
    #     for l in self.model:
    #         if hasattr(l, 'mac_power'):
    #             power_mac += l.mac_power
    #     return power_mac

    # @property
    # def power_neg(self):
    #     # convert uW to W
    #     return self.inv.power * 1e-6 * self.soft_count_neg

    # @property
    # def power_act(self):
    #     # print('check power_act ', self.act.power, self.soft_count_act)
    #     # convert uW to W
    #     return self.act.power * 1e-6 * self.soft_count_act

    def UpdateArgs(self, args):
        self.args = args
        self.model[0].UpdateArgs(args)
        self.model[1].UpdateArgs(args)
        self.model[2].UpdateArgs(args)

    def UpdateVariation(self, N, epsilon):
        self.model[0].UpdateVariation(N, epsilon)
        self.model[1].UpdateVariation(N, epsilon)
        self.model[2].UpdateVariation(N, epsilon)


# ===============================================================================
# ======================== Printed Neural Network ===============================
# ===============================================================================

class PrintedNeuralNetwork(torch.nn.Module):
    def __init__(self, args, N_channel, N_class, N_layer, N_feature, random_state=True):
        super().__init__()
        self.args = args
        self.N = args.N_train
        self.epsilon = args.e_train

        self.ACT = TanhRT(args)
        self.INV = InvRT(args)

        # create pNN with learnable filters and weighted-sum
        self.model = torch.nn.Sequential()
        self.model.add_module('0_pLayer', pRecurrentLayer(
            self.args, N_channel, N_class, self.ACT, self.INV, N_feature))
        for i in range(N_layer-1):
            self.model.add_module(str(
                i+1)+'_pLayer', pRecurrentLayer(self.args, N_class, N_class, self.ACT, self.INV))

    @property
    def DEVICE(self):
        return self.args.DEVICE

    def forward(self, x):
        x_extend = (x.repeat(self.N, 1, 1, 1)).to(self.DEVICE)
        return self.model(x_extend)

    def UpdateArgs(self, args):
        self.args = args
        for layer in self.model:
            if hasattr(layer, 'UpdateArgs'):
                layer.UpdateArgs(args)

    def UpdateVariation(self, N, epsilon):
        for layer in self.model:
            if hasattr(layer, 'UpdateVariation'):
                layer.UpdateVariation(N, epsilon)
        self.N = N
        self.epsilon = epsilon
        self.ACT.N = N
        self.ACT.epsilon = epsilon
        self.INV.N = N
        self.INV.epsilon = epsilon

    def GetParam(self):
        weights = [p for name, p in self.named_parameters() if name.endswith(
            'theta_') or name.endswith('R_') or name.endswith('C_')]
        return weights

# ===============================================================================
# ============================= Loss Functin ====================================
# ===============================================================================


class LossFN(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

    def standard(self, prediction, label):
        label = label.reshape(-1, 1)
        fy = prediction.gather(1, label).reshape(-1, 1)
        fny = prediction.clone()
        fny = fny.scatter_(1, label, -10 ** 10)
        fnym = torch.max(fny, axis=1).values.reshape(-1, 1)
        l = torch.max(self.args.m + self.args.T - fy, torch.tensor(0)
                      ) + torch.max(self.args.m + fnym, torch.tensor(0))
        L = torch.mean(l)
        return L

    def celoss(self, prediction, label):
        lossfn = torch.nn.CrossEntropyLoss().to(self.args.DEVICE)
        return lossfn(prediction.to(self.args.DEVICE), label.to(self.args.DEVICE))

    def forward(self, prediction, label):
        if self.args.loss == 'pnnloss':
            return self.standard(prediction, label)
        elif self.args.loss == 'celoss':
            return self.celoss(prediction, label)


class LFLoss(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.loss_fn = LossFN(args)

    def forward(self, model, x, label):
        prediction = model(x)
        L = []
        for variation in range(prediction.shape[0]):
            for step in range(prediction.shape[3]):
                L.append(self.loss_fn(
                    prediction[variation, :, :, step], label))
        return torch.stack(L).mean()
