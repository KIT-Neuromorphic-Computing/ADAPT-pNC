import os
import torch
from torch.utils.data import Dataset, DataLoader
import sys
sys.path.append(os.path.join(os.getcwd()))
sys.path.append('../')


class dataset(Dataset):
    def __init__(self, dataset, DataPath, datapath,
                 N_time,
                 R_test,
                 R_train,
                 InputNoise,
                 DEVICE,
                 mode='train', temporal=False):

        if datapath is None:
            datapath = os.path.join(DataPath, dataset)
        else:
            datapath = os.path.join(datapath, dataset)

        data = torch.load(datapath)

        self.N_time = N_time
        self.R_test = R_test
        self.R_train = R_train
        self.InputNoise = InputNoise
        self.DEVICE = DEVICE

        X_train = data['X_train']
        y_train = data['y_train']
        X_valid = data['X_valid']
        y_valid = data['y_valid']
        X_test = data['X_test']
        y_test = data['y_test']

        if temporal:
            dimension = X_train.shape
            if len(dimension) == 3:
                pass
            elif len(dimension) == 2:
                self.N_time = N_time
                X_train = X_train.repeat(N_time, 1, 1).permute(1, 2, 0)
                X_valid = X_valid.repeat(N_time, 1, 1).permute(1, 2, 0)
                X_test = X_test.repeat(N_time, 1, 1).permute(1, 2, 0)

        if mode == 'train':
            self.X_train = torch.cat(
                [X_train for _ in range(R_train)], dim=0).to(DEVICE)
            self.y_train = torch.cat(
                [y_train for _ in range(R_train)], dim=0).to(DEVICE)
        elif mode == 'valid':
            self.X_valid = torch.cat(
                [X_valid for _ in range(R_train)], dim=0).to(DEVICE)
            self.y_valid = torch.cat(
                [y_valid for _ in range(R_train)], dim=0).to(DEVICE)
        elif mode == 'test':
            self.X_test = torch.cat(
                [X_test for _ in range(R_test)], dim=0).to(DEVICE)
            self.y_test = torch.cat(
                [y_test for _ in range(R_test)], dim=0).to(DEVICE)

        self.data_name = data['name']
        self.N_class = data['n_class']
        self.N_feature = data['n_feature']
        self.N_train = X_train.shape[0]
        self.N_valid = X_valid.shape[0]
        self.N_test = X_test.shape[0]
        if temporal:
            self.N_time = X_train.shape[2]

        self.mode = mode

    @property
    def noisy_X_train(self):
        noise = torch.randn(self.X_train.shape) * self.InputNoise + 1.
        return self.X_train * noise.to(self.DEVICE)

    @property
    def noisy_X_valid(self):
        noise = torch.randn(self.X_valid.shape) * self.InputNoise + 1.
        return self.X_valid * noise.to(self.DEVICE)

    @property
    def noisy_X_test(self):
        noise = torch.randn(self.X_test.shape) * self.IN_test + 1.
        return self.X_test * noise.to(self.DEVICE)

    def __getitem__(self, index):
        if self.mode == 'train':
            x = self.noisy_X_train[index, :]
            y = self.y_train[index]
        elif self.mode == 'valid':
            x = self.noisy_X_valid[index, :]
            y = self.y_valid[index]
        elif self.mode == 'test':
            x = self.noisy_X_test[index, :]
            y = self.y_test[index]
        return x, y

    def __len__(self):
        if self.mode == 'train':
            return self.N_train * self.R_train
        elif self.mode == 'valid':
            return self.N_valid * self.R_train
        elif self.mode == 'test':
            return self.N_test * self.R_test


def GetDataLoader(DATASET,
                  DataPath,
                  datapath,
                  N_time,
                  R_test,
                  R_train,
                  InputNoise,
                  DEVICE,
                  task,
                  mode="train", path=None):
    normal_datasets = ['Dataset_acuteinflammation.ds',
                       'Dataset_balancescale.ds',
                       'Dataset_breastcancerwisc.ds',
                       'Dataset_cardiotocography3clases.ds',
                       'Dataset_energyy1.ds',
                       'Dataset_energyy2.ds',
                       'Dataset_iris.ds',
                       'Dataset_mammographic.ds',
                       'Dataset_pendigits.ds',
                       'Dataset_seeds.ds',
                       'Dataset_tictactoe.ds',
                       'Dataset_vertebralcolumn2clases.ds',
                       'Dataset_vertebralcolumn3clases.ds']

    temporized_datasets = normal_datasets

    temporal_datasets = ['Dataset_cbf.tsds',
                         'Dataset_distalphalanxtw.tsds',
                         'Dataset_freezerregulartrain.tsds',
                         'Dataset_freezersmalltrain.tsds',
                         'Dataset_gunpointagespan.tsds',
                         'Dataset_gunpointmaleversusfemale.tsds',
                         'Dataset_gunpointoldversusyoung.tsds',
                         'Dataset_middlephalanxoutlineagegroup.tsds',
                         'Dataset_mixedshapesregulartrain.tsds',
                         'Dataset_powercons.tsds',
                         'Dataset_proximalphalanxoutlinecorrect.tsds',
                         'Dataset_selfregulationscp2.tsds',
                         'Dataset_slope.tsds',
                         'Dataset_smoothsubspace.tsds',
                         'Dataset_symbols.tsds']

    split_manufacture = ['Dataset_acuteinflammation.ds',
                         'Dataset_acutenephritis.ds',
                         'Dataset_balancescale.ds',
                         'Dataset_blood.ds',
                         'Dataset_breastcancer.ds',
                         'Dataset_breastcancerwisc.ds',
                         'Dataset_breasttissue.ds',
                         'Dataset_ecoli.ds',
                         'Dataset_energyy1.ds',
                         'Dataset_energyy2.ds',
                         'Dataset_fertility.ds',
                         'Dataset_glass.ds',
                         'Dataset_habermansurvival.ds',
                         'Dataset_hayesroth.ds',
                         'Dataset_ilpdindianliver.ds',
                         'Dataset_iris.ds',
                         'Dataset_mammographic.ds',
                         'Dataset_monks1.ds',
                         'Dataset_monks2.ds',
                         'Dataset_monks3.ds',
                         'Dataset_pima.ds',
                         'Dataset_pittsburgbridgesMATERIAL.ds',
                         'Dataset_pittsburgbridgesSPAN.ds',
                         'Dataset_pittsburgbridgesTORD.ds',
                         'Dataset_pittsburgbridgesTYPE.ds',
                         'Dataset_seeds.ds',
                         'Dataset_teaching.ds',
                         'Dataset_tictactoe.ds',
                         'Dataset_vertebralcolumn2clases.ds',
                         'Dataset_vertebralcolumn3clases.ds']

    normal_datasets.sort()
    temporized_datasets.sort()
    split_manufacture.sort()

    if path is None:
        path = DataPath

    datasets = os.listdir(path)
    datasets = [f for f in datasets if (
        f.startswith('Dataset') and f.endswith('.ds'))]
    datasets.sort()

    if task == 'normal':
        None

    elif task == 'temporal':
        dataset_name = temporal_datasets[DATASET]
        # data
        trainset = dataset(dataset_name,
                           DataPath,
                           datapath,
                           N_time,
                           R_test,
                           R_train,
                           InputNoise,
                           DEVICE,
                           mode='train', temporal=True)
        validset = dataset(dataset_name,
                           DataPath,
                           datapath,
                           N_time,
                           R_test,
                           R_train,
                           InputNoise,
                           DEVICE,
                           mode='valid', temporal=True)
        testset = dataset(dataset_name,
                          DataPath,
                          datapath,
                          N_time,
                          R_test,
                          R_train,
                          InputNoise,
                          DEVICE,
                          mode='test', temporal=True)

        # batch
        train_loader = DataLoader(trainset, batch_size=len(trainset))
        valid_loader = DataLoader(validset, batch_size=len(validset))
        test_loader = DataLoader(testset,  batch_size=len(testset))

        # message
        info = {}
        info['dataname'] = trainset.data_name
        info['N_feature'] = trainset.N_feature
        info['N_class'] = trainset.N_class
        info['N_train'] = len(trainset)
        info['N_valid'] = len(validset)
        info['N_test'] = len(testset)
        info['N_time'] = trainset.N_time

        if mode == 'train':
            return train_loader, info
        elif mode == 'valid':
            return valid_loader, info
        elif mode == 'test':
            return test_loader, info
