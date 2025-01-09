import os
import torch
from torch.utils.data import Dataset, DataLoader
import sys
import tsaug
sys.path.append(os.path.join(os.getcwd()))
sys.path.append('../')


class dataset(Dataset):
    def __init__(self, dataset, max_drift, scale, n_speed_change, max_speed_ratio,
                 args, datapath, mode='train', temporal=False, augment=False,):
        self.args = args
        self.augment = augment
        self.max_drift = max_drift
        self.scale = scale
        self.n_speed_change = n_speed_change
        self.max_speed_ratio = max_speed_ratio

        if datapath is None:
            datapath = os.path.join(args.DataPath, dataset)
        else:
            datapath = os.path.join(datapath, dataset)

        data = torch.load(datapath)

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
                self.N_time = args.N_time
                X_train = X_train.repeat(args.N_time, 1, 1).permute(1, 2, 0)
                X_valid = X_valid.repeat(args.N_time, 1, 1).permute(1, 2, 0)
                X_test = X_test.repeat(args.N_time, 1, 1).permute(1, 2, 0)

        if mode == 'train':
            self.X_train = torch.cat(
                [X_train for _ in range(args.R_train)], dim=0).to(args.DEVICE)
            self.y_train = torch.cat(
                [y_train for _ in range(args.R_train)], dim=0).to(args.DEVICE)
        elif mode == 'valid':
            self.X_valid = torch.cat(
                [X_valid for _ in range(args.R_train)], dim=0).to(args.DEVICE)
            self.y_valid = torch.cat(
                [y_valid for _ in range(args.R_train)], dim=0).to(args.DEVICE)
        elif mode == 'test':
            self.X_test = torch.cat(
                [X_test for _ in range(args.R_test)], dim=0).to(args.DEVICE)
            self.y_test = torch.cat(
                [y_test for _ in range(args.R_test)], dim=0).to(args.DEVICE)

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
        noise = torch.randn(self.X_train.shape) * self.args.InputNoise + 1.
        return self.X_train * noise.to(self.args.DEVICE)

    @property
    def noisy_X_valid(self):
        noise = torch.randn(self.X_valid.shape) * self.args.InputNoise + 1.
        return self.X_valid * noise.to(self.args.DEVICE)

    @property
    def noisy_X_test(self):
        noise = torch.randn(self.X_test.shape) * self.args.IN_test + 1.
        return self.X_test * noise.to(self.args.DEVICE)

    @property
    def augment_X_train(self):
        time_series = torch.permute(self.X_train, (0, 2, 1))
        # Convert the time series to a PyTorch tensor
        # Create the Drift augmenter
        augmenter = (
            # max_drift = 0.2
            tsaug.Drift(max_drift=self.max_drift)
            # scale=0.02
            + tsaug.AddNoise(scale=self.scale)
            # n_speed_change = 5, max_speed_ratio=3
            + tsaug.TimeWarp(n_speed_change=self.n_speed_change,
                             max_speed_ratio=self.max_speed_ratio)
            # + tsaug.Convolve(window="flattop", size=2)
            # + tsaug.Pool(kind='ave', size=2)
        )
        # Convert tensor to numpy
        # Apply the augmentation
        augmented_np_array = augmenter.augment(time_series.numpy())
        # Convert back to tensor
        return torch.permute(torch.tensor(
            augmented_np_array, dtype=torch.float32), (0, 2, 1))

    @property
    def augment_X_valid(self):
        time_series = torch.permute(self.X_valid, (0, 2, 1))
        # Convert the time series to a PyTorch tensor
        # Create the Drift augmenter
        augmenter = (
            tsaug.Drift(max_drift=0.2)
            + tsaug.AddNoise(scale=0.02)
            + tsaug.TimeWarp(n_speed_change=5, max_speed_ratio=3)
            + tsaug.Convolve(window="flattop", size=2)
            + tsaug.Pool(kind='ave', size=2)
        )
        # Convert tensor to numpy
        # Apply the augmentation
        augmented_np_array = augmenter.augment(time_series.numpy())
        # Convert back to tensor
        return torch.permute(torch.tensor(
            augmented_np_array, dtype=torch.float32), (0, 2, 1))

    @property
    def augment_X_test(self):
        time_series = torch.permute(self.X_test, (0, 2, 1))
        # Convert the time series to a PyTorch tensor
        # Create the Drift augmenter
        augmenter = (
            tsaug.Drift(max_drift=0.2)
            + tsaug.AddNoise(scale=0.02)
            + tsaug.TimeWarp(n_speed_change=5, max_speed_ratio=3)
            + tsaug.Convolve(window="flattop", size=2)
            + tsaug.Pool(kind='ave', size=2)
        )
        # Convert tensor to numpy
        # Apply the augmentation
        augmented_np_array = augmenter.augment(time_series.numpy())
        # Convert back to tensor
        return torch.permute(torch.tensor(
            augmented_np_array, dtype=torch.float32), (0, 2, 1))

    def __getitem__(self, index):
        if self.mode == 'train':
            if self.augment == True:
                x = self.augment_X_train[index, :]
            else:
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
            return self.N_train * self.args.R_train
        elif self.mode == 'valid':
            return self.N_valid * self.args.R_train
        elif self.mode == 'test':
            return self.N_test * self.args.R_test


def GetDataLoader(args, mode, max_drift=0.2, scale=0.02, n_speed_change=5, max_speed_ratio=3, path=None):
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
        path = args.DataPath

    datasets = os.listdir(path)
    datasets = [f for f in datasets if (
        f.startswith('Dataset') and f.endswith('.ds'))]
    datasets.sort()

    if args.task == 'normal':
        dataname = normal_datasets[args.DATASET]
        # data
        trainset = dataset(dataname, args, path, mode='train', augment=False)
        validset = dataset(dataname, args, path, mode='valid')
        testset = dataset(dataname, args, path, mode='test')

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

        if mode == 'train':
            return train_loader, info
        elif mode == 'valid':
            return valid_loader, info
        elif mode == 'test':
            return test_loader, info

    elif args.task == 'split':
        train_loaders = []
        valid_loaders = []
        test_loaders = []
        infos = []
        for dataname in split_manufacture:
            # data
            trainset = dataset(dataname, args, path, mode='train')
            validset = dataset(dataname, args, path, mode='valid')
            testset = dataset(dataname, args, path, mode='test')
            # batch
            train_loaders.append(DataLoader(
                trainset, batch_size=len(trainset)))
            valid_loaders.append(DataLoader(
                validset, batch_size=len(validset)))
            test_loaders.append(DataLoader(testset,  batch_size=len(testset)))
            # message
            info = {}
            info['dataname'] = trainset.data_name
            info['N_feature'] = trainset.N_feature
            info['N_class'] = trainset.N_class
            info['N_train'] = len(trainset)
            info['N_valid'] = len(validset)
            info['N_test'] = len(testset)
            infos.append(info)

        if mode == 'train':
            return train_loaders, infos
        elif mode == 'valid':
            return valid_loaders, infos
        elif mode == 'test':
            return test_loaders, infos

    elif args.task == 'temporized':
        dataname = temporized_datasets[args.DATASET]
        # data
        trainset = dataset(dataname, args, path, mode='train', temporal=True)
        validset = dataset(dataname, args, path, mode='valid', temporal=True)
        testset = dataset(dataname, args, path, mode='test', temporal=True)

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

    elif args.task == 'temporal':
        dataset_name = temporal_datasets[args.DATASET]
        # data
        trainset = dataset(dataset_name, args=args, datapath=path,
                           mode='train', temporal=True, augment=False,
                           max_drift=max_drift, scale=scale,
                           n_speed_change=n_speed_change, max_speed_ratio=max_speed_ratio)

        validset = dataset(dataset_name, args=args, datapath=path,
                           mode='valid', temporal=True, augment=False,
                           max_drift=max_drift, scale=scale,
                           n_speed_change=n_speed_change, max_speed_ratio=max_speed_ratio)

        testset = dataset(dataset_name, args=args, datapath=path,
                          mode='test', temporal=True, augment=False,
                          max_drift=max_drift, scale=scale,
                          n_speed_change=n_speed_change, max_speed_ratio=max_speed_ratio)

        # message
        info = {}
        info['dataname'] = validset.data_name
        info['N_feature'] = validset.N_feature
        info['N_class'] = validset.N_class
        info['N_time'] = validset.N_time

        if args.augment == True:
            augmented_trainset = dataset(dataset_name, args=args, datapath=path,
                                         mode='train', temporal=True, augment=True,
                                         max_drift=max_drift, scale=scale,
                                         n_speed_change=n_speed_change, max_speed_ratio=max_speed_ratio)
            trainset = torch.utils.data.ConcatDataset(
                [trainset, augmented_trainset])

            augmented_validset = dataset(dataset_name, args=args, datapath=path,
                                         mode='valid', temporal=True, augment=True,
                                         max_drift=max_drift, scale=scale,
                                         n_speed_change=n_speed_change, max_speed_ratio=max_speed_ratio)
            validset = torch.utils.data.ConcatDataset(
                [validset, augmented_validset])

            augmented_testset = dataset(dataset_name, args=args, datapath=path,
                                        mode='test', temporal=True, augment=True,
                                        max_drift=max_drift, scale=scale,
                                        n_speed_change=n_speed_change, max_speed_ratio=max_speed_ratio)
            testset = torch.utils.data.ConcatDataset(
                [testset, augmented_testset])

        # batch
        train_loader = DataLoader(trainset, batch_size=len(trainset))
        valid_loader = DataLoader(validset, batch_size=len(validset))
        test_loader = DataLoader(testset,  batch_size=len(testset))

        # message
        # info = {}
        # info['dataname'] = validset.data_name
        # info['N_feature'] = validset.N_feature
        # info['N_class'] = validset.N_class
        info['N_train'] = len(trainset)
        info['N_valid'] = len(validset)
        info['N_test'] = len(testset)
        # info['N_time'] = validset.N_time

        if mode == 'train':
            return train_loader, info
        elif mode == 'valid':
            return valid_loader, info
        elif mode == 'test':
            return test_loader, info
