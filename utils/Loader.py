from torch.utils.data import Dataset
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import sys
import tsaug
sys.path.append(os.path.join(os.getcwd()))
sys.path.append('../')


class CustomDataset(Dataset):
    def __init__(self, dataset, args, datapath=None, mode='train', temporal=False, augment=False,
                 noise_level=0.01, warp_factor=0.1, scaling_factor_range=(0.8, 1.2), crop_size=100):
        super(CustomDataset, self).__init__()
        self.args = args
        self.augment = augment

        # Augmentation hyperparameters
        self.noise_level = noise_level
        self.warp_factor = warp_factor
        self.scaling_factor_range = scaling_factor_range
        self.crop_size = crop_size

        # Load dataset
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

        # Handle temporal data augmentation
        if temporal:
            dimension = X_train.shape
            if len(dimension) == 3:
                pass
            elif len(dimension) == 2:
                self.N_time = args.N_time
                X_train = X_train.repeat(args.N_time, 1, 1).permute(1, 2, 0)
                X_valid = X_valid.repeat(args.N_time, 1, 1).permute(1, 2, 0)
                X_test = X_test.repeat(args.N_time, 1, 1).permute(1, 2, 0)

        # Handle different modes
        if mode == 'train':
            self.X = torch.cat([X_train for _ in range(
                args.R_train)], dim=0).to(args.DEVICE)
            self.y = torch.cat([y_train for _ in range(
                args.R_train)], dim=0).to(args.DEVICE)
        elif mode == 'valid':
            self.X = torch.cat([X_valid for _ in range(
                args.R_train)], dim=0).to(args.DEVICE)
            self.y = torch.cat([y_valid for _ in range(
                args.R_train)], dim=0).to(args.DEVICE)
        elif mode == 'test':
            self.X = torch.cat([X_test for _ in range(
                args.R_test)], dim=0).to(args.DEVICE)
            self.y = torch.cat([y_test for _ in range(
                args.R_test)], dim=0).to(args.DEVICE)

        # Initialize other attributes
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
    def noisy_X(self):
        """Add noise to X data depending on the mode."""
        noise = torch.randn(self.X.shape) * self.args.InputNoise + 1.0
        return self.X * noise.to(self.args.DEVICE)

    @staticmethod
    def jittering(time_series, noise_level):
        """Add Gaussian noise to the time series data.

        Args:
            time_series (torch.Tensor): The input time series data of shape (B, D, T).
            noise_level (float): The standard deviation of the Gaussian noise.

        Returns:
            torch.Tensor: The time series data with added Gaussian noise.
        """
        # Generate Gaussian noise with the same shape and device as the input time series
        noise = torch.normal(mean=0, std=noise_level,
                             size=time_series.shape, device=time_series.device)

        # Add the noise to the time series data
        return time_series + noise

    @staticmethod
    def time_warping(batch_time_series, warp_factor):
        """Apply random time warping to a batch of multivariate time series data."""
        # Assume batch_time_series is of shape (B, D, T)
        B, D, T = batch_time_series.size()

        # Generate the original time points
        time_points = torch.arange(T, device=batch_time_series.device, dtype=torch.float).repeat(
            B, D, 1)  # Shape: (B, D, T)

        # Generate random warping factors for each time series in the batch
        random_factors = torch.tensor(np.random.uniform(
            1 - warp_factor, 1 + warp_factor, size=(B, D, 1)), device=batch_time_series.device)

        # Generate warped time points
        warped_time_points = time_points * random_factors

        # Ensure warped time points are within bounds
        warped_time_points = torch.clamp(warped_time_points, 0, T - 1)

        # Perform linear interpolation manually for each time series and each feature
        # Lower indices for interpolation
        lower_indices = torch.floor(warped_time_points).long()
        # Upper indices for interpolation
        upper_indices = torch.ceil(warped_time_points).long()
        # Ensure indices are within valid range
        upper_indices = torch.clamp(upper_indices, max=T - 1)

        # Calculate interpolation weights
        # Weight for lower indices
        lower_weights = (upper_indices.float() - warped_time_points)
        # Weight for upper indices
        upper_weights = (warped_time_points - lower_indices.float())

        # Gather time series values for interpolation
        batch_time_series_warped = (lower_weights * batch_time_series.gather(2, lower_indices) +
                                    upper_weights * batch_time_series.gather(2, upper_indices))

        return batch_time_series_warped

    @staticmethod
    def magnitude_scaling(time_series, scaling_factor_range):
        """Randomly scale the magnitude of the time series data.

        Args:
            time_series (torch.Tensor): The input time series data of shape (B, D, T).
            scaling_factor_range (tuple): A tuple specifying the range (min, max) 
                                        from which to draw the random scaling factor.

        Returns:
            torch.Tensor: The time series data scaled by the random factor.
        """
        # Generate a random scaling factor from the specified range for each time series in the batch
        B, D, T = time_series.shape  # Get the shape of the time series data
        scaling_factors = torch.tensor(
            np.random.uniform(*scaling_factor_range, size=(B, 1, 1)),
            device=time_series.device, dtype=time_series.dtype
        )

        # Scale each time series by its corresponding random scaling factor
        return time_series * scaling_factors

    @staticmethod
    def random_cropping(batch_time_series, crop_size):
        """Randomly crop the time series data along the time dimension.

        Args:
            batch_time_series (torch.Tensor): The input time series data of shape (B, D, T).
            crop_size (int): The desired number of time steps after cropping.

        Returns:
            torch.Tensor: The cropped time series data of shape (B, D, crop_size).
        """
        B, D, T = batch_time_series.shape

        # Ensure crop size is not greater than the number of time steps
        if crop_size > T:
            raise ValueError(
                "crop_size must be less than or equal to the number of time steps (T).")

        # Generate random start points for cropping for each sample in the batch
        start_indices = np.random.randint(0, T - crop_size + 1, size=B)

        # Initialize an empty tensor for cropped time series
        cropped_series = torch.empty(
            (B, D, crop_size), device=batch_time_series.device, dtype=batch_time_series.dtype)

        # Perform cropping for each time series in the batch
        for i in range(B):
            start = start_indices[i]
            cropped_series[i] = batch_time_series[i,
                                                  :, start:start + crop_size]

        return cropped_series

    @staticmethod
    def frequency_domain_augmentation(time_series, noise_level):
        """Add noise in the frequency domain to the time series data.

        Args:
            time_series (torch.Tensor): The input time series data of shape (B, D, T).
            noise_level (float): The standard deviation of the Gaussian noise to be added in the frequency domain.

        Returns:
            torch.Tensor: The augmented time series data of shape (B, D, T).
        """
        # Compute the FFT of the time series along the time dimension
        fft_series = torch.fft.fft(time_series, dim=-1)

        # Generate Gaussian noise in the frequency domain with the same shape and device as fft_series
        noise = torch.normal(mean=0, std=noise_level, size=fft_series.shape,
                             device=fft_series.device, dtype=fft_series.dtype)

        # Add the noise to the FFT of the time series
        augmented_fft = fft_series + noise

        # Perform the inverse FFT to get the augmented time series
        augmented_series = torch.fft.ifft(augmented_fft, dim=-1)

        # Return only the real part since the time series should be real-valued
        return augmented_series.real

    @property
    def augment_X(self):
        """Apply augmentations to the training data."""
        augmented_series = self.X.clone()  # Clone to avoid in-place modifications
        if 'Dataset_powercons' in self.data_name or 'Dataset_smoothsubspace' in self.data_name:
            augmented_series = self.frequency_domain_augmentation(
                augmented_series, self.noise_level)
        if 'Dataset_mixedshapesregulartrain' in self.data_name or 'Dataset_symbols' in self.data_name:
            augmented_series = self.random_cropping(
                augmented_series, self.crop_size)
        augmented_series = self.jittering(augmented_series, self.noise_level)
        augmented_series = self.time_warping(
            augmented_series, self.warp_factor)
        augmented_series = self.magnitude_scaling(
            augmented_series, self.scaling_factor_range)
        return augmented_series

    def __getitem__(self, index):
        """Return a single item (x, y) from the dataset."""
        if self.mode == 'train':
            # Randomly decide whether to return augmented or unaugmented data
            # 50% chance for augmentation
            if self.augment and torch.rand(1).item() > 0.5:
                x = self.augment_X[index, :]  # Augmented version
            else:
                x = self.noisy_X[index, :]  # Unaugmented version (with noise)
            y = self.y[index]
        elif self.mode == 'valid':
            x = self.noisy_X[index, :]  # Use unaugmented validation data
            y = self.y[index]
        elif self.mode == 'test':
            # Simulate variations in the test dataset
            x = self.simulate_variation(
                self.X[index, :], ds_var=self.args.DS_VAR)
            y = self.y[index]
        return x.float(), y

    def simulate_variation(self, x, ds_var='none'):
        """Simulate realistic variations in the test data."""
        if ds_var == 'jittering':
            # Example: Apply controlled Gaussian nois
            x = self.jittering(x, noise_level=self.noise_level)
        elif ds_var == 'time_warping':
            # Example: Apply time warping if needed
            x = self.time_warping(x, warp_factor=self.warp_factor)
        elif ds_var == 'magnitude_scaling':
            # Example: Apply magnitude scaling if needed
            x = self.magnitude_scaling(
                x, scaling_factor_range=self.scaling_factor_range)

        return x

    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.X)

    # def __len__(self):
    #     if self.mode == 'train':
    #         return self.N_train * self.args.R_train
    #     elif self.mode == 'valid':
    #         return self.N_valid * self.args.R_train
    #     elif self.mode == 'test':
    #         return self.N_test * self.args.R_test


def GetDataLoader(args, mode, path=None):
    normal_datasets = ['Dataset_acuteinflammation.ds',  # 0
                       'Dataset_balancescale.ds',  # 1
                       'Dataset_breastcancerwisc.ds',  # 2
                       'Dataset_cardiotocography3clases.ds',  # 3
                       'Dataset_energyy1.ds',  # 4
                       'Dataset_energyy2.ds',  # 5
                       'Dataset_iris.ds',  # 6
                       'Dataset_mammographic.ds',  # 7
                       'Dataset_pendigits.ds',  # 8
                       'Dataset_seeds.ds',  # 9
                       'Dataset_tictactoe.ds',  # 10
                       'Dataset_vertebralcolumn2clases.ds',  # 11
                       'Dataset_vertebralcolumn3clases.ds']  # 12

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
        trainset = CustomDataset(dataname, args, path,
                                 mode='train', augment=False)
        validset = CustomDataset(dataname, args, path,
                                 mode='valid', augment=False)
        testset = CustomDataset(dataname, args, path,
                                mode='test', augment=False)

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
            trainset = CustomDataset(
                dataname, args, path, mode='train', augment=False)
            validset = CustomDataset(
                dataname, args, path, mode='valid', augment=False)
            testset = CustomDataset(
                dataname, args, path, mode='test', augment=False)
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
        trainset = CustomDataset(dataname, args, path,
                                 mode='train', temporal=True, augment=False)
        validset = CustomDataset(dataname, args, path,
                                 mode='valid', temporal=True, augment=False)
        testset = CustomDataset(dataname, args, path,
                                mode='test', temporal=True, augment=False)

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
        trainset = CustomDataset(dataset_name, args=args, datapath=path,
                                 mode='train', temporal=True, augment=args.augment,
                                 noise_level=args.NOISE_LEVEL, warp_factor=args.WARP_FACTOR,
                                 scaling_factor_range=(args.SFR_down, args.SFR_up), crop_size=args.CROP_SIZE)

        validset = CustomDataset(dataset_name, args=args, datapath=path,
                                 mode='valid', temporal=True, augment=False,)

        testset = CustomDataset(dataset_name, args=args, datapath=path,
                                mode='test', temporal=True, augment=False,)

        # message
        info = {}
        info['dataname'] = validset.data_name
        info['N_feature'] = validset.N_feature
        info['N_class'] = validset.N_class
        info['N_time'] = validset.N_time

        # batch
        train_loader = DataLoader(trainset, batch_size=len(trainset))
        valid_loader = DataLoader(validset, batch_size=len(validset))
        test_loader = DataLoader(testset,  batch_size=len(testset))

        # message
        info['N_train'] = len(trainset)
        info['N_valid'] = len(validset)
        info['N_test'] = len(testset)

        if mode == 'train':
            return train_loader, info
        elif mode == 'valid':
            return valid_loader, info
        elif mode == 'test':
            return test_loader, info
