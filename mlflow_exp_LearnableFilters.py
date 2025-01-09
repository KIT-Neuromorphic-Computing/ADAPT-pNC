import PrintedLearnableFilter as pNN
from utils import *
import pprint
import torch
from configuration import *
import os
import sys

from functools import partial
import os
import tempfile
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import random_split
import torchvision
import torchvision.transforms as transforms
import ray
from ray import tune
from ray import train
from ray.train import Checkpoint, get_checkpoint
from ray.tune.schedulers import ASHAScheduler
import ray.cloudpickle as pickle

sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), 'utils'))

args = parser.parse_args()
args.DATASET = 14
args.N_train = 20
args.task = 'temporal'
args.augment = True
args.loss = 'celoss'
args.opt = 'adamw'
args.LR = 0.1
args.metric = 'temporal_acc'
args.projectname = 'MLflowLearnableFilters'
args.DEVICE = 'cpu'
# args.EPOCH = 100

args = FormulateArgs(args)

print(f'Training network on device: {args.DEVICE}.')
MakeFolder(args)

# Store the original working directory
original_path = os.getcwd()


def train_pnn_progressive_here(nn, train_loader, valid_loader, lossfunction, optimizer, args, logger, UUID='default'):
    start_training_time = time.time()

    evaluator = Evaluator(args)

    best_valid_loss = math.inf
    current_lr = args.LR
    patience_lr = 0
    patience = 0

    lr_update = False
    early_stop = False

    current_epoch = 0

    for epoch in range(current_epoch, args.EPOCH):
        start_epoch_time = time.time()

        msg = ''

        for x_train, y_train in train_loader:
            msg += f'{current_lr}'
            msg += f'hyperparameters in printed neural network for training :\nepoch : {
                epoch:-6d} |\n'

            # x_train = x_train.to(args.DEVICE)
            # y_train = y_train.to(args.DEVICE)

            L_train = lossfunction(nn, x_train, y_train)
            train_acc = evaluator(nn, x_train, y_train)
            optimizer.zero_grad()
            L_train.backward()
            optimizer.step()

        with torch.no_grad():
            for x_valid, y_valid in valid_loader:
                msg += f'hyperparameters in printed neural network for validation :\nepoch : {
                    epoch:-6d} |\n'

                # x_valid = x_valid.to(args.DEVICE)
                # y_valid = y_valid.to(args.DEVICE)

                L_valid = lossfunction(nn, x_valid, y_valid)
                valid_acc = evaluator(nn, x_valid, y_valid)

        logger.debug(msg)

        if args.recording:
            record_checkpoint(epoch, nn, L_train, L_valid,
                              UUID, args.recordpath)

        # Report intermediate results
        ray.train.report(
            {"loss": L_valid.item(), "accuracy": valid_acc.item()})

        if L_valid.item() < best_valid_loss:
            best_valid_loss = L_valid.item()
            save_checkpoint(epoch, nn, optimizer,
                            best_valid_loss, UUID, args.temppath)
            patience_lr = 0
            patience = 0
        else:
            patience_lr += 1
            patience += 1

        if patience_lr > args.LR_PATIENCE:
            print('lr update')
            lr_update = True

        if lr_update:
            lr_update = False
            patience_lr = 0
            _, nn, _, _ = load_checkpoint(UUID, args.temppath)
            logger.info(
                'load best network to warm start training with lower lr.')
            for g in optimizer.param_groups:
                g['params'] = nn.GetParam()
                g['lr'] = g['lr'] * args.LR_DECAY
                current_lr = g['lr']
            logger.info(f'lr update to {current_lr}.')

        if current_lr < args.LR_MIN:
            early_stop = True
            print('early stop (lr).')
            logger.info('Early stop (lr).')
            break

        # if patience > args.PATIENCE:
        #     print('Early stop (patience).')
        #     logger.info('Early stop (patience).')
        #     early_stop = True
        #     break

        end_epoch_time = time.time()
        end_training_time = time.time()
        if (end_training_time - start_training_time) >= args.TIMELIMITATION*60*60:
            print('Time limination reached.')
            logger.warning('Time limination reached.')
            break

        if epoch % args.report_freq == 0:
            print(f'| Epoch: {epoch:-6d} | Train loss: {L_train.item():.4e} | Valid loss: {L_valid.item():.4e} | Train acc: {train_acc:.4f} | Valid acc: {
                  valid_acc:.4f} | patience: {patience_lr:-3d} | lr: {current_lr} | Epoch time: {end_epoch_time-start_epoch_time:.1f} |')
            logger.info(f'| Epoch: {epoch:-6d} | Train loss: {L_train.item():.4e} | Valid loss: {L_valid.item():.4e} | Train acc: {train_acc:.4f} | Valid acc: {
                        valid_acc:.4f} | patience: {patience_lr:-3d} | lr: {current_lr} | Epoch time: {end_epoch_time-start_epoch_time:.1f} |')

    _, resulted_nn, _, _ = load_checkpoint(UUID, args.temppath)

    if early_stop:
        os.remove(f'{args.temppath}/{UUID}.ckp')

    return resulted_nn, early_stop


def train_model(config, args, train_loader, valid_loader, test_loader, datainfo):
    args.NOISE_LEVEL = config["NOISE_LEVEL"]
    args.WARP_FACTOR = config["WARP_FACTOR"]
    args.SFR_down = config["SFR_down"]
    args.SFR_up = config["SFR_up"]
    args.CROP_SIZE = config["CROP_SIZE"]

    print('--------------------------------------------------')
    print(f"Current working directory: {os.getcwd()}")

    try:
        # Reset to the original path before loading files
        os.chdir(original_path)

        print('--------------------------------------------------')
        print(f"Current working directory after: {os.getcwd()}")

        pprint.pprint(datainfo)

        # SetSeed(args.SEED)

        setup = f"pLF_data_{args.DATASET:02d}_{
            datainfo['dataname']}_seed_{args.SEED:02d}.model"
        print(f'Training setup: {setup}.')

        msglogger = GetMessageLogger(args, setup)
        msglogger.info(f'Training network on device: {args.DEVICE}.')
        msglogger.info(f'Training setup: {setup}.')
        msglogger.info(args.augment)
        msglogger.info(datainfo)

        if os.path.isfile(f'{args.savepath}/{setup}'):
            print(f'{setup} exists, skip this training.')
            msglogger.info('Training was already finished.')
        else:
            pnn = pNN.PrintedNeuralNetwork(
                args, datainfo['N_feature'], datainfo['N_class'], args.N_Channel, N_feature=args.N_feature).to(args.DEVICE)

            msglogger.info(f'Number of parameters that are learned in this experiment: {
                len(pnn.GetParam())}.')

            lossfunction = pNN.LFLoss(args).to(args.DEVICE)
            optimizer = torch.optim.Adam(pnn.GetParam(), lr=args.LR)

            pnn, best = train_pnn_progressive_here(
                pnn, train_loader, valid_loader, lossfunction, optimizer, args, msglogger, UUID=setup)

            if best:
                if not os.path.exists(f'{args.savepath}/'):
                    os.makedirs(f'{args.savepath}/')
                torch.save(pnn, f'{args.savepath}/{setup}')
                msglogger.info('Training if finished.')
            else:
                msglogger.warning('Time out, further training is necessary.')

        CloseLogger(msglogger)
    except Exception as e:
        print(f"Error during training: {e}")
        raise e


def main(num_samples=10, max_num_epochs=10):

    train_loader, datainfo = GetDataLoader(args, 'train')
    valid_loader, datainfo = GetDataLoader(args, 'valid')
    test_loader, datainfo = GetDataLoader(args, 'test')

    ray.init(num_cpus=10, num_gpus=0)

    # Define the search space for the augmentation hyperparameters
    config = {
        "NOISE_LEVEL": tune.uniform(0.01, 0.1),  # Range for noise level
        # Range for time warping factor
        "WARP_FACTOR": tune.uniform(0.05, 0.2),
        # Lower bound of scaling range
        "SFR_down": tune.uniform(0.8, 1.0),
        # Upper bound of scaling range
        "SFR_up": tune.uniform(1.0, 1.2),
        "CROP_SIZE": tune.choice([45, 50, 55, 60])  # Different crop sizes
    }

    # Define the scheduler for Ray Tune
    scheduler = ASHAScheduler(
        metric="loss",
        mode="min",
        max_t=max_num_epochs,
        grace_period=1,
        reduction_factor=2,
    )

    # Run hyperparameter tuning
    result = tune.run(
        partial(train_model, args=args, train_loader=train_loader,
                valid_loader=valid_loader, test_loader=test_loader, datainfo=datainfo),
        config=config,
        num_samples=num_samples,
        scheduler=scheduler,)

    best_trial = result.get_best_trial("loss", "min", "last")
    print(f"Best trial config for {args.DATASET}: {best_trial.config}")
    print(f"Best trial final validation loss: {
          best_trial.last_result['loss']}")
    print(f"Best trial final validation accuracy: {
          best_trial.last_result['accuracy']}")


if __name__ == "__main__":
    main(num_samples=10, max_num_epochs=10)
