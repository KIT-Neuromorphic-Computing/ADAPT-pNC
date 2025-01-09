import RNN
from utils import *
import pprint
import torch
from configuration import *
import os
import sys
sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), 'utils'))

args = parser.parse_args([])
args.DEVICE = 'cpu'
args.task = 'temporal'

args = FormulateArgs(args)

print(f'Training network on device: {args.DEVICE}.')
MakeFolder(args)

train_loader, datainfo = GetDataLoader(args, 'train')
valid_loader, datainfo = GetDataLoader(args, 'valid')
test_loader, datainfo = GetDataLoader(args, 'test')
pprint.pprint(datainfo)

SetSeed(args.SEED)

setup = f"pLF_data_{args.DATASET:02d}_{
    datainfo['dataname']}_seed_{args.SEED:02d}.model"
print(f'Training setup: {setup}.')

msglogger = GetMessageLogger(args, setup)
msglogger.info(f'Training network on device: {args.DEVICE}.')
msglogger.info(f'Training setup: {setup}.')
msglogger.info(datainfo)

if os.path.isfile(f'{args.savepath}/{setup}'):
    print(f'{setup} exists, skip this training.')
    msglogger.info('Training was already finished.')
else:
    baseline = RNN.SimpleRNNModel(
        datainfo['N_feature'], datainfo['N_class']).to(args.DEVICE)

    msglogger.info(f'Number of parameters that are learned in this experiment: {
                   len(baseline.GetParam())}.')

    lossfunction = RNN.LFLoss(args).to(args.DEVICE)
    optimizer = torch.optim.Adam(baseline.GetParam(), lr=args.LR)

    if args.PROGRESSIVE:
        baseline, best = train_pnn_progressive(
            baseline, train_loader, valid_loader, lossfunction, optimizer, args, msglogger, UUID=setup)
    else:
        baseline, best = train_pnn(
            baseline, train_loader, valid_loader, lossfunction, optimizer, args, msglogger, UUID=setup)

    if best:
        if not os.path.exists(f'{args.savepath}/'):
            os.makedirs(f'{args.savepath}/')
        torch.save(baseline, f'{args.savepath}/{setup}')
        msglogger.info('Training if finished.')
    else:
        msglogger.warning('Time out, further training is necessary.')

CloseLogger(msglogger)
