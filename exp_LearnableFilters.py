import PrintedLearnableFilter as pNN
from utils import *
import pprint
import torch
from configuration import *
import os
import sys
sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), 'utils'))

args = parser.parse_args()

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
    if args.opt == "adam":
        optimizer = torch.optim.Adam(pnn.GetParam(), lr=args.LR)
    elif args.opt == "adamw":
        optimizer = torch.optim.AdamW(pnn.GetParam(), lr=args.LR)
    else:
        optimizer = torch.optim.Adam(pnn.GetParam(), lr=args.LR)

    if args.PROGRESSIVE:
        pnn, best = train_pnn_progressive(
            pnn, train_loader, valid_loader, lossfunction, optimizer, args, msglogger, UUID=setup)
    else:
        pnn, best = train_pnn(pnn, train_loader, valid_loader,
                              lossfunction, optimizer, args, msglogger, UUID=setup)

    if best:
        if not os.path.exists(f'{args.savepath}/'):
            os.makedirs(f'{args.savepath}/')
        torch.save(pnn, f'{args.savepath}/{setup}')
        msglogger.info('Training if finished.')
    else:
        msglogger.warning('Time out, further training is necessary.')

CloseLogger(msglogger)
