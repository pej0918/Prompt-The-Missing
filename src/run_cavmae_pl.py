# -*- coding: utf-8 -*-
# @Time    : 6/11/21 12:57 AM
# @Author  : Yuan Gong
# @Affiliation  : Massachusetts Institute of Technology
# @Email   : yuangong@mit.edu
# @File    : run.py

import argparse
import os
os.environ['MPLCONFIGDIR'] = './plt/'
import ast
import pickle
import sys
import time
import torch
from torch.utils.data import WeightedRandomSampler
basepath = os.path.dirname(os.path.dirname(sys.path[0]))
sys.path.append(basepath)
import dataloader as dataloader
import models
import numpy as np
import warnings
import json
from sklearn import metrics
from traintest_pl import train_pl, validate_pl
from traintest_ft import validate
# finetune cav-mae model

print("I am process %s, running on %s: starting (%s)" % (os.getpid(), os.uname()[1], time.asctime()))

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--data-train", type=str, default='', help="training data json")
parser.add_argument("--data-val", type=str, default='', help="validation data json")
parser.add_argument("--data-eval", type=str, default=None, help="evaluation data json")
parser.add_argument("--label-csv", type=str, default='', help="csv with class labels")
parser.add_argument("--n_class", type=int, default=527, help="number of classes")
parser.add_argument("--model", type=str, default='ast', help="the model used")
parser.add_argument("--dataset", type=str, default="audioset", help="the dataset used", choices=["audioset", "esc50", "speechcommands", "fsd50k", "vggsound", "epic", "k400"])
parser.add_argument("--dataset_mean", type=float, help="the dataset mean, used for input normalization")
parser.add_argument("--dataset_std", type=float, help="the dataset std, used for input normalization")
parser.add_argument("--target_length", type=int, help="the input length in frames")
parser.add_argument("--noise", help='if use balance sampling', type=ast.literal_eval)

parser.add_argument("--exp-dir", type=str, default="", help="directory to dump experiments")
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float, metavar='LR', help='initial learning rate')
parser.add_argument("--optim", type=str, default="adam", help="training optimizer", choices=["sgd", "adam"])
parser.add_argument('-b', '--batch-size', default=48, type=int, metavar='N', help='mini-batch size')
parser.add_argument('-w', '--num-workers', default=32, type=int, metavar='NW', help='# of workers for dataloading (default: 32)')
parser.add_argument("--n-epochs", type=int, default=10, help="number of maximum training epochs")
# not used in the formal experiments, only in preliminary experiments
parser.add_argument("--lr_patience", type=int, default=1, help="how many epoch to wait to reduce lr if mAP doesn't improve")
parser.add_argument("--lr_adapt", help='if use adaptive learning rate', type=ast.literal_eval)
parser.add_argument("--metrics", type=str, default="mAP", help="the main evaluation metrics in finetuning", choices=["mAP", "acc"])
parser.add_argument("--loss", type=str, default="BCE", help="the loss function for finetuning, depend on the task", choices=["BCE", "CE"])
parser.add_argument('--warmup', help='if use warmup learning rate scheduler', type=ast.literal_eval, default='True')
parser.add_argument("--lrscheduler_start", default=2, type=int, help="when to start decay in finetuning")
parser.add_argument("--lrscheduler_step", default=1, type=int, help="the number of step to decrease the learning rate in finetuning")
parser.add_argument("--lrscheduler_decay", default=0.5, type=float, help="the learning rate decay ratio in finetuning")
parser.add_argument('--freqm', help='frequency mask max length', type=int, default=0)
parser.add_argument('--timem', help='time mask max length', type=int, default=0)

parser.add_argument("--wa", help='if do weight averaging in finetuning', type=ast.literal_eval)
parser.add_argument("--wa_start", type=int, default=1, help="which epoch to start weight averaging in finetuning")
parser.add_argument("--wa_end", type=int, default=10, help="which epoch to end weight averaging in finetuning")

parser.add_argument("--n-print-steps", type=int, default=100, help="number of steps to print statistics")
parser.add_argument('--save_model', help='save the model or not', type=ast.literal_eval)

parser.add_argument("--mixup", type=float, default=0, help="how many (0-1) samples need to be mixup during training")
parser.add_argument("--bal", type=str, default=None, help="use balanced sampling or not")

parser.add_argument("--label_smooth", type=float, default=0.1, help="label smoothing factor")
parser.add_argument("--weight_file", type=str, default=None, help="path to weight file")
parser.add_argument("--finetuned_path", type=str, default='None', help="pretrained model path")
parser.add_argument("--ftmode", type=str, default='prompt_learning', help="how to fine-tune the model")

parser.add_argument("--head_lr", type=float, default=50.0, help="learning rate ratio the newly initialized layers / pretrained weights")
parser.add_argument('--freeze_base', help='freeze the backbone or not', type=ast.literal_eval)
parser.add_argument('--skip_frame_agg', help='if do frame agg', type=ast.literal_eval)

parser.add_argument('--proportion', help='proportion of data to use for prompt learning', type=float, default=0.3)
parser.add_argument('--save_data', help='save the data or not', type=ast.literal_eval)

parser.add_argument('--mode', type=str, default='train', help='train or eval')

parser.add_argument('--noise_to_audio', help='if add noise to audio', action='store_true')
parser.add_argument('--noise_to_vision', help='if add noise to vision', action='store_true')

args = parser.parse_args()

def train_run(mode):
    # mode 0: complete
    # mode 1: vision only
    # mode 2: audio only
    # mode 3: noise to both
    
    # all exp in this work is based on 224 * 224 image
    im_res = 224
    audio_conf = {'num_mel_bins': 128, 'target_length': args.target_length, 'freqm': args.freqm, 'timem': args.timem, 'mixup': args.mixup,
                'dataset': args.dataset, 'mode':'train', 'mean':args.dataset_mean, 'std':args.dataset_std,
                'noise':args.noise, 'label_smooth': args.label_smooth, 'im_res': im_res}
    
    train_loader = torch.utils.data.DataLoader(
            dataloader.AudiosetDataset(args.data_train, label_csv=args.label_csv, audio_conf=audio_conf),
            batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True)

    if args.model == 'cav-mae-ft':
        audio_model = models.CAVMAEFT(label_dim=args.n_class, modality_specific_depth=11)
    else:
        raise ValueError('model not supported')

    if args.finetuned_path == 'None':
        raise ValueError('finetuned model path is not provided')

    # finetune based on a CAV-MAE pretrained model, which is the default setting unless for ablation study
    if args.finetuned_path != 'None':
        # TODO: change this to a wget link
        mdl_weight = torch.load(args.finetuned_path)
        if not isinstance(audio_model, torch.nn.DataParallel):
            audio_model = torch.nn.DataParallel(audio_model)
        miss, unexpected = audio_model.load_state_dict(mdl_weight, strict=False)
        print('now load cav-mae fine-tuned weights from ', args.finetuned_path)
        print(miss, unexpected)

    print('Now starting training for {:d} epochs.'.format(args.n_epochs))
    if mode == 0:
        # Case 1. Complete
        train_pl(audio_model, train_loader, args)
    if mode == 1:
        # Case 2. Vision_only
        train_pl(audio_model, train_loader, args, noise_to_audio=True)
    if mode == 2:
        # Case 3. Audio_only
        train_pl(audio_model, train_loader, args, noise_to_vision=True)
    if mode == 3:
        # Case 4. Noise_to_both
        train_pl(audio_model, train_loader, args, noise_to_audio=True, noise_to_vision=True)

# prompt learning 때는 weight average를 하지 않음
# # average the model weights of checkpoints, note it is not ensemble, and does not increase computational overhead
# def wa_model(exp_dir, start_epoch, end_epoch):
#     sdA = torch.load(exp_dir + '/models/audio_model.' + str(start_epoch) + '.pth', map_location='cpu')
#     model_cnt = 1
#     for epoch in range(start_epoch+1, end_epoch+1):
#         sdB = torch.load(exp_dir + '/models/audio_model.' + str(epoch) + '.pth', map_location='cpu')
#         for key in sdA:
#             sdA[key] = sdA[key] + sdB[key]
#         model_cnt += 1
#     print('wa {:d} models from {:d} to {:d}'.format(model_cnt, start_epoch, end_epoch))
#     for key in sdA:
#         sdA[key] = sdA[key] / float(model_cnt)
#     return sdA

def eval_run():
    audio_model = models.CAVMAEFT(label_dim=args.n_class, modality_specific_depth=11, dir_path=args.exp_dir)
    val_audio_conf = {'num_mel_bins': 128, 'target_length': args.target_length, 'freqm': 0, 'timem': 0, 'mixup': 0, 'dataset': args.dataset,
                    'mode':'eval', 'mean': args.dataset_mean, 'std': args.dataset_std, 'noise': False}
    
    if args.finetuned_path == 'None':
        raise ValueError('finetuned model path is not provided')
    mdl_weight = torch.load(args.finetuned_path)
    if not isinstance(audio_model, torch.nn.DataParallel):
        audio_model = torch.nn.DataParallel(audio_model)
    miss, unexpected = audio_model.load_state_dict(mdl_weight, strict=False)
    print('now load cav-mae fine-tuned weights from ', args.finetuned_path)
    print(miss, unexpected)
    
    val_loader = torch.utils.data.DataLoader(
        dataloader.AudiosetDataset(args.data_val, label_csv=args.label_csv, audio_conf=val_audio_conf),
        batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    if not isinstance(audio_model, torch.nn.DataParallel):
        audio_model = torch.nn.DataParallel(audio_model)
    audio_model = audio_model.to('cuda')
    audio_model.eval()
    stats, loss= validate_pl(audio_model, val_loader, args)
    
    AP_res = np.mean([stat['AP'] for stat in stats])
    print("*** Prompt Learning ***")
    print('mAP is {:.4f}'.format(AP_res))
    acc_res = stats[0]['acc']
    print('acc is {:.4f}'.format(acc_res))
    print('loss is {:.4f}'.format(loss))
    
    stats, loss = validate(audio_model, val_loader, args)
    
    print("*** Fine-tuning ***")
    AP_res = np.mean([stat['AP'] for stat in stats])
    print('mAP is {:.4f}'.format(AP_res))
    acc_res = stats[0]['acc']
    print('acc is {:.4f}'.format(acc_res))
    print('loss is {:.4f}'.format(loss))
    
        
if __name__ == '__main__':
    if args.exp_dir == '':
        raise ValueError('exp_dir is not provided')
    if not os.path.exists(args.exp_dir):
        os.makedirs(args.exp_dir)
        
    if args.mode == 'train':
        with open("%s/args.pkl" % args.exp_dir, "wb") as f:
            pickle.dump(args, f)
        with open(args.exp_dir + '/args.json', 'w') as f:
            json.dump(args.__dict__, f, indent=2)
            
        train_run(mode=0)
        train_run(mode=1)
        train_run(mode=2)
        train_run(mode=3)
    if args.mode == 'eval':
        eval_run()