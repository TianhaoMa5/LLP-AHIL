
from __future__ import print_function
import random

import time
import argparse
import os
import sys

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import Counter,OrderedDict
from WideResNet import WideResnet
from datasets.cifar import get_train_loader, get_val_loader
from utils import accuracy, setup_default_logging, AverageMeter, WarmupCosineLrScheduler
import tensorboard_logger

device = torch.device("cuda:0")

def set_model(args):
    model = WideResnet(n_classes=args.n_classes, k=args.wresnet_k, n=args.wresnet_n, proj=False)

    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint)
        model_weights = checkpoint['model'] if 'model' in checkpoint else checkpoint

        # 使用strict=False来允许不完全匹配的权重加载
        msg = model.load_state_dict(model_weights, strict=True)

        # 打印丢失和多余的键，而不是使用断言
        if msg.missing_keys:
            print("Missing keys:", msg.missing_keys)
        if msg.unexpected_keys:
            print("Unexpected keys:", msg.unexpected_keys)

        print('loaded from checkpoint: %s' % args.checkpoint)

    model.train()
    model = model.cuda()

    if args.eval_ema:
        ema_model = WideResnet(n_classes=args.n_classes, k=args.wresnet_k, n=args.wresnet_n, proj=True)
        for param_q, param_k in zip(model.parameters(), ema_model.parameters()):
            param_k.data.copy_(param_q.detach().data)  # initialize
            param_k.requires_grad = False  # not update by gradient for eval_net
        ema_model.cuda()
        ema_model.eval()
    else:
        ema_model = None

    criteria_x = nn.CrossEntropyLoss().cuda()
    return model, criteria_x, ema_model


@torch.no_grad()
def ema_model_update(model, ema_model, ema_m):
    """
    Momentum update of evaluation model (exponential moving average)
    """
    for param_train, param_eval in zip(model.parameters(), ema_model.parameters()):
        param_eval.copy_(param_eval * ema_m + param_train.detach() * (1 - ema_m))

    for buffer_train, buffer_eval in zip(model.buffers(), ema_model.buffers()):
        buffer_eval.copy_(buffer_train)

def llp_loss(labels_proportion, y):
    x = torch.tensor(labels_proportion, device=y.device, dtype=torch.float32)

    cross_entropy = torch.sum(-x * (torch.log(y) + 1e-7))


    return cross_entropy
def weight_decay_with_mask(mask, initial_weight, max_mask_count):
    mask_count = mask.sum().item()  # 计算当前 mask 中的元素数量
    weight_decay = max(0, 1 - mask_count / max_mask_count)  # 线性衰减
    return initial_weight * weight_decay

def train_one_epoch(epoch,
                    model,
                    ema_model,
                    criteria_x,
                    optim,
                    lr_schdlr,
                    dltrain_u,
                    args,
                    n_iters,
                    logger
                    ):

    model.train()
    loss_u_meter = AverageMeter()

    epoch_start = time.time()  # start time
    dl_u =  iter(dltrain_u)
    n_iter = len(dltrain_u)
    for it in range(len(dltrain_u)):
        (var1, var2, var3, var4) = next(dl_u)
        var1 = var1[0]
        # var2 = torch.stack(var2)
        # print(var2)
        # print(f'var1:{var1.shape};\n var2: {var2.shape};\n var3: {var3.shape};\n var4: {var4.shape}')
        length = len(var2[0])

        """
        pseudo_counter = Counter(selected_label.tolist())
        for i in range(args.n_classes):
            classwise_acc[i] = pseudo_counter[i] / max(pseudo_counter.values())

        """
        ims_u_weak1 = var1
        ims_u_weak1 = ims_u_weak1.permute(0, 1, 3, 2, 4)
        imsw, labels_real, labels_idx = [], [], []  # $$$$$$$$$$$$$
        for i in range(length):
            imsw.append(ims_u_weak1[i])
            labels_real.append(var3[i])
            labels_idx.append(var4[i])
        ims_u_weak = torch.cat(imsw, dim=0)
        lbs_u_real = torch.cat(labels_real, dim=0)
        label_proportions = [[] for _ in range(length)]
        lbs_u_real = lbs_u_real.cuda()




        # --------------------------------------
        btu = ims_u_weak.size(0)
        bt=0
        imgs = ims_u_weak
        labels=lbs_u_real
        imgs=imgs.cuda()
        labels=labels.cuda()
        logits = model(imgs)

        #logits_x = logits[:bt]

        # feats_x = features[:bt]

        # loss_x = criteria_x(logits_x, lbs_x)


        labels=labels.cuda()

        loss_u=criteria_x(logits, labels)


        # 计算比例


        optim.zero_grad()
        loss_u.backward()
        optim.step()
        lr_schdlr.step()

        if args.eval_ema:
            with torch.no_grad():
                ema_model_update(model, ema_model, args.ema_m)
        loss_u_meter.update(loss_u.item())


        if (it + 1) % n_iter == 0:
            t = time.time() - epoch_start

            lr_log = [pg['lr'] for pg in optim.param_groups]
            lr_log = sum(lr_log) / len(lr_log)
            logger.info("{}-x{}-s{}, {} | epoch:{}, iter: {}. loss_u: {:.3f}".format(
                args.dataset, args.n_labeled, args.seed, args.exp_dir, epoch, it + 1, loss_u_meter.avg,
                ))

            epoch_start = time.time()

    return loss_u_meter.avg

def evaluate(model, ema_model, dataloader):

    model.eval()

    top1_meter = AverageMeter()
    ema_top1_meter = AverageMeter()

    with torch.no_grad():
        for ims, lbs in dataloader:
            ims = ims.cuda()
            ims = ims.permute(0, 2, 1, 3)

            lbs = lbs.cuda()

            logits = model(ims)
            scores = torch.softmax(logits, dim=1)
            top1, top5 = accuracy(scores, lbs, (1, 5))
            top1_meter.update(top1.item())

            if ema_model is not None:
                logits, _ = ema_model(ims)
                scores = torch.softmax(logits, dim=1)
                top1, top5 = accuracy(scores, lbs, (1, 5))
                ema_top1_meter.update(top1.item())

    return top1_meter.avg, ema_top1_meter.avg


def main():
    parser = argparse.ArgumentParser(description='L^2P-AHIL')
    parser.add_argument('--root', default='./data', type=str, help='dataset directory')
    parser.add_argument('--wresnet-k', default=2, type=int,
                        help='width factor of wide resnet')
    parser.add_argument('--wresnet-n', default=28, type=int,
                        help='depth of wide resnet')
    parser.add_argument('--dataset', type=str, default='FashionMNIST',
                        help='number of classes in dataset')
    parser.add_argument('--n-classes', type=int, default=10,
                         help='number of classes in dataset')
    parser.add_argument('--n-labeled', type=int, default=10,
                        help='number of labeled samples for training')
    parser.add_argument('--n-epoches', type=int, default=1024,
                        help='number of training epoches')
    parser.add_argument('--batchsize', type=int, default=1024,
                        help='train batch size of bag samples')
    parser.add_argument('--bagsize',type=int, default=1,
                        help='train bag size of labeled samples')
    parser.add_argument('--n-imgs-per-epoch', type=int, default= 1024,
                        help='number of training images for each epoch')

    parser.add_argument('--eval-ema', default=True, help='whether to use ema model for evaluation')
    parser.add_argument('--ema-m', type=float, default=0.999)

    parser.add_argument('--lam-u', type=float, default=1.,
                        help='coefficient of unlabeled loss')
    parser.add_argument('--lr', type=float, default=0.003,
                        help='learning rate for training')
    parser.add_argument('--weight-decay', type=float, default=5e-4,
                        help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum for optimizer')
    parser.add_argument('--seed', type=int, default=1,
                        help='seed for random behaviors, no seed if negtive')

    parser.add_argument('--temperature', default=0.2, type=float, help='softmax temperature')
    parser.add_argument('--low-dim', type=int, default=64)
    parser.add_argument('--lam-c', type=float, default=1,
                        help='coefficient of contrastive loss')
    parser.add_argument('--contrast-th', default=0.8, type=float,
                        help='pseudo label graph threshold')
    parser.add_argument('--thr', type=float, default=0.95,
                        help='pseudo label threshold')
    parser.add_argument('--alpha', type=float, default=0.9)
    parser.add_argument('--queue-batch', type=float, default=5,
                        help='number of batches stored in memory bank')
    parser.add_argument('--exp-dir', default='LLP-AHIL', type=str, help='experiment id')
    parser.add_argument('--checkpoint', default='', type=str, help='use pretrained model')

    args = parser.parse_args()

    logger, output_dir = setup_default_logging(args)
    logger.info(dict(args._get_kwargs()))

    tb_logger = tensorboard_logger.Logger(logdir=output_dir, flush_secs=2)

    if args.seed > 0:
        torch.manual_seed(args.seed)
        random.seed(args.seed)
        np.random.seed(args.seed)

    n_iters_per_epoch = args.n_imgs_per_epoch  # 1024
    n_iters_all = 48 * args.n_epoches  # 1024 * 200

    logger.info("***** Running training *****")
    logger.info(f"  Task = {args.dataset}@{args.n_labeled}")

    model, criteria_x, ema_model = set_model(args)
    logger.info("Total params: {:.2f}M".format(
        sum(p.numel() for p in model.parameters()) / 1e6))

    dltrain_u = get_train_loader(args.n_classes,
                                 args.dataset, args.batchsize, args.bagsize, root=args.root,
                                 method='DLLP',supervised=True)
    dlval = get_val_loader(dataset=args.dataset, batch_size=64, num_workers=2, root=args.root)

    wd_params, non_wd_params = [], []
    for name, param in model.named_parameters():
        if 'bn' in name:
            non_wd_params.append(param)
        else:
            wd_params.append(param)
    param_list = [
        {'params': wd_params}, {'params': non_wd_params, 'weight_decay': 0}]
    optim = torch.optim.SGD(param_list, lr=args.lr, weight_decay=args.weight_decay,
                            momentum=args.momentum, nesterov=True)

    lr_schdlr = WarmupCosineLrScheduler(optim, n_iters_all, warmup_iter=0)

    # memory bank
    args.queue_size = 5120

    train_args = dict(
        model=model,
        ema_model=ema_model,

        criteria_x=criteria_x,
        optim=optim,
        lr_schdlr=lr_schdlr,
        dltrain_u=dltrain_u,
        args=args,
        n_iters=n_iters_per_epoch,
        logger=logger
    )

    best_acc = -1
    best_epoch = 0
    logger.info('-----------start training--------------')
    for epoch in range(args.n_epoches):
        loss_u = \
        train_one_epoch(epoch, **train_args)

        top1, ema_top1 = evaluate(model, ema_model, dlval)

        tb_logger.log_value('loss_u', loss_u, epoch)
        if best_acc < top1:
            best_acc = top1
            best_epoch = epoch

        logger.info("Epoch {}. Acc: {:.4f}. Ema-Acc: {:.4f}. best_acc: {:.4f} in epoch{}".
                    format(epoch, top1, ema_top1, best_acc, best_epoch))
        tb_logger.log_value('test_acc', top1, epoch)
        tb_logger.log_value('test_ema_acc', ema_top1, epoch)
        if epoch%10==0:
            save_obj = {
                'model': model.state_dict(),
                'ema_model': ema_model.state_dict(),
                'optimizer': optim.state_dict(),
                'lr_scheduler': lr_schdlr.state_dict(),
                'epoch': epoch,
            }
            torch.save(save_obj, os.path.join(output_dir, 'checkpoint_%02d.pth'%epoch))

if __name__ == '__main__':
    main()
