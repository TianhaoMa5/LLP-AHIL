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
from collections import Counter, OrderedDict
from WideResNet import WideResnet
from datasets.cifar import get_train_loader, get_val_loader
from utils import accuracy, setup_default_logging, AverageMeter, CurrentValueMeter, WarmupCosineLrScheduler
import tensorboard_logger
import torch.multiprocessing as mp
from LeNet import LeNet5
from torchvision import models

import torch
import math





def cross_entropy_loss_torch(softmax_matrix, onehot_labels):
    log_softmax = torch.log(softmax_matrix + 1e-12)

    cross_entropy = -torch.sum(onehot_labels * log_softmax, dim=1)

    mean_loss = torch.mean(cross_entropy)
    return mean_loss



def set_model(args):
    if args.dataset in ['CIFAR10', 'SVHN','CIFAR100','miniImageNet']:
        model = WideResnet(
            n_classes=args.n_classes,
            k=args.wresnet_k,
            n=args.wresnet_n,
            proj=False
        )
        #model = models.resnet18(pretrained=True)
        #model.fc = nn.Linear(model.fc.in_features, args.n_classes)

        #model = resnet34(num_classes=10)
    else:
        model = LeNet5()
    #model = ResNet18CIFAR10(num_classes=args.n_classes)

    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint)

        msg = model.load_state_dict(checkpoint, strict=False)


        assert set(msg.missing_keys) == {"classifier.weight", "classifier.bias"}
        print('loaded from checkpoint: %s' % args.checkpoint)
    model.train()
    model.cuda()

    if args.eval_ema:
        if args.dataset in ['CIFAR10','SVHN', 'CIFAR100']:
            ema_model = WideResnet(
                n_classes=args.n_classes,
                k=args.wresnet_k,
                n=args.wresnet_n,
                proj=False
            )
           # model = models.resnet18(pretrained=True)
            #model.fc = nn.Linear(model.fc.in_features, 10)
            #ema_model = resnet34(num_classes=10)

        else:
            ema_model = LeNet5()
        #ema_model = ResNet18CIFAR10(num_classes=args.n_classes)

        for param_q, param_k in zip(model.parameters(), ema_model.parameters()):
            param_k.data.copy_(param_q.detach().data)  # initialize
            param_k.requires_grad = False  # not update by gradient for eval_net
        ema_model.cuda()
        ema_model.eval()
    else:
        ema_model = None

    criteria_x = nn.CrossEntropyLoss().cuda()
    criteria_u = nn.CrossEntropyLoss(reduction='none').cuda()

    return model, criteria_x, criteria_u, ema_model



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
    x = torch.tensor(labels_proportion, dtype=torch.float64).cuda()
    x = x.squeeze(0)

    # Ensure y is also double

    y = y.double()
    cross_entropy = torch.sum(-x * (torch.log(y) + 1e-7))
    mse_loss = torch.mean((x - y) ** 2)

    return cross_entropy


def custom_loss(probs, lambda_val=1.0):
    # probs is assumed to be a 2D tensor of shape (n, N_i)
    # where n is the number of rows and N_i is the number of columns

    # Compute the log of probs
    log_probs = torch.log(probs)

    # Multiply probs with log_probs element-wise
    product = -probs * log_probs

    # Compute the double sum
    loss = torch.sum(product)

    # Multiply by lambda
    loss = lambda_val * loss

    return loss


def compute_CC_loss(softmax_p, proportions, epsilon=1e-12):
    s, C = softmax_p.shape  # s: bag size, C: number of classes
    device = softmax_p.device
    dtype = torch.double

    k_c = (proportions * s).long()  # [C]

    k_max = k_c.max().item()
    E = torch.zeros(C, k_max + 1, device=device, dtype=dtype)
    E[:, 0] = 1.0  # E[c, 0] = 1

    for j in range(s):
        a = softmax_p[j].to(dtype)
        E_new = E.clone()
        E_new[:, 1:k_max+1] = E[:, 1:k_max+1] * (1 - a.unsqueeze(1)) + E[:, 0:k_max] * a.unsqueeze(1)
        E = E_new

    class_indices = torch.arange(C, device=device)
    E_kc = E[class_indices, k_c]  # [C]

    log_E_kc = torch.log(E_kc + epsilon)  # [C]

    loss = -torch.sum(log_E_kc)  # scalar

    return loss
def compute_CC_loss_logDP(labels_p, proportions, epsilon=1e-12):
    s, C = labels_p.shape  # s: bag size, C: number of classes
    device = labels_p.device
    dtype = labels_p.dtype

    k_c = (proportions * s).long()  # [C]

    k_max = k_c.max().item()

    logE = torch.full((C, k_max + 1), -1e30, device=device, dtype=dtype)
    logE[:, 0] = 0.0  # log(1) = 0

    #   logE_new[c, l] = log_sum_exp( logE[c, l] + log(1 - a_c), logE[c, l-1] + log(a_c) )
    for j in range(s):
        p_j = labels_p[j]  # [C]
        log_p_j   = torch.log(p_j.clamp_min(epsilon))          # log( f_c(x_j) )
        log_1mp_j = torch.log((1 - p_j).clamp_min(epsilon))    # log(1 - f_c(x_j))

        logE_new = torch.full((C, k_max + 1), -1e30, device=device, dtype=dtype)

        # logE_new[:, 0] = logE[:, 0] + log(1 - p_j)
        logE_new[:, 0] = logE[:, 0] + log_1mp_j

        for l in range(1, k_max + 1):
            v1 = logE[:, l]   + log_1mp_j
            v2 = logE[:, l-1] + log_p_j
            max_v12 = torch.maximum(v1, v2)
            log_sum = max_v12 + torch.log(torch.exp(v1 - max_v12) + torch.exp(v2 - max_v12) + epsilon)
            logE_new[:, l] = log_sum

        logE = logE_new

    class_indices = torch.arange(C, device=device)
    logE_kc = logE[class_indices, k_c]  # [C]

    loss = -torch.sum(logE_kc)  # scalar

    return loss

def compute_CC_loss_simplified_gpu(labels_p, proportions, epsilon=1e-12):
    s, C = labels_p.shape  # s: bag size, C: number of classes
    device = labels_p.device
    dtype = labels_p.dtype

    k_c = (proportions * s).long()  # [C]

    f_c = labels_p


    product_sum = f_c.sum(dim=0)

    log_product_sum = torch.log(product_sum + epsilon)  # [C]

    loss = -log_product_sum.sum()

    return loss


def multi_instance_loss_dp_gpu(labels_p: torch.Tensor,
                               proportions: torch.Tensor) -> torch.Tensor:
    device = labels_p.device
    s, C = labels_p.shape

    k_list = torch.round(proportions * s).long()  # shape (C,)

    total_loss = torch.zeros([], device=device, dtype=torch.float32)

    for c in range(C):
        k_c = k_list[c].item()
        if k_c <= 0:
            continue

        p = labels_p[:, c]

        dp = torch.zeros((s + 1, k_c + 1), device=device, dtype=torch.float32)
        dp[0, 0] = 1.0

        for t in range(s):
            # dp[t+1, 0] = dp[t, 0]
            dp[t + 1, 0] = dp[t, 0]
            # dp[t+1, r] = dp[t, r] + p[t]*dp[t, r-1]
            # dp[t+1, 1:] = dp[t, 1:] + p[t] * dp[t, :-1]
            dp[t + 1, 1:] = dp[t, 1:] + p[t] * dp[t, :-1]

        sum_of_products = dp[s, k_c]

        c_loss = torch.log(sum_of_products)

        total_loss += c_loss

    return total_loss


def thre_ema(thre, sum_values, ema):
    return thre * ema + (1 - ema) * sum_values


def weight_decay_with_mask(mask, initial_weight, max_mask_count):
    mask_count = mask.sum().item()
    weight_decay = max(0, 1 - mask_count / max_mask_count)
    return initial_weight * weight_decay


def train_one_epoch(epoch,
                    bagsize,
                    n_classes,
                    model,
                    ema_model,
                    prob_list,
                    criteria_x,
                    criteria_u,
                    optim,
                    lr_schdlr,
                    dltrain_u,
                    args,
                    n_iters,
                    logger,
                    samp_ran
                    ):
    model.train()
    loss_u_meter = AverageMeter()
    loss_prop_meter = AverageMeter()
    thre_meter = AverageMeter()
    kl_meter = AverageMeter()
    kl_hard_meter = AverageMeter()
    loss_contrast_meter = AverageMeter()
    # the number of correct pseudo-labels
    n_correct_u_lbs_meter = AverageMeter()
    # the number of confident unlabeled data
    n_strong_aug_meter = AverageMeter()
    mask_meter = AverageMeter()
    # the number of edges in the pseudo-label graph
    pos_meter = AverageMeter()
    samp_lb_meter, samp_p_meter = [], []
    for i in range(0, bagsize):
        x = CurrentValueMeter()
        y = CurrentValueMeter()
        samp_lb_meter.append(x)
        samp_p_meter.append(y)
    epoch_start = time.time()  # start time
    dl_u = iter(dltrain_u)
    n_iter = len(dltrain_u)

    for it in range(len(dltrain_u)):
        (var1, var2, var3, var4, var5) = next(dl_u)
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

        imsw, labels_real, labels_idx = [], [], []  # $$$$$$$$$$$$$

        for i in range(length):
            imsw.append(ims_u_weak1[i])
            labels_real.append(var3[i])
            labels_idx.append(var4[i])
        ims_u_weak = torch.cat(imsw, dim=0)
        lbs_u_real = torch.cat(labels_real, dim=0)
        label_proportions = [[] for _ in range(length)]
        lbs_u_real = lbs_u_real.cuda()
        lbs_idx = torch.cat(labels_idx, dim=0)
        lbs_idx = lbs_idx.cuda()

        positions = torch.nonzero(lbs_idx == 37821).squeeze()

        if positions.numel() != 0:
            head = positions - positions % bagsize
            rear = head + bagsize - 1

        for i in range(length):
            labels = []
            for j in range(n_classes):
                labels.append(var2[j][i])
            label_proportions[i].append(labels)

        # --------------------------------------
        btu = ims_u_weak.size(0)
        #ims_u_weak = ims_u_weak.permute(0, 2, 1, 3)

        bt = 0
        imgs = torch.cat([ims_u_weak], dim=0).cuda()
        logits = model(imgs)

        # logits_x = logits[:bt]
        logits_u_w = torch.split(logits[0:], btu)
        logits_u_w = logits_u_w[0]


        # loss_x = criteria_x(logits_x, lbs_x)

        chunk_size = len(logits_u_w) // length
        batch_size = length
        chunks = [logits_u_w[i * chunk_size:(i + 1) * chunk_size] for i in range(length)]

        proportion = torch.empty((0, n_classes), dtype=torch.float64).cuda()
        batch_size = length

        for i in range(length):
            pr = label_proportions[i][0]
            pr = torch.stack(pr).cuda()
            proportion = torch.cat((proportion, pr.unsqueeze(0)))
        proportion = proportion.view(length, n_classes, 1)
        proportion = proportion.squeeze(-1)
        proportion = proportion.double()
        loss_prop = torch.Tensor([]).cuda()
        loss_prop = loss_prop.double()
        kl_divergence = torch.Tensor([]).cuda()
        kl_divergence = kl_divergence.double()
        kl_divergence_hard = torch.Tensor([]).cuda()
        kl_divergence_hard = kl_divergence_hard.double()
        for i, chunk in enumerate(chunks):
            labels_p = torch.softmax(chunk, dim=1)
            scores, lbs_u_guess = torch.max(labels_p, dim=1)
            #opt_onehot = solve_optimal_onehot_with_proportions_torch(labels_p, proportion[i], bagsize, n_classes).float()
            #opt_onehot=opt_onehot.cuda()
            labels_p = torch.mean(labels_p, dim=0)

            loss_p = llp_loss(proportion[i], labels_p)

            #loss_p = compute_CC_loss_logDP(labels_p,proportion[i])
            #loss_p= compute_CC_loss_simplified_gpu(labels_p,proportion[i])

            label_prop = torch.tensor(label_proportions[i], dtype=torch.float64).cuda()
            loss_prop = torch.cat((loss_prop, loss_p.view(1)))

            label_prop += 1e-9
            labels_p += 1e-9
            log_labels_p = torch.log(labels_p)
            one_hot_matrix = F.one_hot(lbs_u_guess, num_classes=n_classes)
            one_hot_matrix = one_hot_matrix.float()
            one_hot_matrix = torch.mean(one_hot_matrix, dim=0)

            one_hot_matrix += 1e-9
            log_one_hot_matrix = torch.log(one_hot_matrix)

            kl_soft = F.kl_div(log_labels_p, label_prop, reduction='batchmean')

            kl_hard = F.kl_div(log_one_hot_matrix, label_prop, reduction='batchmean')
            kl_divergence = torch.cat((kl_divergence, kl_soft.view(1)))
            kl_divergence_hard = torch.cat((kl_divergence_hard, kl_hard.view(1)))
        kl_divergence = kl_divergence.mean()
        kl_divergence_hard = kl_divergence_hard.mean()
        loss_prop = loss_prop.mean()
        probs = torch.softmax(logits_u_w, dim=1)
        probs = probs.mean(dim=0)
        prior = torch.full_like(probs, 0.1).detach()
        prior = proportion.mean(dim=0).detach()

        loss_debais = llp_loss(prior, probs)
        x=loss_prop * bagsize
        loss = loss_prop
        with torch.no_grad():

            probs = torch.softmax(logits_u_w, dim=1)

            """
            max_probs, max_idx = torch.max(probs, dim=-1)
            # mask = max_probs.ge(p_cutoff * (class_acc[max_idx] + 1.) / 2).float()  # linear
            # mask = max_probs.ge(p_cutoff * (1 / (2. - class_acc[max_idx]))).float()  # low_limit
            mask = max_probs.ge(0.2+0.75 * (classwise_acc[max_idx] / (2. - classwise_acc[max_idx]))).float()  # convex
            thre=0.2+0.75 * (classwise_acc[max_idx] / (2. - classwise_acc[max_idx]))

            thre_col = thre.view(-1, 1)
            thre_row = thre.view(1, -1)

            thre = torch.mm(thre_col, thre_row)
            delta = thre + (1 - thre) / (n_classes-1) * (1 - thre)
            thre=delta

            # mask = max_probs.ge(p_cutoff * (torch.log(class_acc[max_idx] + 1.) + 0.5)/(math.log(2) + 0.5)).float()  # concave
            select = max_probs.ge(args.thr).long()
            pseudo_lb=max_idx.long()
            pseudo_lb=pseudo_lb.cuda()
            if lbs_idx[select == 1].nelement() != 0:
                selected_label[lbs_idx[select == 1]] = pseudo_lb[select == 1]


            """
            # DA
            """
            prob_list.append(probs.mean(0))
            if len(prob_list)>32:
                prob_list.pop(0)
            prob_avg = torch.stack(prob_list,dim=0).mean(0)
            probs = probs / prob_avg
            probs = probs / probs.sum(dim=1, keepdim=True)
            """


            """
            probs_orig = probs.clone()

            if epoch>0 or it>args.queue_batch: # memory-smoothing
                A = torch.exp(torch.mm(feats_u_w, queue_feats.t())/args.temperature)
                A = A/A.sum(1,keepdim=True)
                probs = args.alpha*probs + (1-args.alpha)*torch.mm(A, queue_probs)
            """

            scores, lbs_u_guess = torch.max(probs, dim=1)
            mask = scores.ge(args.thr).float()
            if positions.numel() != 0:
                for i in range(0, bagsize):
                    samp_lb_meter[i].update(lbs_u_guess[head + i].item())
                    samp_p_meter[i].update(scores[head + i].item())
            """
            feats_w=feats_u_w
            probs_w=probs_orig

            # update memory bank
            n = bt+btu
            queue_feats[queue_ptr:queue_ptr + n,:] = feats_w
            queue_probs[queue_ptr:queue_ptr + n,:] = probs_w
            queue_ptr = (queue_ptr+n)%args.queue_size
            """

        optim.zero_grad()
        loss.backward()
        optim.step()
        lr_schdlr.step()

        if args.eval_ema:
            with torch.no_grad():
                ema_model_update(model, ema_model, args.ema_m)
        loss_prop_meter.update(loss.item())
        mask_meter.update(mask.mean().item())
        kl_meter.update(kl_divergence.mean().item())
        kl_hard_meter.update(kl_divergence_hard.mean().item())
        corr_u_lb = (lbs_u_guess == lbs_u_real).float() * mask
        n_correct_u_lbs_meter.update(corr_u_lb.sum().item())
        n_strong_aug_meter.update(mask.sum().item())

        if (it + 1) % n_iter == 0:
            t = time.time() - epoch_start

            lr_log = [pg['lr'] for pg in optim.param_groups]
            lr_log = sum(lr_log) / len(lr_log)
            logger.info("{}-x{}-s{}, {} | epoch:{}, iter: {}.  loss: {:.3f}. kl: {:.3f}. kl_hard:{:.3f}."
                        "LR: {:.3f}. Time: {:.2f}".format(
                args.dataset, args.n_labeled, args.seed, args.exp_dir, epoch, it + 1, loss_prop_meter.avg, kl_meter.avg,
                kl_hard_meter.avg, lr_log, t))

            epoch_start = time.time()

    return loss_prop_meter.avg, n_correct_u_lbs_meter.avg, n_strong_aug_meter.avg, mask_meter.avg, kl_meter.avg, kl_hard_meter.avg


def evaluate(model, ema_model, dataloader):
    model.eval()

    top1_meter = AverageMeter()
    ema_top1_meter = AverageMeter()
    top5_meter = AverageMeter()
    ema_top5_meter = AverageMeter()
    loss_meter = AverageMeter()

    with torch.no_grad():
        for ims, lbs in dataloader:
            ims = ims.cuda()
            #ims = ims.permute(0, 2, 1, 3)

            lbs = lbs.cuda()

            logits = model(ims)
            loss = torch.nn.CrossEntropyLoss()(logits, lbs)

            loss_meter.update(loss.item())
            scores = torch.softmax(logits, dim=1)
            top1, top5 = accuracy(scores, lbs, (1, 5))
            top1_meter.update(top1.item())
            top5_meter.update(top5.item())

            if ema_model is not None:
                logits = ema_model(ims)
                scores = torch.softmax(logits, dim=1)
                top1, top5 = accuracy(scores, lbs, (1, 5))
                ema_top1_meter.update(top1.item())

    return top1_meter.avg, ema_top1_meter.avg, top5_meter.avg, ema_top5_meter.avg, loss_meter.avg


def main():
    parser = argparse.ArgumentParser(description='DLLP Cifar Training')
    parser.add_argument('--root', default='./data', type=str, help='dataset directory')
    parser.add_argument('--wresnet-k', default=2, type=int,
                        help='width factor of wide resnet')
    parser.add_argument('--wresnet-n', default=28, type=int,
                        help='depth of wide resnet')
    parser.add_argument('--dataset', type=str, default="SVHN",
                        help='number of classes in dataset')
    parser.add_argument('--n-classes', type=int, default=10,
                        help='number of classes in dataset')
    parser.add_argument('--n-labeled', type=int, default=10,
                        help='number of labeled samples for training')
    parser.add_argument('--n-epoches', type=int, default=1024,
                        help='number of training epoches')
    parser.add_argument('--batchsize', type=int, default=32,
                        help='train batch size of bag samples')
    parser.add_argument('--bagsize', type=int, default=16,
                        help='train bag size of samples')
    parser.add_argument('--n-imgs-per-epoch', type=int, default=1024,
                        help='number of training images for each epoch')

    parser.add_argument('--eval-ema', default=False, help='whether to use ema model for evaluation')
    parser.add_argument('--ema-m', type=float, default=0.999)

    parser.add_argument('--lam-u', type=float, default=1.,
                        help='c oefficient of unlabeled loss')
    parser.add_argument('--lr', type=float, default=0.03,
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
    parser.add_argument('--lam-p', type=float, default=2,
                        help='coefficient of proportion loss')
    parser.add_argument('--contrast-th', default=0.8, type=float,
                        help='pseudo label graph threshold')
    parser.add_argument('--thr', type=float, default=0.95,
                        help='pseudo label threshold')
    parser.add_argument('--alpha', type=float, default=0.9)
    parser.add_argument('--queue_batch', type=float, default=5,
                        help='number of batches stored in memory bank')
    parser.add_argument('--exp-dir', default='DLLP', type=str, help='experiment id')
    parser.add_argument('--checkpoint', default='', type=str, help='use pretrained model')
    parser.add_argument('--folds', default='2', type=str, help='number of dataset')
    args = parser.parse_args()

    logger, output_dir = setup_default_logging(args)
    logger.info(dict(args._get_kwargs()))

    tb_logger = tensorboard_logger.Logger(logdir=output_dir, flush_secs=2)
    samp_ran = 37821
    if args.seed > 0:
        torch.manual_seed(args.seed)
        random.seed(args.seed)
        np.random.seed(args.seed)

    n_iters_per_epoch = args.n_imgs_per_epoch  # 1024

    logger.info("***** Running training *****")
    logger.info(f"  Task = {args.dataset}@{args.n_labeled}")

    model, criteria_x, criteria_u, ema_model = set_model(args)
    logger.info("Total params: {:.2f}M".format(
        sum(p.numel() for p in model.parameters()) / 1e6))

    dltrain_u, dataset_length = get_train_loader(args.n_classes,
                                                 args.dataset, args.batchsize, args.bagsize, root=args.root,
                                                 method='DLLP',
                                                 supervised=False)
    dlval = get_val_loader(dataset=args.dataset, batch_size=64, num_workers=2, root=args.root)
    n_iters_all = len(dltrain_u) * args.n_epoches
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
    queue_feats = torch.zeros(args.queue_size, args.low_dim).cuda()
    queue_probs = torch.zeros(args.queue_size, args.n_classes).cuda()
    queue_ptr = 0

    # for distribution alignment
    prob_list = []

    train_args = dict(
        model=model,
        ema_model=ema_model,
        prob_list=prob_list,
        criteria_x=criteria_x,
        criteria_u=criteria_u,
        optim=optim,
        lr_schdlr=lr_schdlr,
        dltrain_u=dltrain_u,
        args=args,
        n_iters=n_iters_per_epoch,
        logger=logger
    )

    best_acc = -1
    best_acc_5 = -1
    best_epoch_5 = 0

    best_epoch = 0

    logger.info('-----------start training--------------')
    for epoch in range(args.n_epoches):
        loss_prob, n_correct_u_lbs, n_strong_aug, mask_mean, num_pos, samp_lb = \
            train_one_epoch(epoch, bagsize=args.bagsize, n_classes=args.n_classes, **train_args, samp_ran=samp_ran,
                            )

        top1, ema_top1, top5, ema_top5, loss_test = evaluate(model, ema_model, dlval)
        tb_logger.log_value('loss_prob', loss_prob, epoch)
        if (n_strong_aug == 0):
            tb_logger.log_value('guess_label_acc', 0, epoch)
        else:
            tb_logger.log_value('guess_label_acc', n_correct_u_lbs / n_strong_aug, epoch)
        tb_logger.log_value('test_acc', top1, epoch)
        tb_logger.log_value('mask', mask_mean, epoch)

        tb_logger.log_value('loss_test', loss_test, epoch)
        if best_acc < top1:
            best_acc = top1
            best_epoch = epoch
        if best_acc_5 < top5:
            best_acc_5 = top5
            best_epoch_5 = epoch
        logger.info(
            "Epoch {}.loss_test: {:.4f}. Acc: {:.4f}. Ema-Acc: {:.4f}. best_acc: {:.4f} in epoch{},Acc_5: {:.4f}.  best_acc_5: {:.4f} in epoch{},".
            format(epoch, loss_test, top1, ema_top1, best_acc, best_epoch, top5, best_acc_5, best_epoch_5))

        if epoch % 1000 == 0:
            save_obj = {
                'model': model.state_dict(),
                'optimizer': optim.state_dict(),
                'lr_scheduler': lr_schdlr.state_dict(),
                'prob_list': prob_list,
                'queue': {'queue_feats': queue_feats, 'queue_probs': queue_probs, 'queue_ptr': queue_ptr},
                'epoch': epoch,
            }
            torch.save(save_obj, os.path.join(output_dir, 'checkpoint_%02d.pth' % epoch))


if __name__ == '__main__':
    main()