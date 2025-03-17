
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
from resnet import PaPiNet

def set_model(args):
    model = WideResnet(n_classes=args.n_classes, k=args.wresnet_k, n=args.wresnet_n, proj=True)
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint)
        msg = model.load_state_dict(checkpoint, strict=False)
        assert set(msg.missing_keys) == {"classifier.weight", "classifier.bias"}
        print('loaded from checkpoint: %s' % args.checkpoint)
    model.train()
    model.cuda()

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
    x = labels_proportion

    # Ensure y is also double

    y = y.double()
    cross_entropy = torch.sum(-x * (torch.log(y) + 1e-7))

    return cross_entropy

@torch.no_grad()
def distributed_sinkhorn(out, args, proportion):
    Q = out
    K = Q.shape[0]  # 获取原型数量，即 Q 的行数
    B = Q.shape[1]  # 获取样本数量，即 Q 的列数

    for it in range(args.sinkhorn_iterations):
        # 归一化每一行：每个原型的总权重必须为 1
        sum_of_rows = torch.sum(Q, dim=1, keepdim=True)
        sum_of_rows[sum_of_rows == 0] = 1
        Q /= sum_of_rows

        # 归一化每一列：每个样本的总权重必须按 proportion[i] 分配
        sum_of_cols = torch.sum(Q, dim=0, keepdim=True)
        # 避免除以零：将总和为零的列的总和替换为 1
        sum_of_cols[sum_of_cols == 0] = 1
        Q /= sum_of_cols
        Q *= proportion

    # 为了保持比例不变，再次进行归一化
    sum_of_cols = torch.sum(Q, dim=0, keepdim=True)
    # 同样的处理，避免除以零
    sum_of_cols[sum_of_cols == 0] = 1
    Q /= sum_of_cols
    Q *= proportion

    return Q  # 返回转置后的 Q



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


def thre_ema(thre, sum_values, ema):
    return thre * ema + (1 - ema) * sum_values


def weight_decay_with_mask(mask, initial_weight, max_mask_count):
    mask_count = mask.sum().item()  # 计算当前 mask 中的元素数量
    weight_decay = max(0, 1 - mask_count / max_mask_count)  # 线性衰减
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
                    queue_feats,
                    queue_probs,
                    queue_ptr,
                    selected_label,
                    classwise_acc,
                    samp_ran
                    ):
    model.train()
    loss_u_meter = AverageMeter()
    loss_prop_meter = AverageMeter()
    thre_meter = AverageMeter()
    loss_contrast_meter = AverageMeter()
    # the number of correct pseudo-labels
    n_correct_u_lbs_meter = AverageMeter()
    # the number of confident unlabeled data
    n_strong_aug_meter = AverageMeter()
    mask_meter = AverageMeter()
    mask_to_meter =AverageMeter()
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
        (var1, var2, var3, var4,var5) = next(dl_u)
        length = len(var2[0])

        """
        pseudo_counter = Counter(selected_label.tolist())
        for i in range(args.n_classes):
            classwise_acc[i] = pseudo_counter[i] / max(pseudo_counter.values())

        """
        ims_u_weak1, ims_u_strong01 = var1

        imsw, imss0, labels_real, labels_idx = [], [], [], []
        for i in range(length):
            imsw.append(ims_u_weak1[i])
            imss0.append(ims_u_strong01[i])
            labels_real.append(var3[i])
            labels_idx.append(var4[i])
        ims_u_weak = torch.cat(imsw, dim=0)
        ims_u_strong0 = torch.cat(imss0, dim=0)
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
        bt = 0
        imgs = torch.cat([ims_u_weak, ims_u_strong0], dim=0).cuda()
        logits, features = model(imgs)
        # logits_x = logits[:bt]
        logits_u_w, logits_u_s0 = torch.split(logits[0:], btu)

        # feats_x = features[:bt]
        feats_u_w, feats_u_s0 = torch.split(features[0:], btu)

        # loss_x = criteria_x(logits_x, lbs_x)
        chunk_size = len(logits_u_w) // length

        # 分成 length 节
        chunks = [logits_u_w[i * chunk_size:(i + 1) * chunk_size] for i in range(length)]

        # 打印分成的各节数据

        # 创建一个空的 PyTorch 向量用于保存 loss_p
        loss_prop = torch.Tensor([]).cuda()
        loss_prop = loss_prop.double()
        for i, chunk in enumerate(chunks):
            labels_p = torch.softmax(chunk, dim=1)
            labels_p_mean = torch.mean(labels_p, dim=0)
            label_proportion = torch.tensor(label_proportions[i], dtype=torch.float64).cuda()
            loss_p = llp_loss(label_proportion, labels_p_mean)

                # 将 loss_p 添加到 all_loss_p 中
            loss_prop = torch.cat((loss_prop, loss_p.view(1)))

            # all_loss_p 包含了每个 loss_p

        loss_prop = loss_prop.mean()
        with torch.no_grad():
            logits_u_w_1 = logits_u_w.detach()

            # feats_x = feats_x.detach()
            feats_u_w = feats_u_w.detach()

            probs = torch.softmax(logits_u_w_1, dim=1)
            fl = 0
            new_probs = []  # 存储修改后的 probs 块

            for i in range(0, probs.size(0), args.bagsize):
                # 获取当前块
                prob_chunk = probs[i:i + args.bagsize]

                # 获取当前块的列约束比例并调整
                lap = torch.tensor(label_proportions[fl], dtype=torch.float64).cuda()
                lap = lap * args.bagsize

                # 使用 distributed_sinkhorn 函数处理每个块
                adjusted_chunk = distributed_sinkhorn(prob_chunk, args, lap)
                new_probs.append(adjusted_chunk)

                fl += 1

            # 将所有处理过的块重新组合成一个新的 probs 张量
            new_probs = torch.cat(new_probs, dim=0)
            scores, lbs_u_guess = torch.max(new_probs, dim=1)
            mask = scores.ge(args.thr).float()

        """
        伪标签提取更换为纯弱监督
        """
        # pseudo-label graph with self-loop

        # 假设probs是一个[N, 10]的矩阵
        # 例如：probs = torch.rand(5, 10)


        # loss_Ein=custom_loss(probs)
        # unsupervised classification loss

        loss_u = (criteria_u(logits_u_w, lbs_u_guess)).mean()
        loss = args.lam_p * loss_prop + args.lam_u * loss_u
        optim.zero_grad()
        loss.backward()
        optim.step()
        lr_schdlr.step()

        if args.eval_ema:
            with torch.no_grad():
                ema_model_update(model, ema_model, args.ema_m)
        loss_prop_meter.update(loss_prop.item())
        loss_u_meter.update(loss_u.item())
        mask_meter.update(mask.mean().item())
        corr_u_lb = (lbs_u_guess == lbs_u_real).float() *mask
        n_correct_u_lbs_meter.update(corr_u_lb.sum().item())
        n_strong_aug_meter.update(mask.sum().item())
        if (it + 1) % n_iter == 0:
            t = time.time() - epoch_start

            lr_log = [pg['lr'] for pg in optim.param_groups]
            lr_log = sum(lr_log) / len(lr_log)
            logger.info(
                "{}-x{}-s{}, {} | epoch:{}, iter: {}. "
                "loss_u: {:.3f}. loss_prop: {:.3f}.  "
                "n_correct_u: {:.2f}/{:.2f}. Mask_to:{:.3f}. Mask:{:.3f}. Thre:{:.3f}. LR: {:.3f}. Time: {:.2f}. ".format(
                    args.dataset, args.n_labeled, args.seed, args.exp_dir, epoch, it + 1,
                    loss_u_meter.avg, loss_prop_meter.avg,
                    n_correct_u_lbs_meter.avg, n_strong_aug_meter.avg, mask_to_meter.avg,
                    mask_meter.avg, thre_meter.avg, lr_log, t
                )
            )

            epoch_start = time.time()

    return loss_u_meter.avg, loss_prop_meter.avg, n_correct_u_lbs_meter.avg, n_strong_aug_meter.avg, mask_meter.avg, pos_meter.avg, samp_lb_meter, samp_p_meter, queue_feats, queue_probs, queue_ptr, prob_list, selected_label, classwise_acc
def evaluate(model, ema_model, dataloader):
    model.eval()

    top1_meter = AverageMeter()
    ema_top1_meter = AverageMeter()

    with torch.no_grad():
        for ims, lbs in dataloader:
            ims = ims.cuda()
            lbs = lbs.cuda()

            logits, _ = model(ims)
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
    torch.cuda.set_device(0)
    parser = argparse.ArgumentParser(description='ROT Cifar Training')
    parser.add_argument('--root', default='./data', type=str, help='dataset directory')
    parser.add_argument('--wresnet-k', default=2, type=int,
                        help='width factor of wide resnet')
    parser.add_argument('--wresnet-n', default=28, type=int,
                        help='depth of wide resnet')
    parser.add_argument('--dataset', type=str, default='CIFAR10',
                        help='number of classes in dataset')
    parser.add_argument('--n-classes', type=int, default=10,
                        help='number of classes in dataset')
    parser.add_argument('--n-labeled', type=int, default=10,
                        help='number of labeled samples for training')
    parser.add_argument('--n-epoches', type=int, default=1024,
                        help='number of training epoches')
    parser.add_argument('--batchsize', type=int, default=64,
                        help='train batch size of bag samples')
    parser.add_argument('--bagsize', type=int, default=16,
                        help='train bag size of samples')
    parser.add_argument('--n-imgs-per-epoch', type=int, default=1024,
                        help='number of training images for each epoch')

    parser.add_argument('--sinkhorn-iterations', type=float, default=25)
    parser.add_argument('--eval-ema', default=True, help='whether to use ema model for evaluation')
    parser.add_argument('--ema-m', type=float, default=0.999)

    parser.add_argument('--lam-u', type=float, default=1,
                        help='coefficient of unlabeled loss')
    parser.add_argument('--lr', type=float, default=0.05,
                        help='learning rate for training')
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                        help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum for optimizer')
    parser.add_argument('--seed', type=int, default=3,
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
    parser.add_argument('--alpha', type=float, default=0.90)
    parser.add_argument('--queue_batch', type=float, default=5,
                        help='number of batches stored in memory bank')
    parser.add_argument('--exp-dir', default='ROT', type=str, help='experiment id')
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
                                                 method='L^2P-AHIL',
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
    best_epoch = 0
    logger.info('-----------start training--------------')
    selected_label = torch.ones((1 * args.bagsize), dtype=torch.long, ) * -1
    selected_label = selected_label.cuda()
    classwise_acc = torch.zeros((args.n_classes,)).cuda()
    for epoch in range(args.n_epoches):
        loss_u, loss_prob, n_correct_u_lbs, n_strong_aug, mask_mean, num_pos, samp_lb, samp_p, queue_feats, queue_probs, queue_ptr, prob_list, selected_label, classwise_acc = \
            train_one_epoch(epoch, bagsize=args.bagsize, n_classes=args.n_classes, **train_args,
                            queue_feats=queue_feats, queue_probs=queue_probs, queue_ptr=queue_ptr,
                            selected_label=selected_label, classwise_acc=classwise_acc, samp_ran=samp_ran)

        top1, ema_top1 = evaluate(model, ema_model, dlval)
        tb_logger.log_value('loss_u', loss_u, epoch)
        tb_logger.log_value('loss_prob', loss_prob, epoch)
        if (n_strong_aug == 0):
            tb_logger.log_value('guess_label_acc', 0, epoch)
        else:
            tb_logger.log_value('guess_label_acc', n_correct_u_lbs / n_strong_aug, epoch)
        tb_logger.log_value('test_acc', top1, epoch)
        tb_logger.log_value('test_ema_acc', ema_top1, epoch)
        tb_logger.log_value('mask', mask_mean, epoch)
        tb_logger.log_value('num_pos', num_pos, epoch)
        """
        for i in range(0, args.bagsize):
            tb_logger.log_value(f'samp_lb_meter_{i}', samp_lb[i].val, epoch)  # 使用适当的属性来获取值
            tb_logger.log_value(f'samp_p_meter_{i}', samp_p[i].val, epoch)
        """
        if best_acc < top1:
            best_acc = top1
            best_epoch = epoch

        logger.info("Epoch {}. Acc: {:.4f}. Ema-Acc: {:.4f}. best_acc: {:.4f} in epoch{}".
                    format(epoch, top1, ema_top1, best_acc, best_epoch))

        if epoch % 10 == 0:
            save_obj = {
                'model': model.state_dict(),
                'ema_model': ema_model.state_dict(),
                'optimizer': optim.state_dict(),
                'lr_scheduler': lr_schdlr.state_dict(),
                'prob_list': prob_list,
                'queue': {'queue_feats': queue_feats, 'queue_probs': queue_probs, 'queue_ptr': queue_ptr},
                'epoch': epoch,
            }
            torch.save(save_obj, os.path.join(output_dir, 'checkpoint_%02d.pth' % epoch))


if __name__ == '__main__':
    main()