import numpy as np
from random import sample

import torch
import torch.nn as nn
import torch.nn.functional as F





class PaPi(nn.Module):
    def __init__(self, args, base_encoder):
        super().__init__()

        pretrained = False  # 指定是否使用预训练模型，默认为否

        self.proto_weight = args.proto_m  # 设置原型权重

        # 初始化编码器，根据参数设置编码器的结构和特性
        self.encoder = base_encoder(name=args.arch, head='mlp', feat_dim=args.low_dim, num_class=args.num_class,
                                    pretrained=pretrained)

        # 注册一个缓冲区，用于存储原型，原型的大小是类别数乘以特征维度
        self.register_buffer("prototypes", torch.zeros(args.num_class, args.low_dim))

    def set_prototype_update_weight(self, epoch, args):
        start = args.pro_weight_range[0]  # 起始原型更新权重
        end = args.pro_weight_range[1]  # 结束原型更新权重
        # 根据当前epoch计算原型权重
        self.proto_weight = 1. * epoch / args.epochs * (end - start) + start

    def forward(self, img_q, img_k=None, img_q_mix=None, img_k_mix=None, partial_Y=None, Y_true=None, args=None,
                eval_only=False):

        output_q, q = self.encoder(img_q)  # 通过编码器处理查询图像

        if eval_only:
            # 如果是评估模式，复制原型并计算评估时的原型逻辑值
            prototypes_eval = self.prototypes.clone().detach()
            logits_prot_test = torch.mm(q, prototypes_eval.t())
            return output_q, logits_prot_test

        output_k, k = self.encoder(img_k)  # 通过编码器处理键图像

        output_q_mix, q_mix = self.encoder(img_q_mix)  # 通过编码器处理混合后的查询图像
        output_k_mix, k_mix = self.encoder(img_k_mix)  # 通过编码器处理混合后的键图像

        # 计算查询图像的预测分数，并进行归一化处理
        predicetd_scores_q = torch.softmax(output_q, dim=1) * partial_Y
        predicetd_scores_q_norm = predicetd_scores_q / predicetd_scores_q.sum(dim=1).repeat(args.num_class,
                                                                                            1).transpose(0, 1)

        # 计算键图像的预测分数，并进行归一化处理
        predicetd_scores_k = torch.softmax(output_k, dim=1) * partial_Y
        predicetd_scores_k_norm = predicetd_scores_k / predicetd_scores_k.sum(dim=1).repeat(args.num_class,
                                                                                            1).transpose(0, 1)

        # 获取查询和键图像的最大预测分数和对应的伪标签
        max_scores_q, pseudo_labels_q = torch.max(predicetd_scores_q_norm, dim=1)
        max_scores_k, pseudo_labels_k = torch.max(predicetd_scores_k_norm, dim=1)

        # 克隆原型以便更新
        prototypes = self.prototypes.clone().detach()

        # 计算查询和键特征与原型的逻辑值
        logits_prot_q = torch.mm(q, prototypes.t())
        logits_prot_k = torch.mm(k, prototypes.t())

        logits_prot_q_mix = torch.mm(q_mix, prototypes.t())
        logits_prot_k_mix = torch.mm(k_mix, prototypes.t())

        # 更新原型，根据预测的伪标签和特征
        for feat_q, label_q in zip(concat_all_gather(q), concat_all_gather(pseudo_labels_q)):
            self.prototypes[label_q] = self.proto_weight * self.prototypes[label_q] + (1 - self.proto_weight) * feat_q

        for feat_k, label_k in zip(concat_all_gather(k), concat_all_gather(pseudo_labels_k)):
            self.prototypes[label_k] = self.proto_weight * self.prototypes[label_k] + (1 - self.proto_weight) * feat_k

        # 归一化原型
        self.prototypes = F.normalize(self.prototypes, p=2, dim=1)

        # 返回所有输出
        return output_q, output_k, logits_prot_q, logits_prot_k, logits_prot_q_mix, logits_prot_k_mix
