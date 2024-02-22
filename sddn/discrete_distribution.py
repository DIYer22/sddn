#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 19 08:37:01 2023

@author: yanglei

存放专有模块
"""

import boxx
import random
import numpy as np
from boxx import mg
import torch
from torch import nn

eps = 1e-20


class SplitableDiscreteDistribution:
    # TODO 编程 nn.module named_buffers
    def __init__(self, k):
        self.k = k
        self.i_to_idx = np.arange(k)
        self.count = np.zeros(k)
        self.near2 = np.zeros((k, k))
        self.loss_acc = np.zeros(k)
        self.idx_max = k - 1
        self.iter = 0
        self.split_iters = []
        self.batchn = 0

    def add_loss_matrix(self, loss_matrix):
        # b x k
        if isinstance(loss_matrix, torch.Tensor):
            loss_matrix = loss_matrix.detach().cpu().numpy()
        b, k = loss_matrix.shape
        if k == 2:
            i_near_near2s = np.zeros((b, 2), np.int64)
            i_near_near2s[:, 1] = 1
        else:
            i_near_near2s = np.argpartition(loss_matrix, 2)[:, :2]

        i_nears = np.argmin(loss_matrix, 1)

        i_near2s = i_near_near2s[i_near_near2s != i_nears[:, None]]

        # loss_mins = loss_matrix[i_nears]
        self.iter += b
        self.batchn += 1
        unique_, count_ = np.unique(i_nears, return_counts=True)
        self.count[unique_] += count_
        # TODO to vector
        for i_near, i_near2 in zip(i_nears, i_near2s):
            self.near2[i_near, i_near2] += 1
        self.loss_acc += loss_matrix.sum(0)
        return dict(i_nears=i_nears)

    def try_split(self):
        count = self.count
        i_split = np.argmax(count)
        i_disapear = np.argmin(count)
        if count[i_disapear] == 0:
            i_disapear = np.argmax((count == 0) * self.loss_acc)
        ps = count / count.sum()
        ps, pd, P = (
            # count[i_split] / self.iter,
            # count[i_disapear] / self.iter,
            ps[i_split],
            ps[i_disapear],
            1 / self.k,
        )
        # 用 Total Variation 简化的一个子集: ps-p + p-pd >  |ps/2-p| * 2 + pd => ps - pd > |2p-ps| + pd
        # 简化近似: 消失节点处的 output 概率直接置 0 而 GT 仍然为原来的值 pd
        tv_loss_now = ps - pd
        tv_loss_splited = abs(ps / 2 - P) * 2 + pd
        # mg()
        if self.iter > getattr(self, "split_start", 0) and (
            tv_loss_splited < tv_loss_now or pd < P / 2
        ):
            # split_rate = 4
            # if self.iter and (ps > P * split_rate or pd < P / split_rate):
            self.split_iters.append(self.iter)
            d = self.split(i_split, i_disapear)
            d["ps"], d["pd"] = ps, pd
            return d

    def split(self, i_split, i_disapear):
        """"""
        # old_near2 = self.near2.copy()
        if "wo.near2" and 0 or self.k == 2:  # or fix_near2
            # eps = 0.01
            self.near2 = 1 - np.eye(self.k)
            self.near2 = self.near2 / self.near2.sum() * self.iter
        near2_sum = self.near2[i_disapear].sum()

        def get_weight_per_row():
            weight_matrix = self.near2 / np.sum(self.near2, axis=1, keepdims=True)
            idx_finite_2d = np.isfinite(weight_matrix)
            weight_matrix = np.where(idx_finite_2d, weight_matrix, 1 / (self.k - 1))
            weight_matrix[~idx_finite_2d.all(), i_disapear] = 0
            return weight_matrix

        if near2_sum:
            self.near2[i_disapear][i_disapear] = 0
            not_self_sum = self.near2[i_disapear].sum()
            if not_self_sum:
                i_to_weight = self.near2[i_disapear] / not_self_sum
            else:
                i_to_weight = np.ones(self.k) / (self.k - 1)
                i_to_weight[i_disapear] = 0

            self.count += self.count[i_disapear] * i_to_weight
            self.count[i_disapear] = 0
            self.near2 += (near2_sum * i_to_weight)[:, None] * get_weight_per_row()
            self.near2 += self.near2[:, i_disapear : i_disapear + 1] * i_to_weight[None]
            self.near2[:, i_disapear] = 0
            self.near2[i_disapear] = 0
        else:
            as_other_near2 = self.near2[:, i_disapear : i_disapear + 1].copy()
            as_other_near2_sum = as_other_near2.sum()
            if as_other_near2_sum:
                self.near2[:, i_disapear] = 0
                self.near2 += as_other_near2 * get_weight_per_row()
        # removed_near2 = self.near2.copy()
        # boxx.g()
        # assert np.allclose(removed_near2.sum(), old_near2.sum()), [removed_near2.sum(), old_near2.sum()]
        """"""
        self.idx_max += 1
        self.i_to_idx[i_split] = self.idx_max
        self.idx_max += 1
        self.i_to_idx[i_disapear] = self.idx_max
        self.count[i_split] = self.count[i_disapear] = self.count[i_split] / 2
        old_i_near2 = self.near2[i_split].sum()
        self.near2[i_split] = self.near2[i_disapear] = 0
        self.near2[:, i_split] = self.near2[:, i_disapear] = self.near2[:, i_split] / 2

        self.near2[i_split] = self.near2[i_disapear] = 0
        self.near2[i_split, i_disapear] = self.near2[i_disapear, i_split] = (
            old_i_near2 / 2
        )
        self.loss_acc[i_disapear] = self.loss_acc[i_split]

        # mg()  # /0
        # assert np.isfinite(self.near2).all()
        return dict(i_split=i_split, i_disapear=i_disapear)

    def plot_dist(self):
        print(self)
        print(
            "last 10 split@[%s]"
            % ", ".join(
                [
                    f"{round(it/self.iter*100, 1)}%"
                    for it in self.split_iters[-10:][::-1]
                ]
            )
        )
        boxx.plot(self.count / self.count.mean(), True)

    def __str__(self):
        return f"SDD(k={self.k}, splitn={len(self.split_iters)}, iter={self.iter}, last_split_iter={([-1]+self.split_iters)[-1]}, batchn={self.batchn})"

    __repr__ = __str__

    @classmethod
    def test(cls, k=2):
        import tqdm

        sdd = cls(k)
        b = 64
        batchn = 20000
        for batchi in tqdm.tqdm(range(batchn)):
            dm = np.random.rand(b, k) * np.linspace(0.59, 1.1, k)[None] ** 10
            sdd.add_loss_matrix(dm)
            split = sdd.try_split()
            if split:
                print(batchi, split)
            # tree - split

            boxx.g()
            assert np.allclose(sdd.iter, sdd.near2.sum()), [
                sdd.iter,
                sdd.near2.sum(),
                sdd.count.sum(),
            ]


def mse_loss_multi_output(input, target=None):
    # input (b, k, c, h, w) or (b, c, h, w)
    # target (b, c, h, w)
    if target is None:
        sub_diff = input
    else:
        is_multi_input = input.ndim != target.ndim
        if is_multi_input:
            target = target[:, None]
        # return (b, k) if is_multi_input else (b,)
        sub_diff = input - target
    return (sub_diff**2).mean((-1, -2, -3))


def l1_loss_multi_output(input, target=None):
    if target is None:
        sub_diff = input
    else:
        is_multi_input = input.ndim != target.ndim
        if is_multi_input:
            target = target[:, None]
        sub_diff = input - target
    return (torch.abs(sub_diff)).mean((-1, -2, -3))


def forward_one_predict(conv1x1, input, idx_k=None, predict_c=3):
    # Not work for training, PyTorch's compiler is smarter than me!
    # TODO Using in sample for fast genarte
    """
    Just forward one output which index is idx_k in k outputs as predict, instead of forward all k outputs
    """
    batch_size, c, h, w = input.shape
    k = conv1x1.out_channels // predict_c
    if isinstance(idx_k, torch.Tensor):
        idx_k = idx_k.detach().cpu().numpy()
    if idx_k is None:
        idx_k = np.random.randint(0, k, (batch_size,))
    dtype = input.dtype

    slicee = (idx_k[:, None] * predict_c + np.arange(0, predict_c)[None]).flatten()
    if isinstance(conv1x1, AdaptConv2d):
        conv1x1, adapt_conv = conv1x1.conv1x1, conv1x1
        conv_weight_adapt = adapt_conv.get_conv_weight_adapt(input)
        if conv_weight_adapt is not None:
            conv_weight_adapt_selected = conv_weight_adapt.view(
                batch_size, k, predict_c, c, 1, 1
            )[range(batch_size), idx_k]
            # predict = adapt_conv.adapt_conv2d(input, conv1x1.weight[slicee], conv_weight_adapt_selected)
            conv_weight_batch = conv1x1.weight[slicee].view(
                batch_size, predict_c, c, 1, 1
            ).to(dtype) + conv_weight_adapt_selected.view(
                batch_size, predict_c, c, 1, 1
            )
            return torch.einsum("bihw,boihw->bohw", [input, conv_weight_batch]).view(
                batch_size, predict_c, h, w
            )
            return predict.view(batch_size, predict_c, h, w)
    weight = conv1x1.weight[slicee]
    bias = None if conv1x1.bias is None else conv1x1.bias[slicee]
    if weight.dtype != dtype:
        weight, bias = weight.to(dtype), bias if bias is None else bias.to(dtype)

    assert conv1x1.padding_mode == "zeros", conv1x1.padding_mode
    assert conv1x1.groups == 1, conv1x1.groups
    if batch_size > 8:  # fast compute
        # "by groups"
        # view size is not compatible with input, and reshape will consume GPU memory(80MB)
        predict = torch.nn.functional.conv2d(
            input.reshape(1, batch_size * c, h, w),
            weight,
            bias,
            conv1x1.stride,
            conv1x1.padding,
            conv1x1.dilation,
            groups=batch_size,
        ).view(batch_size, predict_c, h, w)
    elif batch_size <= 8:  # save GPU memory
        # "for+cat" and 0: # slow 35% for batch64
        predict = torch.cat(
            [
                torch.nn.functional.conv2d(
                    input[ib : ib + 1],
                    weight[ib * predict_c : ib * predict_c + predict_c],
                    None
                    if bias is None
                    else bias[ib * predict_c : ib * predict_c + predict_c],
                    conv1x1.stride,
                    conv1x1.padding,
                    conv1x1.dilation,
                )
                for ib in range(batch_size)
            ],
            0,
        )
    elif 0:  # blanced but more complicated so give up
        # "conv+idx_eye"
        predict = torch.nn.functional.conv2d(
            input, weight, bias, conv1x1.stride, conv1x1.padding, conv1x1.dilation
        )[
            np.arange(batch_size).repeat(predict_c),
            (
                np.arange(batch_size)[:, None] * predict_c
                + np.arange(0, predict_c)[None]
            ).flatten(),
        ].view(
            batch_size, predict_c, h, w
        )
    return predict


class SplitableModuleMixin:
    def split(module, split_idxs, predict_c, optimizers=None):
        # split/update weight
        with torch.no_grad():
            i_split, i_disapear = split_idxs
            i_split, i_disapear = int(i_split), int(i_disapear)
            weight = module._parameters["weight"]  # (k*predict_c, last_c)
            weight[
                i_disapear * predict_c : i_disapear * predict_c + predict_c
            ] = weight[i_split * predict_c : i_split * predict_c + predict_c]
            assert module.bias is None

            # update optimizer
            if optimizers is not None:
                for optimizer in optimizers:
                    if weight in optimizer.state:
                        for k in optimizer.state[weight]:
                            if optimizer.state[weight][k].shape == weight.shape:
                                optimizer.state[weight][k][
                                    i_disapear * predict_c : i_disapear * predict_c
                                    + predict_c
                                ] = optimizer.state[weight][k][
                                    i_split * predict_c : i_split * predict_c
                                    + predict_c
                                ]


class Conv2dMixedPrecision(nn.Conv2d, SplitableModuleMixin):
    def forward(self, input):
        dtype = input.dtype
        if self.weight.dtype == dtype:
            return super().forward(input)
        weight, bias = self.weight.to(
            dtype
        ), None if self.bias is None else self.bias.to(dtype)
        assert self.padding_mode == "zeros", self.padding_mode
        return nn.functional.conv2d(
            input, weight, bias, self.stride, self.padding, self.dilation, self.groups
        )


class LinearMixedPrecision(nn.Linear, SplitableModuleMixin):
    def forward(self, input):
        dtype = input.dtype
        if input.ndim == 4:
            input = input.reshape(input.shape[0], input.shape[1])
        if self.weight.dtype == dtype:
            return super().forward(input)
        weight, bias = self.weight.to(
            dtype
        ), None if self.bias is None else self.bias.to(dtype)
        return nn.functional.linear(input, weight, bias)


class AdaptConv2d(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=1,
        bias=False,
        se_hidden_layers=0,
        se_reduction=16,
    ):
        super().__init__()
        self.out_channels = out_channels
        self.se_hidden_layers = se_hidden_layers
        self.conv1x1 = Conv2dMixedPrecision(
            in_channels, out_channels, kernel_size=kernel_size, bias=bias
        )
        if se_hidden_layers:
            hiddenn = max(in_channels // se_reduction, 8)
            outputn = in_channels * out_channels
            seqs = [
                nn.AdaptiveAvgPool2d(1),
                LinearMixedPrecision(in_channels, hiddenn, bias=False),
                nn.SiLU(inplace=True),
            ]
            [
                seqs.extend(
                    [
                        LinearMixedPrecision(hiddenn, hiddenn, bias=False),
                        nn.SiLU(inplace=True),
                    ]
                )
                for hidden_layeri in range(se_hidden_layers - 1)
            ]
            self.se_base = nn.Sequential(*seqs)
            self.se_output = LinearMixedPrecision(hiddenn, outputn, bias=False)
            # for k,v in (self.named_parameters()):
            #     v.data.mul_(1e-15)

    def forward(self, input):
        conv_weight_adapt = self.get_conv_weight_adapt(input)
        if conv_weight_adapt is None:
            output = self.conv1x1(input)
        else:
            output = self.adapt_conv2d(input, self.conv1x1.weight, conv_weight_adapt)
        return output

    @staticmethod
    def adapt_conv2d(input, weight, conv_weight_adapt=None):
        dtype = input.dtype
        weight = weight.to(dtype)
        if conv_weight_adapt is None:
            return nn.functional.conv2d(input, weight)
        conv_weight_batch = weight[None] + conv_weight_adapt
        return torch.einsum("bihw,boihw->bohw", [input, conv_weight_batch])

    def get_conv_weight_adapt(self, input):
        if hasattr(self, "se_output"):
            feat = self.se_base(input)
            conv_weight_adapt = (self.se_output(feat)).reshape(
                input.shape[0], -1, input.shape[1], 1, 1
            )
            return conv_weight_adapt

    def split(self, split_idxs, predict_c, optimizers=None):
        self.conv1x1.split(split_idxs, predict_c, optimizers)
        if hasattr(self, "se_output"):
            self.se_output.split(split_idxs, predict_c, optimizers)


class DiscreteDistributionOutput(nn.Module):
    inits = []
    learn_residual = True
    resize_area = True
    l1_loss = False
    leak_feat = True
    # learn_residual = False
    # l1_loss = True
    chain_dropout = 0
    adapt_conv = 0
    # adapt_conv = 3

    def __init__(
        self,
        k=64,
        last_c=None,
        predict_c=3,
        loss_func=None,
        distance_func=None,
        leak_choice=None,
        size=None,
    ):
        super().__init__()
        self.k = k
        self.leak_choice = self.leak_feat if leak_choice is None else leak_choice
        self.size = size
        self.sdd = SplitableDiscreteDistribution(k)
        if last_c is None:
            last_c = max(int(round((k * predict_c) ** 0.5)), 4) * (
                bool(leak_choice) + 1
            )
        self.last_c = last_c
        self.predict_c = predict_c
        self.conv_inc = last_c
        if leak_choice:
            assert not (last_c % 2), last_c
            self.conv_inc = last_c // 2

        # self.multi_out_conv1x1 = Conv2dMixedPrecision(
        #     self.conv_inc, k * predict_c, (1, 1), bias=False
        # )
        self.multi_out_conv1x1 = (
            AdaptConv2d(
                self.conv_inc,
                k * predict_c,
                (1, 1),
                bias=False,
                se_hidden_layers=self.adapt_conv,
            )
            if self.adapt_conv
            else Conv2dMixedPrecision(self.conv_inc, k * predict_c, (1, 1), bias=False)
        )
        self.loss_func = loss_func
        self.distance_func = distance_func
        self.idx = len(self.inits)
        self.register_buffer(
            "split_idxs", -torch.ones((2,), dtype=torch.float32, requires_grad=False)
        )  # int is not supported for NCCL process group
        self.inits.append(self)

    def forward(self, d):
        d["output_level"] = d.get("output_level", -1) + 1
        loss_func = self.loss_func
        if loss_func is None:
            loss_func = l1_loss_multi_output if self.l1_loss else mse_loss_multi_output
        distance_func = d.get("distance_func")
        if distance_func is None:
            distance_func = self.distance_func
        if distance_func is None:
            distance_func = loss_func

        feat_last = d["feat_last"]  # TODO rename to feat no last
        dtype = feat_last.dtype
        device = feat_last.device
        b, c, h, w = feat_last.shape
        if self.leak_choice:
            feat_last = feat_last[..., : self.conv_inc, :, :]
        # with torch.no_grad():
        if 1:
            outputs = self.multi_out_conv1x1(feat_last).reshape(
                b, self.k, self.predict_c, h, w
            )
            if self.learn_residual:
                predcit_shape = (b, self.predict_c, h, w)
                if "predict" in d:
                    predict_last = d["predict"]
                else:
                    predict_last = torch.zeros(
                        predcit_shape, dtype=dtype, device=device
                    )
                if predict_last.shape != predcit_shape:
                    predict_last = nn.functional.interpolate(
                        predict_last, (h, w), mode="bilinear"
                    )
                # outputs.add_(predict_last[:, None])
                outputs = outputs + predict_last[:, None]

        if "target" in d:
            suffix = "" if self.size is None else f"_{self.size}x{self.size}"
            target_key = "target" + suffix
            if target_key not in d:
                d[target_key] = nn.functional.interpolate(
                    d["target"],
                    (self.size, self.size),
                    mode="area" if self.resize_area else "bilinear",
                )
            targets = d[target_key]
            if "max_distance" not in d:
                with torch.no_grad():
                    distance_matrix = distance_func(outputs, targets)  # (b, k)
        if self.training:  # train
            # del outputs
            # torch.cuda.empty_cache()
            if "max_distance" in d:  # random sample for DivergeShaping
                idx_k = torch.randint(0, self.k, (b,))
                predicts = predicts_resized = outputs[torch.arange(b), idx_k]
                if self.size != d["random_start_size"]:
                    predicts_resized = nn.functional.interpolate(
                        predicts,
                        (d["random_start_size"], d["random_start_size"]),
                        mode="area" if self.resize_area else "bilinear",
                    )

                l1 = torch.abs(predicts_resized - d["random_start_target"])
                expand_threshold = 0
                d["loss"] = loss_func((l1 - d["max_distance"]).clip(expand_threshold))

            else:
                add_loss_d = self.sdd.add_loss_matrix(distance_matrix)
                idx_k = add_loss_d["i_nears"]
                # idx_k = torch.from_numpy(idx_k).to(device)
                if random.random() < self.chain_dropout:
                    idx_k = torch.randint(0, self.k, (b,))
                predicts = outputs[torch.arange(b), idx_k]
                # predicts = forward_one_predict(self.multi_out_conv1x1, feat_last, idx_k, predict_c=self.predict_c)
                # if self.learn_residual:
                #     predicts = predict_last + predicts
                # boxx.increase("sd")>8 and  boxx.g()/0
                d["loss"] = loss_func(predicts, targets)
            d["losses"] = d.get("losses", []) + [d["loss"].mean()]

            if d["output_level"] == d.get("random_start_level", -1):
                with torch.no_grad():
                    # d["max_distance"] = (
                    #     torch.abs(outputs - targets[:, None]).max(1)[0].detach()
                    # )
                    d["max_distance"] = (predicts - targets).abs() + torch.abs(
                        outputs - predicts[:, None]
                    ).mean(1)
                d["random_start_target"], d["target_raw"] = targets, d.pop("target")
                d["outputs_for_max_distance"] = outputs.cpu().detach()
                d["random_start_size"] = self.size
        else:
            idx_ks = d.get("idx_ks", [])  # code
            if len(idx_ks) == d["output_level"]:
                if "target" in d:  # find nearst code to target
                    idx_k = distance_matrix.argmin(1)  # .detach().cpu().numpy()
                elif "sampler" in d:  # guided sampler
                    d["output"] = outputs
                    idx_k = d["sampler"](d)
                else:  # random sample
                    idx_k = torch.randint(0, self.k, (b,))
                idx_ks.append(idx_k)
                d["idx_ks"] = idx_ks
            else:  # predefine code
                idx_k = idx_ks[d["output_level"]]
                idx_k = torch.from_numpy(np.array(idx_k)).to(device)
                if idx_k.dtype.is_floating_point:
                    if "idx_ks_raw" not in d:
                        import copy

                        d["idx_ks_raw"] = copy.deepcopy(d["idx_ks"])
                    idx_k = (self.k * idx_k).long().clip(0, self.k - 1)
                    idx_k_ = idx_ks[d["output_level"]]
                    idx_k_[:] = (
                        idx_k if isinstance(idx_k_, torch.Tensor) else idx_k.tolist()
                    )
            predicts = outputs[torch.arange(b), idx_k]
            d["outputs"] = d.get("outputs", []) + [outputs.cpu()]
            if "target" in d:
                d["distances"] = d.get("distances", []) + [
                    distance_matrix[torch.arange(b), idx_k]
                ]
        if self.leak_choice:
            # TODO not need gen all feat_leak
            detach_conv_to_leak = False
            # detach_conv_to_leak = True
            if detach_conv_to_leak:
                raise NotImplementedError()
                weight = self.multi_out_conv1x1.weight.detach().to(dtype)
                feat_leak = torch.nn.functional.conv2d(
                    d["feat_last"][..., self.conv_inc :, :, :],
                    weight,
                )
                if self.multi_out_conv1x1.bias is not None:
                    feat_leak += (
                        self.multi_out_conv1x1.bias.detach().view(1, -1, 1, 1).to(dtype)
                    )
                d["feat_leak"] = feat_leak.reshape(b, self.k, self.predict_c, h, w)[
                    torch.arange(b), idx_k
                ]
            else:
                # d["feat_leak"] = self.multi_out_conv1x1(
                #     d["feat_last"][..., self.conv_inc :, :, :]
                # ).reshape(b, self.k, self.predict_c, h, w)[torch.arange(b), idx_k]
                d["feat_leak"] = forward_one_predict(
                    self.multi_out_conv1x1,
                    d["feat_last"][..., self.conv_inc :, :, :],
                    idx_k,
                    predict_c=self.predict_c,
                )
        d["predict"] = predicts
        d["predicts"] = d.get("predicts", []) + [predicts.detach().cpu()]
        return d

    def try_split(self, optimizers=None):
        import torch.distributed as dist

        rank = int(dist.is_initialized()) and dist.get_rank()
        if isinstance(optimizers, torch.optim.Optimizer):
            optimizers = [optimizers]
        if rank == 0:
            splitd = self.sdd.try_split()
            if splitd:
                self.split_idxs[:] = torch.Tensor(
                    [splitd["i_split"], splitd["i_disapear"]]
                )
        if dist.is_initialized():
            torch.distributed.barrier()
            with torch.no_grad():
                dist.broadcast(self.split_idxs, src=0)
        if self.split_idxs[0] != -1:
            self.multi_out_conv1x1.split(self.split_idxs, self.predict_c, optimizers)
            self.split_idxs[:] = torch.Tensor([-1, -1])
        if dist.is_initialized():
            torch.distributed.barrier()

    @classmethod
    def try_split_all(cls, optimizers=None):
        for self in cls.inits:
            self.try_split(optimizers=optimizers)


# class DiscreteDistributionNetwork(nn.Module):


# class HierarchicalDiscreteDistributionNetwork(nn.Module):


# class HierarchicalDiscreteDistributionPyramidNetwork(nn.Module):


class DivergeShapingManager:
    def __init__(self, seed="diverge_shaping"):
        self.fix_rand_gen = random.Random(seed)

    def __call__(self, d, diverge_shaping_rate=0):
        if hasattr(self, "total_output_level"):
            if self.fix_rand_gen.random() < diverge_shaping_rate:
                d["total_output_level"] = self.total_output_level
                d["random_start_level"] = self.fix_rand_gen.randint(
                    0, self.total_output_level - 2
                )  # last level don't need
        else:
            self.d = d
        return self

    def __enter__(self):
        return self

    def __exit__(self, *args):
        if hasattr(self, "d"):
            d = self.__dict__.pop("d")
            self.total_output_level = d.get("output_level", -2) + 1

    def set_total_output_level(self, d):
        # DistributedDataParallel will copy a new d
        self.total_output_level = d.get("output_level", -2) + 1


diverge_shaping_manager = DivergeShapingManager()

if __name__ == "__main__":
    from boxx.ylth import *

    SplitableDiscreteDistribution.test()
    if 0:
        import torchvision.datasets

        transform01 = torchvision.transforms.Compose(
            [
                # torchvision.transforms.Resize(32),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.5), (0.5)),
            ]
        )
        dataset = torchvision.datasets.cifar.CIFAR10(
            os.path.expanduser("~/dataset"),
            train=True,
            transform=transform01,
            download=True,
        )
