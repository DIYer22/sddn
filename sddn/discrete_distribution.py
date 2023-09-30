#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 19 08:37:01 2023

@author: yanglei

存放专有模块
"""

import boxx
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
        i_near_near2s = np.argpartition(loss_matrix, 2)[:, :2]
        i_nears = np.argmin(loss_matrix, 1)

        i_near2s = i_near_near2s[i_near_near2s != i_nears[:, None]]

        # loss_mins = loss_matrix[i_nears]
        self.iter += b
        self.batchn += 1
        unique_, count_ = np.unique(i_nears, return_counts=True)
        self.count[unique_] += count_
        # TODO make near2 work again
        # for i_near, i_near2 in zip(i_nears, i_near2s):
        #     self.near2[i_near, i_near2] += 1
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
        mg()
        if self.iter and (tv_loss_splited < tv_loss_now or pd < P / 2):
            self.split_iters.append(self.iter)
            return self.split(i_split, i_disapear)

    def split(self, i_split, i_disapear):
        """"""
        eps = 0.01
        self.near2 = 1 - np.eye(self.k)
        self.near2 = self.near2 / self.near2.sum() * self.iter
        near2_sum = self.near2[i_disapear].sum()
        if near2_sum:
            self.near2[i_disapear][i_disapear] = 0
            i_to_weight = self.near2[i_disapear] / max(
                self.near2[i_disapear].sum(), eps
            )
            self.count += self.count[i_disapear] * i_to_weight
            self.count[i_disapear] = 0

            self.near2 += (
                (near2_sum * i_to_weight)[None]
                * self.near2
                / np.linalg.norm(self.near2, axis=1, keepdims=True).clip(eps)
            )

            self.near2 += self.near2[:, i_disapear : i_disapear + 1] * i_to_weight[None]
            self.near2[:, i_disapear] = 0
            self.near2[i_disapear] = 0

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

        mg()  # /0
        assert np.isfinite(self.near2).all()
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
    def test(cls, k=10):
        import tqdm

        sdd = cls(k)
        b = 5
        batchn = 2000
        for batchi in tqdm.tqdm(range(batchn)):
            dm = np.random.rand(b, k) * np.linspace(0.59, 1.1, k)[None]
            sdd.add_loss_matrix(dm)
            split = sdd.try_split()
            if split:
                print(batchi)
            # tree - split

        boxx.g()
        assert sdd.iter == sdd.near2.sum() == sdd.count.sum(), [
            sdd.iter,
            sdd.near2.sum(),
            sdd.count.sum(),
        ]


def mse_loss_multi_output(input, target):
    # input (b, k, c, h, w) or (b, c, h, w)
    # target (b, c, h, w)
    is_multi_input = input.ndim != target.ndim
    if is_multi_input:
        target = target[:, None]
    # return (b, k) if is_multi_input else (b,)
    return ((input - target) ** 2).mean((-1, -2, -3))


class DiscreteDistributionOutput(nn.Module):
    inits = []
    learn_residual = True
    def __init__(
        self,
        k=64,
        last_c=None,
        predict_c=3,
        loss_func=None,
        distance_func=None,
        leak_choice=False,
        size=None,
    ):
        super().__init__()
        self.k = k
        self.leak_choice = leak_choice
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

        self.multi_out_conv1x1 = nn.Conv2d(
            self.conv_inc, k * predict_c, (1, 1), bias=False
        )
        self.loss_func = loss_func
        self.distance_func = distance_func
        self.idx = len(self.inits)
        self.register_buffer(
            "split_idxs", -torch.ones((2,), dtype=torch.float32, requires_grad=False)
        )  # int is not supported for NCCL process group
        self.inits.append(self)

    def forward(self, d):
        d["ouput_level"] = d.get("ouput_level", -1) + 1
        loss_func = self.loss_func
        distance_func = self.distance_func
        if loss_func is None:
            loss_func = mse_loss_multi_output
        if distance_func is None:
            distance_func = loss_func

        feat_last = d["feat_last"]  # TODO rename to feat no last
        dtype = feat_last.dtype
        device = feat_last.device
        b, c, h, w = feat_last.shape
        if self.leak_choice:
            feat_last = feat_last[..., : self.conv_inc, :, :]
        outputs = self.multi_out_conv1x1(feat_last).reshape(
            b, self.k, self.predict_c, h, w
        )
        if self.learn_residual:
            predcit_shape = (b, self.predict_c, h, w)
            if "predict" in d:
                predict_last = d["predict"]
            else:
                predict_last = torch.zeros(predcit_shape, dtype=dtype, device=device)
            if predict_last.shape != predcit_shape:
                predict_last = nn.functional.interpolate(
                    predict_last, (h, w), mode="bilinear"
                )
            outputs = predict_last[:, None] + outputs
        with torch.no_grad():
            if "target" in d:
                suffix = "" if self.size is None else f"_{self.size}x{self.size}"
                target_key = "target" + suffix
                if target_key not in d:
                    d[target_key] = nn.functional.interpolate(
                        d["target"], (self.size, self.size), mode="area"
                    )
                targets = d[target_key]
                distance_matrix = distance_func(outputs, targets)  # (b, k)
        if self.training:  # train
            add_loss_d = self.sdd.add_loss_matrix(distance_matrix)
            idx_k = add_loss_d["i_nears"]
            idx_k = torch.from_numpy(idx_k).to(device)
            predicts = outputs[torch.arange(b), idx_k]
            d["loss"] = loss_func(predicts, targets)
            d["losses"] = d.get("losses", []) + [d["loss"].mean()]
        else:
            idx_ks = d.get("idx_ks", [])  # code
            if len(idx_ks) == d["ouput_level"]:
                if "target" in d:  # find nearst code to target
                    idx_k = distance_matrix.argmin(1)  # .detach().cpu().numpy()
                else:  # random sample
                    idx_k = torch.randint(0, self.k, (b,))
                idx_ks.append(idx_k)
                d["idx_ks"] = idx_ks
            else:  # predefine code
                idx_k = idx_ks[self.idx]
                idx_k = torch.from_numpy(np.array(idx_k)).to(device)
                if idx_k.dim == 0:
                    idx_k = idx_k[None]
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
                weight = self.multi_out_conv1x1.weight.detach()
                feat_leak = torch.nn.functional.conv2d(
                    d["feat_last"][..., self.conv_inc :, :, :],
                    weight,
                )
                if self.multi_out_conv1x1.bias is not None:
                    feat_leak += self.multi_out_conv1x1.bias.detach().view(1, -1, 1, 1)
                d["feat_leak"] = feat_leak.reshape(b, self.k, self.predict_c, h, w)[
                    torch.arange(b), idx_k
                ]
            else:
                d["feat_leak"] = self.multi_out_conv1x1(
                    d["feat_last"][..., self.conv_inc :, :, :]
                ).reshape(b, self.k, self.predict_c, h, w)[torch.arange(b), idx_k]
        d["predict"] = predicts
        d["predicts"] = d.get("predicts", []) + [predicts]
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
            with torch.no_grad():
                dist.broadcast(self.split_idxs, src=0)
        if self.split_idxs[0] != -1:
            # update multi_out_conv1x1 weight
            predict_c = self.predict_c
            with torch.no_grad():
                i_split, i_disapear = self.split_idxs
                i_split, i_disapear = int(i_split), int(i_disapear)
                weight = self.multi_out_conv1x1._parameters[
                    "weight"
                ]  # (k*predict_c, last_c)
                weight[
                    i_disapear * predict_c : i_disapear * predict_c + predict_c
                ] = weight[i_split * predict_c : i_split * predict_c + predict_c]
                assert self.multi_out_conv1x1.bias is None

                # update multi_out_conv1x1 optimizer
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

            self.split_idxs[:] = torch.Tensor([-1, -1])

    @classmethod
    def try_split_all(cls, optimizers=None):
        for self in cls.inits:
            self.try_split(optimizers=optimizers)


# class DiscreteDistributionNetwork(nn.Module):


# class HierarchicalDiscreteDistributionNetwork(nn.Module):


# class HierarchicalDiscreteDistributionPyramidNetwork(nn.Module):


if __name__ == "__main__":
    from boxx.ylth import *

    from torchvision.datasets import cifar

    transform01 = torchvision.transforms.Compose(
        [
            # torchvision.transforms.Resize(32),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5), (0.5)),
        ]
    )
    dataset = cifar.CIFAR10(
        os.path.expanduser("~/dataset"),
        train=True,
        transform=transform01,
        download=True,
    )
    # SplitableDiscreteDistribution.test()
