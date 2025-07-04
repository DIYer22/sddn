import cv2
import torch
import boxx
import itertools
import numpy as np
from boxx import npa

k8_order = range(8)

t2np = lambda t: np.uint8((npa(t + 1).squeeze() * 128).clip(0, 255))


def draw_board(arr, color=(0, 0, 0), b=None, leveli=None):
    rgb = arr
    if rgb.ndim == 2:
        rgb = np.concatenate([arr[..., None], arr[..., None], arr[..., None]], -1)
    h, w = arr.shape[:2]
    b = b or int(round(h / 32))
    if h <= 32:
        rgb[:b], rgb[-b:], rgb[:, :b], rgb[:, -b:] = [color] * 4
        return rgb
    bg = np.ones((h + 2 * b, h + 2 * b, 3)) * color
    bg[b : b + h, b : b + w] = rgb
    if leveli == 0:
        return bg

    bgg = np.ones((h + 4 * b, h + 4 * b, 3)) * 255 * 0
    bgg[b : b + h + 2 * b, b : b + w + 2 * b] = bg
    bg = bgg

    return bg


def vis_tree_latent_recursively(model, leveln=3):
    # TODO: too slow, using outputs of last layer and pre batch generate then cache
    is_training = model.training
    model.eval()
    using_pre_eval_cache = True
    using_pre_eval_cache = False
    if using_pre_eval_cache:
        all_idx_ks = list(
            npa(list(itertools.product(*[list(range(8))] * (leveln - 1)))).T
        )
        all_idx_ks += [all_idx_ks[0] * 0]
        d_cache = model(dict(idx_ks=all_idx_ks))
        boxx.g()
        # TODO: different batch size has different result with same idx_ks, except first layer outputs and the results conditioned on first output of first layer
        # d_77 = tree/model(dict(idx_ks=[ks[:] for ks in all_idx_ks]));outputs_77 = shows/uint8(norma-npa*d_77["outputs"][-1][-1])
        # d_77 = tree/model(dict(idx_ks=[ks[-1:] for ks in all_idx_ks]));outputs_77 = shows/uint8(norma-npa*d_77["outputs"][-1][-1])

    npa(list(itertools.product(*[list(range(8))] * 3))).T

    def _build_vis_tree_latent_recursively(idx_ks=None, parent=None):
        idx_ks = idx_ks or []
        leveli = len(idx_ks)
        if leveli == leveln:
            if using_pre_eval_cache:
                flat_idx = int("".join(map(str, idx_ks[: leveln - 1])), 8)
                predicts = [
                    pre[flat_idx][None] for pre in d_cache["predicts"][: leveln - 1]
                ]
                predicts += [
                    d_cache["outputs"][leveln - 1][flat_idx][idx_ks[leveln - 1]][None]
                ]
            else:
                d = dict(idx_ks=npa(idx_ks + idx_ks[-1:] * 10)[:, None])
                model(d)
                predicts = d["predicts"]
            parent["predicts"] = t2np(torch.cat(predicts))[:leveln]
            parent["img"] = draw_board(t2np(predicts[leveli - 1]))
            return parent
        for k in range(8):
            idx_ks_ = idx_ks + [k]
            # idx_ks_ = [k] + idx_ks
            parent[k] = {"idx_ks": idx_ks_.copy()}
            _build_vis_tree_latent_recursively(idx_ks_, parent[k])

        imgs = [parent[i]["img"] for i in k8_order]
        h, w = imgs[0].shape[:2]
        b_rate = 1 / 32
        b = int(round(h * b_rate))
        color = (255, 255, 0)
        color = [int(round((leveln - leveli) / (leveln) * 255))] * 2 + [0]
        color = [
            (0,) * 3,
            (
                0,
                0,
                192,
            ),
            (0, 128, 0),
            (255, 0, 0),
        ][leveli]
        # color = [(64,)*3,(128,)*3,(255,)*3,][leveli]
        if leveli:
            # img_this = np.mean([parent[i]["predicts"][leveli - 1] for i in range(8)], 0)
            img_this = parent[0]["predicts"][leveli - 1]
        else:
            # level0 is mean of level1 outputs
            # img_this = np.mean([parent[i]["predicts"][0] for i in range(8)], 0)
            img_this = np.mean([parent[i]["predicts"][leveln - 1] for i in range(8)], 0)
        img_this = draw_board(img_this, color, int(round(img_this.shape[0] * b_rate)))
        img_resized = cv2.resize(img_this, (w, h), interpolation=cv2.INTER_NEAREST)
        imgs = imgs[:4] + [img_resized] + imgs[4:]
        imgs_3x3 = npa(imgs).reshape(3, 3, *imgs[0].shape)
        img = np.concatenate(np.concatenate(imgs_3x3, 2), 0)
        parent["img"] = np.uint8(draw_board(img, color, b, leveli))
        parent["predicts"] = npa([parent[i]["predicts"] for i in range(8)]).mean(0)
        return parent

    root = {}
    _build_vis_tree_latent_recursively(parent=root)
    # img = root["img"]
    if is_training:
        model.train()
    return root


if __name__ == "__main__":
    with boxx.impt("../"):
        from mnist import *

    ptp = (
        "/home/yl/ddn/asset/2023-11-07-11_47_45-mnist_outputk8_repeatn5-shot14959968.pt"
    )
    ptp = "/home/yl/ddn/asset/2025-05-26-00_16_11-mnist_outputk8_repeatn3_chain.dropout0.0-shot2991712.pt"

    model = torch.load(ptp).eval()
    model.repeatn = 3
    with boxx.timeit():
        root = vis_tree_latent_recursively(model, leveln=3)
        vis = root["img"]
    boxx.shows(vis, png=True)
