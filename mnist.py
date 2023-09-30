#!/usr/bin/env python
# coding: utf-8

# In[ ]:
import os
from boxx.ylth import *
import boxx
import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
import matplotlib.pyplot as plt
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from tqdm.notebook import tqdm

# In[ ]:
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tensor2rgb = lambda x: tprgb((x + 1) * 128).clip(0, 255).astype(np.uint8)
showt = lambda *l, **kv: show(*l, tensor2rgb, **kv)

# In[ ]:
def show_images(images, title="show"):
    """Shows the provided images as sub-pictures in a square"""
    images = [im.permute(1, 2, 0).numpy() for im in images]

    # Defining number of rows and columns
    fig = plt.figure(figsize=(8, 8))
    rows = int(len(images) ** (1 / 2))
    cols = round(len(images) / rows)

    # Populating figure with sub-plots
    idx = 0
    for r in range(rows):
        for c in range(cols):
            fig.add_subplot(rows, cols, idx + 1)

            if idx < len(images):
                plt.imshow(tensor2rgb(images[idx]), cmap="gray")
                plt.axis("off")
                idx += 1
    fig.suptitle(title, fontsize=30)

    # Showing the figure
    # plt.show()
    path = path_prifix + title + ".png"
    plt.savefig(path)
    plt.show()


# In[ ]:
def sinusoidal_embedding(n, d):
    # Returns the standard positional embedding
    embedding = torch.tensor(
        [[i / 10_000 ** (2 * j / d) for j in range(d)] for i in range(n)]
    )
    sin_mask = torch.arange(0, n, 2)

    embedding[sin_mask] = torch.sin(embedding[sin_mask])
    embedding[1 - sin_mask] = torch.cos(embedding[sin_mask])

    return embedding


# In[ ]:
class MyConv(nn.Module):
    def __init__(
        self,
        shape,
        in_c,
        out_c,
        kernel_size=3,
        stride=1,
        padding=1,
        activation=None,
        normalize=True,
    ):
        super().__init__()
        self.ln = nn.LayerNorm(shape)
        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size, stride, padding)
        self.activation = nn.SiLU() if activation is None else activation
        self.normalize = normalize

    def forward(self, x):
        out = self.ln(x) if self.normalize else x
        out = self.conv1(out)
        out = self.activation(out)
        return out


def MyTinyBlock(size, in_c, out_c):
    return nn.Sequential(
        MyConv((in_c, size, size), in_c, out_c),
        MyConv((out_c, size, size), out_c, out_c),
        MyConv((out_c, size, size), out_c, out_c),
    )


def MyTinyUp(size, in_c):
    return nn.Sequential(
        MyConv((in_c, size, size), in_c, in_c // 2),
        MyConv((in_c // 2, size, size), in_c // 2, in_c // 4),
        MyConv((in_c // 4, size, size), in_c // 4, in_c // 4),
    )


# In[ ]:
try:
    from sddn import DiscreteDistributionOutput
except ModuleNotFoundError:
    pass
DiscreteDistributionOutput.inits.clear()


def linear_spatial_embedding(shape):
    size = 1
    for s in shape:
        size *= s
    spatial_embedding = (torch.linspace(-1, 1, size).cuda()).reshape(*shape)
    if len(shape) == 3:
        if shape[-1] == shape[-2]:
            spatial_embedding = torch.cat(
                [torch.rot90(e, i)[None] for i, e in enumerate(spatial_embedding)]
            )
        else:
            spatial_embedding = torch.cat(
                [
                    torch.flip(
                        e,
                        ([0] if i % 4 in (1, 3) else [])
                        + ([1] if i % 4 in (2, 3) else []),
                    )[None]
                    for i, e in enumerate(spatial_embedding)
                ]
            )
    return spatial_embedding


class DiscreteDistributionNetwork(nn.Module):
    # Here is a network with 3 down and 3 up with the tiny block
    def __init__(
        self,
        in_c=1,
        out_c=1,
        k=64,
        last_c=None,
        size=32,
        class_n=1000,
        class_emb_dim=100,
        previous_last_c=None,
        leak_choice=False,
        basec=10,
    ):
        super().__init__()
        # Sinusoidal embedding
        self.time_embed = nn.Embedding(class_n, class_emb_dim)
        self.time_embed.weight.data = sinusoidal_embedding(class_n, class_emb_dim)
        self.time_embed.requires_grad_(False)

        # First half
        self.te1 = self._make_te(class_emb_dim, 1)
        self.b1 = MyTinyBlock(size, in_c, 1 * basec)
        self.down1 = nn.Conv2d(1 * basec, 1 * basec, 4, 2, 1)
        self.te2 = self._make_te(class_emb_dim, 1 * basec)
        self.b2 = MyTinyBlock(size // 2, 1 * basec, 2 * basec)
        self.down2 = nn.Conv2d(2 * basec, 2 * basec, 4, 2, 1)
        self.te3 = self._make_te(class_emb_dim, 2 * basec)
        self.b3 = MyTinyBlock(size // 4, 2 * basec, 4 * basec)
        self.down3 = nn.Conv2d(4 * basec, 4 * basec, 4, 2, 1)

        # Bottleneck
        self.te_mid = self._make_te(class_emb_dim, 4 * basec)
        self.b_mid = nn.Sequential(
            MyConv((4 * basec, size // 8, size // 8), 4 * basec, 2 * basec),
            MyConv((2 * basec, size // 8, size // 8), 2 * basec, 2 * basec),
            MyConv((2 * basec, size // 8, size // 8), 2 * basec, 4 * basec),
        )

        # Second half
        self.up1 = nn.ConvTranspose2d(4 * basec, 4 * basec, 4, 2, 1)
        self.te4 = self._make_te(class_emb_dim, 8 * basec)
        self.b4 = MyTinyUp(size // 4, 8 * basec)
        self.up2 = nn.ConvTranspose2d(2 * basec, 2 * basec, 4, 2, 1)
        self.te5 = self._make_te(class_emb_dim, 4 * basec)
        self.b5 = MyTinyUp(size // 2, 4 * basec)
        self.up3 = nn.ConvTranspose2d(1 * basec, 1 * basec, 4, 2, 1)
        self.te_out = self._make_te(class_emb_dim, 2 * basec)
        # self.b_out = MyTinyBlock(size, 2 * basec, 1 * basec)
        # self.conv_out = nn.Conv2d(1 * basec, out_c, 3, 1, 1)

        if last_c is None:
            last_c = basec
            # last_c = max(int(round((k * out_c) ** 0.5)), 4) * (bool(leak_choice) + 1)
        if previous_last_c is None:
            previous_last_c = last_c
        self.previous_last_c = previous_last_c
        self.b1_previous_last = MyTinyBlock(size, previous_last_c, 1 * basec)
        self.out_c = out_c
        self.size = size
        self.leak_choice = leak_choice

        self.b_out = MyTinyBlock(size, 2 * basec, last_c)
        self.ddo = DiscreteDistributionOutput(
            k=k,
            last_c=last_c,
            predict_c=out_c,
            leak_choice=leak_choice,
        )
        if leak_choice:
            self.b1_leak_choice = MyTinyBlock(size, in_c, 1 * basec)

    def forward(self, d):  # x is (bs, in_c, size, size) t is (bs)
        tmp_inp = d
        if not isinstance(d, dict):
            d = {"predict": tmp_inp}
        if "target" in d:
            batch_size = len(d["target"])
        else:
            batch_size = d.get("batch_size", 1)
        feat_leak = d.get("feat_leak")
        feat_last = d.get("feat_last")
        predict = d.get("predict")
        classi = d.get("classi")
        if feat_last is None:
            feat_last = torch.zeros(
                (batch_size, self.previous_last_c, self.size, self.size)
            ).cuda()
            feat_last[:] = linear_spatial_embedding(
                (self.previous_last_c, self.size, self.size)
            ).cuda()
        if predict is None:
            predict = torch.zeros((batch_size, self.out_c, self.size, self.size)).cuda()
            predict[:] = linear_spatial_embedding(
                (self.out_c, self.size, self.size)
            ).cuda()
        x, t = predict, classi
        n = len(x)
        if t is None:
            t = torch.zeros(n, dtype=torch.long).cuda()
        t = self.time_embed(t)
        out1 = self.b1(
            x + self.te1(t).reshape(n, -1, 1, 1)
        )  # (bs, 1 * basec, size/2, size/2)

        out1 = out1 + self.b1_previous_last(feat_last)

        if self.leak_choice:
            feat_leak = d.get(
                "feat_leak", predict
            )  # same as predict (bs, inc, size, size)
            out1 = out1 + self.b1_leak_choice(feat_leak)

        out2 = self.b2(
            self.down1(out1) + self.te2(t).reshape(n, -1, 1, 1)
        )  # (bs, 2 * basec, size/4, size/4)
        out3 = self.b3(
            self.down2(out2) + self.te3(t).reshape(n, -1, 1, 1)
        )  # (bs, 4 * basec, size/8, size/8)

        out_mid = self.b_mid(
            self.down3(out3) + self.te_mid(t).reshape(n, -1, 1, 1)
        )  # (bs, 4 * basec, size/8, size/8)

        out4 = torch.cat(
            (out3, self.up1(out_mid)), dim=1
        )  # (bs, 8 * basec, size/8, size/8)
        out4 = self.b4(
            out4 + self.te4(t).reshape(n, -1, 1, 1)
        )  # (bs, 2 * basec, size/8, size/8)
        out5 = torch.cat(
            (out2, self.up2(out4)), dim=1
        )  # (bs, 4 * basec, size/4, size/4)
        out5 = self.b5(
            out5 + self.te5(t).reshape(n, -1, 1, 1)
        )  # (bs, 1 * basec, size/2, size/2)
        out = torch.cat((out1, self.up3(out5)), dim=1)  # (bs, 2 * basec, size, size)
        d["feat_last"] = self.b_out(
            out + self.te_out(t).reshape(n, -1, 1, 1)
        )  # (bs, 1 * basec, size, size)

        if not isinstance(tmp_inp, dict):
            return d["feat_last"]
        d = self.ddo(d)
        return d

    def _make_te(self, dim_in, dim_out):
        return nn.Sequential(
            nn.Linear(dim_in, dim_out), nn.SiLU(), nn.Linear(dim_out, dim_out)
        )


# In[ ]:
class HierarchicalDiscreteDistributionNetwork(nn.Module):
    def __init__(
        self,
        ddn_seqen,
    ):
        super().__init__()
        # TODO if different stage k middle_c last_c, insert a transfer module
        self.ddn_seqen = nn.Sequential(*ddn_seqen)

    def set_distance_func(self, distance_func):
        distance_func

    def set_loss_func(self, loss_func):
        loss_func

    def forward(self, d):
        for stacki in range(len(self.ddn_seqen)):
            for repeati in range(repeatn):
                d = self.ddn_seqen[stacki](d)
        return d


# In[ ]:
def training_loop(model, dataloader, optimizer, shots, num_timesteps, device=device):
    """Training loop for DDPM"""

    global_step = 0
    shot_num = 0
    epoch = 0
    while shot_num < shots:
        model.train()
        progress_bar = tqdm(total=len(dataloader))
        progress_bar.set_description(f"Epoch {epoch}")
        for step, (target, classi) in enumerate(dataloader):
            # batchd = {k: batchd[k].to(device) for k in batchd}
            batchd = dict(
                target=target.to(device), classi=classi.to(device) * condition
            )
            # target = batchd["target"]
            d = model(batchd)
            loss = sum(d["losses"]) / len(d["losses"])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # DiscreteDistributionOutput.try_split_all()
            DiscreteDistributionOutput.try_split_all(optimizer)

            progress_bar.update(1)
            logs = {"loss": loss.detach().item(), "step": global_step}
            progress_bar.set_postfix(**logs)
            shot_num += target.shape[0]
            if boxx.timegap(60 * logmin, "show-train"):
                print(
                    f"shot_num/shots={shot_num}/{shots}({round(shot_num/shots*100,2)}%)"
                )
                [
                    print(f"task: {task}, classi: {i}, loss: {float(los)}")
                    or showt(
                        pre,
                        tar,
                    )
                    for pre, tar, i, los in zip(
                        d["predict"][:2],
                        target,
                        classi,
                        d["losses"],
                    )
                ]

                print("generate_image:")

                b, c, h, w = target.shape
                img_gened = generate_image(model, 4, c, w)
                showt(img_gened)
                show_images(
                    generate_image(model, 100, 1, 32), "/%09d-result" % shot_num
                )
            if not global_step % (shots // dataloader.batch_size // dumpn):
                torch.save(model, f"{path_prifix}/shot{shot_num}.pt")
            if shot_num >= shots:
                break
            global_step += 1
        epoch += 1
        progress_bar.close()
    boxx.mg()


# In[ ]:


def generate_image(model, batch_size, channel, size):
    """Generate the image from the Gaussian noise"""

    frames = []
    model.eval()
    with torch.no_grad():
        classn = len(dataset.classes)
        classi = (
            torch.Tensor(
                list(range(classn)) * (batch_size // classn + 1),
            )
            .long()
            .reshape(-1, classn)
            .T.reshape(-1)
            .cuda()
        ) * condition
        predict = model(dict(batch_size=batch_size, classi=classi[:batch_size]))
        for i in range(batch_size):
            frames.append(predict["predict"][i].detach().cpu())
    boxx.mg()
    model.train()
    return frames


if __name__ == "__main__":
    args, argkv = boxx.getArgvDic()
    # In[ ]:
    basec = 32
    stackn = 16
    repeatn = 10
    batch_size = 8  # int(4096 // (stackn * repeatn)) // 2
    learning_rate = 1e-3
    shots = "300w"
    num_timesteps = 1000
    num_workers = 10
    dumpn = 4
    logmin = 30
    data = "cifar"
    data = "mnist"
    condition = False + 1
    task = f"{data}-default"

    cudan = torch.cuda.device_count()
    debug = not cudan or torch.cuda.get_device_capability("cuda") <= (6, 9)
    if data == "mnist":
        basec = 16
        repeatn = 10
        batch_size = 256
        stackn = 1
        shots = "3000w"
        logmin = 30
        condition = False
        task = "mnist-split.opt-3000w"

    if argkv.get("debug"):
        debug = True
    if debug:
        basec = 8
        batch_size = 2
        shots = 100
        num_timesteps = 4
        num_workers = 0
        dumpn = 2
        stackn = 2
        repeatn = 1

    shots = argkv.get("shots", shots)
    if isinstance(shots, str):
        shots = int(shots.lower().replace("w", "0000").replace("k", "000"))
    batch_size = argkv.get("batch_size", batch_size)

    # data/exp dir
    root_dir = boxx.relfile("../data/")
    if os.path.isdir(os.path.expanduser("~/dataset")):
        root_dir = os.path.expanduser("~/dataset")
    path_prifix = (
        os.path.join(root_dir, "exps/minst_ddpm", boxx.localTimeStr(1)) + "-" + task
    )
    # os.makedirs(os.path.dirname(path_prifix), exist_ok=True)
    os.makedirs((path_prifix), exist_ok=True)
    if not os.path.exists("/tmp/boxxTmp"):
        os.system(f"ln -sf {path_prifix} /tmp/boxxTmp")
    if not os.path.exists("/tmp/boxxTmp/showtmp"):
        os.system(f"ln -sf {path_prifix} /tmp/boxxTmp/showtmp")

    if data == "cifar":
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
    if data == "mnist":
        transform01 = torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize(32),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.5), (0.5)),
            ]
        )
        dataset = torchvision.datasets.MNIST(
            root=root_dir, train=True, transform=transform01, download=True
        )
    channeln = len(dataset[0][0])
    dataloader = torch.utils.data.DataLoader(
        dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )

    # In[ ]:
    gen_ddn = lambda: DiscreteDistributionNetwork(
        channeln,
        channeln,
        class_n=len(dataset.class_to_idx),
        leak_choice=True,
        basec=basec,
    )
    ddn_seqen = [gen_ddn() for _ in range(stackn)]
    ddn = ddn_seqen[0]
    network = HierarchicalDiscreteDistributionNetwork(ddn_seqen)
    network = network.to(device)
    model = network
    optimizer = torch.optim.Adam(
        [v for k, v in model.named_parameters()],
        lr=learning_rate,
    )
    training_loop(model, dataloader, optimizer, shots, num_timesteps, device=device)

    for b in dataloader:
        batch = b[0]
        break

    bn = [b for b in batch[:100]]
    show_images(bn, "origin")
    for i in range(4):
        generated = generate_image(model, 100, 1, 32)
        show_images(generated, "result")
    sdd = DiscreteDistributionOutput.inits[-1].sdd
    sdd.plot_dist()

    # import torchsummary
    # torchsummary.summary(network.eval().ddn_seqen[0], (channeln, 32, 32))
