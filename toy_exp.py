#!/usr/bin/env python3
import boxx
import numpy as np

if __name__ == "__main__":
    from boxx import *

    # with boxx.impt("/home/dl/mygit/distribution_playground/"):
    from distribution_playground.source_distribution import *
    from sddn import SplitableDiscreteDistribution

    no_split_for_ablation = ""

    class GeneratorModel:
        def __init__(self, k=100):
            self.k = k
            self.param = np.random.random((k, 2)) * 2 - 1
            self.sdd = SplitableDiscreteDistribution(k)

        def train(self, inp):
            gts = inp["gts"]
            # xys = inp["xys"]
            y = self.param[None]
            loss_matrix = np.abs(y - gts[:, None]).sum(-1)  # b, k
            if no_split_for_ablation:
                i_nears = loss_matrix.argmin(1)
            else:
                i_nears = self.sdd.add_loss_matrix(loss_matrix)["i_nears"]

                splited = self.sdd.try_split()
                if splited:
                    i_split, i_disapear = splited["i_split"], splited["i_disapear"]
                    self.param[i_disapear] = self.param[i_split]

            # backward
            for i_near, gt in zip(i_nears, gts):
                to_gt = gt - self.param[i_near]
                lr = 0.2
                self.param[i_near] += 2 * (to_gt) * lr
                # lr = 0.05
                # self.param[i_near] += to_gt/np.linalg.norm(to_gt) * lr

        def sample(self, n=None):
            n = n or self.k
            idxs = np.random.choice(self.k, n, replace=n > self.k)
            return self.k[idxs]

    no_split_for_ablation = "_no.split"
    bins = (100,) * 2  # 10000 可以扫码
    # bins = (128,) * 2
    # bins = (256,) * 2
    distd = {}

    distd["blurs"] = get_test_dist(bins[0])
    dist = distd["blurs"]

    maps = glob(os.path.expanduser("~/discrete_distribution/info/asset/density/*.png"))
    for dmap in maps:

        densty = cv2.resize(
            imread(dmap),
            bins,
            interpolation=cv2.INTER_AREA,
        )
        if densty.ndim == 3:
            densty = densty[..., 1]
        eps = 1e-20
        densty = densty / densty.sum()
        densty += eps
        densty = densty / densty.sum()
        dist = DistributionByDensityArray(densty)

        distd[filename(dmap)] = dist

    k = 1000
    itern = 600
    batch = 40

    # densty = boxx.resize(sda.astronaut(), (50, 50)).mean(-1)
    k = 5000
    itern = 6000
    batch = 40

    k = 10000
    itern = 50000
    batch = 2

    # itern = 6

    dirr = (
        os.path.expanduser("~/junk/ddn_toy_exp/")
        + f"bin{bins[0]}_k{k}_itern{itern}_batch{batch}{no_split_for_ablation}/"
    )
    os.makedirs(dirr, exist_ok=True)
    strr = dirr + "\n"
    kls = []

    names = list(distd)
    names = __import__("brainpp_yl").split_keys_by_replica(names)
    # for name,dist in distd.items():
    for name in names:
        print(name, dirr)
        dist = distd[name]
        gen = GeneratorModel(k)
        sdd = gen.sdd

        diver_gt = dist.divergence(dist.sample(k))
        print("gt:", dist.str_divergence(diver_gt))

        def log(big=False):
            diver = dist.divergence(gen.param)
            if "yanglei-docker" not in boxx.sysi.host:
                (shows if big else show)(
                    uint8, histEqualize, diver, dist.density, diver_gt
                )
            print(dist.str_divergence(diver))

        for iteri in range(itern):
            gts = dist.sample(batch)
            inp = dict(gts=gts)
            gen.train(inp)
            if not increase(
                "show log",
            ) % (itern // 5):
                log()
            # break
        log()
        print("gt:", dist.str_divergence(dist.sample(k)))
        # sdd.plot_dist()
        # print("near2.sum(), iter=", sdd.near2.sum(), sdd.iter)
        globals().update(gen.sdd.__dict__)
        #     show - dist.divergence(gen.param[:100])
        #%%
        if "save":
            # TODO: dump param and save checkpoint

            max_v = dist.density.max()
            density_to_rgb1 = lambda d: cv2.applyColorMap(
                uint8(norma(d.clip(0, max_v))), cv2.COLORMAP_VIRIDIS
            )[..., ::-1]
            density_to_rgb2 = lambda d: cv2.applyColorMap(
                uint8(histEqualize(d)), cv2.COLORMAP_VIRIDIS
            )[..., ::-1]
            density_to_rgb = density_to_rgb1
            for coveri, density_to_rgb in enumerate(
                [
                    density_to_rgb1,
                    density_to_rgb2,
                ],
                1,
            ):
                d_gt = density_to_rgb(dist.density)
                d_sample = density_to_rgb(
                    dist.divergence(dist.sample(k * 2))["estimated"]
                )
                d_gen = density_to_rgb(dist.divergence(gen.param)["estimated"])
                # show(d_gt,d_gen,d_sample, figsize=(8,8))
                imsave(
                    dirr + f"2d_density.{coveri}_{name}_gt{no_split_for_ablation}.png",
                    d_gt,
                )
                imsave(
                    dirr
                    + f"2d_density.{coveri}_{name}_sample{no_split_for_ablation}.png",
                    d_sample,
                )
                imsave(
                    dirr + f"2d_density.{coveri}_{name}_gen{no_split_for_ablation}.png",
                    d_gen,
                )
            print("Save to:", dirr)

            diverg_gt = dist.divergence(dist.sample(k))
            diverg_gen = dist.divergence(gen.param)
            kls.append([diverg_gen["kl"], diverg_gt["kl"]])
            strr += f"{name}:\n\tgen:{dist.str_divergence(diverg_gen)}\n\tgt:{dist.str_divergence(diverg_gt)}\n\n"

        # if "QR" in name:
        #     1/0
    logp = dirr + "KL.gen%s_gt%s.txt" % tuple(np.mean(kls, 0).round(3))
    openwrite(
        strr,
        logp,
    )
# cmap = ["gist_earth", "turbo",  "viridis"][-1]
