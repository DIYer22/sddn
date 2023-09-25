#!/usr/bin/env python3
import boxx
import numpy as np

if __name__ == "__main__":
    from boxx import *

    # with boxx.impt("/home/dl/mygit/distribution_playground/"):
    from distribution_playground.source_distribution import *
    from sddn import SplitableDiscreteDistribution

    class GeneratorModel:
        def __init__(self, k=100):
            self.k = k
            self.param = np.random.random((k, 2))
            self.sdd = SplitableDiscreteDistribution(k)

        def train(self, inp):
            gts = inp["gts"]
            # xys = inp["xys"]
            y = self.param[None]
            loss_matrix = np.abs(y - gts[:, None]).sum(-1)
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

    dist = get_test_dist(50)
    # densty = boxx.resize(sda.astronaut(), (50, 50)).mean(-1)
    # densty = densty / densty.sum()
    # dist = DistributionByDensityArray(densty)
    k = 1000
    itern = 2000
    batch = 4
    gen = GeneratorModel(k)
    sdd = gen.sdd

    diver_gt = dist.divergence(dist.sample(k))
    print("gt:", dist.str_divergence(diver_gt))

    def log(big=False):
        diver = dist.divergence(gen.param)
        (shows if big else show)(uint8, histEqualize, diver, dist.density, diver_gt)
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
    log(True)
    print("gt:", dist.str_divergence(dist.sample(k)))
    sdd.plot_dist()
    print("near2.sum(), iter=", sdd.near2.sum(), sdd.iter)
    globals().update(gen.sdd.__dict__)
