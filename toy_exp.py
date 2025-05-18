#!/usr/bin/env python3
import boxx
import numpy as np
from sddn import SplitableDiscreteDistribution


class GeneratorModel:
    def __init__(self, k=100):
        self.k = k
        self.param = (
            np.random.random((k, 2)) * 2 - 1
        )  # init nodes in uniform of [-1, +1]
        self.sdd = SplitableDiscreteDistribution(k)

    def train(self, inp):
        gts = inp["gts"]
        # xys = inp["xys"]
        y = self.param[None]
        loss_matrix = np.abs(y - gts[:, None]).sum(-1)  # b, k
        if inp.get("no_split_for_ablation"):
            i_nears = loss_matrix.argmin(1)
        else:
            i_nears = self.sdd.add_loss_matrix(loss_matrix)["i_nears"]

            splited = self.sdd.try_split()
            if splited:
                i_split, i_disapear = splited["i_split"], splited["i_disapear"]
                self.param[i_disapear] = self.param[i_split]

        # backward of gradient descent
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


if __name__ == "__main__":
    from boxx import *
    from distribution_playground import *  # pip install distribution_playground

    # experments hyper-parameters

    bins = (
        100,
    ) * 2  # Quantization to a discrete distribution of h and w resolutions(i.e. shape), 10000 could scan QR_code
    # bins = (128,) * 2
    # bins = (256,) * 2

    k = 2000  # outputk
    itern = 1800  # iter number of each density
    # itern = 600
    batch = 40
    framen = 96  # GIF frames number of each density
    fps = 24  # for GIF play

    # density = boxx.resize(sda.astronaut(), (50, 50)).mean(-1)
    # k = 5000
    # itern = 6000
    # batch = 40
    k10000 = False
    if k10000:
        k = 10000
        itern = 10000
        batch = 10
        framen = 1000
        fps = 50

    no_split_for_ablation = ""
    # no_split_for_ablation = "_no.split"  # without split-and-prune tag
    density_name_seq = [
        "gaussian",
        "blur_circles",
        "QR_code",
        "sprial",
        "words",
        "gaussian",
        "uniform",  # the param initial from uniform, better end by uniform for close loop
    ]
    output_root = os.path.expanduser("~/junk/ddn_toy_exp")
    # output_root = os.path.expanduser("/tmp/junk/ddn_toy_exp")

    dirr = (
        output_root
        + f"/bin{bins[0]}_k{k}_itern{itern}_batch{batch}{no_split_for_ablation}/"
    )
    os.makedirs(dirr, exist_ok=True)
    strr = dirr + "\n"
    kls = []

    gen = GeneratorModel(k)
    previous_name = "uniform"
    frames_data_root = os.path.expanduser(
        f"{output_root}/frames_bin{bins[0]}_k{k}_itern{itern}_batch{batch}{no_split_for_ablation}"
    )
    for namei, name in enumerate(density_name_seq):
        density_map = density_map_builders[name](bins)
        dist = DistributionByDensityArray(density_map["density"])
        if "reset_count":
            last_param = gen.param
            gen = GeneratorModel(k)
            if "conintue_last_param":
                gen.param = last_param
        sdd = gen.sdd

        diver_gt = dist.divergence(dist.sample(k))
        print("gt:", dist.str_divergence(diver_gt))

        def log(big=False):
            diver = dist.divergence(gen.param)
            (shows if big else show)(
                uint8, histEqualize, diver, dist.density, diver_gt, density_to_rgb
            )
            print("divergence:", dist.str_divergence(diver))

        frames_data_dir = os.path.expanduser(
            f"{frames_data_root}/data/{namei}_{previous_name}-to-{name}"
        )
        os.makedirs(frames_data_dir, exist_ok=True)
        for iteri in range(itern):  # train loop
            # print(iteri)
            gts = dist.sample(batch)
            inp = dict(gts=gts, no_split_for_ablation=no_split_for_ablation)
            gen.train(inp)
            if not increase(
                "show log",
            ) % (itern // 5):
                log()
            if framen and not increase(
                "dump frame data",
            ) % (itern // framen):
                saveData(
                    dist.divergence(gen.param),
                    frames_data_dir + p / f"/{previous_name}-to-{name}_i{iteri:06}.pkl",
                )
        previous_name = name
        # break
        log()
        print("gt:", dist.str_divergence(dist.sample(k)))
        # sdd.plot_dist()
        # print("near2.sum(), iter=", sdd.near2.sum(), sdd.iter)
        globals().update(gen.sdd.__dict__)
        #     show - dist.divergence(gen.param[:100])
        # %%
        if "save vis and divergence":
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
                d_sample = density_to_rgb(dist.divergence(dist.sample(k))["estimated"])
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
# %%
if "convert_to_png_frames_and_gif":  # frames_data_root, k, bins = '.', 10000, [100,100]
    from distribution_playground import density_to_rgb, density_map_builders
    from boxx import *

    gt_density_maps = {n: f(bins) for n, f in density_map_builders.items()}

    print(frames_data_root)
    globp = f"{frames_data_root}/data/[12345]_*/*.pkl"
    png_dir = frames_data_root + "/png"
    os.system(f"rm {png_dir}/*.png")
    os.makedirs(png_dir, exist_ok=True)
    pklps = sorted(glob(globp))
    for pkli, pklp in enumerate(pklps):
        divergence = loadData(pklp)
        fname = filename(pklp)
        iteri = int(fname[fname.rindex("_i") + 2 :])
        target_name = fname[fname.index("-to-") + 4 : fname.rindex("_i")]
        source_name = fname[: fname.index("-to-")]
        target_density_map = gt_density_maps[target_name]
        vis_max_probability = gt_density_maps["gaussian"]["density"].max() * 1.5
        # vis_max_probability = target_density_map["density"].max()

        vis_max_probability = max(
            vis_max_probability, 1.5 / k
        )  # minimum probability 1.5/k to make density distinguishable
        vis = density_to_rgb(divergence["estimated"], vis_max_probability)
        if "add_target":
            vis_target = density_to_rgb(
                target_density_map["density"], vis_max_probability
            )
            pad = bins[0] // 32
            vis = np.concatenate([vis, np.ones_like(vis)[:, :pad] * 255, vis_target], 1)
            vis = padding(vis, pad)
            vis[(vis == 0).all(-1)] = 255
        png_path = f"{png_dir}/{pkli:05}_{basename(pklp.replace('.pkl', '.png'))}"
        imsave(png_path, vis)
    show - vis

    png_paths = sorted(glob(dirname(png_path) + "/*.png"))
    gif_path = frames_data_root + "/2d-density-estimation-DDN.gif"
    import imageio

    imageio.mimsave(gif_path, [imread(pa) for pa in png_paths], fps=fps, loop=0)
    print("Saving GIF to:", gif_path)
