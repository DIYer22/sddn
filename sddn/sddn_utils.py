import torch


def sinusoidal_embedding(n, d):
    # Returns the standard positional embedding
    embedding = torch.tensor(
        [[i / 10_000 ** (2 * j / d) for j in range(d)] for i in range(n)]
    )
    sin_mask = torch.arange(0, n, 2)

    embedding[sin_mask] = torch.sin(embedding[sin_mask])
    embedding[1 - sin_mask] = torch.cos(embedding[sin_mask])

    return embedding


def build_init_feature(shape_bchw=(1, 256, 64, 64)):
    """

    Parameters
    ----------
    shape_bchw : TYPE, optional
        DESCRIPTION. The default is (1, 256, 64, 64).

    Returns
    -------
    sinusoidal_embedding on spatial with last 4 channal plus linear

    """
    b, c, h, w = shape_bchw
    if h == 1:
        return torch.cat([torch.linspace(-1, 1, c * h * w).reshape(1, c, h, w)] * b)
    emb_sin = sinusoidal_embedding(h * w, c).reshape(h, w, c).permute(2, 0, 1)
    emb_linear = torch.linspace(-1, 1, h * w).reshape(h, w)
    emb_sin[-4:] += torch.cat([torch.rot90(emb_linear, k)[None] for k in range(4)])
    feature_init = torch.cat([emb_sin[None]] * b)
    return feature_init


if __name__ == "__main__":
    from boxx import *

    feature_init = build_init_feature()
    show(feature_init[0, :3])
    show(feature_init[0, -6:])
