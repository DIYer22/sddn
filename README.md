# `sddn`: Core Library of Discrete Distribution Networks
<small>

`sddn` is the abbreviation of "Splitable Discrete Distribution Networks".
</small>

This repo only includes the core implementation of DDN and simple experiments (2D density estimation and MNIST example).

More info about DDN at: [https://discrete-distribution-networks.github.io/](https://discrete-distribution-networks.github.io/)


```bash
# Install by pip
pip install sddn
```
## ▮ Toy Example for 2D density estimation
Need install [`distribution_playground`](https://github.com/DIYer22/distribution_playground): 2D probability distribution playground for generative Models

```bash
pip install distribution_playground
git clone https://github.com/DIYer22/sddn.git
python sddn/toy_exp.py
```
[toy_exp.py](toy_exp.py) includes:
- Training a tiny DDN to fit probability densities
- Logging and recording divergence metrics between sampling results and GT density maps
- Saving visualization images of final sampling results
- Creating cool "optimization process GIFs":

<br>
<div align="center">
  <a target="_blank" href="https://discrete-distribution-networks.github.io/2d-density-estimation-gif-with-10000-nodes-ddn.html">
    <img src="https://discrete-distribution-networks.github.io/img/frames_bin100_k2000_itern1800_batch40_framen96_2d-density-estimation-DDN.gif" style="height:">
  </a>
  <small><br>DDN optimization process of 2D density estimation <a target="_blank" href="https://discrete-distribution-networks.github.io/2d-density-estimation-gif-with-10000-nodes-ddn.html"><small>[details]</small></a><br>Left: All samples; Right: GT density</small>
</div>
<br>


## ▮ MNIST example
TBD

## ▮ Citation
```bibtex
@inproceedings{yang2025discrete,
  title     = {Discrete Distribution Networks},
  author    = {Lei Yang},
  booktitle = {The Thirteenth International Conference on Learning Representations},
  year      = {2025},
  url       = {https://openreview.net/forum?id=xNsIfzlefG}}
```
**License:** MIT License