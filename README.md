# Multi-Hierarchical GCN Surrogate

[![Python](https://img.shields.io/badge/python-3.10%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![PyG](https://img.shields.io/badge/PyG-2.0%2B-3c90cc)](https://pyg.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Paper](https://img.shields.io/badge/paper-DOI%3A10.1007%2Fs00466%2D024%2D02553%2D6-b31b1b)](https://doi.org/10.1007/s00466-024-02553-6)

> **Warning:** This is not the original **code** used in the corresponding paper. It is a re-implementation in PyTorch based on different tools.

A PyTorch implementation of a multi-hierarchical graph convolutional network surrogate model for structural mechanics simulations. The model learns to predict full-field displacement responses from simulation parameters by exploiting a coarse-to-fine mesh hierarchy, with each level learning a residual correction on top of its coarser neighbour.

## Method

The surrogate is built on a cascade of `MLPAutoencoder` modules, one per coarsening level, trained sequentially from coarsest to finest:

```
parameters μ  ──►  MLP  ──►  z  ──►  decoder  ──►  x_coarse
                                                      │
                                                  upsampler 
                                             (U + learned residual)
                                                      │
                                                      ▼
                                                    x_fine  
                                           (ground truth supervision)
```

Each autoencoder combines:
- A **graph encoder/decoder** based on Chebyshev spectral convolutions ([ChebConv](https://arxiv.org/abs/1606.09375), PyTorch Geometric).
- A small **MLP** that maps the parameter vector μ directly to the latent code z.
- An optional **upsampler** (fixed nearest-neighbour interpolation + learned linear residual) that maps decoded output to the next finer mesh.

### Training loss

```
total = λ_rec · L_rec  +  λ_x · L_x  +  λ_z · L_z  +  λ_up · L_up

L_rec = MSE(x,  decode(encode(x)))          # autoencoder reconstruction
L_x   = MSE(x,  decode(mlp(μ)))             # parameter prediction
L_z   = MSE(encode(x).detach(), mlp(μ))     # latent alignment
L_up  = MSE(x_fine, decode_fine(encode(x)))
      + MSE(x_fine, decode_fine(mlp(μ)))    # upsampling (zero when x_fine absent)
```

## Repository layout

```
src/
  data/           dataset utilities
  layers/         ChebConv wrapper (batched [B, N, F] interface)
  models/
    graph_autoencoder.py   GraphEncoder / GraphDecoder
    surrogate.py           MLP, MLPAutoencoder
    multi_hierarchical.py  MultiHierarchicalSurrogate (training orchestration)
  utils/          mesh sampling, sparse utilities
examples/
  plate_bending.py         self-contained synthetic example (no external data)
tests/            pytest test suite
train.py          CLI training script
train_debug.py    PyCharm / IDE run script
```

## Installation

```bash
pip install -e ".[dev]"
```

> PyTorch and PyTorch Geometric must be installed separately following the [PyG installation guide](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html) since the right CUDA/CPU variant depends on your hardware.

## Quick start

```python
from src.models.multi_hierarchical import MultiHierarchicalSurrogate

surrogate = MultiHierarchicalSurrogate(
    adjacency_list=adjacency_list,       # [A_fine, ..., A_coarse]
    downsampling_list=downsampling_list, # D_i: [N_{i+1}, N_i]
    upsampling_list=upsampling_list,     # U_i: [N_i, N_{i+1}]
    reduced_order=4,
    param_dim=mu_train.shape[1],
)

surrogate.fit(
    mu_train=mu_train, x_train=x_train,
    mu_val=mu_val,     x_val=x_val,
    coarsening_level=3,
    n_epochs=1500,
    batch_size=32,
)

x_pred = surrogate.predict(mu_test, level=0)  # full-resolution prediction
```

## Self-contained example

`examples/plate_bending.py` demonstrates the full pipeline on a **synthetic parametric plate-bending problem** — no external data required:

| Property | Value |
|---|---|
| Mesh | 25 × 25 regular grid → 625 nodes, ~1 152 triangles |
| Parameters | amplitude *A* ∈ [0.5, 2], wavenumber *k* ∈ [1, 3], time *t* ∈ [0, 1] |
| Displacement | sinusoidal bending field (analytical ground truth) |
| Hierarchy | 2 coarsening levels (25 % triangles kept each step) |

```bash
python examples/plate_bending.py
```

Output PNG figures are written to `checkpoints/plate_bending/` showing prediction vs. ground truth at every hierarchy level. See `train_debug.py` for a real-data example.

## Reference

If you use this code, please cite the original paper:

```bibtex
@article{kneifl24,
 title = {Multi-hierarchical surrogate learning for explicit structural dynamical systems using graph convolutional neural networks},
 author = {Kneifl, Jonas and Fehr, Jörg and Brunton, Steven L. and Kutz, J. Nathan},
 doi = {10.1007/s00466-024-02553-6},
 journal = {Computational Mechanics},
 month = {October},
 year = {2024}
}
```
