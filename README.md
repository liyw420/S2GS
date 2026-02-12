# Sparsity Matters: Efficient On-Device Streaming of Free-Viewpoint Videos via Structured Sparse Gaussian Splatting

### Project Page | Paper | Supplementary Material

<div align="center">
  <img src="assets/teaser.png"/>
</div><br/>
**This repository is the official implementation of "Sparsity Matters: Efficient On-Device Streaming of Free-Viewpoint Videos via Structured Sparse Gaussian Splatting".** In this paper, we propose S²GS, a novel on-device FVV streaming framework that integrates end-to-end learnable structured sparsity into Gaussian Splatting. Notably, compared with [QUEEN](https://github.com/NVlabs/queen), S²GS reduces Gaussian primitives by 67.6%, storage costs by 84.9%, and training time by 59.5%, while achieving 480+ FPS rendering on the N3DV dataset.

---
## 🔥 News
- 2026/02: Initial code release!

---
## 🛠️ Pipeline
<div align="center">
  <img src="assets/pipeline.png"/>
</div><br/>
Overview of the S²GS framework. (a) The streaming octree representation is initialized using root Gaussian primitives from the first frame and subsequently represents FVVs with multi-resolution hierarchical grids, enabling Level-of-Motion modeling and efficient point queries. (b) A learnable structured gating mechanism is introduced to sparsify Gaussian residuals via hierarchical feature propagation, differentiable sampling with Gumbel Sigmoid, and gate discretization using a straight-through estimator (STE). (c) A sparse regularization loss, together with compact linear decoding, is incorporated to achieve efficient end-to-end training of structured gates, latent codes, decoders, and positional offset residuals.

## 🌟 Get started

### Environment

The hardware and software requirements are the same as those of the [QUEEN](https://github.com/NVlabs/queen), which this code is built upon. To setup the environment, please follow:

```shell
git clone https://github.com/S2GS
cd S2GS
# Install dependencies from requirements.txt
pip install -r requirements.txt

# Install CUDA-dependent submodules
pip install --no-build-isolation ./submodules/simple-knn
pip install --no-build-isolation ./submodules/diff-gaussian-rasterization
pip install --no-build-isolation ./submodules/gaussian-rasterization-grad
```

### Data preparation

**N3DV dataset:**

Download the [Neural 3D Video dataset](https://github.com/facebookresearch/Neural_3D_Video) and extract vidoes of each scene to `./data/dynerf/scene_name`.

**Meet Room dataset:**

Download the [Meet Room dataset](https://github.com/AlgoHunt/StreamRF?tab=readme-ov-file) and extract videos of each scene to `./data/meetroom/scene_name`.

**ENeRF Outdoor dataset:**

Please reach out to the authors of the paper [Efficient neural radiance fields for interactive free-viewpoint video](https://github.com/zju3dv/ENeRF) for access to the dataset. Please extract videos of each scene to `./data/enerf/scene_name`.

### Running

For data preprocessing, model training, rendering, and evaluation on the N3DV Dataset:

```
bash ./scripts/pre_dynerf/run_train_eval.sh
```
For data preprocessing, model training, rendering, and evaluation on the Meet Room Dataset:

```
bash ./scripts/pre_meetroom/run_train_eval.sh
```
For data preprocessing, model training, rendering, and evaluation on the ENeRF Outdoor Dataset:

```
bash ./scripts/pre_enerf/run_train_eval.sh
```
---

## 🙏 Acknowledgements

GC-4DGS builds upon the original [QUEEN](https://github.com/NVlabs/queen) codebase. Besides, we thank all authors from [Octree-GS](https://github.com/city-super/Octree-GS) and [3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting) for presenting such excellent works.
