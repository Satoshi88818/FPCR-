# FPCRv6: Advanced Front Point Cloud Reconstruction Network

**An advanced prototype for 3D human body reconstruction from partial front-facing point clouds using diffusion-based back completion and strong anthropomorphic priors.**

## Overview

FPCRv6 is a research-oriented PyTorch implementation that addresses the challenging task of reconstructing a full 3D human body (pose and shape via SMPL parameters) from only a single-view front point cloud. This is highly relevant for consumer-grade depth cameras (e.g., Kinect, iPhone LiDAR) where only the front is visible due to self-occlusion.

Unlike prior works (e.g., FPCR-Net, which uses separate equivariant processing but lacks explicit completion), this version introduces a **diffusion-based back completion module** conditioned on the observed front, initialized and regularized with **bilateral symmetry priors** — a first-principles anthropomorphic constraint rooted in human anatomy (near-perfect external left-right mirroring).

The pipeline:
1. **Completes the invisible back** using a conditional denoising diffusion process on point clouds.
2. Extracts **SO(3)-equivariant features** via e3nn's GatePointsNetwork (richer irreps up to l=2 for expressive geometric reasoning).
3. Predicts **per-point part segmentation** (24 SMPL parts) on front and completed back.
4. Aggregates part-wise features via soft attention for robust regression of **SMPL pose** (6D rotations → axis-angle) and **shape** parameters.

This design combines modern geometric DL (equivariance), generative modeling (diffusion for structured completion), and strong priors (symmetry) for stable, high-quality reconstruction from highly partial inputs.

## Key Innovations

- **Diffusion Back Completion with Symmetry Priors**:
  - Initializes noisy back as mirrored front (across YZ plane).
  - Conditional score network processes combined (noisy back + clean front) points.
  - Soft symmetry enforcement every sampling step: average with mirror + light noise.
  - Enables plausible completion without full-back supervision during inference.

- **Fully Equivariant Feature Extraction**:
  - Uses e3nn for true rotation/translation equivariance, critical for arbitrary orientations.

- **Part-Aware Regression**:
  - Soft part segmentation + weighted feature aggregation mimics structural reasoning.

- **Stable Pose Regression**:
  - 6D continuous rotation representation.

## Dependencies

- Python 3.8+
- PyTorch (tested on 2.x)
- torch-geometric
- e3nn
- smplx (`pip install smplx`)
- numpy

**Note**: Requires a valid SMPL model file (neutral gender recommended). Download from [SMPL website](https://smpl.is.tue.mpg.de/) and update the path in the code.

## Usage

### Inference (Front-only Completion + SMPL Regression)

```python
import torch
from FPCRv6 import FPCRNet  # assuming code saved as FPCRv6.py

model = FPCRNet().cuda().eval()

# P_F: [B, N, 3] front point cloud (e.g., N=1024)
P_F = torch.randn(1, 1024, 3).cuda()

theta, beta, P_B_completed, I_F_logit, I_B_logit, _ = model(P_F)

print(theta.shape)  # [B, 72] axis-angle pose
print(beta.shape)   # [B, 10] shape parameters
print(P_B_completed.shape)  # [B, N, 3] completed back points
```

To get full SMPL mesh:
```python
vertices, joints = model.smpl(theta, beta)
```

### Training (with Ground-Truth Back for Supervision)

The toy loop in the code demonstrates training on synthetic SMPL data with z-split front/back.

Add real datasets (e.g., CAPE partial scans, RenderPeople) for better generalization.

Supervision:
- Diffusion denoising loss on back.
- Pose/shape MSE.
- (Optional) Add cross-entropy on part segmentation, Chamfer on completed back.

## Limitations & Future Work

- Toy data uses simplistic z-splitting; real front scans have more complex occlusions.
- Diffusion sampling is basic (Euler, 50 steps) — replace with DDIM for faster inference.
- Point density fixed at 1024; scale for higher-res.
- No clothed human support yet (naked SMPL only).

Potential extensions:
- Integrate clothing via additional displacement or implicit fields.
- Multi-view consistency.
- Real-time optimization.

## Citation

This is an experimental prototype (v6). If you find it useful, consider citing related works:

- Original FPCR-Net: "FPCR-Net: Front Point Cloud Regression Network for End-to-End SMPL Parameter Estimation" (2024).
- e3nn: Geiger et al., "e3nn: Euclidean Neural Networks" (2022).
- Diffusion on points: Various recent works (e.g., Point-E, MHCDiff).

## License

MIT License (for the code structure). Note: SMPL model has its own academic license.

---

Built with first-principles reasoning: leveraging symmetry as a core axiom of human form, equivariance for geometric truth, and diffusion for probabilistic completion of the unknown. Enjoy experimenting!