# SCEB-UNet

PyTorch implementation for *SCEB-UNet: Spatial energy modulation and calibrated multi-scale fusion for medical image segmentation*. The full model uses the Scale–Context Energy Block (SCEB), Residual-Calibrated Multi-scale Fusion (RCMF), and Uncertainty-Guided Boundary Fusion (UGBF).

## Installation

Use Python 3.10. The reference environment uses PyTorch 2.11.0 and torchvision 0.26.0. Install the matching CPU or CUDA wheels from the [PyTorch installation guide](https://pytorch.org/get-started/locally/), then install the remaining dependencies.

```bash
python -m pip install -r requirements.txt
python SCEB_UNet/run.py smoke
```

The smoke command runs the full network on a synthetic 224 × 224 input on CPU and checks the output shape and the 16,305,518 trainable parameters. The formal training preset requires a CUDA GPU with native BF16 support.

## Data and usage

See [dataset setup and frozen partitions](Reproducibility/README.md). Each `--data-root` points to the prepared root of one dataset. Obtain images and annotations from the dataset providers. The repository contains split identifiers and source code.

```bash
python SCEB_UNet/run.py check-data --dataset GlaS --data-root data/GlaS
python SCEB_UNet/run.py train --dataset GlaS --data-root data/GlaS --seed 2 --variant full
python SCEB_UNet/run.py evaluate --dataset GlaS --data-root data/GlaS --checkpoint "path/to/best_model.pth.tar"
```

The training command reports its timestamped run directory under `runs/`. Replace the example checkpoint path with that run's `models/best_model.pth.tar`. Use `--resume` with the same training configuration and a trusted `last_model.pth.tar` to continue a run. Only load checkpoints you trust.

Evaluation saves per-image and macro/global Dice and IoU under `runs/<dataset>/evaluation-*/`. Add `--save-predictions` to export masks, or `--cpu` for FP32 inference. Use the reference CUDA BF16 path and validation-selected checkpoints when comparing against the formal scores. CPU results can differ due to precision.

## Manuscript configurations

- `full` runs SCEB-UNet on any of the six datasets. It uses width 32, seed 2 by default, a physical batch of 4 with accumulation over 3 steps, and a maximum of 1,000 epochs. Validation hard Dice at 0.5 selects the checkpoint. The stable protocol uses BF16 network execution with FP32 spatial statistics and logits-based losses.
- `backbone` uses ordinary SimAM, MLFC and direct skip concatenation. For the paired GlaS evaluation, run `full` and `backbone` independently with seeds 0, 1 and 2.
- ClinicDB presets `without_sceb`, `rcmf_calibration_zero`, `without_ugbf`, and `concat_matched_aux` implement the component-removal settings. `backbone_aux` uses plain concatenation with FP32 fused output and retains the auxiliary heads. Use `full` and `backbone_aux` with seeds 0, 1 and 2 for the supervision-matched setting.
- GlaS design presets `glas_b0`, `glas_b1`, `glas_b2`, `glas_b3` and `glas_design_full` use global-energy attention, MLFC and the archived experimental settings. The archived study used the legacy probability-based FP16 protocol. The portable launcher applies these settings to the supplied source snapshot.

The release covers the SCEB-UNet model family, core training and overlap evaluation. Historical checkpoints, third-party comparison implementations and the manuscript's figure-generation pipeline are separate from this release.

## Attribution

- **HANC and MLFC.** SCEB-UNet draws on these designs for neighbourhood context aggregation and cross-level feature alignment. RCMF adds image-dependent source calibration before the MLFC core. [ACC-UNet, MICCAI 2023](https://doi.org/10.1007/978-3-031-43898-1_66)
- **ResPath.** The residual skip paths follow the design introduced in MultiResUNet. [MultiResUNet, Neural Networks 2020](https://doi.org/10.1016/j.neunet.2019.08.025)
- **SimAM.** The global-energy attention draws on SimAM's spatial energy formulation. [SimAM, ICML 2021](https://proceedings.mlr.press/v139/yang21o.html)

This release contains adapted code from the [ACC-UNet implementation](https://github.com/kiharalab/ACC-UNet). The adapted source and portable entry point were prepared on 2026-11-18. The upstream GNU GPL version 3 license text is in [LICENSE](LICENSE).
