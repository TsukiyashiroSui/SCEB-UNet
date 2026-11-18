"""Single source of truth for reproducible ACC-UNet experiments."""

from pathlib import Path
import os
import time

import ml_collections
import torch


# The default is the risk-controlled v2 candidate: an MLFC-equivalent RCMF
# initialization plus decoder-only bounded SimAM refinement. Legacy runners
# explicitly pin their own settings for checkpoint reproduction.
ablation_mode = os.environ.get("ACC_ABLATION_MODE", "full").strip().lower()
_ABLATIONS = {
    # name: (context-disagreement residual, scale-calibrated energy, UGBFv2)
    "none": (False, False, False),
    "base": (False, False, True),
    "context": (True, False, True),
    "scale": (False, True, True),
    "sce_no_ugbf": (True, True, False),
    "full": (True, True, True),
}
if ablation_mode not in _ABLATIONS:
    raise ValueError(
        f"ACC_ABLATION_MODE must be one of {sorted(_ABLATIONS)}, got {ablation_mode!r}"
    )
context_residual, scale_calibration, boundary_fusion = _ABLATIONS[ablation_mode]
context_energy = context_residual and scale_calibration  # legacy metadata
attention_type = os.environ.get(
    "ACC_ATTENTION_TYPE", "bdc_simam"
).strip().lower()
feature_fusion_type = os.environ.get(
    "ACC_FEATURE_FUSION", "rcmf"
).strip().lower()
rcmf_calibration_limit = float(
    os.environ.get("ACC_RCMF_CALIBRATION_LIMIT", "0.05")
)
if not 0.0 <= rcmf_calibration_limit <= 0.25:
    raise ValueError("ACC_RCMF_CALIBRATION_LIMIT must be in [0, 0.25]")
attention_scope = os.environ.get("ACC_ATTENTION_SCOPE", "decoder").strip().lower()
if attention_type not in {
    "se", "simam", "dc_simam", "bdc_simam", "bdc_simam_single"
}:
    raise ValueError(
        "ACC_ATTENTION_TYPE must be se, simam, dc_simam, bdc_simam, "
        "or bdc_simam_single"
    )
if feature_fusion_type not in {"mlfc", "rsrf", "rcmf"}:
    raise ValueError("ACC_FEATURE_FUSION must be mlfc, rsrf, or rcmf")
if attention_scope not in {"all", "encoder", "skip", "decoder"}:
    raise ValueError("ACC_ATTENTION_SCOPE must be all, encoder, skip, or decoder")

legacy_structure_version = {
    "none": "acc_hanc_plain",
    "base": "acc_hanc",
    "context": "context_disagreement_v1",
    "scale": "scale_calibrated_simam_v1",
    "sce_no_ugbf": "sce_hanc_v1_no_ugbf",
    "full": "sce_hanc_v1",
}[ablation_mode]
legacy_model_name = {
    "none": "ACC_UNet_Plain",
    "base": "ACC_UNet_ABL_BaseUGBFv2",
    "context": "ACC_UNet_ABL_ContextUGBFv2",
    "scale": "ACC_UNet_ABL_ScaleUGBFv2",
    "sce_no_ugbf": "ACC_UNet_ABL_SCE_NoUGBF",
    "full": "ACC_UNet_SCEH_UGBFv2",
}[ablation_mode]
if feature_fusion_type != "mlfc" or attention_type in {"dc_simam", "bdc_simam"}:
    scope_suffix = "" if attention_scope == "all" else f"_{attention_scope}"
    structure_version = (
        f"sceb_v2_{feature_fusion_type}_{attention_type}"
        f"{scope_suffix}_{ablation_mode}"
    )
    model_name = (
        f"SCEB_UNet_v2_{feature_fusion_type.upper()}_"
        f"{attention_type.upper()}{scope_suffix}_{ablation_mode}"
    )
else:
    structure_version = legacy_structure_version
    model_name = legacy_model_name
experiment_id = "".join(
    character for character in os.environ.get("ACC_EXPERIMENT_ID", "").strip()
    if character.isalnum() or character in {"-", "_"}
)
release_variant = os.environ.get("ACC_RELEASE_VARIANT", "")
if experiment_id:
    model_name = f"SCEB_UNet_Ablation_{experiment_id}"
n_channels = 3
n_labels = 1
n_filts = 32

# Uncertainty-Guided Boundary Fusion (UGBF). Set False for the SimAM-only
# ablation; the network then follows the original forward path exactly.
boundary_fusion_version = "v2_detached_low_aux" if boundary_fusion else "none"
boundary_kernel_size = 5
boundary_loss_weight = 0.03 if boundary_fusion else 0.0
deep_supervision_weight = 0.02 if boundary_fusion else 0.0
ugbf_residual_limit = float(os.environ.get("ACC_UGBF_RESIDUAL_LIMIT", "1.0"))
if not 0.0 < ugbf_residual_limit <= 1.0:
    raise ValueError("ACC_UGBF_RESIDUAL_LIMIT must be in (0, 1]")
if boundary_fusion and ugbf_residual_limit < 1.0:
    boundary_fusion_version = f"v3_bounded_{ugbf_residual_limit:g}"

preprocessing_version = os.environ.get(
    "ACC_PREPROCESSING_VERSION", "legacy_v1"
).strip().lower()
if preprocessing_version not in {"legacy_v1", "corrected_v2"}:
    raise ValueError(
        "ACC_PREPROCESSING_VERSION must be legacy_v1 or corrected_v2"
    )

# Paper-aligned GlaS protocol. Physical batch 4 with accumulation preserves
# the paper's effective batch 12 while fitting a 16 GB GPU.
seed = int(os.environ.get("ACC_SEED", "2"))
epochs = int(os.environ.get("ACC_EPOCHS", "1000"))
img_size = 224
batch_size = int(os.environ.get("ACC_BATCH_SIZE", "4"))
paper_batch_size = 12
gradient_accumulation_steps = int(os.environ.get("ACC_ACCUM_STEPS", "3"))
if batch_size < 1 or gradient_accumulation_steps < 1:
    raise ValueError("Batch size and gradient accumulation must be positive")
if batch_size * gradient_accumulation_steps != paper_batch_size:
    raise ValueError(
        "The shared GlaS protocol requires physical batch times accumulation "
        f"to equal {paper_batch_size}, got "
        f"{batch_size} x {gradient_accumulation_steps}"
    )
use_amp = True
numerical_protocol = os.environ.get(
    "ACC_NUMERICAL_PROTOCOL", "legacy_probability_v1"
).strip().lower()
if numerical_protocol not in {"legacy_probability_v1", "stable_logits_v2"}:
    raise ValueError("Unknown ACC_NUMERICAL_PROTOCOL")
amp_dtype = os.environ.get(
    "ACC_AMP_DTYPE", "bfloat16" if numerical_protocol == "stable_logits_v2" else "float16"
).strip().lower()
if amp_dtype not in {"float16", "bfloat16"}:
    raise ValueError("ACC_AMP_DTYPE must be float16 or bfloat16")
if numerical_protocol == "stable_logits_v2" and amp_dtype != "bfloat16":
    raise ValueError("stable_logits_v2 freezes AMP to bfloat16")
amp_init_scale = 1024.0
activation_checkpointing = os.environ.get("ACC_CHECKPOINTING", "0") == "1"
matched_rng_protocol = os.environ.get("ACC_MATCHED_RNG_PROTOCOL", "0") == "1"
data_order_seed = seed + 1_000_003
learning_rate = float(os.environ.get("ACC_LEARNING_RATE", "1e-3"))
optimizer_name = "adam"
weight_decay = 0.0
cosineLR = True
eta_min = 1e-4
validate_every = 1
early_stopping_patience = int(os.environ.get("ACC_PATIENCE", "100"))
initialization_checkpoint = os.environ.get("ACC_INIT_CHECKPOINT", "").strip()
calibration_only = os.environ.get("ACC_CALIBRATION_ONLY", "0").strip() == "1"
if epochs < 1 or early_stopping_patience < 1:
    raise ValueError("Epochs and early-stopping patience must be positive")
num_workers = int(os.environ.get("ACC_NUM_WORKERS", "4"))
gpu_duty_cycle = float(os.environ.get("ACC_GPU_DUTY_CYCLE", "1.0"))
if not 0.05 <= gpu_duty_cycle <= 1.0:
    raise ValueError("ACC_GPU_DUTY_CYCLE must be in [0.05, 1.0]")
print_frequency = 17
vis_frequency = 100
save_model = True
tensorboard = os.environ.get("ACC_TENSORBOARD", "1").strip() == "1"
deterministic = True
pretrain = False

# Dataset. Supported layouts are resolved explicitly; no fuzzy mask matching.
task_name = os.environ.get("ACC_TASK_NAME", "GlaS").strip()
PROJECT_ROOT = Path(__file__).resolve().parent.parent
dataset_base_override = os.environ.get("ACC_DATASET_BASE", "").strip()
DATASET_BASE = (
    Path(dataset_base_override).resolve()
    if dataset_base_override
    else PROJECT_ROOT / "dataset_split" / task_name
)
reproducibility_split = "exp1"
_REPRO_TASK_DIRS = {
    "GlaS": "GlaS",
    "ISIC2018": "ISIC18",
    "BUSI": "BUSI",
    "ClinicDB": "ClinicDB",
    "DSB2018": "DSB2018",
}
REPRO_SPLIT_DIR = (
    PROJECT_ROOT / "Reproducibility" / "dataset_split" /
    _REPRO_TASK_DIRS.get(task_name, task_name) /
    reproducibility_split
)

_DATASET_SPLITS = {
    "ISIC2018": ("Train_Folder", "Val_Folder", "Test_Folder"),
    "BUSI": ("Train_Folder", "Val_Folder", "Test_Folder"),
    # ClinicDB is stored once as 612 image/mask pairs.  The official
    # ACC-UNet reproducibility lists select the disjoint exp1 subsets.
    "ClinicDB": (".", ".", "."),
    # DSB2018 uses the 670-case official training release for development and
    # the 65-case official stage-1 test release as the untouched test set.
    "DSB2018": ("Train_Folder", "Train_Folder", "Test_Folder"),
    "KvasirSEG": ("Train_Folder", "Val_Folder", "Test_Folder"),
    # DRIVE's official test set must never be used for validation.  Validation
    # is a frozen subset of the 20 official training cases.
    "DRIVE": ("Train_Folder", "Train_Folder", "Val_Folder"),
    "GlaS": ("train", "testA", "testB"),
}
if task_name not in _DATASET_SPLITS:
    raise ValueError(f"Unsupported task_name: {task_name}")

_train_split, _val_split, _test_split = _DATASET_SPLITS[task_name]
train_dataset = str(DATASET_BASE / _train_split)
if task_name == "GlaS":
    # Keep both official challenge test subsets untouched. Validation is drawn
    # reproducibly from the training set by train_model.py.
    val_dataset = train_dataset
    test_datasets = {
        "testA": str(DATASET_BASE / "testA"),
        "testB": str(DATASET_BASE / "testB"),
    }
    test_dataset = test_datasets["testB"]  # compatibility for external scripts
else:
    val_dataset = str(DATASET_BASE / _val_split)
    test_dataset = str(DATASET_BASE / _test_split)
    test_datasets = {"test": test_dataset}

# ACC-UNet's published ISIC18 exp1 protocol selects train/validation/test
# samples from the labelled challenge-training pool by fixed name lists.
if task_name in {"ISIC2018", "ClinicDB"}:
    val_dataset = train_dataset
    test_dataset = train_dataset
    test_datasets = {"test": train_dataset}

# A timestamped run prevents accidental deletion or mixing of experiments.
experiment_tag = experiment_id or f"{attention_type}-{attention_scope}-{ablation_mode}"
if numerical_protocol != "legacy_probability_v1":
    experiment_tag = f"{experiment_tag}_{numerical_protocol}"
session_name = f"{experiment_tag}_{task_name}_{time.strftime('%Y%m%d-%H%M%S')}"
output_root_text = os.environ.get("ACC_OUTPUT_ROOT", "").strip()
OUTPUT_ROOT = Path(output_root_text) if output_root_text else PROJECT_ROOT / "Experiments"
save_path = str(OUTPUT_ROOT / task_name / model_name / session_name)
model_path = str(Path(save_path) / "models")
tensorboard_folder = str(Path(save_path) / "tensorboard_logs")
logger_path = str(Path(save_path) / f"{session_name}.log")
visualize_path = str(Path(save_path) / "visualize_val")

use_cuda = torch.cuda.is_available()
pin_memory = use_cuda


def get_CTranS_config():
    """Retained for the original comparison models shipped with ACC-UNet."""
    config = ml_collections.ConfigDict()
    config.transformer = ml_collections.ConfigDict()
    config.KV_size = 960
    config.transformer.num_heads = 4
    config.transformer.num_layers = 4
    config.expand_ratio = 4
    config.transformer.embeddings_dropout_rate = 0.1
    config.transformer.attention_dropout_rate = 0.1
    config.transformer.dropout_rate = 0
    config.patch_sizes = [16, 8, 4, 2]
    config.base_channel = 64
    config.n_classes = 1
    return config
