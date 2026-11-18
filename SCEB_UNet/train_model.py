"""Reproducible training entry point for SCEB-UNet and its ablations."""

from datetime import datetime
import json
import logging
import os
from pathlib import Path
import platform
import random
import shutil
import sys

import numpy as np
import torch
from tensorboardX import SummaryWriter
from torch.backends import cudnn
from torch.utils.data import DataLoader, Subset

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from release_models import build_model
import Config as config
from Load_Dataset import ImageToImage2D, RandomGenerator, ValGenerator
from Train_one_epoch import train_one_epoch
from utils import CosineAnnealingWarmRestarts, WeightedDiceBCE


def logger_config(log_path):
    logger = logging.getLogger("acc_unet_train")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    for handler in logger.handlers[:]:
        handler.close()
        logger.removeHandler(handler)

    formatter = logging.Formatter("%(message)s")
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    cudnn.benchmark = not config.deterministic
    cudnn.deterministic = config.deterministic


def worker_init_fn(worker_id):
    random.seed(config.seed + worker_id)
    np.random.seed(config.seed + worker_id)


def validate_resume_numerics(checkpoint):
    """A numerical-protocol correction must start a separate experiment."""
    previous = checkpoint.get("training_config", {}).get(
        "numerical_protocol", "legacy_probability_v1"
    )
    if previous != config.numerical_protocol:
        raise ValueError(
            f"Cannot resume {previous} checkpoint under {config.numerical_protocol}; "
            "start a fresh experiment instead of mixing numerical protocols"
        )
    if previous == "stable_logits_v2":
        if checkpoint.get("training_config", {}).get("amp_dtype") != config.amp_dtype:
            raise ValueError("Stable resume requires the same recorded AMP dtype")


def dataloader_kwargs(shuffle, generator=None):
    kwargs = {
        "batch_size": config.batch_size,
        "shuffle": shuffle,
        "num_workers": config.num_workers,
        "pin_memory": config.pin_memory,
        "worker_init_fn": worker_init_fn,
    }
    if generator is not None:
        kwargs["generator"] = generator
    if config.num_workers > 0:
        kwargs.update(persistent_workers=True, prefetch_factor=2)
    return kwargs


def save_model_checkpoint(
    model,
    optimizer,
    epoch,
    val_loss,
    val_dice,
    save_path,
    filename="best_model.pth.tar",
    scheduler=None,
    scaler=None,
    best_dice=None,
    best_epoch=None,
    train_generator=None,
):
    checkpoint = {
        "epoch": epoch,
        "model": config.model_name,
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "scaler": scaler.state_dict() if scaler is not None else None,
        "val_loss": float(val_loss),
        "val_dice": float(val_dice),
        "best_dice": float(val_dice if best_dice is None else best_dice),
        "best_epoch": int(epoch + 1 if best_epoch is None else best_epoch),
        "model_config": {
            "release_variant": config.release_variant,
            "n_channels": config.n_channels,
            "n_classes": config.n_labels,
            "n_filts": config.n_filts,
            "attention_type": config.attention_type,
            "attention_scope": config.attention_scope,
            "feature_fusion_type": config.feature_fusion_type,
            "rcmf_calibration_limit": config.rcmf_calibration_limit,
            "boundary_fusion": config.boundary_fusion,
            "boundary_fusion_version": config.boundary_fusion_version,
            "ugbf_residual_limit": config.ugbf_residual_limit,
            "context_energy": config.context_energy,
            "context_residual": config.context_residual,
            "scale_calibration": config.scale_calibration,
            "ablation_mode": config.ablation_mode,
            "structure_version": config.structure_version,
            "preserve_boundary_rng": config.matched_rng_protocol,
            "numerical_protocol": config.numerical_protocol,
            "activation_checkpointing": config.activation_checkpointing,
        },
        "training_config": {
            "task_name": config.task_name,
            "seed": config.seed,
            "batch_size": config.batch_size,
            "paper_batch_size": config.paper_batch_size,
            "gradient_accumulation_steps": config.gradient_accumulation_steps,
            "use_amp": config.use_amp,
            "amp_init_scale": config.amp_init_scale,
            "learning_rate": config.learning_rate,
            "optimizer": config.optimizer_name,
            "reproducibility_split": config.reproducibility_split,
            "boundary_kernel_size": config.boundary_kernel_size,
            "boundary_loss_weight": config.boundary_loss_weight,
            "deep_supervision_weight": config.deep_supervision_weight,
            "initialization_checkpoint": config.initialization_checkpoint,
            "calibration_only": config.calibration_only,
            "preprocessing_version": config.preprocessing_version,
            "matched_rng_protocol": config.matched_rng_protocol,
            "data_order_seed": config.data_order_seed,
            "numerical_protocol": config.numerical_protocol,
            "amp_dtype": config.amp_dtype,
            "grad_scaler_enabled": (config.use_amp and config.use_cuda
                and (config.numerical_protocol == "legacy_probability_v1"
                     or config.amp_dtype == "float16")),
            "gpu_duty_cycle": config.gpu_duty_cycle,
            "activation_checkpointing": config.activation_checkpointing,
        },
        "rng_state": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch_cpu": torch.get_rng_state(),
            "torch_cuda": (
                torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
            ),
            "train_generator": (
                train_generator.get_state() if train_generator is not None else None
            ),
        },
    }
    destination = Path(save_path) / filename
    temporary = destination.with_name(f"{destination.name}.tmp")
    torch.save(checkpoint, temporary)
    os.replace(temporary, destination)
    return destination


def save_best_model(model, optimizer, epoch, val_loss, val_dice, save_path):
    """Backward-compatible best-checkpoint writer used by existing runs."""
    return save_model_checkpoint(
        model, optimizer, epoch, val_loss, val_dice, save_path
    )


def restore_rng_state(checkpoint, train_generator=None):
    """Restore stochastic state for exact continuation when it is available."""
    state = checkpoint.get("rng_state")
    if not state:
        return False
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"].cpu())
    cuda_state = state.get("torch_cuda")
    if cuda_state is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([item.cpu() for item in cuda_state])
    generator_state = state.get("train_generator")
    if train_generator is not None and generator_state is not None:
        train_generator.set_state(generator_state.cpu())
    return True


def _normalise_checkpoint_key(key, target_keys):
    """Map a proven MLFC checkpoint into the RCMF stable-path namespace."""
    if key.startswith("module."):
        key = key[len("module."):]
    if key in target_keys:
        return key
    for prefix in ("mlfc1.", "mlfc2.", "mlfc3."):
        if key.startswith(prefix) and not key.startswith(f"{prefix}core."):
            candidate = f"{prefix}core.{key[len(prefix):]}"
            if candidate in target_keys:
                return candidate
    return key


def load_predecessor_initialization(model, checkpoint_path, device):
    """Load every predecessor parameter while leaving only new calibrators absent."""
    checkpoint_path = Path(checkpoint_path).resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Initialization checkpoint does not exist: {checkpoint_path}"
        )
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    source_state = checkpoint.get("state_dict", checkpoint)
    target_keys = set(model.state_dict())
    mapped_state = {
        _normalise_checkpoint_key(key, target_keys): value
        for key, value in source_state.items()
    }
    incompatible = model.load_state_dict(mapped_state, strict=False)
    allowed_missing = {
        name for name, _ in model.named_parameters()
        if ".router." in name
        or name.endswith(".local_gain")
        or name.endswith(".temperature_gain")
    }
    actual_missing = set(incompatible.missing_keys)
    if incompatible.unexpected_keys or actual_missing != allowed_missing:
        raise RuntimeError(
            "Unsafe predecessor mapping: "
            f"missing={sorted(actual_missing)}, "
            f"expected_missing={sorted(allowed_missing)}, "
            f"unexpected={sorted(incompatible.unexpected_keys)}"
        )
    return checkpoint_path, checkpoint, sorted(actual_missing)


def is_calibration_parameter(name):
    """Parameters introduced by RCMF or decoder BDC-SimAM."""
    return (
        ".router." in name
        or name.endswith(".local_gain")
        or name.endswith(".temperature_gain")
    )


def enter_training_mode(model):
    """Train calibrators without drifting frozen BatchNorm running statistics."""
    model.train()
    if config.calibration_only:
        for module in model.modules():
            if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                module.eval()


def build_datasets(train_transform, val_transform, logger):
    """Build fixed author splits when available, otherwise use split folders."""
    if config.task_name not in {
        "GlaS", "ISIC2018", "ClinicDB", "DRIVE", "DSB2018"
    }:
        return (
            ImageToImage2D(config.train_dataset, train_transform, image_size=config.img_size),
            ImageToImage2D(config.val_dataset, val_transform, image_size=config.img_size),
        )

    train_source = ImageToImage2D(
        config.train_dataset, train_transform, image_size=config.img_size
    )
    val_source = ImageToImage2D(
        config.train_dataset, val_transform, image_size=config.img_size
    )
    def read_stems(filename):
        path = config.REPRO_SPLIT_DIR / filename
        if not path.is_file():
            raise FileNotFoundError(f"Missing author split file: {path}")
        stems = [
            line.strip() for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if len(stems) != len(set(stems)):
            raise RuntimeError(f"Duplicate sample names in {path}")
        return stems

    index_by_stem = {
        image_path.stem: index
        for index, (image_path, _) in enumerate(train_source.samples)
    }
    train_stems = read_stems("Train.txt")
    val_stems = read_stems("Val.txt")
    missing = sorted((set(train_stems) | set(val_stems)) - set(index_by_stem))
    if missing:
        raise RuntimeError(f"Author split references missing images: {missing}")
    overlap = sorted(set(train_stems) & set(val_stems))
    if overlap:
        raise RuntimeError(f"Train/validation overlap in author split: {overlap}")
    train_indices = [index_by_stem[stem] for stem in train_stems]
    val_indices = [index_by_stem[stem] for stem in val_stems]
    expected_sizes = {
        "GlaS": (68, 17),
        "ISIC2018": (1557, 519),
        "ClinicDB": (368, 122),
        "DRIVE": (16, 4),
        "DSB2018": (536, 134),
    }
    expected_train, expected_val = expected_sizes[config.task_name]
    if len(train_indices) != expected_train or len(val_indices) != expected_val:
        raise RuntimeError(
            f"Unexpected {config.task_name} exp1 sizes: "
            f"{len(train_indices)} train / {len(val_indices)} val"
        )

    manifest = {
        "strategy": "authors_reproducibility_split",
        "experiment": config.reproducibility_split,
        "source_pairs": len(train_source),
        "train_samples": [train_source.samples[index][0].name for index in train_indices],
        "validation_samples": [val_source.samples[index][0].name for index in val_indices],
        "dataset_note": (
            "train_57.bmp was restored after byte-level mask-pair verification."
            if config.task_name == "GlaS"
            else "Exact ACC-UNet author exp1 names; train, validation and test are disjoint."
        ),
    }
    manifest_path = Path(config.save_path) / "split_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)
    logger.info(
        f"{config.task_name} author {config.reproducibility_split}: "
        f"{len(train_indices)} train / {len(val_indices)} validation; "
        f"manifest={manifest_path}"
    )
    return Subset(train_source, train_indices), Subset(val_source, val_indices)


def main_loop(logger):
    device = torch.device("cuda" if config.use_cuda else "cpu")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    train_transform = RandomGenerator(
        (config.img_size, config.img_size),
        preprocessing_version=config.preprocessing_version,
    )
    val_transform = ValGenerator(
        (config.img_size, config.img_size),
        preprocessing_version=config.preprocessing_version,
    )
    train_dataset, val_dataset = build_datasets(train_transform, val_transform, logger)
    train_generator = None
    if config.matched_rng_protocol:
        train_generator = torch.Generator()
        train_generator.manual_seed(config.data_order_seed)
    train_loader = DataLoader(
        train_dataset,
        **dataloader_kwargs(shuffle=True, generator=train_generator),
    )
    val_loader = DataLoader(val_dataset, **dataloader_kwargs(shuffle=False))

    model = build_model(
        variant=config.release_variant,
        n_channels=config.n_channels,
        n_classes=config.n_labels,
        n_filts=config.n_filts,
        attention_type=config.attention_type,
        attention_scope=config.attention_scope,
        boundary_fusion=config.boundary_fusion,
        context_energy=config.context_energy,
        context_residual=config.context_residual,
        scale_calibration=config.scale_calibration,
        feature_fusion_type=config.feature_fusion_type,
        rcmf_calibration_limit=config.rcmf_calibration_limit,
        activation_checkpointing=config.activation_checkpointing,
        ugbf_residual_limit=config.ugbf_residual_limit,
        preserve_boundary_rng=config.matched_rng_protocol,
        numerical_protocol=config.numerical_protocol,
    ).to(device)

    resume_path_text = os.environ.get("ACC_RESUME_CHECKPOINT", "").strip()
    if resume_path_text and config.initialization_checkpoint:
        raise ValueError(
            "ACC_RESUME_CHECKPOINT and ACC_INIT_CHECKPOINT are mutually exclusive"
        )

    initialization = None
    if config.initialization_checkpoint:
        initialization = load_predecessor_initialization(
            model, config.initialization_checkpoint, device
        )

    if config.calibration_only:
        if initialization is None and not resume_path_text:
            raise ValueError(
                "Calibration-only training requires ACC_INIT_CHECKPOINT or "
                "ACC_RESUME_CHECKPOINT"
            )
        for name, parameter in model.named_parameters():
            parameter.requires_grad_(is_calibration_parameter(name))

    trainable_parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    if not trainable_parameters:
        raise RuntimeError("No trainable parameters selected")

    if config.optimizer_name.lower() != "adam":
        raise ValueError("The clean validated protocol currently supports optimizer_name='adam' only")
    optimizer = torch.optim.Adam(
        trainable_parameters,
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    criterion = WeightedDiceBCE(dice_weight=0.5, BCE_weight=0.5, n_labels=config.n_labels)
    scheduler = (
        CosineAnnealingWarmRestarts(
            optimizer, T_0=10, T_mult=1, eta_min=config.eta_min
        )
        if config.cosineLR else None
    )
    writer = SummaryWriter(config.tensorboard_folder) if config.tensorboard else None
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=(config.use_amp and device.type == "cuda"
                 and (config.numerical_protocol == "legacy_probability_v1"
                      or config.amp_dtype == "float16")),
        init_scale=config.amp_init_scale,
    )

    logger.info(f"Session: {config.session_name}")
    logger.info(f"Dataset: {config.task_name}")
    logger.info(f"Device: {device} ({platform.node()})")
    logger.info(
        f"Model: ACC_UNet, n_filts={config.n_filts}, "
        f"attention={config.attention_type}, scope={config.attention_scope}, "
        f"feature_fusion={config.feature_fusion_type}, "
        f"RCMF_calibration_limit={config.rcmf_calibration_limit:g}, "
        f"UGBF={config.boundary_fusion}, version={config.boundary_fusion_version}, "
        f"UGBF_limit={config.ugbf_residual_limit:g}, "
        f"ablation={config.ablation_mode}, context_residual={config.context_residual}, "
        f"scale_calibration={config.scale_calibration}, structure={config.structure_version}"
    )
    logger.info(
        f"Matched RNG protocol={config.matched_rng_protocol}, "
        f"data_order_seed={config.data_order_seed}"
    )
    logger.info(
        f"Optimizer: Adam, lr={config.learning_rate}, batch={config.batch_size}, "
        f"seed={config.seed}"
    )
    logger.info(
        f"Protocol: author split={config.reproducibility_split}, epochs={config.epochs}, "
        f"patience={config.early_stopping_patience}, AMP={config.use_amp} "
        f"dtype={config.amp_dtype} "
        f"(init_scale={config.amp_init_scale:g}); physical batch={config.batch_size}, "
        f"accumulation={config.gradient_accumulation_steps}, "
        f"effective batch={config.batch_size * config.gradient_accumulation_steps}"
    )
    logger.info(f"Activation checkpointing={config.activation_checkpointing}")
    logger.info(f"Numerical protocol={config.numerical_protocol}; "
                f"GradScaler enabled={scaler.is_enabled()}")
    logger.info(f"GPU duty-cycle target={config.gpu_duty_cycle:.0%}")
    total_parameter_count = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameter_count = sum(
        parameter.numel() for parameter in trainable_parameters
    )
    logger.info(
        f"Parameters: total={total_parameter_count:,}, "
        f"trainable={trainable_parameter_count:,}; "
        f"calibration_only={config.calibration_only}"
    )
    if initialization is not None:
        init_path, init_checkpoint, missing_keys = initialization
        logger.info(
            f"Predecessor initialization: {init_path}; "
            f"loaded={len(model.state_dict()) - len(missing_keys)} tensors, "
            f"new={len(missing_keys)} tensors"
        )
    logger.info(
        f"UGBF losses: boundary={config.boundary_loss_weight}, "
        f"deep_supervision={config.deep_supervision_weight}, "
        f"boundary_kernel={config.boundary_kernel_size}"
    )

    best_dice = float("-inf")
    best_epoch = 0
    start_epoch = 0
    if resume_path_text:
        resume_path = Path(resume_path_text).resolve()
        if not resume_path.is_file():
            raise FileNotFoundError(f"Resume checkpoint does not exist: {resume_path}")
        checkpoint = torch.load(resume_path, map_location=device, weights_only=False)
        validate_resume_numerics(checkpoint)
        previous_training = checkpoint.get("training_config", {})
        previous_model = checkpoint.get("model_config", {})
        if any(previous_training.get(key, value) != value for key, value in (
            ("task_name", config.task_name), ("seed", config.seed)
        )):
            raise ValueError("Resume checkpoint dataset or seed does not match this run")
        if previous_model.get("release_variant") and previous_model["release_variant"] != config.release_variant:
            raise ValueError("Resume checkpoint variant does not match this run")
        for key in ("attention_type", "attention_scope", "context_residual",
                    "scale_calibration", "boundary_fusion", "feature_fusion_type",
                    "rcmf_calibration_limit", "ugbf_residual_limit"):
            if key in previous_model and previous_model[key] != getattr(config, key):
                raise ValueError(f"Resume checkpoint uses a different {key}")
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_epoch = int(checkpoint.get("best_epoch", start_epoch))
        best_dice = float(checkpoint.get("best_dice", checkpoint["val_dice"]))
        if scheduler is not None:
            scheduler_state = checkpoint.get("scheduler")
            if scheduler_state is not None:
                scheduler.load_state_dict(scheduler_state)
            else:
                scheduler.step(start_epoch)
        scaler_state = checkpoint.get("scaler")
        if scaler_state is not None:
            scaler.load_state_dict(scaler_state)
        rng_restored = restore_rng_state(checkpoint, train_generator)
        baseline_path = Path(config.model_path) / "best_model.pth.tar"
        previous_best_path = resume_path.parent / "best_model.pth.tar"
        if resume_path.name == "last_model.pth.tar" and previous_best_path.is_file():
            shutil.copyfile(previous_best_path, baseline_path)
        else:
            baseline_path = save_best_model(
                model,
                optimizer,
                start_epoch - 1,
                checkpoint["val_loss"],
                best_dice,
                config.model_path,
            )
        logger.info(
            f"Resumed from {resume_path}; next epoch={start_epoch + 1}, "
            f"best Dice={best_dice:.4f} at epoch {best_epoch}; "
            f"baseline copied to {baseline_path}; RNG restored={rng_restored}"
        )
    elif initialization is not None:
        logger.info("Validation of exact predecessor-equivalent initialization")
        model.eval()
        with torch.inference_mode():
            initial_val_loss, initial_val_dice = train_one_epoch(
                val_loader, model, criterion, optimizer, writer, -1,
                None, config.model_name, logger, device=device, scaler=scaler,
            )
        best_dice = initial_val_dice
        best_epoch = 0
        baseline_path = save_best_model(
            model,
            optimizer,
            -1,
            initial_val_loss,
            initial_val_dice,
            config.model_path,
        )
        logger.info(
            f"Protected fallback: initial Dice={initial_val_dice:.4f}; "
            f"saved {baseline_path}. A calibrated checkpoint replaces it only "
            "after improving validation Dice."
        )
    started_at = datetime.now()
    last_val_loss = float("nan")
    last_val_dice = float("nan")

    for epoch in range(start_epoch, config.epochs):
        logger.info(f"\n========= Epoch [{epoch + 1}/{config.epochs}] =========")
        enter_training_mode(model)
        train_one_epoch(
            train_loader, model, criterion, optimizer, writer, epoch,
            None, config.model_name, logger, device=device, scaler=scaler,
        )

        if (epoch + 1) % config.validate_every == 0:
            logger.info("Validation")
            model.eval()
            with torch.inference_mode():
                val_loss, val_dice = train_one_epoch(
                    val_loader, model, criterion, optimizer, writer, epoch,
                    None, config.model_name, logger, device=device, scaler=scaler,
                )
            last_val_loss = val_loss
            last_val_dice = val_dice

            if val_dice > best_dice:
                best_dice = val_dice
                best_epoch = epoch + 1
                checkpoint_path = save_best_model(
                    model, optimizer, epoch, val_loss, val_dice, config.model_path
                )
                logger.info(f"New best Dice {best_dice:.4f}; saved {checkpoint_path}")

        if scheduler is not None:
            scheduler.step()

        last_checkpoint_path = save_model_checkpoint(
            model,
            optimizer,
            epoch,
            last_val_loss,
            last_val_dice,
            config.model_path,
            filename="last_model.pth.tar",
            scheduler=scheduler,
            scaler=scaler,
            best_dice=best_dice,
            best_epoch=best_epoch,
            train_generator=train_generator,
        )
        logger.info(
            f"Latest completed epoch {epoch + 1}; saved {last_checkpoint_path}"
        )

        if epoch + 1 - best_epoch > config.early_stopping_patience:
            logger.info("Early stopping")
            break

    if writer is not None:
        writer.close()
    elapsed = datetime.now() - started_at
    logger.info(f"Training completed in {elapsed}; best Dice={best_dice:.4f} at epoch {best_epoch}")
    if device.type == "cuda":
        gibibyte = 1024 ** 3
        logger.info(
            "CUDA peak memory: "
            f"allocated={torch.cuda.max_memory_allocated(device) / gibibyte:.3f} GiB, "
            f"reserved={torch.cuda.max_memory_reserved(device) / gibibyte:.3f} GiB"
        )
    return model


def main():
    seed_everything(config.seed)
    save_path = Path(config.save_path)
    if save_path.exists():
        raise FileExistsError(
            f"Run directory already exists; change the session timestamp instead of overwriting it: {save_path}"
        )
    Path(config.model_path).mkdir(parents=True)
    Path(config.tensorboard_folder).mkdir(parents=True)
    Path(config.visualize_path).mkdir(parents=True)

    run_config = {
        name: getattr(config, name) for name in (
            "task_name", "model_name", "n_filts", "attention_type", "attention_scope",
            "feature_fusion_type", "rcmf_calibration_limit",
            "boundary_fusion", "boundary_kernel_size", "boundary_loss_weight",
            "deep_supervision_weight", "boundary_fusion_version",
            "ugbf_residual_limit", "preprocessing_version",
            "context_energy", "structure_version",
            "context_residual", "scale_calibration", "ablation_mode",
            "experiment_id", "release_variant",
            "seed", "epochs", "batch_size", "learning_rate", "optimizer_name",
            "paper_batch_size", "gradient_accumulation_steps", "use_amp", "amp_dtype",
            "matched_rng_protocol", "data_order_seed",
            "activation_checkpointing",
            "amp_init_scale", "weight_decay", "img_size", "validate_every",
            "early_stopping_patience", "reproducibility_split",
            "initialization_checkpoint", "calibration_only",
            "numerical_protocol",
            "gpu_duty_cycle",
        )
    }
    with (save_path / "run_config.json").open("w", encoding="utf-8") as handle:
        json.dump(run_config, handle, indent=2, ensure_ascii=False)

    logger = logger_config(config.logger_path)
    main_loop(logger)


if __name__ == "__main__":
    main()
