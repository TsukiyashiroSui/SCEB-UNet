"""Evaluate an ACC-UNet checkpoint on every configured held-out test split."""

import argparse
import csv
import json
from pathlib import Path
import sys

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from release_models import build_model
import Config as config
from Load_Dataset import ImageToImage2D, ValGenerator


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Checkpoint to evaluate. If omitted, the newest checkpoint for the configured task is used.",
    )
    parser.add_argument("--save-predictions", action="store_true")
    parser.add_argument("--output-dir", type=Path, help="New directory for evaluation results")
    return parser.parse_args(argv)


def find_latest_checkpoint():
    search_root = PROJECT_ROOT / "Experiments" / config.task_name / config.model_name
    if not search_root.is_dir():
        raise FileNotFoundError(f"Experiment directory does not exist: {search_root}")

    checkpoints = {
        path.resolve()
        for pattern in ("**/models/best_model.pth.tar", "**/models/best_model*.pth.tar")
        for path in search_root.glob(pattern)
        if path.is_file()
    }
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoint found below: {search_root}")
    return max(checkpoints, key=lambda path: path.stat().st_mtime)


def infer_legacy_attention(state_dict):
    se_markers = (".sqe.fc1.", ".sqe.attention.fc1.", ".sqes.")
    if any(any(marker in key for marker in se_markers) and ".fc1." in key for key in state_dict):
        return "se"
    return "simam"


def normalize_legacy_keys(state_dict):
    normalized = {}
    for key, value in state_dict.items():
        if key.startswith("module."):
            key = key[len("module."):]
        # Some early checkpoints wrapped ChannelSELayer under an ``attention``
        # attribute.  RSRF, however, intentionally owns a ModuleList named
        # ``attention``; removing that segment corrupts valid v2 keys.
        if not key.startswith("rsrf.attention."):
            key = key.replace(".attention.", ".")
        normalized[key] = value
    return normalized


def infer_n_filts(state_dict):
    key = "cnv11.conv3.weight"
    if key not in state_dict:
        raise KeyError(f"Cannot infer n_filts because {key} is absent from checkpoint")
    return int(state_dict[key].shape[0])


def binary_scores(prediction, target):
    prediction = prediction.astype(bool)
    target = target.astype(bool)
    intersection = np.logical_and(prediction, target).sum(dtype=np.float64)
    pred_sum = prediction.sum(dtype=np.float64)
    target_sum = target.sum(dtype=np.float64)
    union = np.logical_or(prediction, target).sum(dtype=np.float64)
    dice = 1.0 if pred_sum + target_sum == 0 else 2.0 * intersection / (pred_sum + target_sum)
    iou = 1.0 if union == 0 else intersection / union
    return dice, iou, intersection, pred_sum, target_sum, union


def evaluate_split(
    model, device, split_name, dataset_path, output_dir, save_predictions,
    preprocessing_version,
):
    transform = ValGenerator(
        (config.img_size, config.img_size),
        preprocessing_version=preprocessing_version,
    )
    dataset = ImageToImage2D(dataset_path, transform, image_size=config.img_size)
    if config.task_name in {"ISIC2018", "ClinicDB", "DRIVE", "DSB2018"}:
        index_by_stem = {
            image_path.stem: index
            for index, (image_path, _) in enumerate(dataset.samples)
        }
        split_file = config.REPRO_SPLIT_DIR / "Test.txt"
        stems = [
            line.strip() for line in split_file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        missing = sorted(set(stems) - set(index_by_stem))
        if missing:
            raise RuntimeError(
                f"{config.task_name} test split references missing images: {missing}"
            )
        dataset = torch.utils.data.Subset(
            dataset, [index_by_stem[stem] for stem in stems]
        )
        expected_test_sizes = {
            "ISIC2018": 518,
            "ClinicDB": 122,
            "DRIVE": 20,
            "DSB2018": 65,
        }
        expected_test_size = expected_test_sizes[config.task_name]
        if len(dataset) != expected_test_size:
            raise RuntimeError(
                f"Unexpected {config.task_name} exp1 test size: {len(dataset)}"
            )
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=config.num_workers)

    prediction_dir = output_dir / "predictions" / split_name
    if save_predictions:
        prediction_dir.mkdir(parents=True, exist_ok=True)

    image_dice = []
    image_iou = []
    records = []
    total_intersection = total_pred = total_target = total_union = 0.0

    with torch.inference_mode():
        for sampled_batch, names in tqdm(loader, desc=f"Testing {split_name}", unit="image"):
            images = sampled_batch["image"].to(device, non_blocking=True)
            targets = sampled_batch["label"].cpu().numpy()[0, 0]
            with torch.autocast(
                device_type=device.type,
                dtype=(
                    torch.bfloat16
                    if config.amp_dtype == "bfloat16"
                    else torch.float16
                ),
                enabled=config.use_amp and device.type == "cuda",
            ):
                probabilities = model(images)
            predictions = (probabilities >= 0.5).cpu().numpy()[0, 0]

            dice, iou, intersection, pred_sum, target_sum, union = binary_scores(
                predictions, targets
            )
            image_dice.append(dice)
            image_iou.append(iou)
            records.append((names[0], dice, iou))
            total_intersection += intersection
            total_pred += pred_sum
            total_target += target_sum
            total_union += union

            if save_predictions:
                name = Path(names[0]).stem + "_prediction.png"
                cv2.imwrite(str(prediction_dir / name), predictions.astype(np.uint8) * 255)

    with (output_dir / f"{split_name}_per_image.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("image", "dice", "iou"))
        writer.writerows(records)

    metrics = {
        "samples": len(dataset),
        "macro_dice": float(np.mean(image_dice)),
        "macro_iou": float(np.mean(image_iou)),
        "global_dice": float(2 * total_intersection / (total_pred + total_target + 1e-12)),
        "global_iou": float(total_intersection / (total_union + 1e-12)),
    }
    totals = {
        "image_dice": image_dice,
        "image_iou": image_iou,
        "intersection": total_intersection,
        "prediction": total_pred,
        "target": total_target,
        "union": total_union,
    }
    return metrics, totals


def main(argv=None):
    args = parse_args(argv)
    checkpoint_path = args.checkpoint.resolve() if args.checkpoint else find_latest_checkpoint()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    print(f"Automatically selected checkpoint: {checkpoint_path}" if args.checkpoint is None
          else f"Using checkpoint: {checkpoint_path}")

    device = torch.device("cuda" if config.use_cuda else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = normalize_legacy_keys(checkpoint["state_dict"])
    model_config = checkpoint.get("model_config", {})
    training_config = checkpoint.get("training_config", {})
    checkpoint_task = training_config.get("task_name")
    if checkpoint_task and checkpoint_task != config.task_name:
        raise ValueError(f"Checkpoint task is {checkpoint_task}, requested {config.task_name}")
    variant = model_config.get("release_variant") or model_config.get("control_variant", "")
    if variant.startswith("plain_s"):
        variant = "backbone_aux"
    attention_type = model_config.get("attention_type", infer_legacy_attention(state_dict))
    attention_scope = model_config.get("attention_scope", "all")
    feature_fusion_type = model_config.get("feature_fusion_type", "mlfc")
    rcmf_calibration_limit = model_config.get("rcmf_calibration_limit", 0.05)
    boundary_fusion = model_config.get("boundary_fusion", False)
    context_energy = model_config.get("context_energy", False)
    context_residual = model_config.get("context_residual", context_energy)
    scale_calibration = model_config.get("scale_calibration", context_energy)
    ugbf_residual_limit = model_config.get("ugbf_residual_limit", 1.0)
    preserve_boundary_rng = model_config.get("preserve_boundary_rng", False)
    preprocessing_version = training_config.get(
        "preprocessing_version", "legacy_v1"
    )
    numerical_protocol = model_config.get(
        "numerical_protocol", training_config.get("numerical_protocol", "legacy_probability_v1")
    )
    config.amp_dtype = training_config.get(
        "amp_dtype", "bfloat16" if numerical_protocol == "stable_logits_v2" else "float16"
    )

    model = build_model(
        variant=variant or config.release_variant,
        n_channels=model_config.get("n_channels", config.n_channels),
        n_classes=model_config.get("n_classes", config.n_labels),
        n_filts=model_config.get("n_filts", infer_n_filts(state_dict)),
        attention_type=attention_type,
        attention_scope=attention_scope,
        boundary_fusion=boundary_fusion,
        context_energy=context_energy,
        context_residual=context_residual,
        scale_calibration=scale_calibration,
        feature_fusion_type=feature_fusion_type,
        rcmf_calibration_limit=rcmf_calibration_limit,
        ugbf_residual_limit=ugbf_residual_limit,
        preserve_boundary_rng=preserve_boundary_rng,
        numerical_protocol=numerical_protocol,
    ).to(device)
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    output_dir = args.output_dir or checkpoint_path.parent.parent / "test_results"
    output_dir.mkdir(parents=True, exist_ok=args.output_dir is None)

    split_results = {}
    all_dice = []
    all_iou = []
    total_intersection = total_pred = total_target = total_union = 0.0
    for split_name, dataset_path in config.test_datasets.items():
        metrics, totals = evaluate_split(
            model, device, split_name, dataset_path, output_dir,
            args.save_predictions, preprocessing_version,
        )
        split_results[split_name] = metrics
        all_dice.extend(totals["image_dice"])
        all_iou.extend(totals["image_iou"])
        total_intersection += totals["intersection"]
        total_pred += totals["prediction"]
        total_target += totals["target"]
        total_union += totals["union"]

    combined = {
        "samples": len(all_dice),
        "macro_dice": float(np.mean(all_dice)),
        "macro_iou": float(np.mean(all_iou)),
        "global_dice": float(2 * total_intersection / (total_pred + total_target + 1e-12)),
        "global_iou": float(total_intersection / (total_union + 1e-12)),
    }
    results = {
        "checkpoint": str(checkpoint_path),
        "dataset": config.task_name,
        "evaluation_device": device.type,
        "evaluation_precision": config.amp_dtype if device.type == "cuda" and config.use_amp else "float32",
        "variant": variant or config.release_variant,
        "training_seed": training_config.get("seed"),
        "numerical_protocol": numerical_protocol,
        "amp_dtype": config.amp_dtype,
        "training_config": training_config,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "trainable_parameters": sum(
            parameter.numel() for parameter in model.parameters()
            if parameter.requires_grad
        ),
        "attention_type": attention_type,
        "attention_scope": attention_scope,
        "feature_fusion_type": feature_fusion_type,
        "rcmf_calibration_limit": rcmf_calibration_limit,
        "boundary_fusion": boundary_fusion,
        "ugbf_residual_limit": ugbf_residual_limit,
        "preprocessing_version": preprocessing_version,
        "matched_rng_protocol": training_config.get(
            "matched_rng_protocol", False
        ),
        "context_energy": context_energy,
        "context_residual": context_residual,
        "scale_calibration": scale_calibration,
        "ablation_mode": model_config.get("ablation_mode", "legacy"),
        "structure_version": model_config.get("structure_version", "acc_hanc"),
        "boundary_fusion_version": model_config.get(
            "boundary_fusion_version", config.boundary_fusion_version
        ),
        "splits": split_results,
        "combined": combined,
    }
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2, ensure_ascii=False)
    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
