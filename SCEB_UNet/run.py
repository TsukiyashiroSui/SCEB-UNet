"""Portable entry point for the SCEB-UNet manuscript configuration."""

import argparse
from datetime import datetime
import os
from pathlib import Path


DATASETS = ("GlaS", "ClinicDB", "DRIVE", "ISIC2018", "BUSI", "DSB2018")
STABLE_MODELS = {
    # Config ablation mode, attention, attention scope, fusion, calibration bound.
    "full": ("full", "bdc_simam", "decoder", "rcmf", "0.05"),
    "backbone": ("none", "simam", "all", "mlfc", "0.05"),
    "backbone_aux": ("base", "simam", "all", "mlfc", "0.0"),
    "without_sceb": ("base", "simam", "all", "rcmf", "0.05"),
    "rcmf_calibration_zero": ("full", "bdc_simam", "decoder", "rcmf", "0.0"),
    "without_ugbf": ("sce_no_ugbf", "bdc_simam", "decoder", "rcmf", "0.05"),
    "concat_matched_aux": ("full", "bdc_simam", "decoder", "rcmf", "0.05"),
}
GLAS_DESIGN = {
    "glas_b0": "base",
    "glas_b1": "context",
    "glas_b2": "scale",
    "glas_b3": "sce_no_ugbf",
    "glas_design_full": "full",
}
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def configure(args):
    design = args.variant in GLAS_DESIGN
    if design:
        mode, attention, scope, fusion, bound = (
            GLAS_DESIGN[args.variant], "simam", "all", "mlfc", "0.05"
        )
    else:
        mode, attention, scope, fusion, bound = STABLE_MODELS[args.variant]

    # Prevent local launchers or experiment overrides from leaking into a run.
    for name in tuple(os.environ):
        if name.startswith("ACC_"):
            del os.environ[name]
    os.environ.update({
        "ACC_TASK_NAME": args.dataset,
        "ACC_DATASET_BASE": str(args.data_root.resolve()),
        "ACC_OUTPUT_ROOT": str(args.output_root.resolve()),
        "ACC_SEED": str(args.seed),
        "ACC_RELEASE_VARIANT": args.variant,
        "ACC_EXPERIMENT_ID": f"{args.dataset}_s{args.seed}_{args.variant}",
        "ACC_ABLATION_MODE": mode,
        "ACC_ATTENTION_TYPE": attention,
        "ACC_ATTENTION_SCOPE": scope,
        "ACC_FEATURE_FUSION": fusion,
        "ACC_RCMF_CALIBRATION_LIMIT": bound,
        "ACC_UGBF_RESIDUAL_LIMIT": "1.0",
        "ACC_PREPROCESSING_VERSION": "legacy_v1",
        "ACC_NUMERICAL_PROTOCOL": "legacy_probability_v1" if design else "stable_logits_v2",
        "ACC_AMP_DTYPE": "float16" if design else "bfloat16",
        "ACC_CHECKPOINTING": "0" if design else "1",
        "ACC_MATCHED_RNG_PROTOCOL": "0" if design else "1",
        "ACC_NUM_WORKERS": "4" if design else "0",
        "ACC_TENSORBOARD": "0",
    })


def check_data():
    """Check exact identifiers, image/mask pairing and split coverage."""
    import Config as config
    from Load_Dataset import ImageToImage2D

    splits = {
        part: set((config.REPRO_SPLIT_DIR / f"{part}.txt").read_text(
            encoding="utf-8"
        ).splitlines())
        for part in ("Train", "Val", "Test")
    }
    train_ids, val_ids, test_ids = (splits[p] for p in ("Train", "Val", "Test"))
    if (train_ids & val_ids) or (train_ids & test_ids) or (val_ids & test_ids):
        raise ValueError("Dataset identifiers overlap among the frozen splits")

    def require_ids(path, expected):
        dataset = ImageToImage2D(path)
        actual = {image.stem for image, _ in dataset.samples}
        missing = expected - actual
        extra = actual - expected
        if missing or extra or len(actual) != len(dataset):
            raise ValueError(f"{path}: missing {sorted(missing)[:5]}, "
                             f"unexpected {sorted(extra)[:5]} or duplicate stems")

    if config.task_name == "BUSI":
        require_ids(config.train_dataset, train_ids)
        require_ids(config.val_dataset, val_ids)
    elif config.task_name in {"ISIC2018", "ClinicDB"}:
        require_ids(config.train_dataset, train_ids | val_ids | test_ids)
    else:
        require_ids(config.train_dataset, train_ids | val_ids)

    if config.task_name == "GlaS":
        for name, path in config.test_datasets.items():
            require_ids(path, {stem for stem in test_ids if stem.startswith(name + "_")})
    elif config.task_name not in {"ISIC2018", "ClinicDB"}:
        require_ids(config.test_dataset, test_ids)
    print(f"{config.task_name}: train/validation/test {len(train_ids)}/{len(val_ids)}/{len(test_ids)}")


def smoke():
    import torch
    from release_models import build_model

    torch.set_num_threads(2)
    model = build_model(
        n_channels=3, n_classes=1, n_filts=32, attention_type="bdc_simam",
        attention_scope="decoder", context_energy=True, context_residual=True,
        scale_calibration=True, boundary_fusion=True, feature_fusion_type="rcmf",
        rcmf_calibration_limit=0.05, ugbf_residual_limit=1.0,
        preserve_boundary_rng=True, activation_checkpointing=True,
        numerical_protocol="stable_logits_v2",
    ).eval()
    with torch.inference_mode():
        prediction = model(torch.zeros((1, 3, 224, 224)))
    parameters = sum(p.numel() for p in model.parameters())
    if parameters != 16_305_518 or prediction.shape != (1, 1, 224, 224):
        raise RuntimeError("The Full model does not match its reference size")
    if not torch.isfinite(prediction).all():
        raise RuntimeError("Nonfinite model output")
    print(f"SCEB-UNet CPU smoke passed | parameters={parameters:,} | shape={tuple(prediction.shape)}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    commands.add_parser("smoke", help="Run the Full model once on CPU using a synthetic input")
    for action in ("check-data", "train", "evaluate"):
        command = commands.add_parser(action)
        command.add_argument("--dataset", choices=DATASETS, required=True)
        command.add_argument("--data-root", type=Path, required=True)
        command.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "runs")
        command.add_argument("--variant", choices=tuple(STABLE_MODELS) + tuple(GLAS_DESIGN), default="full")
        command.add_argument("--seed", type=int, choices=(0, 1, 2), default=2)
        if action == "train":
            command.add_argument("--resume", type=Path, help="Last checkpoint for the same dataset and variant")
        if action == "evaluate":
            command.add_argument("--checkpoint", type=Path, required=True)
            command.add_argument("--save-predictions", action="store_true")
            command.add_argument("--cpu", action="store_true", help="Use FP32 CPU inference")
    args = parser.parse_args()

    if args.action == "smoke":
        smoke()
        return
    if args.variant in GLAS_DESIGN and args.dataset != "GlaS":
        parser.error("The component-design presets apply to GlaS")
    if args.variant not in {"full", "backbone"} and args.variant in STABLE_MODELS and args.dataset != "ClinicDB":
        parser.error("The matched controls apply to ClinicDB")
    if getattr(args, "cpu", False):
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    configure(args)
    check_data()
    if args.action == "train":
        import torch
        import train_model

        if not torch.cuda.is_available():
            parser.error("Training requires a CUDA GPU. Use smoke or evaluate --cpu for a CPU check.")
        if args.variant in STABLE_MODELS and not torch.cuda.is_bf16_supported(including_emulation=False):
            parser.error("The stable training protocol requires native CUDA BF16 support.")
        if args.resume:
            os.environ["ACC_RESUME_CHECKPOINT"] = str(args.resume.resolve())
        train_model.main()
    elif args.action == "evaluate":
        import test_model

        stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        output = args.output_root / args.dataset / f"evaluation-{stamp}"
        flags = ["--checkpoint", str(args.checkpoint.resolve()), "--output-dir", str(output)]
        if args.save_predictions:
            flags.append("--save-predictions")
        test_model.main(flags)


if __name__ == "__main__":
    main()
