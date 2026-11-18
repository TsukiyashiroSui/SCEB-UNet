"""Validate and prepare the CC0 BBBC038v1 semantic-segmentation dataset."""

from __future__ import annotations

import argparse
from io import BytesIO
import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from PIL import Image


SPLITS = (
    Path(__file__).resolve().parent.parent
    / "Reproducibility" / "dataset_split" / "DSB2018" / "exp1"
)
SEED = 2


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def load_and_validate(path: Path, expected_rows: int, expected_split: str):
    table = pq.read_table(path)
    if table.num_rows != expected_rows:
        raise RuntimeError(f"{path.name}: expected {expected_rows} rows, got {table.num_rows}")
    rows = table.to_pylist()
    ids = [row["image_id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise RuntimeError(f"Duplicate image_id in {path}")
    if {row["split"] for row in rows} != {expected_split}:
        raise RuntimeError(f"Unexpected split values in {path}")
    return rows


def prepare_rows(rows, output: Path, split_folder: str):
    image_dir = output / split_folder / "img"
    mask_dir = output / split_folder / "labelcol"
    image_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)
    records = []

    for row in rows:
        image_id = row["image_id"]
        image_bytes = row["image"]["bytes"]
        mask_bytes = row["mask"]["bytes"]
        if not image_bytes or not mask_bytes:
            raise RuntimeError(f"Missing image or mask bytes for {image_id}")

        with Image.open(BytesIO(image_bytes)) as source_image:
            image = source_image.convert("RGB")
        with Image.open(BytesIO(mask_bytes)) as source_mask:
            mask_array = np.asarray(source_mask.convert("L"))

        expected_shape = (int(row["height"]), int(row["width"]))
        if (image.height, image.width) != expected_shape:
            raise RuntimeError(f"Image shape mismatch for {image_id}")
        if mask_array.shape != expected_shape:
            raise RuntimeError(f"Mask shape mismatch for {image_id}")
        mask_values = set(np.unique(mask_array).tolist())
        if not mask_values <= {0, 255}:
            raise RuntimeError(f"Non-binary mask values for {image_id}: {mask_values}")

        image_path = image_dir / f"{image_id}.png"
        mask_path = mask_dir / f"{image_id}.png"
        image.save(image_path, format="PNG", optimize=False)
        Image.fromarray(mask_array.astype(np.uint8), mode="L").save(
            mask_path, format="PNG", optimize=False
        )
        records.append(
            {
                "image_id": image_id,
                "source_image_sha256": digest(image_bytes),
                "source_mask_sha256": digest(mask_bytes),
                "height": image.height,
                "width": image.width,
                "num_nuclei": int(row["num_nuclei"]),
                "is_grayscale": bool(row["is_grayscale"]),
            }
        )
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="Directory containing both Parquet files")
    parser.add_argument("--output", type=Path, required=True, help="New directory for prepared image/mask pairs")
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        parser.error("Choose a new output directory so existing data cannot be overwritten.")
    train_path = args.source / "train-00000-of-00001.parquet"
    test_path = args.source / "stage1_test-00000-of-00001.parquet"
    train_rows = load_and_validate(train_path, 670, "stage1_train")
    test_rows = load_and_validate(test_path, 65, "stage1_test")

    train_ids = {row["image_id"] for row in train_rows}
    test_ids = {row["image_id"] for row in test_rows}
    if train_ids & test_ids:
        raise RuntimeError("Train/test image_id overlap")

    train_hashes = {digest(row["image"]["bytes"]) for row in train_rows}
    test_hashes = {digest(row["image"]["bytes"]) for row in test_rows}
    if train_hashes & test_hashes:
        raise RuntimeError("Train/test source-image hash overlap")

    sorted_ids = np.array(sorted(train_ids))
    shuffled = np.random.RandomState(SEED).permutation(sorted_ids)
    val_ids = sorted(shuffled[:134].tolist())
    development_ids = sorted(shuffled[134:].tolist())
    final_test_ids = sorted(test_ids)
    if len(development_ids) != 536 or len(val_ids) != 134 or len(final_test_ids) != 65:
        raise RuntimeError("Unexpected frozen split sizes")

    for part, ids in (("Train", development_ids), ("Val", val_ids), ("Test", final_test_ids)):
        if (SPLITS / f"{part}.txt").read_text(encoding="utf-8").splitlines() != ids:
            raise ValueError(f"The source produces a different {part} split")

    output.mkdir(parents=True, exist_ok=False)
    train_records = prepare_rows(train_rows, output, "Train_Folder")
    test_records = prepare_rows(test_rows, output, "Test_Folder")

    audit = {
        "dataset": "BBBC038v1 / DSB2018",
        "license": "CC0",
        "official_source": "https://bbbc.broadinstitute.org/BBBC038",
        "mirror": "https://huggingface.co/datasets/MedOtter/2018-Data-Science-Bowl",
        "seed": SEED,
        "train": len(development_ids),
        "validation": len(val_ids),
        "test": len(final_test_ids),
        "source_parquet_sha256": {
            train_path.name: digest(train_path.read_bytes()),
            test_path.name: digest(test_path.read_bytes()),
        },
        "cross_split_id_overlap": 0,
        "cross_source_image_hash_overlap": 0,
        "train_records": train_records,
        "test_records": test_records,
    }
    (output / "preparation_audit.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(
        f"DSB2018_PREPARED train={len(development_ids)} "
        f"val={len(val_ids)} test={len(final_test_ids)}"
    )


if __name__ == "__main__":
    main()
