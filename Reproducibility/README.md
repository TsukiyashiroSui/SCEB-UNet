# Dataset setup

Every dataset folder consumed by the loader contains `img/` for source images and `labelcol/` for matching masks. Use the filename stems in `dataset_split/<dataset>/exp1/`. Leave images at their source resolution. The loader reads images in BGR order, resizes the pairs and thresholds masks.

| Dataset and source | Layout under `--data-root` | Train / validation / test |
| --- | --- | --- |
| [GlaS](https://warwick.ac.uk/fac/sci/dcs/research/tia/glascontest/) | `train/`, `testA/`, `testB/`. For example, `train/img/train_1.bmp` pairs with `train/labelcol/train_1_anno.bmp`. Keep the original 85 training pairs together. | 68 / 17 / 80 |
| [CVC-ClinicDB](https://polyp.grand-challenge.org/CVCClinicDB/) | Place all 612 pairs in `img/` and `labelcol/` beneath the root. Keep numeric stems consistent with the manifest. The experiment uses the frame-level split. | 368 / 122 / 122 |
| [DRIVE](https://drive.grand-challenge.org/) | Put official training pairs in `Train_Folder/` and official test pairs in `Val_Folder/`. The latter is the held-out test folder in this project. Use the first observer's annotations. | 16 / 4 / 20 |
| [ISIC2018](https://challenge.isic-archive.com/data/) | Put the 2,594 labeled Task 1 challenge-training pairs in `Train_Folder/`. Preserve the `ISIC_…` image names and `_segmentation.png` mask suffixes. The local manifest directory is `ISIC18`. | 1,557 / 519 / 518 |
| [BUSI](https://doi.org/10.1016/j.dib.2019.104863) | Place image/mask pairs in `Train_Folder/`, `Val_Folder/` and `Test_Folder/` using the supplied lists. Preserve all `_mask*.png` files for each case. The loader unions multiple lesion annotations. Keep empty masks for the normal images. | 468 / 156 / 156 |
| [DSB2018 / BBBC038v1](https://bbbc.broadinstitute.org/BBBC038) | Put all 670 stage-1 development pairs in `Train_Folder/` and the 65 stage-1 evaluation pairs in `Test_Folder/`. Use one binary foreground mask per image. | 536 / 134 / 65 |

For DSB2018, the [Parquet mirror](https://huggingface.co/datasets/MedOtter/2018-Data-Science-Bowl) provides the stage-1 annotated pairs used by the preparation script. Download `train-00000-of-00001.parquet` and `stage1_test-00000-of-00001.parquet` to the same input directory. The script checks the resulting case IDs against the packaged lists before writing any pairs.

```bash
python -m pip install pyarrow==21.0.0
python SCEB_UNet/prepare_dsb2018.py --source data/DSB2018-parquet --output data/DSB2018
python SCEB_UNet/run.py check-data --dataset DSB2018 --data-root data/DSB2018
```

GlaS, ClinicDB and ISIC2018 use the project copies of the ACC-UNet `exp1` lists. DRIVE and DSB2018 use the fixed study lists. BUSI lists were exported from the actual 780-image folder split used in the formal experiments. The bundled upstream BUSI `exp1` files in the training workspace covered 647 images, so those files do not define the 780-image evaluation reported here.

The `check-data` command verifies image/mask pair names, exact split identifiers and split coverage. The public code archive includes identifier lists. Obtain the image and mask files from the cited providers under their respective use terms.
