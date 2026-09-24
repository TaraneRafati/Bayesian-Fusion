# Probabilistic Fusion of Global Spatial Attention and Local Multi-Component Intensity Statistics for Hemorrhage Detection

Code for weakly supervised localization of intracranial hemorrhage in non-contrast head CT. A slice-level classifier is trained with image-level labels only. Its class activation map (or ViT attention map) is then fused with Hounsfield-unit statistics through a closed-form ternary Bayesian posterior (hemorrhage, cranial bone, background), and thresholded with hysteresis to give a pixel-level mask.

## Repository layout

```
train.py                    classifier training (image-level labels)
fit_gmm.py                  per-class HU Gaussian mixtures fit on the training split
explain.py                  CAM / attention extraction, threshold search, GMM fusion, metrics, figures
callbacks/  losses/         training callbacks and loss
metrics/  models/           pixel, box and classification metrics; EfficientNetV2-B3, ResNet-18, DINOv2 ViT-S/14
optimizers/  plugins/       optimizer registry; dataset and CT windowing
utils/settings.py           shared defaults (windowing, augmentations, input sizes)
utils/xai_utils/            CAM and attention extraction, fusion, thresholds, metrics, statistics, paper figures
scripts/                    standalone plotting scripts for the paper figures
```

## Setup

```
pip install -r requirements.txt
```

DINOv2 weights are fetched through `torch.hub` (`facebookresearch/dinov2`) on first use.

## Data

Slice-level CSVs need the columns `patient_name`, `slice_number` and `hemorrhage_binary`. Each slice is a `.npy` file named `<patient_name>_<slice_number>.npy` in `--image_dir` and holds Hounsfield units. Ground-truth masks are `.npy` files with the same names in `--masks_dir`. No dataset paths are stored in the repository; every path is a command-line argument.

- Hemorica is used for classifier training, GMM estimation and in-distribution evaluation.
- Seg-CQ500 is used only as a cross-dataset test set. `utils/xai_utils/make_test_subsets.py --csv <test csv> --out_dir <dir>` writes patient-level subsets and chunks of its test CSV.

## 1. Train the classifier

```
python train.py --image_dir <CT dir> --train_csv <train.csv> --val_csv <val.csv> --backbone EfficientNetV2B3
```

The best epoch by validation loss is written to `<ckpt_dir>/binary classification_<backbone>__<size>_<lr>/best.pth`. Defaults: 512 px (518 px for DINOv2), batch size 16, learning rate 0.001, 40 epochs, seed 51, Adam, BCE, window 40/80 HU.

## 2. Fit the HU mixtures

```
python fit_gmm.py --original_ct_dir <unstripped CT dir> --stripped_ct_dir <skull-stripped CT dir> --masks_dir <masks dir> --train_csv <train.csv> --output gmm_parameters.csv
```

Fits `--n_components` Gaussians (default 2) to each of hemorrhage, skull and background, using training-split pixels only. Hemorrhage pixels come from the mask in 0-100 HU, background from the skull-stripped CT outside the mask in 0-100 HU, and skull from unstripped pixels of at least 200 HU that the stripping removed. The result is written as a CSV with one row per component.

## 3. Explanations and fusion

The protocol runs in two passes.

```
python explain.py --image_dir <CT dir> --masks_dir <masks dir> --test_csv <test.csv> --train_csv <train.csv> --checkpoint <best.pth> --evaluate_optimal_threshold --eval_opt_thresh_on_train
python explain.py --image_dir <CT dir> --masks_dir <masks dir> --test_csv <test.csv> --checkpoint <best.pth> --cam_threshold <selected> --fusion --gmm_params gmm_parameters.csv
```

1. The first pass sweeps the attention threshold on 50 values in [0.1, 0.9], selects the best by pixel Dice and reports the baseline metrics. With `--eval_opt_thresh_on_train` the threshold is searched on the training split and reported on the test split.
2. The second pass applies the selected threshold and reports baseline and `+GMM fusion` metrics. Add `--find_opt_fu_thresh` to sweep the fusion threshold as well.

| Argument | Meaning |
| --- | --- |
| `--backbone`, `--checkpoint`, `--image_size` | `EfficientNetV2B3`, `ResNet18` or `DINOv2ViTS14`, and the trained weights to load |
| `--cam_method`, `--target_layer` | `HiResCAM`, `GradCAM` or `AblationCAM`, and one block index (default -2) |
| `--dino_mode` | For DINOv2: `cam`, `attention` (last-layer CLS-to-patch) or `rollout` |
| `--fusion`, `--fusion_threshold`, `--tau_low`, `--spatial_prior_sigma` | Enable fusion; hysteresis thresholds (defaults 0.5 and 0.35); sigma of the distance-transform skull prior (default 15) |
| `--calc_subtypes_metrics` | Also report metrics per hemorrhage subtype (needs the subtype columns in the CSVs) |
| `--calculate_statistics` | Write per-slice baseline-versus-fusion differences to `<output_dir>/dice_statistics`; the violin scripts read `dice_differences.csv` from there |
| `--save_vis_for_paper` | Write per-panel PDF figures instead of the combined PNG |

Outputs go to `--output_dir` (default `outputs`).

## Figure scripts

`scripts/models_plot.py` and `scripts/legends.py` run directly with `python`. The violin scripts take their input on the command line, for example `python scripts/violin-plots/diff-violin.py --csv outputs/dice_statistics/dice_differences.csv`; `subtype.py` takes `--dice_csv` and `--subtype_csv`.
