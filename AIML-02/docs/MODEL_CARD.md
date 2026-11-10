# Model card - chest X-ray finding classifier and hippocampus segmenter

> **FOR EDUCATION ONLY - NOT FOR CLINICAL USE.** These models are student-project demonstrations. They have not been
> clinically validated, are not medical devices and must never inform real patient care.

## 1. Models
| | Chest X-ray classifier | Hippocampus segmenter |
|---|---|---|
| Task | Multi-label: 14 thoracic findings (NIH ChestX-ray14 label set) | 3-D MRI: anterior and posterior hippocampus |
| Architecture | DenseNet-121, ImageNet pre-trained, new 14-way head (PyTorch) | 3-D U-Net, 3 down-sampling levels (MONAI) |
| Input | 128 x 128 grayscale (repeated to 3 channels, ImageNet normalisation) | MRI volume padded/cropped to 48 x 64 x 48 |
| Training | 20,000 images, 4 epochs, AdamW, light augmentation (rotation, scale, shift, brightness/contrast) | 182 volumes, 60 epochs, Dice + cross-entropy loss |
| Version | `densenet121-chestmnist128-v1` | `unet3d_hippocampus.pt` |

## 2. Intended use
Learning how a DICOM -> AI -> explanation -> human review pipeline fits together. Out of scope: diagnosis, triage,
screening, any use on real patients, any use outside a classroom or demo.

## 3. Data
* **Classification:** ChestMNIST from MedMNIST v3 = NIH ChestX-ray14, down-sampled to 128 px. We used the dataset's own
  train/val/test split (sub-sampled to 20,000 / 4,000 / 10,000 images for speed). About 54% of images have no finding;
  finding prevalence ranges from about 0.2% (hernia) to about 18% (infiltration) - see `results/dataset_summary.json`.
* **Segmentation:** Medical Segmentation Decathlon Task04 Hippocampus (260 labelled MRI volumes), randomly split by us
  182 / 39 / 39 (train / validation / test).

## 4. Performance (held-out test data)
* Classifier: **mean AUROC 0.799** over 14 findings (10,000 test images). Best: cardiomegaly 0.898, hernia 0.896,
  effusion 0.870, edema 0.869. Weakest: infiltration 0.680, nodule 0.697, pneumonia 0.722.
* Segmenter: **mean Dice 0.874** (anterior 0.881, posterior 0.867) on 39 test volumes.
* AUROC measures ranking quality, not clinical usefulness. We did not measure sensitivity/specificity at an
  operating point, calibration error, or subgroup performance.

## 5. Limitations
* **Low resolution.** 128 px images lose small findings (nodules, subtle pneumothorax). Real systems use 1000+ px.
* **Noisy labels.** NIH labels were text-mined from radiology reports by an automated tool; they contain errors and do
  not equal a radiologist reading the image.
* **Single source, no external validation.** All chest data come from one US institution. Performance on other
  hospitals, scanners, protocols or populations is unknown and is likely worse.
* **Uncalibrated probabilities.** The scores shown in the viewer are raw model outputs trained with plain binary
  cross-entropy. A "59%" does not mean a 59% chance of disease.
* **Weak findings.** Infiltration and nodule detection were only modestly better than chance-level-plus (AUROC about 0.7).
* **Segmentation scope.** The hippocampus model was trained on a narrow, curated research dataset (cropped volumes around
  the structure) and is not a general brain-MRI tool.

## 6. Bias and fairness
* NIH ChestX-ray14 reflects a single hospital's patient mix; age, sex and race distributions are not controlled and
  were **not analysed** here, so we cannot say how performance varies across groups. Differences across groups are a
  documented problem for chest X-ray models.
* Chest X-ray models are known to use shortcuts such as devices, tubes, markers, text labels and portable-view
  artifacts rather than anatomy. We saw signs of this: in some Grad-CAM examples the heatmap sits on a corner text
  label (e.g. the "PORTABLE" marker area) or a small chest marker instead of lung tissue
  (`results/gradcam_examples.png`, `docs/screenshots/`).
* Class imbalance: rare findings (hernia ~0.2%, pneumonia ~1.3%, fibrosis ~1.4% in training data) have few training
  examples, so their estimates are unstable (hernia AUROC is based on very few positive test images).

## 7. Explanation (Grad-CAM) caveats
Grad-CAM shows regions that increased the model's score for a finding; it does **not** prove the model reasons like a
clinician, and heatmaps are coarse (8 x 8 feature map up-sampled). On low-confidence cases the heatmap can be diffuse or
point outside the lungs. It is a debugging and review aid only.

## 8. Human oversight and logging
Every AI result must be approved or rejected by a human reviewer in the viewer. Each decision is written to an
append-only, hash-chained audit log together with the AI finding and confidence the reviewer saw; the chain can be
verified for tampering (`GET /api/audit`).
