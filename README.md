# Human Protein Atlas Image Classification

CS 444 Deep Learning for Computer Vision final project (UIUC) — multi-label
classification of protein subcellular localization from fluorescence microscopy
images.

Team: Sylvey Lin, Zhi-Yi Cao.

## The problem

Each sample is four grayscale channels — nucleus (blue), microtubules (red),
endoplasmic reticulum (yellow), and the protein of interest (green) — labeled
with one or more of 28 localization categories. A protein can sit in several
compartments at once, so this is a multi-label task scored by macro F1, which
punishes models that ignore rare classes. 31,072 samples across 27 cell types;
heavy class imbalance.

## Approach

- **Baselines:** ImageNet-pretrained ResNet18 / ResNet34 (torchvision),
  fine-tuned with `BCEWithLogitsLoss`, AdamW, 30 epochs, batch size 64,
  90/10 train/validation split. Custom head adapted to the 4-channel input.
- **Loss exploration:** implemented a multi-label ArcFace loss (angular margin
  penalty on cosine similarities, `s=30, m=0.5`) to improve separation between
  overlapping localization classes; compared against BCE and focal-loss
  variants. CurricularFace / AdaFace were scoped as next steps.
- **Training:** Google Colab Pro+ (A100). Trained weights are not checked in;
  see the Kaggle dataset below to reproduce.

## Results

| Model | Val macro F1 | Kaggle public | Kaggle private |
|---|---|---|---|
| ResNet18 + BCE | 0.6383 | 0.45702 | 0.43363 |
| ResNet34 + BCE | 0.6258 | 0.44622 | 0.42232 |

Both baselines submitted to the official Kaggle leaderboard, confirming the
end-to-end pipeline.

## What's here

- **`notebooks/ResNet_bce_with_logits.ipynb`** — baseline training: ImageNet-
  pretrained ResNet18/ResNet34 fine-tuned with `BCEWithLogitsLoss` (AdamW,
  30 epochs, batch 64, 90/10 split), dataloader with augmentation, macro-F1
  evaluation. Originally run on Colab.
- **`notebooks/Transformer.ipynb`** — transformer-backbone experiments (Swin /
  ViT via `timm`) with 4-channel input adaptation, building on the ResNet
  training setup. Originally run on Colab.
- **`notebooks/Submission.ipynb`** — inference pipeline: loads the trained
  ResNet18 weights, runs the 4-channel test images through preprocessing
  (resize 512×512, tensor), and writes the Kaggle submission CSV. Originally
  run on Colab; paths point at the original Drive layout.
- **`reports/DLCV_project.pdf`** — full project report (problem, method,
  results, ablations, references).
- **`reports/project_proposal.pdf`** — original project proposal.

## Data

Images: [Human Protein Atlas Image Classification](https://www.kaggle.com/c/human-protein-atlas-image-classification)
(Kaggle). Not included here due to size.

## My role

Two-person team: I authored the project report and collaborated on experiment
design (backbone comparison, loss-function selection); my teammate led code
implementation.
