# Banana Instance Segmentation and Health Classification

[![YOLO11](https://img.shields.io/badge/Model-YOLO11m--seg-blue)](https://github.com/ultralytics/ultralytics)
[![Ultralytics](https://img.shields.io/badge/Framework-Ultralytics%208.4.115-7B68EE)](https://github.com/ultralytics/ultralytics)
[![License: AGPL v3](https://img.shields.io/badge/License-AGPL%20v3-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.12.13-blue)](https://www.python.org/)

**Author:** Huzaifa Waqar Butt  
**Final model:** YOLO11m-seg  
**Framework:** Ultralytics 8.4.115  
**Task:** Instance Segmentation  
**Classes:** `healthy_banana`, `unhealthy_banana`  
**Project status:** Finalized and evaluated  

---

## Overview

This project implements a two-class **banana instance-segmentation and health-classification system** using **Ultralytics YOLO11m-seg** and transfer learning.

For every detected banana, the model can provide:

- a predicted class;
- confidence score;
- bounding box;
- pixel-level instance mask.

The two final classes are:

```text
0  healthy_banana
1  unhealthy_banana
```

The released model is specialized for these two banana classes. It is not intended to behave as a general-purpose object detector after fine-tuning.

---

## Project Resources

| Resource | Link |
|---|---|
| GitHub Repository | [banana-instance-segmentation-yolo11](https://github.com/Huzaifa-170504/banana-instance-segmentation-yolo11) |
| Final Model Weights | [Google Drive - Model Files](https://drive.google.com/drive/folders/1ifpUlrHQ-1E-OsR8DP8POm_bxVpTWVjF?usp=drive_link) |
| Prepared Dataset | [Google Drive - Prepared Dataset](https://drive.google.com/drive/folders/1DKcyoLGbVsmpsWNJbIsEsNHLAEFzw_OE?usp=drive_link) |
| Official Ultralytics Repository | [ultralytics/ultralytics](https://github.com/ultralytics/ultralytics) |
| Ultralytics Documentation | [docs.ultralytics.com](https://docs.ultralytics.com/) |

> The Google Drive folders should be shared with **Viewer** access so reviewers can download the model and dataset without being able to modify the source files.

---

## Final Evaluation Results

The final `best.pt` checkpoint was evaluated on the held-out research test split containing:

- **359 test images**
- **2,513 annotated banana instances**

| Metric | Bounding Boxes | Segmentation Masks |
|---|---:|---:|
| Precision | 99.994% | 99.994% |
| Recall | 100.000% | 100.000% |
| mAP@50 | 99.500% | 99.500% |
| mAP@50-95 | 91.190% | 78.962% |

> **Important:** `99.5%` is **mAP@50 on the internal research test split**, not a guaranteed real-world accuracy value.

The stricter mask mAP@50-95 of **78.962%** shows that precise mask localization is more difficult than achieving high detection performance at IoU 0.50.

---

## Final Model

The recommended checkpoint for prediction, validation, export, demonstration, and deployment testing is:

```text
best.pt
```

### Checkpoint Integrity

```text
best.pt SHA-256:
cde8ddf49e2c9cdd8a360a3da82822c24c1ddfbb36cd3667c572222f37286bba

last.pt SHA-256:
de3437ffe0516dde27bed420ea40891001292e087526657b502772ccada389cc
```

### Download Model Weights

[Download / View Model Files on Google Drive](https://drive.google.com/drive/folders/1ifpUlrHQ-1E-OsR8DP8POm_bxVpTWVjF?usp=drive_link)

Use `best.pt` for normal inference and evaluation. `last.pt` represents the state from the final training epoch and is mainly useful for recovery or continued training.

---

## Dataset

The project uses **YOLO instance-segmentation polygon annotations**.

### Dataset Composition

| Dataset Item | Count |
|---|---:|
| Original source images | 8 |
| Augmented image-label pairs | 2,872 |
| Total raw image-label pairs | 2,880 |

### Research Split Used for Final Reporting

| Split | Images |
|---|---:|
| Train | 1,795 |
| Validation | 718 |
| Test | 359 |
| **Total** | **2,872** |

The research split was organized by augmentation family rather than simply performing a random image-level split. This reduced direct leakage between augmentations produced from the same original source image.

### Secondary Benchmark Split

| Split | Images |
|---|---:|
| Train | 2,297 |
| Validation | 287 |
| Test | 288 |

The benchmark split was retained as a secondary conventional comparison. The final reported model uses the research split.

### Download Prepared Dataset

[Download / View Prepared Dataset on Google Drive](https://drive.google.com/drive/folders/1DKcyoLGbVsmpsWNJbIsEsNHLAEFzw_OE?usp=drive_link)

### Dataset Limitation

Although the project contains thousands of augmented examples, those examples originated from only **eight independent source images**.

Augmentation improves variation but does not replace genuinely independent data from different:

- cameras;
- farms and environments;
- lighting conditions;
- backgrounds;
- banana varieties;
- maturity levels;
- disease appearances.

External evaluation on genuinely new images is required before making broad real-world deployment claims.

---

## Development Environment

| Component | Verified Environment |
|---|---|
| Platform | Google Colab with Google Drive |
| Python | 3.12.13 |
| PyTorch | 2.11.0+cu128 |
| CUDA | 12.8 |
| GPU | NVIDIA A100-SXM4-80GB |
| Ultralytics | 8.4.115 |
| Image size | 640 × 640 |
| Random seed | 42 |
| Deterministic mode | Enabled |
| AMP | Enabled |

The original training environment used an NVIDIA A100 GPU. Inference and training performance on other hardware will differ.

---

## Training Configuration

| Parameter | Final Value |
|---|---|
| Model | YOLO11m-seg |
| Task | Instance segmentation |
| Epochs | 100 |
| Batch size | 16 |
| Image size | 640 |
| Workers | 2 |
| Device | CUDA device 0 |
| Optimizer | Auto |
| AMP | True |
| Deterministic | True |
| Seed | 42 |
| Patience | 50 |
| Save period | Every 10 epochs |
| Cache | False |

The final training run completed all **100 epochs** in approximately **63.30 minutes** on an NVIDIA A100-SXM4-80GB GPU.

---

## Project Structure

The complete development project was organized as:

```text
Segmentation_Project/
├── 01_Datasets/
├── 02_Repositories/
│   └── ultralytics/
├── 03_Pretrained_Weights/
│   └── yolo11m-seg.pt
├── 04_Training_Runs/
│   └── banana_segmentation_full_v1_20260805_184146/
├── 05_Logs/
├── 06_Scripts/
├── 07_Results/
├── 08_Visualizations/
├── AUTHOR.txt
├── README.md
└── documentation files
```

For the public GitHub repository, large datasets and externally hosted model assets may be kept outside GitHub and linked through the **Project Resources** section above.

---

## Project Scripts

The finalized project contains 12 Python scripts under `06_Scripts`:

- `initialize_project.py`
- `bootstrap.py`
- `config.py`
- `prepare_dataset.py`
- `verify_environment.py`
- `verify_dataset.py`
- `training_config.py`
- `model_factory.py`
- `experiment_manager.py`
- `trainer.py`
- `logger.py`
- `utils.py`

These scripts cover project initialization, environment checks, dataset preparation, configuration, model loading, experiment management, training, logging, and reproducibility utilities.

---

## Installation

### 1. Clone This Repository

```bash
git clone https://github.com/Huzaifa-170504/banana-instance-segmentation-yolo11.git
cd banana-instance-segmentation-yolo11
```

### 2. Install Dependencies

If the repository contains `requirements.txt`:

```bash
pip install -r requirements.txt
```

A typical environment requires:

```text
ultralytics==8.4.115
torch
torchvision
numpy
opencv-python
PyYAML
pandas
matplotlib
```

Install the PyTorch build that matches your CPU/GPU and CUDA environment.

### 3. Official Ultralytics Source

Official repository:

https://github.com/ultralytics/ultralytics

The project was developed using **Ultralytics 8.4.115**.

---

## Using the Final Model

After downloading `best.pt` from the model link, place it in a convenient local directory such as:

```text
weights/best.pt
```

Then run:

```python
from ultralytics import YOLO

model = YOLO("weights/best.pt")

results = model.predict(
    source="path/to/image_or_folder",
    imgsz=640,
    conf=0.25,
    save=True,
)

for result in results:
    print(result.boxes)
    print(result.masks)
```

The output can include:

- detected banana class;
- confidence score;
- bounding box;
- instance segmentation mask.

---

## Re-evaluate the Test Split

Download the prepared dataset and update the path to its research `dataset.yaml`.

Example:

```python
from ultralytics import YOLO

model = YOLO("weights/best.pt")

metrics = model.val(
    data="path/to/Prepared_Dataset/research/dataset.yaml",
    split="test",
    imgsz=640,
    batch=16,
    device=0,
    plots=True,
)

print(metrics.results_dict)
```

Use an appropriate `device` value for your environment. For CPU evaluation, for example, use:

```python
device="cpu"
```

---

## Results and Evidence

The finalized project stores evidence including:

```text
07_Results/
├── 01_Final_Training/
├── 02_Final_Test_Evaluation/
└── final_metrics.json

05_Logs/
08_Visualizations/
```

Important files include:

- `07_Results/01_Final_Training/results.csv` — epoch-by-epoch losses and metrics;
- `07_Results/01_Final_Training/args.yaml` — final Ultralytics runtime arguments;
- `07_Results/02_Final_Test_Evaluation/test_evaluation.json` — final test metrics;
- `07_Results/final_metrics.json` — clean machine-readable summary;
- `05_Logs/` — project verification and finalization logs;
- `08_Visualizations/` — selected project visualizations.

---

## Reproducibility

The project was developed with reproducibility controls including:

- fixed random seed (`42`);
- deterministic mode enabled;
- saved training arguments;
- exact project configuration;
- checkpoint SHA-256 hashes;
- persistent logs;
- final evaluation reports;
- separate research train/validation/test splits.

The final checkpoint hash is provided above so reviewers can verify that the model they downloaded is the same checkpoint used for the reported evaluation.

---

## Responsible Interpretation

For instance segmentation, no single ordinary classification-style "accuracy" value fully describes performance.

The most useful headline metric for this project is:

```text
Mask mAP@50 = 99.5%
```

However:

- it is measured on the internal research test split;
- it must not be presented as guaranteed 99.5% real-world accuracy;
- mask mAP@50-95 is lower at **78.962%**;
- the source dataset contains only eight independent original images;
- external validation is required for broad deployment claims.

---

## Repository and External Storage

This repository is intended to contain the project code, documentation, results, and reproducibility information.

Large project assets are distributed separately:

### Model Weights

[Google Drive - Model Files](https://drive.google.com/drive/folders/1ifpUlrHQ-1E-OsR8DP8POm_bxVpTWVjF?usp=drive_link)

### Prepared Dataset

[Google Drive - Prepared Dataset](https://drive.google.com/drive/folders/1DKcyoLGbVsmpsWNJbIsEsNHLAEFzw_OE?usp=drive_link)

### Ultralytics Framework

[Official Ultralytics GitHub Repository](https://github.com/ultralytics/ultralytics)

This arrangement keeps the GitHub repository lightweight while still giving reviewers access to the data and final model required for evaluation.

---

## License

This project is released under the **GNU Affero General Public License v3.0 (AGPL-3.0)**.

See the repository [`LICENSE`](LICENSE) file for details.

Ultralytics software and models are also subject to their applicable licensing terms.

---

## Citation

When referring to this project in academic or technical work, use:

```text
Huzaifa Waqar Butt.
Banana Instance Segmentation and Health Classification using YOLO11m-seg.
2026.
GitHub: https://github.com/Huzaifa-170504/banana-instance-segmentation-yolo11
```

---

## Author

**Huzaifa Waqar Butt**

GitHub: [Huzaifa-170504](https://github.com/Huzaifa-170504)

---

## Acknowledgements

This project was built using:

- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)
- [PyTorch](https://pytorch.org/)
- [OpenCV](https://opencv.org/)
- Google Colab
- Google Drive
