# Multi-Scale Hierarchical Graph Fusion for Skin Lesion Diagnosis  
Author: Pham Le Thanh Nhan

--------------------------------------------------------------------------------

TABLE OF CONTENTS

- Project Overview
- Installation
- How to Run
  - Train & Test with Each Backbone
  - Expected Outputs
- File Descriptions
- Troubleshooting
- Citation & Contact

--------------------------------------------------------------------------------

PROJECT OVERVIEW

This repository implements a Multi-Scale Hierarchical Graph Fusion (MHGF) framework for skin lesion classification using the PAD-UFES-20 dataset.  
The model fuses multi-scale CNN features and patient metadata through a hierarchical graph module.  
Four popular CNN backbones are supported: DenseNet201, ResNet50, ConvNeXt-Tiny, EfficientNetB0.

Main features:
- Modular, extensible codebase.
- Stratified 5-fold training, checkpointing, and robust evaluation.
- Easy switching between backbone architectures.
- Automatically handles class imbalance.
- Clean outputs for reporting and thesis writing.

--------------------------------------------------------------------------------

INSTALLATION

1. Python Version:
- Recommended: Python 3.8 or later

2. Install dependencies:
Make sure PyTorch is installed for your hardware.

    pip install -r requirements.txt

- If using a GPU, follow official PyTorch instructions to install CUDA-enabled torch and torchvision.

--------------------------------------------------------------------------------

HOW TO RUN

Train & Test with Each Backbone

Run any of the following for 5-fold cross-validation training + test set evaluation:

    python run_dense201.py         # DenseNet201 backbone
    python run_resnet50.py         # ResNet50 backbone
    python run_convnext_tiny.py    # ConvNeXt-Tiny backbone
    python run_efficientnetb0.py   # EfficientNetB0 backbone

- Each script will save checkpoints and results to its respective subfolder in outputs/.
- Training may take several hours depending on your GPU/CPU and dataset size.
- The code automatically detects GPU if available.

Expected Outputs

After successful runs, you will see results like:

outputs/
 |-- padufes20_hf1/
 |     |-- best_model_fold1.pth
 |     |-- ...
 |     |-- classification_report_fold1.json
 |     |-- test_metrics_summary.csv
 |-- mhgf2/ ...
- best_model_foldX.pth : model checkpoint for each fold
- classification_report_foldX.json : detailed sklearn report for each fold
- test_metrics_summary.csv : summary of all test metrics (for easy reporting)

--------------------------------------------------------------------------------

FILE DESCRIPTIONS

run_main_template.py      : Generic training/evaluation logic, imported by all runner scripts.
run_dense201.py           : Run with DenseNet201 backbone
run_resnet50.py           : Run with ResNet50 backbone
run_convnext_tiny.py      : Run with ConvNeXt-Tiny backbone
run_efficientnetb0.py     : Run with EfficientNetB0 backbone

src/
- dataset.py              : Custom PyTorch dataset loader for images + metadata
- fusion_modules.py       : Contains metadata encoder, graph fusion, pooling modules
- model_dense201.py       : DenseNet201 multi-scale model definition
- model_resnet50.py       : ResNet50 multi-scale model definition
- model_convnext_tiny.py  : ConvNeXt-Tiny multi-scale model definition
- model_efficientnetb0.py : EfficientNetB0 multi-scale model definition

--------------------------------------------------------------------------------

TROUBLESHOOTING

- Out-of-memory (OOM) error: Reduce batch size in run_main_template.py (change batch_size=32 to 16 or 8).
- Missing CUDA: The code will run on CPU if CUDA is not available, but will be slower.
- Data shape errors: Make sure CSV files contain all the columns listed in cols and images are in RGB format.
- Module not found: Check your project structure matches the tree above.

--------------------------------------------------------------------------------

CONTACT

Pham Le Thanh Nhan
Email: pltnhan07@gmail.com
