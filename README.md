# Visual Prompting Evaluation for Vision-Language Models in Radiology

This repository contains the implementation for our MIDL 2025 paper ["Visual Prompt Engineering for Vision Language Models in Radiology"](https://openreview.net/pdf?id=0Jn1d4gYRS).

## Overview

This project aims to evaluate how different visual prompting techniques affect the performance of vision-language models (VLMs) in the domain of radiology image analysis. We assess the impact of different visual annotations (bounding boxes, circles, arrows, and crops) on model performance across several common chest X-ray datasets. We demonstrate, that visual markers, particularly a red circle, improve AUROC by up to 0.185. 

## Features

- Evaluation of multiple vision-language models (BiomedCLIP, BMC_CLIP_CF) on radiology datasets
- Implementation of various visual prompting techniques:
  - Bounding boxes
  - Circles
  - Arrows
  - Region cropping
- Support for multiple radiology datasets:
  - PadChest-GR
  - VinDr-CXR
  - NIH14 (ChestX-ray14)
  - JSRT (Japanese Society of Radiological Technology)

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/MIC-DKFZ/VPE-in-Radiology.git
   cd VPE-in-Radiology
   ```

2. Make sure you have Python 3.12 installed:
   ```bash
   python --version
   ```

3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

4. Download the required datasets:
   - **PadChest-GR**: [Download from BIMCV website](https://bimcv.cipf.es/bimcv-projects/padchest-gr/) - Requires registration for the grounded reports version
   - **VinDr-CXR**: [Download from PhysioNet](https://physionet.org/content/vindr-cxr/1.0.0/) - Requires PhysioNet credentialed access
   - **NIH ChestX-ray14**: [Download from NIH](https://nihcc.app.box.com/v/ChestXray-NIHCC/folder/36938765345)
   - **JSRT Dataset**: [Download from JSRT website](http://db.jsrt.or.jp/eng.php) - Requires registration

5. Configure dataset paths in `default_config.json` to match your local environment. Update the following fields for each dataset:
   ```json
   {
     "PadChestGRTrainDataset": {
       "metadata_file": "/path/to/padchest/metadata.csv",
       "images_dir": "/path/to/padchest/images",
       "grounded_reports": "/path/to/padchest/grounded_reports.json"
     },
     "VinDrCXRTrainDataset": {
       "metadata_file": "/path/to/vindrcxr/annotations_train.csv",
       "images_dir": "/path/to/vindrcxr/pngs/train"
     }
   }
   ```

   Alternatively, you can create a custom config file (e.g., `custom_config.json`) and specify it using the `CONFIG` environment variable:
   ```bash
   export CONFIG=/path/to/custom_config.json
   python src/process.py --model_name BiomedCLIPModel --dataset_name PadChestGRTrain
   ```

## Usage

### Basic Usage

Run the core processing script with different parameters:

```bash
python src/process.py --model_name BiomedCLIPModel --dataset_name PadChestGRTrain
```

### Visual Prompting Options

Add visual annotations to images:

```bash
# Add bounding box
python src/process.py --model_name BiomedCLIPModel --dataset_name PadChestGRTrain --image_annotation_type bbox --color red

# Add circle
python src/process.py --model_name BiomedCLIPModel --dataset_name PadChestGRTrain --image_annotation_type circle --color red

# Add arrow
python src/process.py --model_name BiomedCLIPModel --dataset_name PadChestGRTrain --image_annotation_type arrow --color red

# Use cropping
python src/process.py --model_name BiomedCLIPModel --dataset_name PadChestGRTrain --image_annotation_type crop
```

### Text Annotation

You can add text annotation suffixes that describe the visual prompts:

```bash
python src/process.py --model_name BiomedCLIPModel --dataset_name PadChestGRTrain --image_annotation_type bbox --color red --text_annotation_suffix ' indicated by a red bounding box'
```

### Reproducing Our Experiments

To reproduce all experiments from our paper, run:

```bash
bash reproduce_paper.sh
```

## Project Structure

- `src/`: Core source code
  - `models.py`: Implementation of vision-language models
  - `process.py`: Main processing script for running experiments
  - `datasets/`: Dataset-specific implementations
    - `base_dataloader.py`: Base class for all datasets
    - `padchestgr/`, `vindrcxr/`, `nih14/`, `jsrt/`: Dataset-specific modules
  - `utils/`: Utility functions
- `experiments/`: Output directory for experiment results
- `default_config.json`: Configuration file for dataset paths
- `requirements.txt`: Required Python packages
- `reproduce_paper.sh`: Script to reproduce all experiments from the paper

## Results and Output

Experiment results are saved under the `experiments/` directory with the following structure:

```
experiments/
    {DATASET_NAME}/
        {MODEL_NAME}/
            {ANNOTATION_TYPE}/
                {COLOR}/
                    {LINE_WIDTH}/
                        {TEXT_ANNOTATION}/
                            metrics.csv
                            metrics.json
                            results.csv
```

Each experiment directory contains:
- `metrics.json`: Detailed performance metrics
- `metrics.csv`: Summary of performance metrics in CSV format
- `results.csv`: Raw prediction results for each image

## Citation

If you use this code or our findings in your research, please cite our paper:

[Paper Link](https://openreview.net/pdf?id=0Jn1d4gYRS)

```bibtex
@inproceedings{denner2025visual,
  title={Visual Prompt Engineering for Vision Language Models in Radiology},
  author={Denner, Stefan and Bujotzek, Markus Ralf and Bounias, Dimitrios and Zimmerer, David and Stock, Raphael and Maier-Hein, Klaus},
  booktitle={Medical Imaging with Deep Learning},
  year={2025}
}
```