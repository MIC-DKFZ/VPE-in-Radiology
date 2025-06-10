#!/usr/bin/env python3
"""
Main entry point for processing radiology images with visual prompts.
This module provides a unified interface for evaluating different
visual language models on radiology datasets using various visual prompting techniques.
"""

import os
import json
import argparse
from pathlib import Path
from typing import Optional, Dict, Any

import pandas as pd
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    roc_auc_score,
    matthews_corrcoef,
    balanced_accuracy_score,
)

from datasets.jsrt.datasets import JSRTDataset
from datasets.padchestgr.datasets import (
    PadChestGRTrainDataset,
    PadChestGRValDataset,
    PadChestGRTestDataset,
)
from datasets.nih14.datasets import NIH14Dataset
from datasets.vindrcxr.datasets import VinDrCXRTrainDataset, VinDrCXRTestDataset

from models import BiomedCLIPModel, BMC_CLIP_CF_Model

DATASETS = {
    "JSRT": JSRTDataset,
    "NIH14": NIH14Dataset,
    "VinDrCXRTrain": VinDrCXRTrainDataset,
    "VinDrCXRTest": VinDrCXRTestDataset,
    "PadChestGRTrain": PadChestGRTrainDataset,
    "PadChestGRVal": PadChestGRValDataset,
    "PadChestGRTest": PadChestGRTestDataset,
}

MODELS = {
    "BiomedCLIPModel": BiomedCLIPModel,
    "BMC_CLIP_CF_Model": BMC_CLIP_CF_Model,
}


def setup_output_directory(
    output_dir: str,
    dataset_name: str,
    model_name: str,
    image_annotation_type: Optional[str] = None,
    color: Optional[str] = None,
    line_width: Optional[int] = None,
    text_annotation_suffix: Optional[str] = None,
) -> Path:
    """
    Set up output directory for experiment results.

    Args:
        base_path: Base path of the project
        dataset_name: Name of the dataset
        model_name: Name of the model
        image_annotation_type: Type of image annotation
        color: Color for annotation
        line_width: Width of annotation line
        text_annotation_suffix: Text suffix for annotation

    Returns:
        Path to the output directory
    """
    output_dir = Path(output_dir)
    experiment_dir = output_dir / dataset_name / model_name.upper()

    if image_annotation_type:
        output_dir = experiment_dir / image_annotation_type
        if image_annotation_type in ["bbox", "circle", "arrow"]:
            if color:
                output_dir = output_dir / color
                if line_width:
                    output_dir = output_dir / str(line_width)

            if text_annotation_suffix is not None:
                # Create a subdirectory for text annotation, sanitize the name
                text_dir = text_annotation_suffix.strip().replace(" ", "_")
                output_dir = output_dir / text_dir
    else:
        output_dir = experiment_dir / "None"

    # Create directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    return output_dir


def process_dataset(
    output_dir: str,
    model_name: str,
    dataset_name: str,
    image_annotation_type: Optional[str] = None,
    color: Optional[str] = None,
    line_width: Optional[int] = None,
    text_annotation_suffix: Optional[str] = None,
    batch_size: int = 16,
) -> Dict[str, Any]:
    """
    Process a dataset with given parameters using the unified dataset loaders.

    Args:
        base_path: Base path of the project
        model_name: Name of the model to use
        dataset_name: Name of the dataset (if None, will be detected from metadata_file)
        image_annotation_type: Type of annotation (bbox, circle, arrow, crop, None)
        color: Color of the annotation
        line_width: Width of the annotation line
        text_annotation_suffix: Text suffix for annotation
        output_dir: Output directory (if None, will be generated)
        batch_size: Batch size for data loading

    Returns:
        Dictionary containing results and metrics
    """
    # Get model instance
    model = MODELS.get(model_name)()

    # Setup output directory
    output_path: Path = setup_output_directory(
        output_dir,
        dataset_name,
        model_name,
        image_annotation_type,
        color,
        line_width,
        text_annotation_suffix,
    )
    print(f"Output directory set to: {output_path}")

    # Create dataset-specific parameters
    dataset_params = {
        "transform": model.preprocess,
        "image_annotation_type": image_annotation_type,
        "color": color,
        "line_width": line_width,
        "text_annotation_suffix": text_annotation_suffix,
    }
    dataset = DATASETS.get(dataset_name)(**dataset_params)

    # Create dataloader
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=4
    )

    # Prepare the model with text prompts
    model.prepare_text_prompts(dataset.generate_text_prompt())

    # Run evaluation
    print(f"Evaluating {dataset_name} with model {model_name}...")
    if image_annotation_type and image_annotation_type != "crop":
        print(
            f"Using image annotation type: {image_annotation_type}, color: {color}, line width: {line_width}, text suffix: '{text_annotation_suffix}'"
        )
    elif image_annotation_type == "crop":
        print("Using image annotation type: crop (no additional parameters needed)")
    else:
        print("No image annotation type specified, using default settings.")

    all_results = []
    all_true_labels = []
    all_pred_labels = []
    all_raw_probs = []

    class_labels = dataset.class_labels

    for image_ids, batch_images, batch_labels in tqdm(
        dataloader, desc="Processing images"
    ):
        # Convert batch labels to class indices
        batch_label_indices = [class_labels.index(label) for label in batch_labels]
        all_true_labels.extend(batch_label_indices)

        # Process batch and get predictions
        batch_probs_list = []
        for image_id, img in zip(image_ids, batch_images):
            probs = model.predict(img)
            batch_probs_list.append(probs)

            # Store highest probability class as prediction
            pred_idx = np.argmax(probs)
            all_pred_labels.append(pred_idx)
            all_raw_probs.append(probs)

            # Store result
            all_results.append(
                {
                    "image_id": image_id,
                    "true_label": class_labels[
                        batch_label_indices[len(batch_probs_list) - 1]
                    ],
                    "predicted_label": class_labels[pred_idx],
                    **{
                        class_labels[i]: float(probs[i])
                        for i in range(len(class_labels))
                    },
                }
            )

    # Calculate metrics
    accuracy = np.mean(np.array(all_true_labels) == np.array(all_pred_labels))
    balanced_acc = balanced_accuracy_score(all_true_labels, all_pred_labels)

    # Calculate AUROC and AUPRC if applicable
    try:
        all_raw_probs = np.array(all_raw_probs)
        true_one_hot = np.zeros((len(all_true_labels), len(class_labels)))
        for i, label_idx in enumerate(all_true_labels):
            true_one_hot[i, label_idx] = 1

        auroc = roc_auc_score(
            true_one_hot, all_raw_probs, average="macro", multi_class="ovr"
        )
        auprc = average_precision_score(true_one_hot, all_raw_probs, average="macro")
        mcc = matthews_corrcoef(all_true_labels, all_pred_labels)
    except Exception as e:
        print(f"Warning: Could not compute AUROC/AUPRC: {e}")
        auroc = None
        auprc = None
        mcc = None

    # Create metrics dictionary
    metrics = {
        "accuracy": float(accuracy),
        "balanced_accuracy": float(balanced_acc) if balanced_acc is not None else None,
        "auroc": float(auroc) if auroc is not None else None,
        "auprc": float(auprc) if auprc is not None else None,
        "mcc": float(mcc) if mcc is not None else None,
        "class_report": classification_report(
            all_true_labels,
            all_pred_labels,
            target_names=class_labels,
            output_dict=True,
        ),
    }

    # Create a dataframe with results
    results_df = pd.DataFrame(all_results)

    # Save results
    metrics_path = output_path / "metrics.json"
    csv_path = output_path / "results.csv"
    metrics_csv_path = output_path / "metrics.csv"

    # Save metrics as JSON
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)

    # Save results as CSV
    results_df.to_csv(csv_path, index=False)

    # Save metrics as CSV for easier comparison
    metrics_flat = {
        "dataset": dataset_name,
        "model": model_name,
        "annotation_type": image_annotation_type or "None",
        "color": color or "None",
        "line_width": line_width or 0,
        "text_suffix": text_annotation_suffix or "None",
        "accuracy": metrics["accuracy"],
        "balanced_accuracy": metrics["balanced_accuracy"],
        "auroc": metrics["auroc"],
        "auprc": metrics["auprc"],
        "mcc": metrics["mcc"],
    }
    pd.DataFrame([metrics_flat]).to_csv(metrics_csv_path, index=False)

    print("Metrics:")
    for key, value in metrics.items():
        if isinstance(value, dict):
            print(f"{key}: {json.dumps(value, indent=2)}")
        else:
            print(f"{key}: {value}")

    print(f"Results saved to {output_path}")

    return {"metrics": metrics, "results": all_results, "output_dir": output_path}


def main():
    parser = argparse.ArgumentParser(description="VPE-VLM-in-Radiology Processor")

    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        choices=list(MODELS.keys()),
        help="Model to use for evaluation",
    )

    parser.add_argument(
        "--dataset_name",
        required=True,
        type=str,
        choices=list(DATASETS.keys()),
        help="Dataset name",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="experiments",
        help="Output directory",
    )

    parser.add_argument(
        "--image_annotation_type",
        type=str,
        default=None,
        choices=["bbox", "circle", "arrow", "crop"],
        help="Type of image annotation",
    )
    parser.add_argument(
        "--color",
        type=str,
        default=None,
        choices=["red", "green", "blue", "yellow", "orange", "white", "black"],
        help="Color for annotations",
    )
    parser.add_argument(
        "--line_width", type=int, default=1, help="Width of lines for annotations"
    )
    parser.add_argument(
        "--text_annotation_suffix",
        type=str,
        default="",
        help="Suffix to add to text descriptions, like ' indicated by a red bounding box'",
    )
    parser.add_argument(
        "--batch_size", type=int, default=16, help="Batch size for data loading"
    )

    # Parse arguments and call process_dataset
    args = parser.parse_args()
    process_dataset(**vars(args))


if __name__ == "__main__":
    main()
