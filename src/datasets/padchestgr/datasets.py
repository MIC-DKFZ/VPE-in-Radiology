import json
from pathlib import Path

import pandas as pd

from datasets.base_dataset import BaseImageDataset


class PadchestDataset(BaseImageDataset):
    """
    Dataset class for BIMCV-Padchest-GR images with annotation capabilities.

    Args:
        transform: Optional transform to apply to images
        image_annotation_type: Type of annotation ('bbox', 'circle', 'arrow', 'crop', None)
        color: Color for annotations (if applicable)
        line_width: Line width for annotations (if applicable)
        text_annotation_suffix: Suffix to append to text annotations
    """

    def __init__(
        self,
        transform=None,
        image_annotation_type=None,
        color=None,
        line_width=None,
        text_annotation_suffix="",
    ):
        super().__init__(
            transform=transform,
            image_annotation_type=image_annotation_type,
            text_annotation_suffix=text_annotation_suffix,
            color=color,
            line_width=line_width,
        )
        grounded_reports_path = Path(self.load_class_config()["grounded_reports"])
        assert (
            grounded_reports_path.exists()
        ), f"Grounded reports file {grounded_reports_path} does not exist"
        with grounded_reports_path.open("r") as f:
            grounded_data = json.load(f)
        self.grounded_dict = {
            entry["ImageID"]: entry for entry in grounded_data if "ImageID" in entry
        }

    @property
    def class_labels(self):
        return sorted(self.df["label_group"].unique().tolist())

    def __getitem__(self, idx):
        """
        Get an image and its label from the dataset.

        Args:
            idx: Index of the item to retrieve

        Returns:
            Tuple of (image_id, image, label)
        """
        row = self.df.iloc[idx]
        image_id = row["ImageID"]
        label_group = row["label_group"]
        label = row["label"]
        image_path = self.images_dir / image_id

        findings_info = self.grounded_dict[image_id]
        image = self.annotate_image(
            image_path=image_path,
            item_info=findings_info,
            label=label,
            annotation_type=self.annotation_type,
            color=self.color,
            line_width=self.line_width,
        )
        # Apply transformation if specified
        if self.transform:
            image = self.transform(image)

        return image_id, image, label_group

    def get_box_coordinates(self, image, image_path, item_info, label):
        """
        Get bounding box coordinates for NIH14 findings.

        Args:
            image: PIL Image with target size
            image_path: Path to the image file
            item_info: Dictionary with annotation information
            label: Label for which to find bounding box

        Returns:
            Tuple (x1, y1, x2, y2) of box coordinates
        """
        if "findings" not in item_info:
            raise ValueError(
                f"No findings in item_info for {item_info.get('ImageID', 'unknown')}"
            )

        # Find the matching finding for this label
        for finding in item_info["findings"]:
            if "labels" in finding and label in finding["labels"]:
                if "boxes" in finding and len(finding["boxes"]) > 0:
                    matched_box = finding["boxes"][0]
                    w, h = image.size

                    # Scale normalized coordinates to image dimensions
                    x1, y1, x2, y2 = matched_box
                    return (int(x1 * w), int(y1 * h), int(x2 * w), int(y2 * h))

        raise ValueError(
            f"No bounding box found for {label} in image {item_info.get('ImageID', 'unknown')}"
        )


class PadChestGRTrainDataset(PadchestDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @classmethod
    def load_metadata(cls, metadata_file):
        """
        Load metadata from a metadata file.

        Args:
            metadata_file: Path to the metadata file containing metadata

        Returns:
            DataFrame with metadata
        """
        df = pd.read_csv(metadata_file)

        # load train_indices.txt from the directory where also this file is located
        train_indices = Path(__file__).parent / "train_indices.txt"
        assert (
            train_indices.exists()
        ), f"Train indices file {train_indices} does not exist"
        train_indices = pd.read_csv(train_indices, header=None)[0].tolist()
        df = df[df.index.isin(train_indices)].reset_index(drop=True)
        return df


class PadChestGRTestDataset(PadchestDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @classmethod
    def load_metadata(cls, metadata_file):
        """
        Load metadata from a metadata file.

        Args:
            metadata_file: Path to the metadata file containing metadata

        Returns:
            DataFrame with metadata
        """
        df = pd.read_csv(metadata_file)

        # load test_indices.txt from the directory where also this file is located
        test_indices = Path(__file__).parent / "test_indices.txt"
        assert (
            test_indices.exists()
        ), f"Test indices file {test_indices} does not exist"
        test_indices = pd.read_csv(test_indices, header=None)[0].tolist()
        df = df[df.index.isin(test_indices)].reset_index(drop=True)
        return df


class PadChestGRValDataset(PadchestDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @classmethod
    def load_metadata(cls, metadata_file):
        """
        Load metadata from a metadata file.

        Args:
            metadata_file: Path to the metadata file containing metadata

        Returns:
            DataFrame with metadata
        """
        df = pd.read_csv(metadata_file)

        # load val_indices.txt from the directory where also this file is located
        val_indices = Path(__file__).parent / "val_indices.txt"
        assert (
            val_indices.exists()
        ), f"Validation indices file {val_indices} does not exist"
        val_indices = pd.read_csv(val_indices, header=None)[0].tolist()
        df = df[df.index.isin(val_indices)].reset_index(drop=True)
        return df