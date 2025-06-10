from pathlib import Path
import pandas as pd

from datasets.base_dataset import BaseImageDataset


class VinDrCXRDataset(BaseImageDataset):
    """
    Dataset class for VinDrCXR images

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

    @property
    def class_labels(self):
        """
        Get unique class labels from the dataset.

        Returns:
            List of unique class labels
        """
        return sorted(self.df["class_name"].unique().tolist())

    @classmethod
    def load_metadata(cls, metadata_file):
        """
        Loads and processes JSRT metadata from metadata file.

        Args:
            metadata_file: Path to CSV metadata file

        Returns:
            DataFrame with metadata information
        """
        df = pd.read_csv(metadata_file)

        # remove all 'No finding' class_name
        df = df[df["class_name"] != "No finding"]

        grouping_criteria = ["image_id", "class_name"]

        if "rad_id" in df.columns:
            grouping_criteria.append("rad_id")

        # Step 1: Count occurrences of each class_name per image_id
        class_counts = df.groupby(grouping_criteria).size().reset_index(name="count")

        # Step 2: Identify image_ids where any class_name appears more than once
        images_to_exclude = class_counts[class_counts["count"] > 1]["image_id"].unique()

        # Step 3: Filter out rows with these image_ids
        df = df[~df["image_id"].isin(images_to_exclude)].reset_index(drop=True)
        return df

    def __getitem__(self, idx):
        """
        Get an image and its label from the dataset.

        Args:
            idx: Index of the item to retrieve

        Returns:
            Tuple of (image_id, image, label)
        """
        row = self.df.iloc[idx]
        image_id = row["image_id"]
        label = row["class_name"]

        image_path = self.images_dir / f"{image_id}.png"  

        item_info = {
            "x_min": row["x_min"],
            "y_min": row["y_min"],
            "x_max": row["x_max"],
            "y_max": row["y_max"],
        }

        image = self.annotate_image(
            image_path=image_path,
            item_info=item_info,
            label=label,
            annotation_type=self.annotation_type,
            color=self.color,
            line_width=self.line_width,
        )

        # Apply transformation if provided
        if self.transform:
            image = self.transform(image)

        return image_id, image, label

    def get_box_coordinates(self, image, image_path, item_info, label):
        """
        Get bounding box coordinates for VinDrCXR findings.

        Args:
            image: PIL Image with target size
            image_path: Path to the image file
            item_info: Dictionary with annotation information
            label: Label for which to find bounding box

        Returns:
            Tuple (x1, y1, x2, y2) of box coordinates
        """
        orig_image = self.load_image(image_path, resize=None)
        w, h = orig_image.size
        t_h, t_w = image.size
        w_ratio, h_ratio = w / t_w, h / t_h

        x_min, y_min = item_info["x_min"], item_info["y_min"]
        x_max, y_max = item_info["x_max"], item_info["y_max"]

        x1, y1, x2, y2 = (
            int(x_min / w_ratio),
            int(y_min / h_ratio),
            int(x_max / w_ratio),
            int(y_max / h_ratio),
        )
        return (x1, y1, x2, y2)


class VinDrCXRTrainDataset(VinDrCXRDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @classmethod
    def load_metadata(cls, metadata_file):
        """
        Loads and processes VinDrCXR training metadata from metadata file.

        Args:
            metadata_file: Path to CSV metadata file
        Returns:
            DataFrame with metadata information
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


class VinDrCXRTestDataset(VinDrCXRDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @classmethod
    def load_metadata(cls, metadata_file):
        """
        Loads and processes VinDrCXR test metadata from metadata file.

        Args:
            metadata_file: Path to CSV metadata file
        Returns:
            DataFrame with metadata information
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
