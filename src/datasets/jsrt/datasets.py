import pandas as pd
from pathlib import Path
import numpy as np
from PIL import Image
from datasets.base_dataset import BaseImageDataset


class JSRTDataset(BaseImageDataset):
    """
    Dataset class for JSRT images with annotation capabilities.

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
        return sorted(self.df["malignancy"].unique().tolist())

    def generate_text_prompt(self):
        # get all unique labels from the dataset
        unique_labels = self.class_labels
        # create a text prompt for the dataset
        prompts = [
            f"A chest X-ray with a {l} lung nodule{self.text_annotation_suffix}."
            for l in unique_labels
        ]
        print(f"Generated {len(prompts)} text prompts for JSRT dataset: {prompts}")
        return prompts

    def __getitem__(self, idx):
        """
        Get an image and its label from the dataset.

        Args:
            idx: Index of the item to retrieve

        Returns:
            Tuple of (image_id, image, label)
        """
        row = self.df.iloc[idx]
        image_id = row["image_filename"]
        label = row["malignancy"]

        # Get raw image and its corresponding field for nodule locations
        image_path = self.images_dir / Path(image_id)

        # Get nodule information for annotation
        item_info = {
            "x": row["x_coordinate"],
            "y": row["y_coordinate"],
            "nodule_size_mm": row["nodule_size_mm"],
        }

        # Annotate the image using the base class method
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

        # Return image and label by default
        return image_id, image, label

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
        # Get image dimensions
        orig_image = self.load_image(image_path, resize=None)
        w, h = orig_image.size
        t_h, t_w = image.size
        w_ratio, h_ratio = w / t_w, h / t_h

        # Extract nodule position and size
        x = item_info["x"]
        y = item_info["y"]
        nodule_size_mm = item_info["nodule_size_mm"]

        nodule_size_pixels = 5 * int(nodule_size_mm / 0.175)
        orig_x1, orig_y1, orig_x2, orig_y2 = (
            x - nodule_size_pixels,
            y - nodule_size_pixels,
            x + nodule_size_pixels,
            y + nodule_size_pixels,
        )

        # scale box coordinates to target_image size
        x1 = int(orig_x1 / w_ratio)
        y1 = int(orig_y1 / h_ratio)
        x2 = int(orig_x2 / w_ratio)
        y2 = int(orig_y2 / h_ratio)

        return (x1, y1, x2, y2)

    @classmethod
    def load_metadata(cls, metadata_file):
        """
        Loads and processes JSRT metadata from metadata file.

        Args:
            metadata_file: Path to CSV metadata file

        Returns:
            DataFrame with metadata information
        """
        try:
            columns = [
                "image_filename",
                "nodule_size_mm",
                "degree_of_subtlety",
                "age",
                "sex",
                "x_coordinate",
                "y_coordinate",
                "malignancy",
                "anatomic_location",
                "diagnosis"
            ]
            df = pd.read_csv(
                metadata_file, sep="\t", header=None, names=columns, index_col=False
            )
            return df
        except Exception as e:
            print(f"Error loading JSRT metadata: {e}")
            return None

    def load_image(self, image_path, resize=256):
        def adjust_window_level(image, window_center=2048, window_width=4096):
            min_val = window_center - window_width // 2
            max_val = window_center + window_width // 2
            image = np.clip(image, min_val, max_val)
            image = (image - min_val) / (max_val - min_val) * 255.0
            image = np.clip(image, 0, 255).astype(np.uint8)
            return image

        def invert_image(image):
            return 255 - image

        raw_image = np.fromfile(image_path, dtype=">i2").reshape((2048, 2048))
        image = adjust_window_level(raw_image)
        image = np.array(invert_image(image))
        # Normalize to [0, 1]
        image = (image - image.min()) / (image.max() - image.min())
        image = Image.fromarray((image * 255).astype(np.uint8)).convert("RGB")
        if resize:
            image = image.resize((resize, resize))
        return image
